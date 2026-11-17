"""Network-guided MCTS with chance nodes for Cheqqers (track 33, docs/specs/cheqqers_gpu_mcts.md).

The tree lives in C++ (native/cheqqers/mcts.h): decision nodes with the network's move priors (PUCT), chance nodes
with every measurement outcome and its probability (expectation over outcomes). Python only runs the network:
`cq_mcts_collect` returns a batch of encoded positions and their legal moves, the network (torch, CUDA if available)
gives values and priors, `cq_mcts_apply` backs them up. No Python loop over tree nodes.

    from search.gpu_mcts_cheqqers import MctsPlayer
    player = MctsPlayer("model.pt")                       # CUDA if available
    res = player.search(env, legal_moves, time_ms=2200)   # res.move, res.sims, res.evals, res.visits
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.cheqqers_policy_value import FLAT_IN, load_policy_value, split_flat  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402

MAX_MOVES = 256


class MctsNet:
    """The policy-value network as the tree's evaluator: values = P(win) - P(loss) for the side to move, priors =
    softmax of the move logits (0 at padding)."""

    def __init__(self, model: Any, device: Optional[str] = None, half: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_policy_value(model, device) if not isinstance(model, torch.nn.Module) else model.to(device)
        self.model.eval()
        self.half = half and device.startswith("cuda")

    CHUNK = 1024  # rows per forward pass: bounds the move head's (rows x moves x 3 x d) gather

    @torch.no_grad()
    def launch(self, x: np.ndarray, mv: np.ndarray, lens: np.ndarray):
        """Queue the forward pass on the GPU and return without waiting (finish() collects): the caller can run the
        CPU side of another batch meanwhile. The inputs are copied before this returns."""
        dev = self.device
        outs = []
        for i in range(0, x.shape[0], self.CHUNK):
            xt = torch.from_numpy(np.ascontiguousarray(x[i:i + self.CHUNK])).to(dev)
            mt = torch.from_numpy(np.ascontiguousarray(mv[i:i + self.CHUNK])).to(dev).long()
            lt = torch.from_numpy(np.ascontiguousarray(lens[i:i + self.CHUNK])).to(dev).long()
            sq, gl = split_flat(xt)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                logits, vlog = self.model(sq, gl, mt, lt)
            pri = torch.nan_to_num(torch.softmax(logits.float(), -1), nan=0.0)
            wdl = torch.softmax(vlog.float(), -1)
            outs.append((wdl[:, 0] - wdl[:, 2], pri))
        return outs

    @staticmethod
    def finish(outs) -> tuple[np.ndarray, np.ndarray]:
        vals = np.concatenate([v.cpu().numpy() for v, _ in outs]).astype(np.float32)
        pri = np.concatenate([p.cpu().numpy() for _, p in outs]).astype(np.float32)
        return vals, pri

    @torch.no_grad()
    def __call__(self, x: np.ndarray, mv: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[0] > self.CHUNK:
            parts = [self(x[i:i + self.CHUNK], mv[i:i + self.CHUNK], lens[i:i + self.CHUNK])
                     for i in range(0, x.shape[0], self.CHUNK)]
            return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        dev = self.device
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        mt = torch.from_numpy(mv).to(dev, non_blocking=True).long()
        lt = torch.from_numpy(lens).to(dev, non_blocking=True).long()
        sq, gl = split_flat(xt)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            logits, vlog = self.model(sq, gl, mt, lt)
        pri = torch.softmax(logits.float(), -1)
        pri = torch.nan_to_num(pri, nan=0.0)
        wdl = torch.softmax(vlog.float(), -1)
        val = wdl[:, 0] - wdl[:, 2]
        return val.cpu().numpy().astype(np.float32), pri.cpu().numpy().astype(np.float32)


class GraphNet(MctsNet):
    """MctsNet replayed as CUDA graphs: one graph per move-count bucket (16, 32, ..., 256) at a fixed batch size, rows
    padded (padding rows have no move and are dropped). Removes the per-call kernel-launch cost, which dominates at
    the batch sizes the tree produces (tens of rows)."""

    BUCKETS = (16, 32, 64, 128, 256)

    def __init__(self, model: Any, batch: int, device: Optional[str] = None, half: bool = True):
        super().__init__(model, device, half)
        if not self.device.startswith("cuda"):
            raise RuntimeError("GraphNet needs CUDA")
        self.B = batch
        self.graphs: dict[int, tuple] = {}

    def _graph(self, m: int):
        if m in self.graphs:
            return self.graphs[m]
        dev = self.device
        x = torch.zeros(self.B, FLAT_IN, device=dev)
        mv = torch.zeros(self.B, m, 3, dtype=torch.long, device=dev)
        ln = torch.zeros(self.B, m, dtype=torch.long, device=dev)
        ln[:, 0] = 2

        def run():
            sq, gl = split_flat(x)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                logits, vlog = self.model(sq, gl, mv, ln)
            pri = torch.nan_to_num(torch.softmax(logits.float(), -1), nan=0.0)
            wdl = torch.softmax(vlog.float(), -1)
            return wdl[:, 0] - wdl[:, 2], pri

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            val, pri = run()
        self.graphs[m] = (g, x, mv, ln, val, pri)
        return self.graphs[m]

    @torch.no_grad()
    def __call__(self, x: np.ndarray, mv: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, m0 = x.shape[0], mv.shape[1]
        if n > self.B:
            return super().__call__(x, mv, lens)
        m = next(b for b in self.BUCKETS if b >= m0)
        g, gx, gmv, gln, gval, gpri = self._graph(m)
        gx[:n].copy_(torch.from_numpy(x), non_blocking=True)
        gmv[:n, :m0].copy_(torch.from_numpy(mv), non_blocking=True)
        gln[:n, :m0].copy_(torch.from_numpy(lens), non_blocking=True)
        if m0 < m:
            gln[:n, m0:] = 0
        gln[n:] = 0
        gln[n:, 0] = 2  # padding rows: one dummy move (no all-masked row)
        g.replay()
        val = gval[:n].cpu().numpy().astype(np.float32)
        pri = gpri[:n, :m0].cpu().numpy().astype(np.float32)
        return val, pri


@dataclass
class MctsResult:
    move: Any
    sims: int = 0
    evals: int = 0
    batches: int = 0
    collisions: int = 0
    nodes: int = 0
    max_depth: int = 0
    root_value: float = 0.0  # side to move's view
    elapsed_ms: float = 0.0
    visits: list[tuple[list[int], int, float, float]] = field(default_factory=list)  # move, visits, q (mover), prior
    policy: list[float] = field(default_factory=list)  # Gumbel: the policy target over the moves of `visits`
    gumbel: list[float] = field(default_factory=list)  # Gumbel: the root's Gumbel draws
    candidates: list[int] = field(default_factory=list)  # Gumbel: the root candidates left (indices)
    root_vnet: float = 0.0  # the network's value of the root, side to move's view
    timed_out: int = 0  # 1: the deadline passed while waiting for the network (that batch was discarded)


def _opts(c_puct: float, fpu: float, dir_alpha: float, dir_frac: float, seed: int, max_nodes: int,
          gumbel_m: int = 0, gumbel_n: int = 64, c_visit: float = 50.0, c_scale: float = 1.0) -> nc.MctsOpts:
    return nc.MctsOpts(c_puct=c_puct, fpu_reduction=fpu, dirichlet_alpha=dir_alpha, dirichlet_frac=dir_frac,
                       seed=seed & 0xFFFFFFFFFFFFFFFF, max_nodes=max_nodes, max_moves=MAX_MOVES, gumbel_m=gumbel_m,
                       gumbel_n=gumbel_n, c_visit=c_visit, c_scale=c_scale)


class MctsPlayer:
    def __init__(self, model: Any, device: Optional[str] = None, *, batch: int = 128, c_puct: float = 1.5,
                 fpu: float = 0.25, max_nodes: int = 1_500_000, half: bool = True, graphs: bool = True):
        self.lib = nc.load()
        if not hasattr(self.lib, "cq_mcts_new"):
            raise nc.NativeUnavailable(f"{nc.LIB_PATH} has no MCTS (rebuild it)")
        # model: a model.pt path, a CheqqersPolicyValue, or any callable (x, mv, lens) -> (values, priors)
        plain = callable(model) and not isinstance(model, (str, Path, torch.nn.Module))
        if plain:
            self.net = model
        else:
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            use_graphs = graphs and dev.startswith("cuda")
            self.net = GraphNet(model, batch + 64, dev, half) if use_graphs else MctsNet(model, dev, half)
        self.batch = batch
        self.c_puct, self.fpu, self.max_nodes = c_puct, fpu, max_nodes
        cap = batch + 64
        self.cap = cap
        self._freeing = None  # queue of the background thread that frees finished trees
        self._evaluator = None  # one persistent thread runs the network: a search can stop waiting at its deadline
        self._pending = None  # an abandoned evaluation (deadline passed): the next one queues behind it
        self.latency_ema: Optional[float] = None  # seconds per batch, across searches (warm-up seeds it)
        self.LAUNCH_MARGIN, self.LAUNCH_FACTOR, self.HARD_FRAC = 0.03, 1.5, 0.95
        self.x = np.zeros((cap, FLAT_IN), dtype=np.float32)
        self.mv = np.zeros((cap, MAX_MOVES, 3), dtype=np.int32)
        self.lens = np.zeros((cap, MAX_MOVES), dtype=np.int32)

    def _release(self, h) -> None:
        """Free a finished tree on ONE persistent background thread. Not a thread per tree: the engine keeps a pool of
        recycled positions per thread (up to 16,384, ~230 MB) that a finished thread would leak."""
        import queue
        import threading

        if self._freeing is None:
            q: "queue.Queue" = queue.Queue()

            def worker() -> None:
                while True:
                    self.lib.cq_mcts_free(q.get())
                    q.task_done()

            threading.Thread(target=worker, daemon=True, name="mcts-free").start()
            self._freeing = q
        if self._freeing.qsize() >= 2:  # memory stays bounded; normally the move does not wait for a free
            self._freeing.join()
        self._freeing.put(h)

    def _evaluate(self, n: int, m: int, hard_deadline: float):
        """Run the network on rows [:n] on the evaluator thread; wait at most until hard_deadline. Returns
        (values, priors) or None if the deadline passed (the batch finishes in the background and is discarded).
        Inputs are copied: an abandoned evaluation never reads buffers the next collect writes."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

        if self._evaluator is None:
            self._evaluator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcts-eval")
        x, mv, ln = self.x[:n].copy(), np.ascontiguousarray(self.mv[:n, :m]), np.ascontiguousarray(self.lens[:n, :m])
        t0 = time.perf_counter()
        fut = self._evaluator.submit(self.net, x, mv, ln)  # MctsNet / GraphNet calls are no_grad themselves
        try:
            vals, pri = fut.result(timeout=max(0.0, hard_deadline - t0))
        except FutTimeout:
            self._pending = fut
            return None
        dt = time.perf_counter() - t0
        self.latency_ema = dt if self.latency_ema is None else 0.7 * self.latency_ema + 0.3 * dt
        return vals, pri

    def _ptr(self, a: np.ndarray, t):
        return a.ctypes.data_as(ctypes.POINTER(t))

    def search(self, env: Any, legal_moves: Optional[Sequence[Any]] = None, time_ms: float = 2200.0,
               max_sims: int = 0, seed: int = 1, dirichlet_alpha: float = 0.0, dirichlet_frac: float = 0.25,
               deadline_frac: float = 0.9, gumbel_m: int = 0, gumbel_n: int = 64, c_visit: float = 50.0,
               c_scale: float = 1.0) -> MctsResult:
        """MCTS from `env` (OfficialCheqqersEnv or NativeEnv) among `legal_moves` (default: all) for time_ms *
        deadline_frac, or until max_sims simulations (> 0). The move: most visits, ties to the higher prior."""
        started = time.perf_counter()
        native = env if isinstance(env, nc.NativeEnv) else nc.NativeEnv.from_env(env)
        legal = [list(map(int, m)) for m in (legal_moves if legal_moves is not None else native.legal_moves())]
        if len(legal) <= 1:
            return MctsResult(move=(legal_moves[0] if legal_moves else (legal[0] if legal else None)))
        flat = []
        for m in legal:
            flat += [len(m)] + m + [0] * (3 - len(m))
        err = nc._err()
        h = self.lib.cq_mcts_new(native._h, nc._ints(flat), len(legal),
                                 ctypes.byref(_opts(self.c_puct, self.fpu, dirichlet_alpha, dirichlet_frac, seed,
                                                    self.max_nodes, gumbel_m, gumbel_n, c_visit, c_scale)), err,
                                 nc._ERR)
        if not h:
            nc._raise(err, "mcts")
        h = ctypes.c_void_p(h)
        deadline = started + time_ms * deadline_frac / 1000.0
        hard_deadline = started + time_ms * self.HARD_FRAC / 1000.0  # no wait for the network goes past this
        timed_out = 0
        batches = 0
        last_sims = -1
        mmax = ctypes.c_int32()
        fp, ip = ctypes.c_float, ctypes.c_int32
        try:
            recent: list[float] = []  # the last batches' durations: no batch is started that would end past the deadline
            while True:
                t_batch = time.perf_counter()
                n = self.lib.cq_mcts_collect(h, self.batch, self.cap, self._ptr(self.x, fp), self._ptr(self.mv, ip),
                                             self._ptr(self.lens, ip), ctypes.byref(mmax), err, nc._ERR)
                if n < 0:
                    nc._raise(err, "mcts collect")
                if n == 0 and gumbel_m > 0 and self.lib.cq_mcts_done(h):
                    break
                if n == 0:
                    # no position for the network: the simulations ended on exact values or on pending paths.
                    # Exhausted only if none was backed up (every line decided, or the node cap)
                    st = self._stats(h, 0)[1]
                    if st.sims == last_sims or time.perf_counter() >= deadline or (max_sims and st.sims >= max_sims):
                        break
                    last_sims = st.sims
                    continue
                m = max(1, mmax.value)
                now = time.perf_counter()
                # no batch is launched that is expected to end past the deadline (latency EMA across searches,
                # seeded by the warm-up); the wait itself is capped at the hard deadline (slow or busy GPU)
                if self.latency_ema is not None and \
                        now + self.LAUNCH_FACTOR * self.latency_ema + self.LAUNCH_MARGIN >= deadline:
                    break
                res = self._evaluate(n, m, hard_deadline)
                if res is None:  # deadline passed while waiting: the move from the statistics so far
                    timed_out += 1
                    break
                vals, pri = res
                vals = np.ascontiguousarray(vals, dtype=np.float32)
                pri = np.ascontiguousarray(pri, dtype=np.float32)
                if self.lib.cq_mcts_apply(h, self._ptr(vals, fp), self._ptr(pri, fp), m, n, err, nc._ERR) < 0:
                    nc._raise(err, "mcts apply")
                batches += 1
                st = self._stats(h, 0)[1]
                last_sims = st.sims
                now = time.perf_counter()
                recent = (recent + [now - t_batch])[-3:]
                if now + max(recent) >= deadline or (max_sims and st.sims >= max_sims):
                    break
            k = len(legal)
            moves = np.zeros(4 * k, dtype=np.int32)
            vis = np.zeros(k, dtype=np.int32)
            q = np.zeros(k, dtype=np.float64)
            pr = np.zeros(k, dtype=np.float64)
            st = nc.MctsStats()
            self.lib.cq_mcts_root(h, self._ptr(moves, ip), self._ptr(vis, ip), self._ptr(q, ctypes.c_double),
                                  self._ptr(pr, ctypes.c_double), k, ctypes.byref(st))
            gpol, gdraw, gcand, gchosen = [], [], [], -1
            if gumbel_m > 0:
                pol = np.zeros(k)
                gd = np.zeros(k)
                cm = np.zeros(k, dtype=np.int32)
                gchosen = self.lib.cq_mcts_gumbel(h, self._ptr(pol, ctypes.c_double), self._ptr(gd, ctypes.c_double),
                                                  self._ptr(cm, ip), k)
                gpol, gdraw, gcand = pol.tolist(), gd.tolist(), [i for i in range(k) if cm[i]]
        finally:
            # destroying a tree of ~1e5 nodes (each with its position) takes ~0.1 s: done by a background thread
            # (ctypes releases the GIL), so the move is not late; the next search waits for it only if it is still busy
            self._release(h)
        sign = 1.0 if st.root_turn == 0 else -1.0
        rows = []
        for i in range(k):
            mvl = list(moves[4 * i + 1: 4 * i + 1 + moves[4 * i]])
            rows.append(([int(v) for v in mvl], int(vis[i]), sign * float(q[i]) if not math.isnan(q[i]) else float("nan"),
                         float(pr[i])))
        best = gchosen if gumbel_m > 0 and gchosen >= 0 else max(range(k), key=lambda i: (rows[i][1], rows[i][3], -i))
        key = rows[best][0]
        chosen = next((m for m in (legal_moves if legal_moves is not None else legal) if list(map(int, m)) == key),
                      key)
        return MctsResult(move=chosen, sims=int(st.sims), evals=int(st.evals), batches=batches,
                          collisions=int(st.collisions), nodes=int(st.nodes), max_depth=int(st.max_depth),
                          root_value=sign * float(st.root_value),
                          elapsed_ms=(time.perf_counter() - started) * 1000.0, visits=rows, policy=gpol,
                          gumbel=gdraw, candidates=gcand, root_vnet=sign * float(st.root_vnet), timed_out=timed_out)

    def _stats(self, h, k):
        st = nc.MctsStats()
        z = np.zeros(4, dtype=np.int32)
        d = np.zeros(1, dtype=np.float64)
        self.lib.cq_mcts_root(h, self._ptr(z, ctypes.c_int32), self._ptr(z, ctypes.c_int32),
                              self._ptr(d, ctypes.c_double), self._ptr(d, ctypes.c_double), k, ctypes.byref(st))
        return None, st


_players: dict[tuple, MctsPlayer] = {}


# The play-time configuration measured in the panels; server.py and the hub agents use the same defaults.
# fpu 0.25 -> 0.0 (15 Nov sweep S1, seed 1115, 152 pairs each vs FW: paired score diff +0.036 [+0.005,+0.067] over fpu 0.25;
# c_puct 1.0 and 2.5 not better than 1.5)
PLAY_BATCH, PLAY_C_PUCT, PLAY_FPU = 256, 1.5, 0.0


def play_settings(batch: Optional[int] = None, c_puct: Optional[float] = None,
                  fpu: Optional[float] = None) -> tuple[int, float, float]:
    """The play-time MCTS settings: the given ones, else CHEQQERS_MCTS_BATCH / _CPUCT / _FPU, else the panels'."""
    return (int(batch if batch is not None else os.getenv("CHEQQERS_MCTS_BATCH", PLAY_BATCH)),
            float(c_puct if c_puct is not None else os.getenv("CHEQQERS_MCTS_CPUCT", PLAY_C_PUCT)),
            float(fpu if fpu is not None else os.getenv("CHEQQERS_MCTS_FPU", PLAY_FPU)))


def get_mcts_player(model_path: str, batch: Optional[int] = None, c_puct: Optional[float] = None,
                    fpu: Optional[float] = None, device: Optional[str] = None) -> MctsPlayer:
    """THE way to build the play-time MCTS (server.py and the series agents both call it): one MctsPlayer per
    (model, settings, device) in a process, settings from play_settings(), warmed up (CUDA initialisation and the CUDA
    graphs take seconds: two short searches on the start position run here, not in a game)."""
    path = str(Path(model_path).expanduser().resolve())
    b, c, f = play_settings(batch, c_puct, fpu)
    key = (path, b, c, f, device or "")
    if key not in _players:
        player = MctsPlayer(path, device=device, batch=b, c_puct=c, fpu=f)
        if isinstance(player.net, GraphNet):  # every move-count bucket captured now, never inside a game's search
            for bucket in GraphNet.BUCKETS:
                player.net._graph(bucket)
        player.search(nc.NativeEnv.new(6, 2, 2), None, time_ms=300.0)
        player.search(nc.NativeEnv.new(6, 2, 2), None, time_ms=300.0)
        _players[key] = player
    return _players[key]
