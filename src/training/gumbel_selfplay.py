"""Gumbel AlphaZero-style self-play generation for Cheqqers (recipe docs/specs/cheqqers_rl_recipe.md, sections 2-3).

    uv run python src/training/gumbel_selfplay.py --model M.pt --games 6000 --parallel 2048 --out gen/g001 \
        --m 16 --n 64 --threads 3

Thousands of games, each searched by its own Gumbel MCTS tree in C++ (native cq_gz_*: Gumbel-Top-m root +
Sequential Halving over n simulations, completed-Q interior selection, chance nodes by expectation). Two batches of
games alternate (double buffering): while the GPU evaluates one batch's leaves, the C++ threads back up and collect
the other's. Every searched position becomes a sample: the policy target softmax(logits + sigma(completed Q)) over
its legal moves and the game's realised result for the side to move (value class 0 win / 1 draw / 2 loss).

Writes <out>_XXX.npz shards in the trainer's layout (train_cheqqers_transformer Arrays: x, mstart, msq, mlen, target,
value, game, stats) plus "policy" (one probability per legal move); target = the argmax of the policy target.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from search import native_cheqqers as nc  # noqa: E402
from search.gpu_mcts_cheqqers import MAX_MOVES, MctsNet, _opts  # noqa: E402

D = 144
P = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))  # noqa: E731


class GzGroup:
    """One batch of games in C++ (cq_gz_*) with its buffers."""

    def __init__(self, lib, n_games: int, seed: int, m: int, n: int, threads: int, leaves: int, c_visit: float,
                 c_scale: float, open_lo: int, open_hi: int, max_plies: int):
        self.lib, self.threads, self.leaves = lib, threads, leaves
        err = nc._err()
        o = _opts(1.5, 0.25, 0.0, 0.25, seed, 10**6, m, n, c_visit, c_scale)
        h = lib.cq_gz_new(n_games, seed & 0xFFFFFFFFFFFFFFFF, open_lo, open_hi, max_plies, ctypes.byref(o), threads,
                          err, nc._ERR)
        if not h:
            nc._raise(err, "gumbel self-play batch")
        self.h = ctypes.c_void_p(h)
        self.n = n_games
        self.cap = leaves + 64
        self.x = np.zeros((n_games * self.cap, D), np.float32)
        self.mv = np.zeros((n_games * self.cap, MAX_MOVES, 3), np.int32)
        self.ln = np.zeros((n_games * self.cap, MAX_MOVES), np.int32)
        self.counts = np.zeros(n_games, np.int32)
        self.which = np.zeros(n_games, np.int32)
        self.rows, self.m, self.pending = 0, 1, None

    def set_frozen(self, sides: np.ndarray) -> None:
        """Per game: the colour searched by the frozen network (-1: none)."""
        s = np.ascontiguousarray(sides, dtype=np.int32)
        self.lib.cq_gz_set_frozen(self.h, P(s, ctypes.c_int32), len(s))

    def row_nets(self) -> np.ndarray:
        """Per packed row of the last collect: 1 if the frozen network evaluates it."""
        self.lib.cq_gz_which(self.h, P(self.which, ctypes.c_int32))
        return np.repeat(self.which, self.counts)[: self.rows]

    def collect(self) -> int:
        err, mm = nc._err(), ctypes.c_int32()
        r = self.lib.cq_gz_collect(self.h, self.leaves, self.cap, P(self.x, ctypes.c_float), P(self.mv, ctypes.c_int32),
                                   P(self.ln, ctypes.c_int32), P(self.counts, ctypes.c_int32), ctypes.byref(mm),
                                   self.threads, err, nc._ERR)
        if r < 0:
            nc._raise(err, "gumbel collect")
        self.rows, self.m = r, max(1, mm.value)
        return r

    def apply(self, vals: np.ndarray, pri: np.ndarray) -> int:
        prf = np.zeros((max(self.rows, 1), MAX_MOVES), np.float32)
        if self.rows:
            prf[: self.rows, : self.m] = pri
        err = nc._err()
        live = self.lib.cq_gz_apply(self.h, P(np.ascontiguousarray(vals, dtype=np.float32), ctypes.c_float),
                                    P(prf, ctypes.c_float), MAX_MOVES, self.threads, err, nc._ERR)
        if live < 0:
            nc._raise(err, "gumbel apply")
        return live

    def status(self) -> tuple[int, int, int]:
        live, fin, mp = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int64()
        self.lib.cq_gz_status(self.h, ctypes.byref(live), ctypes.byref(fin), ctypes.byref(mp))
        return live.value, fin.value, mp.value

    def take(self) -> Optional[dict]:
        ns, nm = ctypes.c_int64(), ctypes.c_int64()
        self.lib.cq_gz_take(self.h, None, None, None, None, None, None, ctypes.byref(ns), ctypes.byref(nm))
        if ns.value == 0:
            return None
        S, MV = ns.value, nm.value
        x = np.zeros((S, D), np.float32)
        mv = np.zeros((MV, 4), np.int32)
        pol = np.zeros(MV, np.float32)
        mst = np.zeros(S + 1, np.int64)
        z = np.zeros(S, np.int8)
        gid = np.zeros(S, np.int32)
        games = self.lib.cq_gz_take(self.h, P(x, ctypes.c_float), P(mv, ctypes.c_int32), P(pol, ctypes.c_float),
                                    P(mst, ctypes.c_int64), P(z, ctypes.c_int8), P(gid, ctypes.c_int32),
                                    ctypes.byref(ns), ctypes.byref(nm))
        return {"x": x, "moves": mv, "policy": pol, "mstart": mst, "z": z, "game": gid, "games": games}

    def free(self) -> None:
        self.lib.cq_gz_free(self.h)


def to_shard(parts: list[dict], game_base: int, meta: Optional[dict] = None) -> dict:
    x = np.concatenate([p["x"] for p in parts])
    mv = np.concatenate([p["moves"] for p in parts])
    pol = np.concatenate([p["policy"] for p in parts])
    starts, base = [], 0
    for p in parts:
        starts.append(p["mstart"][:-1] + base)
        base += int(p["mstart"][-1])
    mst = np.concatenate(starts + [np.asarray([base])])
    z = np.concatenate([p["z"] for p in parts])
    # argmax of each position's policy target, vectorised (the first of equal maxima)
    cnt = np.diff(mst)
    seg = np.repeat(np.arange(len(z)), cnt)
    mx = np.maximum.reduceat(pol, mst[:-1]) if len(z) else np.zeros(0, np.float32)
    hit = np.flatnonzero(pol == mx[seg])
    first = hit[np.unique(seg[hit], return_index=True)[1]]
    target = (first - mst[:-1]).astype(np.int32)
    value = np.where(z > 0, 0, np.where(z < 0, 2, 1)).astype(np.int8)
    gid = np.concatenate([p["game"] + game_base + 100000 * k for k, p in enumerate(parts)]).astype(np.int32)
    return {"x": x, "mstart": mst, "msq": mv[:, 1:4].astype(np.int8), "mlen": mv[:, 0].astype(np.int8),
            "target": target, "value": value, "policy": pol, "game": gid,
            "stats": np.asarray(json.dumps({**(meta or {}), "games": int(sum(p["games"] for p in parts)), "excluded_positions": 0,
                                            "unfinished": 0, "ply_limit": 0}))}


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def fw_weights(path: str) -> Optional[np.ndarray]:
    """FW's evaluation weights as the C array (features..., scale, logit_per_man)."""
    if not path:
        return None
    from env.cheqqers_official import EvalWeights

    w = EvalWeights.from_json(path)
    return np.ascontiguousarray(list(w.as_vector()) + [w.scale, w.logit_per_man], dtype=np.float64)


class Nets:
    """The current network and, optionally, a frozen opponent: each packed row goes to its game's network."""

    def __init__(self, current: MctsNet, frozen: Optional[MctsNet]):
        self.cur, self.frz = current, frozen

    def launch(self, g: "GzGroup"):
        x, mv, ln = g.x[: g.rows], g.mv[: g.rows, : g.m], g.ln[: g.rows, : g.m]
        if self.frz is None:
            return ("one", self.cur.launch(x, mv, ln))
        which = g.row_nets()
        a, b = np.flatnonzero(which == 0), np.flatnonzero(which == 1)
        ha = self.cur.launch(x[a], mv[a], ln[a]) if len(a) else None
        hb = self.frz.launch(x[b], mv[b], ln[b]) if len(b) else None
        return ("two", a, b, ha, hb, g.rows, g.m)

    def finish(self, h):
        if h[0] == "one":
            return self.cur.finish(h[1])
        _, a, b, ha, hb, rows, m = h
        vals = np.zeros(rows, np.float32)
        pri = np.zeros((rows, m), np.float32)
        if ha is not None:
            va, pa = self.cur.finish(ha)
            vals[a], pri[a] = va, pa
        if hb is not None:
            vb, pb = self.frz.finish(hb)
            vals[b], pri[b] = vb, pb
        return vals, pri


def generate(args) -> int:
    lib = nc.load()
    if not hasattr(lib, "cq_gz_set_refill"):
        raise nc.NativeUnavailable(f"{nc.LIB_PATH} has no Gumbel self-play with refill (rebuild it)")
    if args.gpu_mem_frac > 0 and torch.cuda.is_available():
        # the caching allocator would otherwise keep most of the GPU (28.8 GB seen): leave room for the trainers and
        # the panel's match processes on the same card
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    net = Nets(MctsNet(args.model), MctsNet(args.frozen_model) if args.frozen_model else None)
    cont = Path(args.continuous) if args.continuous else None
    model_sha = file_sha(args.model)
    frozen_path = args.frozen_model
    shard_models = [model_sha]  # every model that searched moves since the last shard cut (games can straddle a swap)
    pointer_sha = None
    if cont is not None and (cont / "latest.json").exists():
        pointer_sha = json.loads((cont / "latest.json").read_text()).get("sha256")
    t_swap = time.time()
    swaps = 0
    frng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # two groups (double buffering), each with half the slots; finished games restart in their slot until each
    # group has started its share of --games (refill keeps the batches full to the end)
    k = min(args.parallel, args.games)
    sizes = [max(1, k // 2), k - max(1, k // 2)] if k > 1 else [1]
    if cont is not None:  # continuous: every finished game restarts in its slot, no generation barrier
        k = args.parallel
        sizes = [max(1, k // 2), k - max(1, k // 2)]
    shares = [args.games * sz // k for sz in sizes]
    shares[-1] = args.games - sum(shares[:-1])
    groups = []
    for j, (sz, share) in enumerate(zip(sizes, shares)):
        if sz <= 0:
            continue
        g = GzGroup(lib, sz, args.seed + j, args.m, args.n, args.threads, args.leaves, args.c_visit, args.c_scale,
                    args.open_lo, args.open_hi, args.max_plies)
        lib.cq_gz_set_refill(g.h, ctypes.c_int64(1 << 50 if cont is not None else max(0, share - sz)))
        # recipe section 4: 20% of the games against a frozen checkpoint, 10% against FW (disjoint slots)
        u = frng.random(sz)
        side = frng.integers(0, 2, sz)
        if args.frozen_model and args.frozen_frac > 0:
            g.set_frozen(np.where(u < args.frozen_frac, side, -1))
        if args.fw_frac > 0:
            fw_sides = np.ascontiguousarray(np.where((u >= args.frozen_frac) & (u < args.frozen_frac + args.fw_frac),
                                                     side, -1), dtype=np.int32)
            w = fw_weights(args.fw_weights)
            lib.cq_gz_set_fw(g.h, P(fw_sides, ctypes.c_int32), sz, float(args.fw_ms), int(args.fw_depth),
                             None if w is None else P(w, ctypes.c_double))
        groups.append(g)
    total_games = total_samples = shard = evals = rounds = 0
    pending: list[dict] = []
    # where the time goes (B, lead 14 Nov): waiting for the GPU result, C++ tree apply + collect, packing + launch
    prof = {"wait_gpu": 0.0, "cpp_tree": 0.0, "launch_h2d": 0.0, "rows": 0, "batches": 0}
    live = {id(g): g.n for g in groups}
    for g in groups:
        g.collect()
        g.pending = net.launch(g) if g.rows else None

    def drain(final: bool) -> None:
        nonlocal total_games, total_samples, shard, pending
        for g in groups:
            part = g.take()
            if part:
                pending.append(part)
                total_games += part["games"]
                total_samples += len(part["z"])
        if pending and (sum(len(p["z"]) for p in pending) >= args.shard_size or final):
            final = f"{out}_{shard:03d}.npz"  # written then renamed: a concurrent trainer never reads a partial shard
            with open(final + ".tmp", "wb") as fh:
                np.savez(fh, **to_shard(pending, game_base=shard * 10_000_000,
                                    meta={"models_sha256": list(shard_models), "frozen": str(frozen_path or "")}))
            os.replace(final + ".tmp", final)
            print(json.dumps({"shard": final, "rows": int(sum(len(p["z"]) for p in pending)) if pending else 0,
                              "models_sha256": list(shard_models)}), flush=True)
            shard_models[:] = [model_sha]
            shard += 1
            pending = []

    while any(v > 0 for v in live.values()):
        for g in groups:
            if live[id(g)] == 0:
                continue
            tw = time.perf_counter()
            if g.pending is not None:
                vals, pri = net.finish(g.pending)
            else:
                vals, pri = np.zeros(0, np.float32), np.zeros((0, g.m), np.float32)
            evals += g.rows
            tc = time.perf_counter()
            prof["wait_gpu"] += tc - tw
            live[id(g)] = g.apply(vals, pri)
            g.pending = None
            if live[id(g)] > 0:
                g.collect()  # its next batch, while the other group's runs on the GPU
                tl = time.perf_counter()
                prof["cpp_tree"] += tl - tc
                if g.rows:
                    g.pending = net.launch(g)
                prof["launch_h2d"] += time.perf_counter() - tl
                if args.sleep_ms > 0:  # duty cycle: the GPU idles a little after each batch (power cap, 14 Nov)
                    time.sleep(args.sleep_ms / 1000.0)
                prof["rows"] += g.rows
                prof["batches"] += 1
        rounds += 1
        if cont is not None and (cont / "STOP").exists():
            break
        if cont is not None and time.time() - t_swap >= args.swap_every:
            t_swap = time.time()
            try:
                rec = json.loads((cont / "latest.json").read_text())
            except (OSError, ValueError):
                rec = None
            if rec and rec.get("sha256") != pointer_sha and Path(rec["path"]).exists():
                # hot swap for new searches: launched batches finish on the old object; the shard is cut first so
                # its rows carry the models that produced them (in-flight games may straddle: both shas are kept)
                drain(True)
                new_net = MctsNet(rec["path"])
                new_sha = file_sha(rec["path"])
                frozen_new = None
                pool = sorted(glob.glob(str(cont / "models" / "model_*.pt")))
                if net.frz is not None and pool:
                    frozen_path = pool[int(frng.integers(0, len(pool)))]
                    frozen_new = MctsNet(frozen_path)
                net.cur = new_net
                if frozen_new is not None:
                    net.frz = frozen_new
                pointer_sha, model_sha = rec.get("sha256"), new_sha
                shard_models[:] = [new_sha] + [m for m in shard_models if m != new_sha]
                swaps += 1
                print(json.dumps({"swap": swaps, "ckpt": rec.get("ckpt"), "model_sha256": new_sha,
                                  "frozen": str(frozen_path or ""), "seconds": round(time.time() - started)}), flush=True)
        if rounds % 50 == 0:
            drain(False)
            el = time.time() - started
            print(json.dumps({"games": total_games, "samples": total_samples, "evals": evals, "seconds": round(el),
                              "games_per_hour": round(total_games / el * 3600), "evals_per_s": round(evals / el),
                              "live": sum(live.values()),
                              "prof_s": {k: round(v, 1) for k, v in prof.items() if k not in ("rows", "batches")},
                              "rows_per_batch": round(prof["rows"] / max(1, prof["batches"]))}), flush=True)
    drain(True)
    for g in groups:
        g.free()
    el = time.time() - started
    print(json.dumps({"games": total_games, "samples": total_samples, "evals": evals, "seconds": round(el),
                      "games_per_hour": round(total_games / el * 3600), "evals_per_s": round(evals / el)}), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--parallel", type=int, default=2048, help="games in flight (two batches of half)")
    parser.add_argument("--out", required=True, help="shard prefix")
    parser.add_argument("--m", type=int, default=16, help="Gumbel root candidates")
    parser.add_argument("--n", type=int, default=64, help="simulations per move")
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--leaves", type=int, default=8, help="leaves per tree per round")
    parser.add_argument("--open-lo", type=int, default=2)
    parser.add_argument("--open-hi", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--continuous", default="", help="generation dir: no barrier (refill forever), hot-swap to "
                        "<dir>/latest.json's checkpoint when it changes, frozen opponent from <dir>/models, stop on <dir>/STOP")
    parser.add_argument("--swap-every", type=float, default=300.0, help="seconds between checks of latest.json")
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="pause after each batch launch (GPU power cap)")
    parser.add_argument("--frozen-model", default="", help="a frozen checkpoint playing one side in --frozen-frac")
    parser.add_argument("--gpu-mem-frac", type=float, default=0.55,
                        help="cap on this process's share of GPU memory (budget with the trainers on one 32 GB card)")
    parser.add_argument("--frozen-frac", type=float, default=0.2)
    parser.add_argument("--fw-frac", type=float, default=0.0, help="share of games with FW playing one side")
    parser.add_argument("--fw-ms", type=float, default=30.0, help="FW opponent's time per move")
    parser.add_argument("--fw-depth", type=int, default=6, help="FW opponent's depth cap")
    parser.add_argument("--fw-weights", default="config/cheqqers_eval_weights_2026-11-13.json")
    args = parser.parse_args(argv)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
