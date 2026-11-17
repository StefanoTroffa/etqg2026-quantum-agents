"""MCTS with chance nodes (native/cheqqers/mcts.h, search/gpu_mcts_cheqqers.py; track 33 gate 1).

With a deterministic evaluator (value = tanh of a fixed linear function of the encoding, uniform priors):
- every root move visited exactly once has Q = sum_k p_k v(child_k) over the Python engine's get_move_outcomes
  (OfficialCheqqersEnv), children's values from the same evaluator (side to move's view converted to the root
  mover's), terminal children exact;
- the move returned is legal, the search stops at max_sims, and two runs give the same visits (no timing in the tree);
- many trees in one batch (cq_mcts_collect_many / apply_many) give the same root visits as one tree at a time.
"""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]

pytest.importorskip("torch")

from search import native_cheqqers as nc  # noqa: E402

pytestmark = pytest.mark.skipif(not nc.available() or not hasattr(nc.load(), "cq_mcts_new"),
                                reason="native library with MCTS not available")

W = np.random.default_rng(5).normal(size=144).astype(np.float32) * 0.2


def dummy(x: np.ndarray, mv: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.tanh(x.astype(np.float64) @ W.astype(np.float64)).astype(np.float32)
    pri = (lens > 0).astype(np.float32)
    return vals, pri


def value_of(native: nc.NativeEnv, root_white: bool) -> float:
    if native.terminal:
        w = native.winner
        v_white = float(w)
    else:
        x = np.asarray(native.encode(), dtype=np.float32)[None]
        v = float(dummy(x, np.zeros((1, 1, 3), np.int32), np.ones((1, 1), np.int32))[0][0])
        v_white = v if native.turn == 0 else -v
    return v_white if root_white else -v_white


@pytest.fixture(scope="module")
def positions() -> list[dict]:
    out = []
    from env.cheqqers_official import OfficialCheqqersEnv

    rng = np.random.default_rng(0)
    for _ in range(12):
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        for _ in range(int(rng.integers(4, 30))):
            moves = env.get_legal_moves()
            if env.done or not moves:
                break
            outs = env.get_move_outcomes(list(moves[int(rng.integers(len(moves)))]))
            env = outs[int(rng.integers(len(outs)))][1]
        if not env.done and len(env.get_legal_moves()) > 1:
            out.append(env)
    assert len(out) >= 6
    return out


def test_chance_expectation_matches_python_outcomes(positions: list) -> None:
    from search.gpu_mcts_cheqqers import MctsPlayer

    player = MctsPlayer(dummy, batch=4)
    checked = 0
    for env in positions:
        legal = env.get_legal_moves()
        res = player.search(env, legal, time_ms=1e9, max_sims=len(legal) // 2 + 1)  # most moves visited once
        assert list(map(int, res.move)) in [list(map(int, m)) for m in legal]
        root_white = nc.NativeEnv.from_env(env).turn == 0
        for move, visits, q, _ in res.visits:
            if visits != 1:
                continue
            outs = env.get_move_outcomes(list(move))
            total = sum(p for p, _ in outs)
            exp = sum((p / total) * value_of(nc.NativeEnv.from_env(child), root_white) for p, child in outs if p > 1e-9)
            assert math.isclose(q, exp, rel_tol=1e-5, abs_tol=1e-5), (move, q, exp)
            checked += 1
    assert checked > 20


def test_deterministic_and_stops_at_max_sims(positions: list) -> None:
    from search.gpu_mcts_cheqqers import MctsPlayer

    player = MctsPlayer(dummy, batch=8)
    env = positions[0]
    a = player.search(env, None, time_ms=1e9, max_sims=200)
    b = player.search(env, None, time_ms=1e9, max_sims=200)
    assert a.visits == b.visits or all(x[:2] == y[:2] for x, y in zip(a.visits, b.visits))
    assert 200 <= a.sims < 200 + 8 + 64


def test_many_trees_in_one_batch_match_single_trees(positions: list) -> None:
    from search.gpu_mcts_cheqqers import MAX_MOVES, MctsPlayer, _opts

    lib = nc.load()
    natives = [nc.NativeEnv.from_env(e) for e in positions[:6]]
    rounds, per = 12, 8
    # one tree at a time, fixed number of collect/apply rounds
    single = []
    for nat in natives:
        p = MctsPlayer(dummy, batch=per)
        err = nc._err()
        h = ctypes.c_void_p(lib.cq_mcts_new(nat._h, None, 0, ctypes.byref(_opts(1.5, 0.25, 0, 0.25, 1, 10**6)), err,
                                            nc._ERR))
        mm = ctypes.c_int32()
        for _ in range(rounds):
            n = lib.cq_mcts_collect(h, per, p.cap, p._ptr(p.x, ctypes.c_float), p._ptr(p.mv, ctypes.c_int32),
                                    p._ptr(p.lens, ctypes.c_int32), ctypes.byref(mm), err, nc._ERR)
            if n <= 0:
                break
            m = max(1, mm.value)
            v, pr = dummy(p.x[:n], p.mv[:n, :m], p.lens[:n, :m])
            pr = np.ascontiguousarray(pr)
            assert lib.cq_mcts_apply(h, p._ptr(v, ctypes.c_float), p._ptr(pr, ctypes.c_float), m, n, err, nc._ERR) == 0
        single.append(_root_visits(lib, h))
        lib.cq_mcts_free(h)
    # all trees together
    T = len(natives)
    cap_each = per + 64
    hs = (ctypes.c_void_p * T)(*[lib.cq_mcts_new(n._h, None, 0, ctypes.byref(_opts(1.5, 0.25, 0, 0.25, 1, 10**6)),
                                                 nc._err(), nc._ERR) for n in natives])
    x = np.zeros((T * cap_each, 144), np.float32)
    mv = np.zeros((T * cap_each, MAX_MOVES, 3), np.int32)
    ln = np.zeros((T * cap_each, MAX_MOVES), np.int32)
    counts = np.zeros(T, np.int32)
    mm = ctypes.c_int32()
    err = nc._err()
    P = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))  # noqa: E731
    for _ in range(rounds):
        tot = lib.cq_mcts_collect_many(hs, T, per, cap_each, P(x, ctypes.c_float), P(mv, ctypes.c_int32),
                                       P(ln, ctypes.c_int32), P(counts, ctypes.c_int32), ctypes.byref(mm), 4, err,
                                       nc._ERR)
        assert tot >= 0
        if tot == 0:
            break
        # the stride of the packed rows is MAX_MOVES; the network sees the first mmax columns
        m = max(1, mm.value)
        v, pr = dummy(x[:tot], mv[:tot, :m], ln[:tot, :m])
        prf = np.zeros((tot, MAX_MOVES), np.float32)
        prf[:, :m] = pr
        assert lib.cq_mcts_apply_many(hs, T, P(counts, ctypes.c_int32), P(v, ctypes.c_float), P(prf, ctypes.c_float),
                                      MAX_MOVES, 4, err, nc._ERR) == 0
    many = [_root_visits(lib, h) for h in hs]
    for h in hs:
        lib.cq_mcts_free(h)
    assert many == single


def _root_visits(lib, h) -> list[int]:
    k = 512
    moves = np.zeros(4 * k, np.int32)
    vis = np.zeros(k, np.int32)
    q = np.zeros(k)
    pr = np.zeros(k)
    n = lib.cq_mcts_root(h, moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                         vis.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                         q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         pr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k, None)
    return vis[:n].tolist()


def test_search_never_exceeds_its_budget_with_a_slow_network() -> None:
    """A slow or busy GPU must not make a move late (clean-install test, RTX 2060: 3,155 ms for a 2,200 ms move):
    the evaluator sleeps a random 20-400 ms per batch, including one batch far longer than the whole budget; every
    search returns a legal move within its time budget, and an abandoned batch does not break the next search."""
    import random
    import time

    from search.gpu_mcts_cheqqers import MctsPlayer

    rng = random.Random(7)
    calls = {"n": 0}

    def slow(x: np.ndarray, mv: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["n"] += 1
        time.sleep(0.9 if calls["n"] == 5 else rng.uniform(0.02, 0.4))
        return dummy(x, mv, lens)

    player = MctsPlayer(slow, batch=32)
    start = nc.NativeEnv.new(6, 2, 2)
    legal = [list(map(int, m)) for m in start.legal_moves()]
    budget = 500.0
    for i in range(8):
        t0 = time.perf_counter()
        r = player.search(start, None, time_ms=budget, seed=i)
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert elapsed <= budget, (i, elapsed, r.batches, r.timed_out)
        assert list(map(int, r.move)) in legal
    assert calls["n"] > 8  # the network was used, not skipped
