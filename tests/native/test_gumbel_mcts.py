"""Gumbel root + Sequential Halving + completed-Q targets in the native MCTS (mcts.h, recipe section 2; gate 1).

With a deterministic evaluator (value = tanh of a fixed linear function of the encoding, uniform priors over the
legal moves):
- the policy target equals softmax(log prior + sigma(completed Q)) recomputed here from the root's priors, visits,
  Q and network value (Danihelka et al. 2022: unvisited moves take the mixed value; Q min-max normalised;
  sigma = (c_visit + max N) * c_scale * q_hat);
- Sequential Halving uses at most gumbel_n root simulations, ends with one candidate (the move played), visits only
  the top-m moves by Gumbel + logit, and with >= 16 moves gives the winner 1 + 2 + 4 + 8 = 15 visits (n = 64);
- the search ends by itself (no deadline) and the same seed gives the same result.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "native")]

pytest.importorskip("torch")

from search import native_cheqqers as nc  # noqa: E402

pytestmark = pytest.mark.skipif(not nc.available() or not hasattr(nc.load(), "cq_mcts_gumbel"),
                                reason="native library with the Gumbel MCTS not available")

from test_gpu_mcts import dummy  # noqa: E402

C_VISIT, C_SCALE, M, N = 50.0, 1.0, 16, 64


@pytest.fixture(scope="module")
def positions() -> list:
    from env.cheqqers_official import OfficialCheqqersEnv

    rng = np.random.default_rng(3)
    out = []
    for _ in range(60):
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        for _ in range(int(rng.integers(2, 25))):
            moves = env.get_legal_moves()
            if env.done or not moves:
                break
            outs = env.get_move_outcomes(list(moves[int(rng.integers(len(moves)))]))
            env = outs[int(rng.integers(len(outs)))][1]
        if not env.done and len(env.get_legal_moves()) >= 3:
            out.append(env)
    assert len(out) >= 20
    return out


def _search(env, seed: int = 1):
    from search.gpu_mcts_cheqqers import MctsPlayer

    return MctsPlayer(dummy, batch=16).search(env, None, time_ms=1e9, seed=seed, gumbel_m=M, gumbel_n=N,
                                              c_visit=C_VISIT, c_scale=C_SCALE)


def _target(res) -> list[float]:
    prior = [r[3] for r in res.visits]
    n = [r[1] for r in res.visits]
    q = [r[2] for r in res.visits]  # side to move's view
    sum_n = sum(n)
    pi_vis = sum(p for p, k in zip(prior, n) if k > 0)
    v_mix = res.root_vnet
    if sum_n > 0 and pi_vis > 0:
        v_mix = (res.root_vnet + sum_n * sum(p * x for p, x, k in zip(prior, q, n) if k > 0) / pi_vis) / (1 + sum_n)
    comp = [x if k > 0 else v_mix for x, k in zip(q, n)]
    lo, hi = min(comp), max(comp)
    scale = (C_VISIT + max(n)) * C_SCALE
    logits = [math.log(max(p, 1e-12)) + scale * (c - lo) / max(hi - lo, 1e-8) for p, c in zip(prior, comp)]
    mx = max(logits)
    z = sum(math.exp(v - mx) for v in logits)
    return [math.exp(v - mx) / z for v in logits]


def test_policy_target_is_softmax_of_logits_plus_sigma_completed_q(positions: list) -> None:
    for env in positions:
        res = _search(env)
        assert res.policy == pytest.approx(_target(res), abs=1e-9)


def test_halving_budget_and_single_winner(positions: list) -> None:
    for env in positions:
        res = _search(env)
        assert sum(r[1] for r in res.visits) <= N
        assert len(res.candidates) == 1 and res.visits[res.candidates[0]][0] == list(map(int, res.move))


def test_only_top_m_moves_by_gumbel_plus_logit_are_visited(positions: list) -> None:
    for env in positions:
        res = _search(env)
        score = [g + math.log(max(r[3], 1e-12)) for g, r in zip(res.gumbel, res.visits)]
        top = set(sorted(range(len(score)), key=lambda i: -score[i])[:M])
        assert {i for i, r in enumerate(res.visits) if r[1] > 0} <= top


def test_winner_gets_fifteen_visits_with_sixteen_candidates(positions: list) -> None:
    wide = [e for e in positions if len(e.get_legal_moves()) >= M]
    for env in wide:
        res = _search(env)
        assert res.visits[res.candidates[0]][1] == 15
    if not wide:
        pytest.skip("no position with 16 legal moves in the sample")


def test_same_seed_same_search(positions: list) -> None:
    a, b = _search(positions[0], seed=7), _search(positions[0], seed=7)
    assert a.visits == b.visits and a.policy == b.policy and a.move == b.move


def test_gumbel_selfplay_samples_are_consistent(tmp_path: Path) -> None:
    """cq_gz_*: one sample per move played, policy targets sum to 1, z in {-1, 0, 1} and the same game's samples
    agree on the result (each sample's z is the result from its side to move: all 0, or +-1)."""
    from training.gumbel_selfplay import GzGroup

    lib = nc.load()
    g = GzGroup(lib, 6, 5, 8, 16, 2, 8, 50.0, 1.0, 2, 6, 300)
    live = g.n
    while live > 0:
        g.collect()
        vals, pri = dummy(g.x[: g.rows], g.mv[: g.rows, : g.m], g.ln[: g.rows, : g.m])
        live = g.apply(vals, pri)
    _, finished, played = g.status()
    part = g.take()
    g.free()
    assert finished == 6 and len(part["z"]) == played
    st, pol = part["mstart"], part["policy"]
    assert all(abs(pol[st[i]:st[i + 1]].sum() - 1.0) < 1e-5 for i in range(len(part["z"])))
    for gid in set(part["game"].tolist()):
        zs = {int(v) for v in part["z"][part["game"] == gid]}
        assert zs <= {-1, 1} or zs == {0}


def test_frozen_side_moves_make_no_samples() -> None:
    """Games with a frozen opponent: the frozen side's searches are flagged for the frozen network and make no
    samples, so a decisive frozen game's samples are all one colour's (one z value); an ordinary decisive game has both
    colours' (z = +1 and -1)."""
    from training.gumbel_selfplay import GzGroup

    lib = nc.load()
    frozen = np.array([0, 1, -1, 0, 1, -1], dtype=np.int32)
    g = GzGroup(lib, len(frozen), 9, 8, 16, 2, 8, 50.0, 1.0, 2, 6, 300)
    g.set_frozen(frozen)
    live, flagged = g.n, 0
    while live > 0:
        g.collect()
        flagged += int(g.row_nets().sum())
        vals, pri = dummy(g.x[: g.rows], g.mv[: g.rows, : g.m], g.ln[: g.rows, : g.m])
        live = g.apply(vals, pri)
    part = g.take()
    g.free()
    assert flagged > 0
    for gid in range(len(frozen)):
        zs = {int(v) for v in part["z"][part["game"] == gid]}
        if zs == {0}:
            continue  # a draw says nothing about colours
        assert len(zs) == (1 if frozen[gid] >= 0 else 2), (gid, zs)


def test_fw_opponent_games_sample_only_the_network_side() -> None:
    """cq_gz_set_fw: that colour's moves are played by the FW alpha-beta search (no tree, no samples), so a decisive
    FW game's samples are one colour's; the other games are unchanged."""
    import ctypes

    from training.gumbel_selfplay import P, GzGroup

    lib = nc.load()
    fw = np.array([0, 1, -1, 1], dtype=np.int32)
    g = GzGroup(lib, len(fw), 13, 8, 16, 2, 8, 50.0, 1.0, 2, 6, 300)
    lib.cq_gz_set_fw(g.h, P(fw, ctypes.c_int32), len(fw), 5.0, 2, None)
    live = g.n
    while live > 0:
        g.collect()
        vals, pri = dummy(g.x[: g.rows], g.mv[: g.rows, : g.m], g.ln[: g.rows, : g.m])
        live = g.apply(vals, pri)
    part = g.take()
    g.free()
    for gid in range(len(fw)):
        zs = {int(v) for v in part["z"][part["game"] == gid]}
        if zs == {0}:
            continue
        assert len(zs) == (1 if fw[gid] >= 0 else 2), (gid, zs)
