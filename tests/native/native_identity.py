"""Shared by the native-engine identity tests and scripts/native_identity_sweep.py: where positions come from and
what "the same" means.

Two positions are the same when their flat exports (native_cheqqers.export_env) are equal, the doubles compared bit
for bit: board, every amplitude of every joint system in its dict order, piece properties and ids, turn, open capture
sequence, counters, repetition history. `move_history` and `last_move` are not compared (no rule reads them).
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path
from typing import Any, Iterator, Optional

from env.cheqqers_official import EvalWeights, OfficialCheqqersEnv, PieceColor
from search import native_cheqqers as nc

HERE = Path(__file__).parent
REPO = HERE.parents[1]
ARENA_SAMPLE = REPO / "tests" / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl"
BENCH = HERE / "fixtures" / "bench_positions_2026-11-13.jsonl"

# Weights unlike the defaults, both on the linear scale and as a logit, so a feature computed in the wrong place
# cannot hide behind a zero weight.
ODD_WEIGHTS = (
    EvalWeights(men=1.1, queens=2.3, advance=0.31, advance_sq=0.17, back_rank=0.09, queen_main_diagonal=0.33,
                queen_near_diagonal=0.21, centre=0.19, superposition=-0.07, multi_jump=0.4, threat_mover=0.6,
                threat_waiter=0.35, tempo=0.11, exposed_mover=-0.27, exposed_waiter=0.29,
                superposition_when_few=-0.5, material_share=1.7, scale=23.0),
    EvalWeights(men=0.9, queens=2.1, exposed_mover=-0.3, exposed_waiter=0.25, material_share=1.2, logit_per_man=0.8),
)


def bits(values: Any) -> bytes:
    values = list(values)
    return struct.pack(f"{len(values)}d", *values)


def same_export(a: tuple[list[int], list[float]], b: tuple[list[int], list[float]]) -> bool:
    return a[0] == b[0] and bits(a[1]) == bits(b[1])


def load_games(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def new_env(game: dict[str, Any]) -> OfficialCheqqersEnv:
    s = game.get("settings") or {}
    return OfficialCheqqersEnv(board_size=int(s.get("boardSize", 6)), start_rows=int(s.get("startRows", 2)),
                               quantumness=int(s.get("quantumness", 2)))


def replay_positions(game: dict[str, Any], upto: Optional[int] = None) -> Iterator[tuple[int, OfficialCheqqersEnv, Optional[dict]]]:
    """(ply, position before the move, recorded move) for every ply, then (n, final position, None)."""
    env = new_env(game)
    moves = game["moves"] if upto is None else game["moves"][:upto]
    for ply, move in enumerate(moves):
        yield ply, env, move
        env.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
    yield len(moves), env, None


def random_positions(seed: int, board_size: int = 6, quantumness: int = 2, max_plies: int = 120) -> Iterator[OfficialCheqqersEnv]:
    """A game of uniformly random moves whose measurement outcomes are drawn with their probabilities, all from
    `seed`: every position, the finished one included."""
    rng = random.Random(seed)
    env = OfficialCheqqersEnv(board_size=board_size, quantumness=quantumness)
    for _ in range(max_plies):
        yield env
        if env.is_terminal():
            return
        legal = env.get_legal_moves()
        if not legal:
            return
        outcomes = env.get_move_outcomes(rng.choice(legal))
        r, acc = rng.random(), 0.0
        nxt = outcomes[-1][1]
        for p, child in outcomes:
            acc += p
            if r <= acc:
                nxt = child
                break
        env = nxt
    yield env


def compare_position(env: OfficialCheqqersEnv, outcomes: bool = True) -> list[str]:
    """Every way the native engine could differ from the Python one at this position; [] when it does not."""
    out: list[str] = []
    native = nc.NativeEnv.from_env(env)
    if not same_export(nc.export_env(env), native.export()):
        return ["the position does not survive the copy to the native engine"]
    py_legal, c_legal = env.get_legal_moves(), native.legal_moves()
    if py_legal != c_legal:
        out.append(f"legal moves {py_legal} != {c_legal}")
    if int(env.state) != int(native.state) or env.winner != native.winner:
        out.append(f"state {env.state} / {native.state}")
    for persp in (PieceColor.WHITE, PieceColor.BLACK):
        if bits(env.eval_features(persp)) != bits(native.eval_features(persp)):
            out.append(f"eval_features({persp.name}) {env.eval_features(persp)} != {native.eval_features(persp)}")
        for w in (None,) + ODD_WEIGHTS:
            env.eval_weights = w
            py_v = env.evaluate_state(persp)
            c_v = native.evaluate_state(persp, w)
            if bits([py_v]) != bits([c_v]):
                out.append(f"evaluate_state({persp.name}, {w}) {py_v!r} != {c_v!r}")
        env.eval_weights = None
    if outcomes:
        for move in py_legal:
            py_out = env.get_move_outcomes(move)
            c_out = native.move_outcomes(move)
            if bits(p for p, _ in py_out) != bits(p for p, _ in c_out):
                out.append(f"outcome probabilities of {move}: {[p for p, _ in py_out]} != {[p for p, _ in c_out]}")
                continue
            for k, ((_, pe), (_, ce)) in enumerate(zip(py_out, c_out)):
                if not same_export(nc.export_env(pe), ce.export()):
                    out.append(f"outcome {k} of {move}: child positions differ")
    return out


def with_cache(env: OfficialCheqqersEnv) -> OfficialCheqqersEnv:
    """The position as the search sees it: a copy with the per-position cache on."""
    copy = env.clone()
    copy.cache_enabled = True
    return copy
