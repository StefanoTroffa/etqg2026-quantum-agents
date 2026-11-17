"""The Cheqqers rules of the organisers' harness of 11 November 2026, on positions recorded from that harness.

The harness changed four things against the version of 30 October: a slide never passes an enemy piece; squares a
move passes over are measured and must be empty ("transit"); a landing or passed-over square that turns out
occupied, or a piece to be jumped that turns out absent, fails the move; a capture with nothing to capture does
nothing. `fixtures/cheqqers_harness_2026-11-11_positions.jsonl` holds 48 positions the harness itself played
(`js/cheqqers_trace.mjs`, 300 games, seed 20261113), twelve of each kind below, with the move it made and the
board it got. No Node is needed here; `test_cheqqers_harness_differential.py` checks against the live harness.
"""

import json
from pathlib import Path

import pytest

from env.cheqqers_official import OfficialCheqqersEnv

FIXTURE = Path(__file__).parent / "fixtures" / "cheqqers_harness_2026-11-11_positions.jsonl"
RECORDS = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
KINDS = ("queen_near_superposed_enemy", "transit_passed", "transit_failed", "capture_failed")


def board_signature(board: list[dict]) -> tuple:
    squares = []
    for square in board:
        presence = round(float(square.get("probability", 0.0)), 3)
        if not square.get("piece") or presence <= 0.0:
            squares.append(None)
        else:
            squares.append((str(square["piece"])[0], bool(square.get("crowned")), presence))
    return tuple(squares)


def records_of(kind: str) -> list[dict]:
    return [r for r in RECORDS if r["kind"] == kind]


# ── Legal moves ────────────────────────────────────────────────────────────────────────────────────────────────


class TestLegalMovesOnRecordedPositions:
    @pytest.mark.parametrize("kind", KINDS)
    def test_simulator_lists_the_harness_moves_when_position_is_of_kind(self, kind):
        records = records_of(kind)
        assert len(records) == 12
        for record in records:
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            env._override_legal_moves = None  # the simulator's own generator, not the payload's list
            ours = sorted(tuple(m) for m in env.get_legal_moves())
            theirs = sorted(tuple(m) for m in record["payload"]["legalMoves"])
            assert ours == theirs, f"move {record['move']}: simulator {ours} | harness {theirs}"


# ── Move outcomes ──────────────────────────────────────────────────────────────────────────────────────────────


class TestMoveOutcomesOnRecordedPositions:
    @pytest.mark.parametrize("kind", KINDS)
    def test_harness_board_is_among_simulator_outcomes_when_position_is_of_kind(self, kind):
        for record in records_of(kind):
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            ours = {board_signature(child.to_tournament_payload()["board"])
                    for _, child in env.get_move_outcomes(record["move"])}
            assert board_signature(record["after"]["board"]) in ours, f"move {record['move']} of kind {kind}"

    def test_passing_over_a_superposed_square_branches_into_pass_and_fail(self):
        # every recorded slide or capture over a square in superposition has at least two outcomes in the
        # simulator: the square measured empty (the move goes on) and measured occupied (the move fails)
        for record in records_of("transit_passed") + records_of("transit_failed"):
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            assert len(env.get_move_outcomes(record["move"])) >= 2, f"move {record['move']}"
