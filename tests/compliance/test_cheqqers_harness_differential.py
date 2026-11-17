"""The Python Cheqqers simulator against the organisers' harness itself.

Random games are played by the harness (`tests/compliance/js/cheqqers_trace.mjs`, Node) and every position is handed
to `OfficialCheqqersEnv`. Two things must agree, because the search plans on them:

* the legal moves the simulator generates for the position;
* the position after the move: what the harness got must be one of the outcomes the simulator enumerates.

Needs Node and a directory holding `local-cheqqers-server.js` with the engine installed beside it
(`npm install @quantumplayed/quantum-game-engine@1.0.2`); set `QG_CHEQQERS_HARNESS_DIR` to it. Skipped otherwise.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from env.cheqqers_official import OfficialCheqqersEnv

DRIVER = Path(__file__).parent / "js" / "cheqqers_trace.mjs"
GAMES = int(os.getenv("QG_DIFFERENTIAL_GAMES", "60"))
SEED = int(os.getenv("QG_DIFFERENTIAL_SEED", "20261113"))


def _harness_file() -> Path:
    folder = os.getenv("QG_CHEQQERS_HARNESS_DIR")
    if not folder:
        pytest.skip("QG_CHEQQERS_HARNESS_DIR is not set")
    harness = Path(folder) / "local-cheqqers-server.js"
    if not harness.is_file() or not (Path(folder) / "node_modules").is_dir():
        pytest.skip(f"{folder} does not hold local-cheqqers-server.js and node_modules")
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    return harness


@pytest.fixture(scope="module")
def harness_records() -> list[dict]:
    """Every move of GAMES random games played by the harness, as the driver prints them."""
    harness = _harness_file()
    run = subprocess.run(
        ["node", str(DRIVER), str(harness), str(GAMES), str(SEED)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    records = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert len(records) > 20 * GAMES // 3, f"only {len(records)} moves in {GAMES} games"
    return records


def signature(board: list[dict], turn: str) -> tuple:
    """A position as the payload shows it: per square colour, crown and presence to three decimals, plus the turn."""
    squares = []
    for square in board:
        piece = square.get("piece")
        presence = round(float(square.get("probability", 0.0)), 3)
        if not piece or presence <= 0.0:
            squares.append(None)
        else:
            squares.append((str(piece)[0], bool(square.get("crowned")), presence))
    return (turn, tuple(squares))


def simulator_signature(env: OfficialCheqqersEnv) -> tuple:
    payload = env.to_tournament_payload()
    return signature(payload["board"], str(payload["currentTurn"]).lower())


def describe(record: dict) -> str:
    board = [
        f"{i}{s['piece'][0]}{'Q' if s['crowned'] else ''}{'' if s['probability'] > 0.999 else round(s['probability'], 3)}"
        for i, s in enumerate(record["payload"]["board"]) if s.get("piece")
    ]
    return (f"game {record['game']} ply {record['ply']} {record['payload']['player']} to move | "
            f"{' '.join(board)} | move {record['move']}")


# ── Legal moves ────────────────────────────────────────────────────────────────────────────────────────────────


class TestLegalMovesAgainstHarness:
    def test_simulator_lists_the_harness_moves_when_no_capture_sequence_is_open(self, harness_records):
        mismatches = []
        checked = 0
        for record in harness_records:
            if record["multiJump"] is not None:  # the payload does not say which piece must go on capturing
                continue
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            env._override_legal_moves = None  # the simulator's own generator, not the payload's list
            ours = sorted(tuple(m) for m in env.get_legal_moves())
            theirs = sorted(tuple(m) for m in record["payload"]["legalMoves"])
            checked += 1
            if ours != theirs:
                only_ours = sorted(set(ours) - set(theirs))
                only_theirs = sorted(set(theirs) - set(ours))
                mismatches.append(f"{describe(record)} | simulator only {only_ours} | harness only {only_theirs}")
        assert checked > 0
        # The payload carries no piece identities. With three or more half-pieces of one colour entangled, the
        # simulator can pair them wrongly, misjudge which piece was already jumped, and so the longest capture.
        # Measured after the port of 11 November 2026: 2 of 28,000 positions. More than 1 in 2,000 is a rules bug.
        assert len(mismatches) <= max(0, checked // 2000), (
            f"{len(mismatches)} of {checked} positions differ; first: {mismatches[0]}")


# ── Move outcomes ──────────────────────────────────────────────────────────────────────────────────────────────


class TestMoveOutcomesAgainstHarness:
    def test_harness_board_is_among_simulator_outcomes_when_no_capture_sequence_is_open(self, harness_records):
        mismatches = []
        checked = 0
        for record in harness_records:
            if record["multiJump"] is not None:
                continue
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            ours = {simulator_signature(child)[1] for _, child in env.get_move_outcomes(record["move"])}
            theirs = signature(record["after"]["board"], record["after"]["turn"])[1]
            checked += 1
            if theirs not in ours:
                mismatches.append(f"{describe(record)} | harness board not among {len(ours)} simulator outcomes")
        assert checked > 0
        assert not mismatches, f"{len(mismatches)} of {checked} moves differ; first: {mismatches[0]}"

    def test_side_to_move_matches_harness_except_for_unknown_piece_identity(self, harness_records):
        """After a capture the same side moves again if it can capture a piece it has not jumped this turn. The
        payload carries no piece identities, so when three or more half-pieces of one colour are entangled the
        simulator can pair them wrongly and miss, or invent, the continuation. That is a limit of the payload, not
        of the rules port: it must stay rare, and the boards must agree (previous test)."""
        wrong_turn = []
        checked = 0
        for record in harness_records:
            if record["multiJump"] is not None:
                continue
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            ours = {simulator_signature(child) for _, child in env.get_move_outcomes(record["move"])}
            theirs = signature(record["after"]["board"], record["after"]["turn"])
            checked += 1
            if theirs not in ours and theirs[1] in {board for _, board in ours}:
                wrong_turn.append(describe(record))
        assert checked > 0
        assert len(wrong_turn) <= max(1, checked // 200), (
            f"{len(wrong_turn)} of {checked} moves leave the wrong side to move; first: {wrong_turn[0]}")


# ── The engine's limit on entangled squares ────────────────────────────────────────────────────────────────────

LOW_LIMIT = 16  # an entangled system of at most four squares, instead of the engine's fifteen


@pytest.fixture(scope="module")
def low_limit_records() -> list[dict]:
    """Games in which the harness's engine is limited to LOW_LIMIT and split moves are preferred, so that the
    engine often refuses a move ("QuantumLimitExceeded"). The harness swallows that error inside applyMove."""
    harness = _harness_file()
    run = subprocess.run(
        ["node", str(DRIVER), str(harness), "40", str(SEED), "200", "0.5", str(LOW_LIMIT)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    records = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert sum(r["engineError"] for r in records) > 100, "the low limit was not reached often enough to test"
    return records


class TestEngineLimitAgainstHarness:
    """With more entangled squares than the engine allows, the engine throws; the harness catches the error and
    goes on with the same side to move. The real limit (15 squares) is rare in random play but our own agent
    reaches it, so the simulator must do what the harness does, not raise."""

    def test_refused_move_leaves_the_same_side_to_move_as_in_the_harness(self, low_limit_records, monkeypatch):
        from env.cheqqers_official import JointSystem

        monkeypatch.setattr(JointSystem, "max_dimension_limit", LOW_LIMIT)
        mismatches = []
        checked = 0
        for record in low_limit_records:
            if record["multiJump"] is not None or not record["engineError"]:
                continue
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            ours = {simulator_signature(child) for _, child in env.get_move_outcomes(record["move"])}
            theirs = signature(record["after"]["board"], record["after"]["turn"])
            assert record["after"]["turn"] == record["payload"]["player"]  # what the harness does
            checked += 1
            if theirs not in ours:
                mismatches.append(describe(record))
        assert checked > 50
        assert len(mismatches) <= checked // 50, (
            f"{len(mismatches)} of {checked} refused moves differ; first: {mismatches[0]}")

    def test_accepted_moves_still_agree_when_the_limit_is_low(self, low_limit_records, monkeypatch):
        from env.cheqqers_official import JointSystem

        monkeypatch.setattr(JointSystem, "max_dimension_limit", LOW_LIMIT)
        mismatches = []
        checked = 0
        for record in low_limit_records:
            if record["multiJump"] is not None or record["engineError"]:
                continue
            env = OfficialCheqqersEnv.from_tournament_json(record["payload"])
            ours = {simulator_signature(child)[1] for _, child in env.get_move_outcomes(record["move"])}
            checked += 1
            if signature(record["after"]["board"], record["after"]["turn"])[1] not in ours:
                mismatches.append(describe(record))
        assert checked > 200
        assert len(mismatches) <= checked // 50, (
            f"{len(mismatches)} of {checked} accepted moves differ; first: {mismatches[0]}")
