"""The tournament adapter over the exact TiqTaqToe tables (src/search/exact_tiqtaqtoe.py).

The tables are generated, not committed, so the tests that need them skip when they are absent.
"""

import copy
from fractions import Fraction

import pytest

from search.exact_tiqtaqtoe import ExactTiqTaqToePlayer, _normalise_payload

# The /move body captured on submission day of the June 2026 tournament (X split [0, 2], O to move). Its amplitudes
# use the keys "re"/"imaginary"; the current local harness sends "real"/"imaginary".
JUNE_PAYLOAD = {
    "gameId": "local-test-game", "roundId": "round-1", "player": "O", "currentTurn": "O",
    "board": ["half-x1", None, "half-x2", None, None, None, None, None, None],
    "probabilities": [[0.5, 0.5, 0], [1, 0, 0], [0.5, 0.5, 0]] + [[1, 0, 0]] * 6,
    "measuredSquares": [False] * 9,
    "emptySquares": [False, True, False, True, True, True, True, True, True],
    "quantumState": {"00X000000": {"re": 0.5, "imaginary": -0.5}, "X00000000": {"re": 0.5, "imaginary": 0.5}},
    "pairs": [[0, 2], None, None, None], "squareToPair": {"0": 0, "2": 0},
    "settings": {"quantumness": 2, "decoherence": 0, "timerEnabled": True, "moveTime": 5},
    "legalMoves": [[1], [3], [4], [5], [6], [7], [8], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [3, 4], [3, 5],
                   [3, 6], [3, 7], [3, 8], [4, 5], [4, 6], [4, 7], [4, 8], [5, 6], [5, 7], [5, 8], [6, 7], [6, 8],
                   [7, 8], [0, "M"], [2, "M"]],
}


def _payload(**settings):
    payload = copy.deepcopy(JUNE_PAYLOAD)
    payload["settings"].update(settings)
    return payload


def _player_with_tables(quantumness):
    player = ExactTiqTaqToePlayer()
    if not player.preload(quantumness, 0):
        pytest.skip(f"exact tables for quantumness {quantumness} are not on disk: {player._load_errors}")
    return player


def test_amplitude_keys_of_both_harness_versions_are_accepted():
    june = _normalise_payload(JUNE_PAYLOAD)["quantumState"]
    current = _normalise_payload({"quantumState": {"X00000000": {"real": 0.5, "imaginary": 0.5}}})["quantumState"]
    assert june["X00000000"] == current["X00000000"] == {"real": 0.5, "imaginary": 0.5}
    with pytest.raises(ValueError):
        _normalise_payload({"quantumState": {"X00000000": {"imaginary": 0.5}}})


def test_configurations_outside_the_tables_return_none_not_a_guess():
    player = ExactTiqTaqToePlayer()
    assert player.get_move(_payload(quantumness=3)) is None
    assert "outside the model" in player.last_reason
    assert player.get_move(_payload(decoherence=2)) is None
    assert "decoherence" in player.last_reason


def test_draw_weight_must_be_one_the_tables_hold():
    with pytest.raises(ValueError):
        ExactTiqTaqToePlayer(w=Fraction(1, 3))


def test_tournament_payload_gets_a_legal_optimal_move():
    player = _player_with_tables(2)
    move = player.get_move(JUNE_PAYLOAD)
    assert move in JUNE_PAYLOAD["legalMoves"], player.last_reason
    assert player.last_reason.startswith("exact q2")


def test_a_move_the_harness_does_not_list_is_never_returned():
    player = _player_with_tables(2)
    best = player.get_move(JUNE_PAYLOAD)
    restricted = copy.deepcopy(JUNE_PAYLOAD)
    restricted["legalMoves"] = [m for m in JUNE_PAYLOAD["legalMoves"] if m != best]
    second = player.get_move(restricted)
    assert second in restricted["legalMoves"] and second != best
