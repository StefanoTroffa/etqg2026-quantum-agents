"""The per-position cache of square probabilities and legal moves (Track-W): faster, and nothing else.

The cache is off unless asked for (`OfficialCheqqersEnv.cache_enabled`); the search switches it on for the copy of
the position it works on. With it the search must choose the same move and visit the same number of positions,
and the simulator must replay arena games exactly as without it. Positions come from the arena games in
`tests/compliance/fixtures/arena_cheqqers_2026-11-12_sample.jsonl`.
"""

import json
from pathlib import Path

import pytest

import env.cheqqers_official as simulator
from arena.arena_records import replay, verify
from env.cheqqers_official import OfficialCheqqersEnv
from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move

FIXTURE = Path(__file__).parents[1] / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl"
GAMES = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def positions(plies=(6, 12, 18), games=14) -> list[OfficialCheqqersEnv]:
    """Positions with more than one legal move, reached in the first `games` arena games after `plies` moves."""
    out = []
    for game in GAMES[:games]:
        for ply in plies:
            if ply >= len(game["moves"]):
                continue
            env = replay({**game, "moves": game["moves"][:ply]})
            if not env.done and len(env.get_legal_moves()) > 1:
                out.append(env)
    return out


def search(env: OfficialCheqqersEnv, depth: int, use_cache: bool) -> tuple:
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=600000.0, use_cache=use_cache)
    move = player.get_move(env.clone())
    return canonicalize_cheqqers_move(move), player._nodes_visited


# ── The search with and without the cache ──────────────────────────────────────────────────────────────────────


class TestSearchWithCache:
    def test_search_chooses_the_same_move_and_visits_as_many_positions_when_cache_is_on(self):
        checked = 0
        for env in positions():
            assert search(env, 2, use_cache=True) == search(env, 2, use_cache=False)
            checked += 1
        assert checked >= 25

    def test_deeper_search_is_unchanged_too_when_cache_is_on(self):
        for env in positions(plies=(8,), games=4):
            assert search(env, 3, use_cache=True) == search(env, 3, use_cache=False)

    def test_search_leaves_the_callers_position_without_cache(self):
        env = positions(plies=(6,), games=1)[0]
        ExpectiCheqqersPlayer(max_depth=1, time_limit_ms=600000.0, use_cache=True).get_move(env)
        assert env.cache_enabled is False


# ── The simulator with the cache ───────────────────────────────────────────────────────────────────────────────


class TestSimulatorWithCache:
    def test_arena_games_replay_exactly_when_cache_is_on_by_default(self, monkeypatch):
        monkeypatch.setattr(simulator, "CACHE_BY_DEFAULT", True)
        assert OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2).cache_enabled is True
        report = verify(GAMES)
        assert report["mismatches"] == []
        assert report["replayed"] == report["same_result"] == report["same_hash"] == len(GAMES)

    def test_legal_moves_are_recomputed_after_a_move_when_cache_is_on(self):
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        env.cache_enabled = True
        before = env.get_legal_moves()
        assert env.get_legal_moves() == before  # served from the cache
        env.step(before[0], return_state=False)
        after = env.get_legal_moves()
        assert after != before
        fresh = env.clone()
        fresh.cache_enabled = False
        assert after == fresh.get_legal_moves()

    def test_clone_keeps_the_setting_and_starts_with_an_empty_cache(self):
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        env.cache_enabled = True
        env.get_legal_moves()
        copy = env.clone()
        assert copy.cache_enabled is True
        assert copy._moves_cache is None and copy._probs_cache is None

    @pytest.mark.parametrize("enabled", [False, True])
    def test_evaluation_is_the_same_number_when_cache_is_on_or_off(self, enabled):
        for env in positions(plies=(10,), games=6):
            reference = env.evaluate_state("white")
            copy = env.clone()
            copy.cache_enabled = enabled
            assert copy.evaluate_state("white") == reference


# ── All squares of an entangled system in one pass (Track-W step 3) ────────────────────────────────────────────


class TestProbabilitiesInOnePass:
    def test_one_pass_gives_every_square_exactly_its_own_probabilities(self):
        entangled = 0
        for game in GAMES:
            for ply in range(0, len(game["moves"]), 3):
                env = replay({**game, "moves": game["moves"][:ply]})
                expected = [sq.get_probabilities() for sq in env.squares]
                env.cache_enabled = True
                env._invalidate_caches()
                assert env._square_probs() == expected  # exact: same additions in the same order
                entangled += any(sq.entangled_system is not None for sq in env.squares)
        assert entangled > 50
