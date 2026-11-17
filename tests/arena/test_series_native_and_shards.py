"""Series with the C++ search (`cpp=1` in an agent spec) and series split into shards that merge (`pair_start`)."""

import json

import pytest

from arena.leiden_sparring_hub import (
    InProcessCheqqersAgent,
    _build_agent_from_spec,
    read_series_files,
    run_cheqqers_series,
    summarize_pairs,
)
from search import native_cheqqers

needs_native = pytest.mark.skipif(not native_cheqqers.available(), reason="native Cheqqers library not loadable")


# ── The engine option in a spec ────────────────────────────────────────────────────────────────────────────────


class TestEngineInSpec:
    def test_cheqqers_agent_searches_in_python_when_engine_not_given(self):
        assert _build_agent_from_spec("P:expecti_heuristic@depth=3", "cheqqers").engine == "python"

    def test_cheqqers_agent_searches_natively_when_cpp_given(self):
        agent = _build_agent_from_spec("C:expecti_heuristic@depth=3,ms=500,cpp=1,q=1,qp=2,ext=6", "cheqqers")
        assert agent.engine == "cpp"
        assert agent.player.quiescence is True and agent.player.quantum_plies == 2
        assert agent.player.max_capture_extension == 6

    def test_spec_raises_when_key_is_unknown(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("C:expecti_heuristic@cpp=1,turbo=1", "cheqqers")

    def test_tiqtaqtoe_agent_rejects_cpp(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("A:pure_mcts@cpp=1", "tiqtaqtoe")

    def test_neural_agent_rejects_cpp(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("N:expecti_neural@cpp=1", "cheqqers")


@needs_native
class TestNativeAgentPlays:
    def test_native_agent_plays_the_python_move_at_fixed_depth(self):
        """The port is exact at a fixed depth (tests/native/): the agent must hand it the same position and options."""
        from env.cheqqers_official import OfficialCheqqersEnv

        env = OfficialCheqqersEnv(board_size=6, start_rows=2, game_type=2)
        for _ in range(2):
            env.step(env.to_tournament_payload()["legalMoves"][0], return_state=False)
        payload = env.to_tournament_payload()
        legal = payload["legalMoves"]
        assert len(legal) > 1
        py = InProcessCheqqersAgent("P", max_depth=3, time_limit_ms=60_000)
        cpp = InProcessCheqqersAgent("C", max_depth=3, time_limit_ms=60_000, engine="cpp")
        assert py.select_move(env.clone(), payload, legal)[0] == cpp.select_move(env.clone(), payload, legal)[0]
        assert cpp.last_depth == 3

    def test_series_with_a_native_agent_records_depth_and_no_errors(self, tmp_path):
        out = tmp_path / "native.jsonl"
        run_cheqqers_series(
            InProcessCheqqersAgent("C", max_depth=2, time_limit_ms=200, engine="cpp"),
            InProcessCheqqersAgent("R", mode="random", seed=3),
            num_pairs=1, harvest_losses=False, hard_cases_path=str(tmp_path / "c.jsonl"), seed=9,
            opening_plies=2, out_path=str(out))
        games = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(games) == 2
        for g in games:
            assert g["fallbacks_a"] == 0 and g["native_errors_a"] == 0
            assert g["mean_depth_a"] > 0


# ── Shards of one series ───────────────────────────────────────────────────────────────────────────────────────


def play(tmp_path, name: str, pair_start: int, num_pairs: int, seed: int = 5, searching: bool = False) -> list[dict]:
    """Random players by default (fast; their own move generator runs on across games, so only the series' own
    definition, pairs, openings and colours, is comparable between runs). `searching`: two deterministic native
    searches at a fixed depth, whose games repeat exactly."""
    out = tmp_path / f"{name}.jsonl"
    if searching:
        agents = (InProcessCheqqersAgent("A", max_depth=2, time_limit_ms=60_000, engine="cpp"),
                  InProcessCheqqersAgent("B", max_depth=1, time_limit_ms=60_000, engine="cpp"))
    else:
        agents = (InProcessCheqqersAgent("A", mode="random", seed=11), InProcessCheqqersAgent("B", mode="random", seed=12))
    run_cheqqers_series(
        *agents,
        num_pairs=num_pairs, harvest_losses=False, hard_cases_path=str(tmp_path / f"{name}_c.jsonl"),
        seed=seed, opening_plies=2, out_path=str(out), pair_start=pair_start)
    return [json.loads(line) for line in out.read_text().splitlines()]


DEFINITION = ("pair", "opening", "a_is_white")
GAME = DEFINITION + ("winner", "plies")


class TestShards:
    def test_shards_define_the_pairs_of_the_whole_series_when_merged(self, tmp_path):
        whole = play(tmp_path, "whole", 0, 4)
        parts = play(tmp_path, "s0", 0, 2) + play(tmp_path, "s1", 2, 2)
        assert [{k: g[k] for k in DEFINITION} for g in parts] == [{k: g[k] for k in DEFINITION} for g in whole]

    @needs_native
    def test_shards_play_the_games_of_the_whole_series_when_agents_are_deterministic(self, tmp_path):
        whole = play(tmp_path, "dwhole", 0, 3, searching=True)
        parts = play(tmp_path, "d0", 0, 1, searching=True) + play(tmp_path, "d1", 1, 2, searching=True)
        assert [{k: g[k] for k in GAME} for g in parts] == [{k: g[k] for k in GAME} for g in whole]

    def test_shard_keeps_both_games_of_a_pair_on_one_opening_with_colours_swapped(self, tmp_path):
        games = play(tmp_path, "s3", 3, 2)
        assert [g["pair"] for g in games] == [3, 3, 4, 4]
        for first, second in zip(games[0::2], games[1::2]):
            assert first["opening"] == second["opening"] and first["a_is_white"] != second["a_is_white"]

    def test_summary_counts_every_pair_once_when_shard_files_are_merged(self, tmp_path):
        play(tmp_path, "m0", 0, 2)
        play(tmp_path, "m1", 2, 3)
        summary = summarize_pairs(read_series_files([tmp_path / "m0.jsonl", tmp_path / "m1.jsonl"]))
        assert summary["pairs"] == 5 and summary["incomplete_pairs"] == 0
        g = summary["games"]
        assert g["won"] + g["drawn"] + g["lost"] == 10

    def test_reading_raises_when_two_shards_overlap(self, tmp_path):
        play(tmp_path, "o0", 0, 2)
        play(tmp_path, "o1", 1, 2)
        with pytest.raises(ValueError):
            read_series_files([tmp_path / "o0.jsonl", tmp_path / "o1.jsonl"])

    def test_different_seeds_stay_apart_when_read_together(self, tmp_path):
        play(tmp_path, "d5", 0, 2, seed=5)
        play(tmp_path, "d6", 0, 2, seed=6)
        assert summarize_pairs(read_series_files([tmp_path / "d5.jsonl", tmp_path / "d6.jsonl"]))["pairs"] == 4
