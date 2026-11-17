"""Cheqqers series of the sparring hub: random openings played once with each colour, and one record per game."""

import json

from arena.leiden_sparring_hub import InProcessCheqqersAgent, run_cheqqers_series


def play(tmp_path, name: str, opening_plies: int = 2, num_pairs: int = 4, seed: int = 5) -> list[dict]:
    """A short series between two seeded random players; the per-game records it wrote."""
    out = tmp_path / f"{name}.jsonl"
    run_cheqqers_series(
        InProcessCheqqersAgent("A", mode="random", seed=11),
        InProcessCheqqersAgent("B", mode="random", seed=12),
        num_pairs=num_pairs,
        harvest_losses=False,
        hard_cases_path=str(tmp_path / f"{name}_cases.jsonl"),
        seed=seed,
        opening_plies=opening_plies,
        out_path=str(out),
    )
    return [json.loads(line) for line in out.read_text().splitlines()]


# ── Paired openings ────────────────────────────────────────────────────────────────────────────────────────────


class TestPairedOpenings:
    def test_both_games_of_a_pair_share_the_opening_when_openings_are_on(self, tmp_path):
        games = play(tmp_path, "pairs")
        assert len(games) == 8
        for first, second in zip(games[0::2], games[1::2]):
            assert first["pair"] == second["pair"]
            assert first["opening"] == second["opening"] and len(first["opening"]) == 2
            assert first["a_is_white"] != second["a_is_white"]

    def test_openings_differ_between_pairs_when_openings_are_on(self, tmp_path):
        games = play(tmp_path, "variety", num_pairs=6)
        assert len({json.dumps(g["opening"]) for g in games}) >= 3

    def test_series_plays_from_the_start_position_when_openings_are_off(self, tmp_path):
        games = play(tmp_path, "plain", opening_plies=0)
        assert all(g["opening"] == [] for g in games)


# ── Reproducibility and records ────────────────────────────────────────────────────────────────────────────────


class TestSeriesRecords:
    def test_series_repeats_game_for_game_when_seed_repeats(self, tmp_path):
        first = play(tmp_path, "run1")
        second = play(tmp_path, "run2")
        keep = ("pair", "opening", "a_is_white", "winner", "result_for_a", "plies")
        assert [{k: g[k] for k in keep} for g in first] == [{k: g[k] for k in keep} for g in second]

    def test_series_differs_when_seed_differs(self, tmp_path):
        first = play(tmp_path, "seed5", seed=5)
        second = play(tmp_path, "seed6", seed=6)
        assert [g["opening"] for g in first] != [g["opening"] for g in second]

    def test_each_game_record_names_result_and_slowest_move(self, tmp_path):
        for game in play(tmp_path, "fields", num_pairs=1):
            assert game["winner"] in ("white", "black", "draw")
            assert game["result_for_a"] in (1.0, 0.5, 0.0)
            assert game["slowest_ms_a"] >= 0.0 and game["slowest_ms_b"] >= 0.0
            assert game["fallbacks_a"] == 0 and game["fallbacks_b"] == 0


# ── Reading a paired series ────────────────────────────────────────────────────────────────────────────────────


def game(pair: int, result_for_a: float, a_is_white: bool) -> dict:
    return {"pair": pair, "result_for_a": result_for_a, "a_is_white": a_is_white, "winner": "x",
            "slowest_ms_a": 10.0, "slowest_ms_b": 20.0, "fallbacks_a": 0, "fallbacks_b": 0}


class TestSummarizePairs:
    def test_summary_has_no_spread_when_every_pair_is_split(self):
        from arena.leiden_sparring_hub import summarize_pairs

        games = [g for p in range(50) for g in (game(p, 1.0, True), game(p, 0.0, False))]
        summary = summarize_pairs(games)
        assert summary["pairs"] == 50 and summary["score_a"] == 0.5
        assert summary["standard_error"] == 0.0
        assert summary["elo"] == 0.0 and summary["elo_low"] == 0.0 and summary["elo_high"] == 0.0

    def test_summary_puts_the_interval_above_zero_when_one_side_wins_most_pairs(self):
        from arena.leiden_sparring_hub import summarize_pairs

        games = []
        for p in range(60):
            second = 1.0 if p % 3 else 0.0  # two pairs in three won 2-0, the third split
            games += [game(p, 1.0, True), game(p, second, False)]
        summary = summarize_pairs(games)
        assert summary["score_a"] > 0.8
        assert summary["elo_low"] > 0.0
        assert summary["pair_counts"] == {"2-0": 40, "1-1": 20, "0-2": 0, "other": 0}

    def test_summary_ignores_a_pair_when_one_of_its_games_is_missing(self):
        from arena.leiden_sparring_hub import summarize_pairs

        games = [game(0, 1.0, True), game(0, 1.0, False), game(1, 1.0, True)]
        summary = summarize_pairs(games)
        assert summary["pairs"] == 1 and summary["incomplete_pairs"] == 1

    def test_summary_reports_slowest_move_and_fallbacks(self):
        from arena.leiden_sparring_hub import summarize_pairs

        summary = summarize_pairs([game(0, 1.0, True), game(0, 0.0, False)])
        assert summary["slowest_ms_a"] == 10.0 and summary["slowest_ms_b"] == 20.0
        assert summary["fallbacks_a"] == 0 and summary["fallbacks_b"] == 0

    def test_pairs_of_two_files_stay_apart_when_read_together(self, tmp_path):
        from arena.leiden_sparring_hub import read_series_files, summarize_pairs

        for name, results in (("a", (1.0, 1.0)), ("b", (0.0, 0.0))):
            (tmp_path / f"{name}.jsonl").write_text(
                "".join(json.dumps(game(0, r, i == 0)) + "\n" for i, r in enumerate(results)))
        summary = summarize_pairs(read_series_files([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]))
        assert summary["pairs"] == 2
        assert summary["pair_counts"] == {"2-0": 1, "1-1": 0, "0-2": 1, "other": 0}
