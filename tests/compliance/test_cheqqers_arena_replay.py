"""Games played in the organisers' arena, replayed through the Python Cheqqers simulator (src/arena/arena_records.py).

`fixtures/arena_cheqqers_2026-11-12_sample.jsonl` holds 44 of the 502 games of the arena session of 12 November 2026
(rules version d691637b68; every thirteenth game that ended normally, and the six longest, without repeats), reduced
to the moves, the measurement outcomes the arena drew, the result and the arena's hash of the final position.
Players' names are left out. On 13 November all 502 replayed with the same result; these keep that true.
"""

import copy
import json
from pathlib import Path

import pytest

from arena.arena_records import ReplayMismatch, forfeits, move_times, pairings, replay, team_games, verify

FIXTURE = Path(__file__).parent / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl"
GAMES = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


# ── Replay ─────────────────────────────────────────────────────────────────────────────────────────────────────


class TestArenaReplay:
    def test_fixture_is_the_sample_it_says(self):
        assert len(GAMES) == 44
        assert {g["rulesVersion"] for g in GAMES} == {"d691637b68"}

    def test_every_sampled_game_replays_with_the_arenas_result_and_hash(self):
        report = verify(GAMES)
        assert report["mismatches"] == []
        assert report["replayed"] == report["same_result"] == report["same_hash"] == 44

    def test_replay_raises_when_a_recorded_move_is_not_legal(self):
        game = copy.deepcopy(GAMES[0])
        game["moves"][0]["move"] = [0, 17]
        with pytest.raises(ReplayMismatch, match="not legal"):
            replay(game)

    def test_replay_raises_when_the_wrong_side_is_recorded_to_move(self):
        game = copy.deepcopy(GAMES[0])
        game["moves"][0]["by"] = "black"
        with pytest.raises(ReplayMismatch, match="to move"):
            replay(game)

    def test_verify_reports_a_game_whose_recorded_result_is_wrong(self):
        game = copy.deepcopy(GAMES[0])
        game["result"] = "black" if game["result"] == "white" else "white"
        report = verify([game])
        assert report["same_result"] == 0 and len(report["mismatches"]) == 1


# ── Summary ────────────────────────────────────────────────────────────────────────────────────────────────────


def record(white: str, black: str, result: str, reason: str = "normal", forfeit_by=None, ms=(100, 200)) -> dict:
    return {"id": f"{white}-{black}", "players": {"white": {"name": white}, "black": {"name": black}},
            "result": result, "reason": reason, "forfeitBy": forfeit_by,
            "moves": [{"by": "white", "move": [13, 10], "outcomes": [], "ms": ms[0]},
                      {"by": "black", "move": [4, 7, 8], "outcomes": [], "ms": ms[1]}]}


class TestArenaSummary:
    def test_pairings_count_wins_by_colour(self):
        games = [record("Us", "Them", "white"), record("Them", "Us", "white"), record("Us", "Them", "draw")]
        row = pairings(games)[("Them", "Us")]
        assert (row["a_wins"], row["b_wins"], row["draws"]) == (1, 1, 1)
        assert (row["a_white_wins"], row["a_white_games"], row["a_black_games"]) == (1, 1, 2)

    def test_move_times_are_per_player_as_the_arena_measured(self):
        times = move_times([record("Us", "Them", "white", ms=(100, 300)), record("Them", "Us", "white", ms=(500, 700))])
        assert times["Us"]["moves"] == 2 and times["Us"]["max_ms"] == 700
        assert times["Them"]["three_square_share"] == 0.5

    def test_forfeits_name_who_forfeited(self):
        games = [record("Us", "Them", "white"), record("Us", "Them", "white", reason="timeout", forfeit_by="black")]
        assert forfeits(games) == [{"id": "Us-Them", "reason": "timeout", "by": "Them", "plies": 2}]

    def test_team_games_are_read_from_the_teams_side(self):
        rows = team_games([record("Us", "Them", "white"), record("Them", "Us", "white")], "Us")
        assert [(r["colour"], r["result"], r["slowest_ms"]) for r in rows] == [("white", "win", 100.0), ("black", "loss", 200.0)]


# ── Review of played games: what each move gave away, as a longer search sees it ───────────────────────────────


def named(game: dict, us_white: bool) -> dict:
    """A fixture game (names removed) with "Us" on one side and "Them" on the other."""
    white, black = ("Us", "Them") if us_white else ("Them", "Us")
    return {**game, "players": {"white": {"name": white}, "black": {"name": black}}, "reason": "normal"}


class TestReview:
    def test_review_has_one_row_per_decision_with_a_choice_and_never_a_negative_loss(self):
        from arena.arena_records import review

        games = [named(GAMES[0], us_white=True), named(GAMES[1], us_white=False)]
        rows = review(games, "Us", judge_seconds=600.0, judge_depth=1, workers=1)
        expected = 0
        for game, colour in zip(games, ("white", "black")):
            for ply, move in enumerate(game["moves"]):
                if move["by"] == colour and len(replay({**game, "moves": game["moves"][:ply]}).get_legal_moves()) > 1:
                    expected += 1
        assert len(rows) == expected > 0
        for row in rows:
            assert row["judge_depth"] == 1
            assert row["loss"] >= 0.0
            assert row["loss"] == pytest.approx(row["best_value"] - row["played_value"], abs=1e-9)
            assert (row["loss"] == 0.0) or (row["played"] != row["best"])

    def test_review_keeps_only_lost_games_when_asked(self):
        from arena.arena_records import review

        games = [named(g, us_white=True) for g in GAMES[:6]]
        lost = {g["id"] for g in games if g["result"] == "black"}
        rows = review(games, "Us", judge_seconds=600.0, judge_depth=1, workers=1, only_losses=True)
        assert 0 < len(lost) < len(games)
        assert {row["game"] for row in rows} == lost

    def test_summary_counts_losses_of_half_a_man_by_stage_of_the_game(self):
        from arena.arena_records import review_summary

        rows = [{"ply": 3, "loss": 0.0, "judge_depth": 7}, {"ply": 5, "loss": 20.0, "judge_depth": 8},
                {"ply": 14, "loss": 12.5, "judge_depth": 9}, {"ply": 30, "loss": 4.0, "judge_depth": 9}]
        summary = review_summary(rows)
        assert summary["decisions"] == 4 and summary["half_man_or_more"] == 2
        assert summary["mean_loss"] == pytest.approx(36.5 / 4)
        assert summary["by_ply"]["0-9"] == {"decisions": 2, "mean_loss": 10.0, "half_man_or_more": 1}
        assert summary["by_ply"]["20+"]["half_man_or_more"] == 0
        assert summary["judge_depth_median"] == 9
