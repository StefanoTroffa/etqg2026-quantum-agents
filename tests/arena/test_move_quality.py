"""Move quality: what an agent's choices give away on recorded positions, as a much longer search sees them.

A series between two versions of the agent needs hundreds of games to see 30 Elo; on 13 November 2026 none of four
candidates separated from the plain search in 20 to 45 pairs. This instrument asks a different question on many
positions at once: at each recorded decision a judge values every legal move, and an agent's loss there is the
best value minus the value of the move it picks at its own time limit.
"""

import json
from pathlib import Path

import pytest

from arena.arena_records import replay
from arena.move_quality import agent_losses, decisions, judge, summarize

FIXTURE = Path(__file__).parents[1] / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl"
GAMES = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()][:3]


def judged_rows(tmp_path, depth: int = 1) -> list[dict]:
    return judge(decisions(GAMES), seconds=600.0, depth=depth, workers=1, cache_path=tmp_path / "judged.jsonl")


# ── The decisions of recorded games ────────────────────────────────────────────────────────────────────────────


class TestDecisions:
    def test_decisions_are_the_positions_with_a_choice(self):
        found = decisions(GAMES)
        expected = 0
        for game in GAMES:
            for ply in range(len(game["moves"])):
                expected += len(replay({**game, "moves": game["moves"][:ply]}).get_legal_moves()) > 1
        assert len(found) == expected > 20
        assert all(d["game"]["moves"] == GAMES[[g["id"] for g in GAMES].index(d["id"])]["moves"][:d["ply"]]
                   for d in found)

    def test_decisions_can_be_limited_to_one_side_by_a_predicate(self):
        white = decisions(GAMES, keep=lambda game, ply: game["moves"][ply]["by"] == "white")
        assert 0 < len(white) < len(decisions(GAMES))


# ── The judge ──────────────────────────────────────────────────────────────────────────────────────────────────


class TestJudge:
    def test_judge_values_every_legal_move_of_every_decision(self, tmp_path):
        rows = judged_rows(tmp_path)
        assert len(rows) == len(decisions(GAMES))
        for row, decision in zip(rows, decisions(GAMES)):
            legal = replay(decision["game"]).get_legal_moves()
            assert (row["id"], row["ply"]) == (decision["id"], decision["ply"])
            assert sorted(move for move, _ in row["values"]) == sorted(legal)
            assert row["judge_depth"] == 1

    def test_judge_does_not_search_again_what_its_file_already_holds(self, tmp_path):
        first = judged_rows(tmp_path)
        lines = (tmp_path / "judged.jsonl").read_text()
        again = judged_rows(tmp_path)
        assert again == first
        assert (tmp_path / "judged.jsonl").read_text() == lines

    def test_judge_searches_again_when_asked_for_a_deeper_judgement(self, tmp_path):
        judged_rows(tmp_path, depth=1)
        deeper = judged_rows(tmp_path, depth=2)
        assert all(row["judge_depth"] == 2 for row in deeper)


# ── An agent against the judge ─────────────────────────────────────────────────────────────────────────────────


class TestAgentLosses:
    def test_agent_that_searches_like_the_judge_loses_nothing(self, tmp_path):
        rows = judged_rows(tmp_path, depth=2)
        losses = agent_losses(rows, GAMES, "Same:expecti_heuristic@depth=2,ms=600000", workers=1)
        assert len(losses) == len(rows)
        assert all(row["loss"] == pytest.approx(0.0, abs=1e-9) for row in losses)

    def test_shallower_agent_loses_something_and_never_gains(self, tmp_path):
        rows = judged_rows(tmp_path, depth=3)
        losses = agent_losses(rows, GAMES, "Shallow:expecti_heuristic@depth=1,ms=600000", workers=1)
        assert all(row["loss"] >= 0.0 for row in losses)
        assert sum(row["loss"] for row in losses) > 0.0

    def test_summary_counts_decisions_and_costly_ones(self):
        rows = [{"loss": 0.0, "depth": 5, "ply": 3}, {"loss": 13.0, "depth": 6, "ply": 12},
                {"loss": 60.0, "depth": 4, "ply": 30}, {"loss": 2.0, "depth": 5, "ply": 31}]
        summary = summarize(rows)
        assert summary["decisions"] == 4
        assert summary["mean_loss"] == pytest.approx(75.0 / 4)
        assert summary["half_man_or_more"] == 2 and summary["two_men_or_more"] == 1
        assert summary["mean_depth"] == pytest.approx(5.0)


# ── A deeper judge where the agents differ ─────────────────────────────────────────────────────────────────────


class TestJudgeWhereAgentsDiffer:
    def table(self, tmp_path) -> dict:
        rows = judged_rows(tmp_path, depth=3)
        return {name: agent_losses(rows, GAMES, spec, workers=1)
                for name, spec in (("Deep", "Deep:expecti_heuristic@depth=3,ms=600000"),
                                   ("Shallow", "Shallow:expecti_heuristic@depth=1,ms=600000"))}

    def test_disagreements_are_the_decisions_with_more_than_one_chosen_move(self, tmp_path):
        from arena.move_quality import disagreements

        table = self.table(tmp_path)
        found = disagreements(table)
        expected = sum(a["move"] != b["move"] for a, b in zip(table["Deep"], table["Shallow"]))
        assert len(found) == expected > 0
        for moves in found.values():
            assert len(moves) == 2

    def test_rejudged_losses_cover_only_the_disagreements_and_the_deeper_agent_loses_less(self, tmp_path):
        from arena.move_quality import disagreements, judge_moves, losses_under

        table = self.table(tmp_path)
        differing = disagreements(table)
        verdicts = judge_moves(differing, GAMES, seconds=600.0, depth=3, workers=1)
        assert {(v["id"], v["ply"]) for v in verdicts} == set(differing)
        assert all(v["judge_depth"] == 3 for v in verdicts)
        deep, shallow = losses_under(verdicts, table["Deep"]), losses_under(verdicts, table["Shallow"])
        assert len(deep) == len(shallow) == len(differing)
        assert all(row["loss"] >= 0.0 for row in deep + shallow)
        assert sum(row["loss"] for row in deep) < sum(row["loss"] for row in shallow)
