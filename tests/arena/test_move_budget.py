"""The search budget comes from the time limit the organisers send in each request.

The arena and the harness put the limit in `settings`: `timeout` in milliseconds for Cheqqers (3000 in the arena),
`moveTime` in seconds for TiqTaqToe (5). The arena measures it with network time included, plus a 500 ms allowance
(arena-setup.md, 8 November 2026). On 13 November a 2,200 ms budget under a 3,000 ms limit gave replies of at most
2,343 ms as the arena measured them.
"""

import pytest

from arena.move_budget import CHEQQERS_DEFAULT_DEPTH, cheqqers_budget, tiqtaqtoe_budget_ms


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch) -> None:
    for name in ("CHEQQERS_TIME_LIMIT_MS", "CHEQQERS_MAX_DEPTH", "TTT_TIME_LIMIT_MS"):
        monkeypatch.delenv(name, raising=False)


# ── Cheqqers ───────────────────────────────────────────────────────────────────────────────────────────────────


class TestCheqqersBudget:
    def test_arena_limit_gives_the_budget_measured_on_13_november(self):
        assert cheqqers_budget({"timeout": 3000}) == (2200.0, CHEQQERS_DEFAULT_DEPTH)

    def test_budget_follows_the_limit_when_the_limit_changes(self):
        assert cheqqers_budget({"timeout": 6000})[0] == pytest.approx(5200.0)
        assert cheqqers_budget({"timeout": 1500})[0] == pytest.approx(750.0)

    def test_budget_keeps_most_of_a_short_limit_for_the_network(self):
        assert 0.0 < cheqqers_budget({"timeout": 400})[0] <= 200.0

    @pytest.mark.parametrize("settings", [None, {}, {"timeout": None}, {"timeout": "fast"}, {"timeout": 0}])
    def test_budget_is_the_arena_one_when_no_usable_limit_is_sent(self, settings):
        assert cheqqers_budget(settings) == (2200.0, CHEQQERS_DEFAULT_DEPTH)

    def test_environment_overrides_win_when_set(self, monkeypatch):
        monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "900")
        monkeypatch.setenv("CHEQQERS_MAX_DEPTH", "5")
        assert cheqqers_budget({"timeout": 3000}) == (900.0, 5)

    def test_depth_cap_does_not_end_the_search_before_the_clock(self):
        assert CHEQQERS_DEFAULT_DEPTH >= 12


# ── TiqTaqToe (the search answers only when the exact tables cannot) ───────────────────────────────────────────


class TestTiqTaqToeBudget:
    def test_move_time_in_seconds_gives_a_budget_with_the_same_margin(self):
        assert tiqtaqtoe_budget_ms({"moveTime": 5}) == pytest.approx(4000.0)

    def test_budget_is_the_arena_one_when_no_move_time_is_sent(self):
        assert tiqtaqtoe_budget_ms({}) == pytest.approx(4000.0)

    def test_environment_override_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("TTT_TIME_LIMIT_MS", "1200")
        assert tiqtaqtoe_budget_ms({"moveTime": 5}) == 1200.0
