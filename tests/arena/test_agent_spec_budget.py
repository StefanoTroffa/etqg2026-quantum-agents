"""Agent specs of the sparring hub CLI: the `@depth=..,ms=..` search budget of an in-process agent."""

import pytest

from arena.leiden_sparring_hub import _build_agent_from_spec, _split_budget

# ── Budget suffix parsing ──────────────────────────────────────────────────────────────────────────────────────


class TestSplitBudget:
    def test_split_returns_spec_unchanged_when_no_budget(self):
        assert _split_budget("Baseline:expecti_heuristic") == ("Baseline:expecti_heuristic", {})

    def test_split_reads_depth_and_ms_when_given(self):
        head, budget = _split_budget("Deep:expecti_heuristic@depth=8,ms=2200")
        assert head == "Deep:expecti_heuristic"
        assert budget == {"depth": 8.0, "ms": 2200.0}

    @pytest.mark.parametrize("spec", ["A:expecti_heuristic@dept=8", "A:expecti_heuristic@8", "A:random@ms"])
    def test_split_raises_when_item_is_not_a_known_key(self, spec):
        with pytest.raises(ValueError):
            _split_budget(spec)


# ── Agents built from a spec ───────────────────────────────────────────────────────────────────────────────────


class TestBuildAgentFromSpec:
    def test_cheqqers_agent_takes_its_own_budget_when_given(self):
        agent = _build_agent_from_spec("Deep:expecti_heuristic@depth=8,ms=2200", "cheqqers", timeout_ms=3000)
        assert agent.name == "Deep"
        assert agent.player.max_depth == 8
        assert agent.player.time_limit_ms == 2200.0

    def test_cheqqers_agent_keeps_series_defaults_when_no_budget(self):
        agent = _build_agent_from_spec("Base:expecti_heuristic", "cheqqers", timeout_ms=3000)
        assert agent.player.max_depth == 4
        assert agent.player.time_limit_ms == 2400.0

    def test_tiqtaqtoe_agent_rejects_depth_when_given(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("A:pure_mcts@depth=3", "tiqtaqtoe")


# ── Evaluation weights in a spec ───────────────────────────────────────────────────────────────────────────────


class TestWeightsInSpec:
    def test_split_keeps_the_weights_path_as_text_when_given(self):
        head, budget = _split_budget("Cand:expecti_heuristic@depth=5,weights=config/w.json")
        assert head == "Cand:expecti_heuristic"
        assert budget == {"depth": 5.0, "weights": "config/w.json"}

    def test_cheqqers_agent_searches_with_the_weights_file_when_given(self, tmp_path):
        import dataclasses
        import json

        from env.cheqqers_official import EvalWeights

        weights = dataclasses.replace(EvalWeights(), queens=1.75, superposition=-0.2)
        path = tmp_path / "w.json"
        path.write_text(json.dumps(weights.to_dict()))
        agent = _build_agent_from_spec(f"Cand:expecti_heuristic@depth=3,weights={path}", "cheqqers")
        assert agent.player.eval_weights == weights

    def test_cheqqers_agent_has_default_weights_when_none_given(self):
        agent = _build_agent_from_spec("Base:expecti_heuristic@depth=3", "cheqqers")
        assert agent.player.eval_weights is None

    def test_tiqtaqtoe_agent_rejects_weights_when_given(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("A:pure_mcts@weights=w.json", "tiqtaqtoe")


# ── Search options in a spec: quiescence, its budget, an opening book ──────────────────────────────────────────


class TestSearchOptionsInSpec:
    def test_cheqqers_agent_searches_with_quiescence_when_asked(self):
        agent = _build_agent_from_spec("Q:expecti_heuristic@depth=3,q=1,ext=8", "cheqqers")
        assert agent.player.quiescence is True
        assert agent.player.max_capture_extension == 8

    def test_cheqqers_agent_searches_without_quiescence_when_not_asked(self):
        agent = _build_agent_from_spec("Plain:expecti_heuristic@depth=3", "cheqqers")
        assert agent.player.quiescence is False
        assert agent.player.max_capture_extension == 4

    def test_cheqqers_agent_reads_its_opening_book_when_given(self, tmp_path):
        from search.opening_book import OpeningBook

        OpeningBook({"T:0|1WM1": [15, 12]}).save(tmp_path / "book.json")
        agent = _build_agent_from_spec(f"B:expecti_heuristic@depth=3,book={tmp_path / 'book.json'}", "cheqqers")
        assert len(agent.player.opening_book) == 1

    def test_cheqqers_agent_limits_three_square_moves_to_its_first_plies_when_asked(self):
        assert _build_agent_from_spec("L:expecti_heuristic@depth=3,qp=2", "cheqqers").player.quantum_plies == 2
        assert _build_agent_from_spec("F:expecti_heuristic@depth=3", "cheqqers").player.quantum_plies is None

    def test_tiqtaqtoe_agent_rejects_cheqqers_search_options(self):
        with pytest.raises(ValueError):
            _build_agent_from_spec("A:pure_mcts@q=1", "tiqtaqtoe")
