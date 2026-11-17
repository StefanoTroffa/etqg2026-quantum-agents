"""The Cheqqers evaluation as data: `EvalWeights`, the feature vector it is linear in, and how a player carries it.

`fixtures/cheqqers_eval_reference_2026-11-13.jsonl` holds 82 positions played by the organisers' harness with the
value the hand-written evaluation gave them on 13 November 2026, before its constants became data. The default
weights must give those values back: the default agent does not change.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from env.cheqqers_official import EVAL_FEATURES, EvalWeights, OfficialCheqqersEnv
from search.expecti_cheqqers import ExpectiCheqqersPlayer

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = [json.loads(line) for line in
             (FIXTURES / "cheqqers_eval_reference_2026-11-13.jsonl").read_text().splitlines() if line.strip()]


def env_of(record: dict) -> OfficialCheqqersEnv:
    return OfficialCheqqersEnv.from_tournament_json(record["payload"])


# ── Default weights reproduce the hand-written evaluation ──────────────────────────────────────────────────────


class TestDefaultWeights:
    def test_evaluate_state_returns_recorded_values_when_weights_are_default(self):
        assert len(REFERENCE) == 82
        for record in REFERENCE:
            env = env_of(record)
            assert env.evaluate_state("white") == pytest.approx(record["white"], abs=1e-9)
            assert env.evaluate_state("black") == pytest.approx(record["black"], abs=1e-9)

    def test_evaluation_is_the_weighted_sum_of_features_when_not_clipped(self):
        weights = EvalWeights()
        for record in REFERENCE:
            env = env_of(record)
            features = env.eval_features("white")
            assert len(features) == len(EVAL_FEATURES)
            total = weights.scale * sum(w * f for w, f in zip(weights.as_vector(), features))
            assert env.evaluate_state("white") == pytest.approx(max(-900.0, min(900.0, total)), abs=1e-9)

    def test_features_change_sign_when_colour_changes(self):
        for record in REFERENCE[:20]:
            env = env_of(record)
            white, black = env.eval_features("white"), env.eval_features("black")
            assert white == pytest.approx([-f for f in black], abs=1e-12)

    def test_new_features_have_zero_weight_when_weights_are_default(self):
        weights = EvalWeights()
        assert weights.exposed_mover == weights.exposed_waiter == 0.0
        assert weights.superposition_when_few == weights.material_share == 0.0


# ── Other weights ──────────────────────────────────────────────────────────────────────────────────────────────


class TestCustomWeights:
    def test_evaluation_changes_only_where_a_piece_is_in_superposition_when_that_weight_changes(self):
        without_bonus = dataclasses.replace(EvalWeights(), superposition=0.0)
        index = EVAL_FEATURES.index("superposition")
        changed = unchanged = 0
        for record in REFERENCE:
            env = env_of(record)
            has_superposition = abs(env.eval_features("white")[index]) > 1e-12
            env.eval_weights = without_bonus
            same = env.evaluate_state("white") == pytest.approx(record["white"], abs=1e-9)
            assert same != has_superposition
            changed += has_superposition
            unchanged += not has_superposition
        assert changed > 0 and unchanged > 0

    def test_weights_round_trip_through_a_json_file(self, tmp_path):
        weights = dataclasses.replace(EvalWeights(), queens=1.75, exposed_waiter=-0.35)
        path = tmp_path / "weights.json"
        path.write_text(json.dumps(weights.to_dict()))
        assert EvalWeights.from_json(path) == weights

    def test_from_dict_raises_when_a_name_is_unknown(self):
        with pytest.raises(ValueError):
            EvalWeights.from_dict({"quens": 1.75})

    def test_clone_keeps_the_weights(self):
        env = env_of(REFERENCE[0])
        env.eval_weights = dataclasses.replace(EvalWeights(), men=2.0)
        assert env.clone().eval_weights is env.eval_weights


# ── The player carries its weights into the search ─────────────────────────────────────────────────────────────


class TestPlayerWeights:
    def test_get_move_sets_the_players_weights_on_the_position(self):
        record = next(r for r in REFERENCE if len(env_of(r).get_legal_moves()) > 1)
        weights = dataclasses.replace(EvalWeights(), superposition=-0.3)
        env = env_of(record)
        ExpectiCheqqersPlayer(max_depth=1, time_limit_ms=60000.0, eval_weights=weights).get_move(env)
        assert env.eval_weights is weights

    def test_get_move_resets_the_position_to_default_weights_when_the_player_has_none(self):
        record = next(r for r in REFERENCE if len(env_of(r).get_legal_moves()) > 1)
        env = env_of(record)
        env.eval_weights = dataclasses.replace(EvalWeights(), men=5.0)  # left there by another player
        ExpectiCheqqersPlayer(max_depth=1, time_limit_ms=60000.0).get_move(env)
        assert env.eval_weights is None


# ── The server's weights come from the environment ─────────────────────────────────────────────────────────────


class TestWeightsFromEnvironment:
    def test_weights_are_none_when_the_variable_is_unset(self, monkeypatch):
        from search.expecti_cheqqers import eval_weights_from_environment

        monkeypatch.delenv("CHEQQERS_EVAL_WEIGHTS", raising=False)
        assert eval_weights_from_environment() is None

    def test_weights_load_from_the_named_file_when_the_variable_is_set(self, monkeypatch, tmp_path):
        from search.expecti_cheqqers import eval_weights_from_environment

        weights = dataclasses.replace(EvalWeights(), queens=1.75)
        path = tmp_path / "fitted.json"
        path.write_text(json.dumps(weights.to_dict()))
        monkeypatch.setenv("CHEQQERS_EVAL_WEIGHTS", str(path))
        assert eval_weights_from_environment() == weights

    def test_loading_raises_when_the_named_file_is_missing(self, monkeypatch, tmp_path):
        from search.expecti_cheqqers import eval_weights_from_environment

        monkeypatch.setenv("CHEQQERS_EVAL_WEIGHTS", str(tmp_path / "nowhere.json"))
        with pytest.raises(FileNotFoundError):
            eval_weights_from_environment()


# ── Evaluation on the win-probability scale ────────────────────────────────────────────────────────────────────


class TestWinProbabilityScale:
    """With `logit_per_man` set, the evaluation is 900 * tanh(logit / 2): the expected result of the game on a
    scale from -900 to 900, so that an average over chance outcomes is an average of results, not of points."""

    def test_default_weights_stay_on_the_linear_scale(self):
        assert EvalWeights().logit_per_man == 0.0

    def test_squashed_value_orders_positions_as_the_linear_value_does(self):
        squashed = dataclasses.replace(EvalWeights(), logit_per_man=0.8)
        pairs = []
        for record in REFERENCE:
            env = env_of(record)
            env.eval_weights = squashed
            pairs.append((record["white"], env.evaluate_state("white")))
        pairs.sort()
        for (linear_a, squashed_a), (linear_b, squashed_b) in zip(pairs, pairs[1:]):
            if linear_b - linear_a > 1e-6 and abs(linear_b) < 899 and abs(linear_a) < 899:
                assert squashed_b > squashed_a

    def test_squashed_value_stays_inside_the_non_terminal_range_and_is_antisymmetric(self):
        squashed = dataclasses.replace(EvalWeights(), logit_per_man=0.8)
        for record in REFERENCE:
            env = env_of(record)
            env.eval_weights = squashed
            white, black = env.evaluate_state("white"), env.evaluate_state("black")
            assert -900.0 < white < 900.0
            assert white == pytest.approx(-black, abs=1e-9)

    def test_squashed_value_matches_the_formula(self):
        import math

        weights = dataclasses.replace(EvalWeights(), logit_per_man=0.8)
        env = env_of(REFERENCE[3])
        men = sum(w * f for w, f in zip(weights.as_vector(), env.eval_features("white")))
        env.eval_weights = weights
        assert env.evaluate_state("white") == pytest.approx(900.0 * math.tanh(0.5 * 0.8 * men), abs=1e-9)
