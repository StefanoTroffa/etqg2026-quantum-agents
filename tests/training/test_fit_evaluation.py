"""Fitting the Cheqqers evaluation weights to games (src/training/fit_evaluation.py)."""

import json

import numpy as np
import pytest

from env.cheqqers_official import EVAL_FEATURES, EvalWeights
from training.fit_evaluation import fit_logistic, fit_weights, generate, load_positions, log_loss, play_game

K = len(EVAL_FEATURES)


def synthetic_games(true_vector: np.ndarray, games: int = 400, positions: int = 30, seed: int = 0):
    """Feature rows drawn at random, with each game's result drawn from the logistic curve of its first row's
    score under `true_vector`: data in which that vector is the right answer."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(games * positions, K))
    game = np.repeat(np.arange(games), positions)
    y = (rng.random(games * positions) < 1.0 / (1.0 + np.exp(-(X @ true_vector)))).astype(float)
    return X, y, game


# ── The regression ─────────────────────────────────────────────────────────────────────────────────────────────


class TestFitLogistic:
    def test_fit_recovers_the_vector_when_data_follow_it(self):
        true_vector = np.linspace(-1.0, 1.5, K)
        X, y, _ = synthetic_games(true_vector)
        fitted = fit_logistic(X, y, l2=1e-6)
        cosine = fitted @ true_vector / (np.linalg.norm(fitted) * np.linalg.norm(true_vector))
        assert cosine > 0.99

    def test_fit_gives_opposite_scores_to_a_position_and_its_mirror(self):
        X, y, _ = synthetic_games(np.ones(K), games=50)
        fitted = fit_logistic(X, y)
        assert X[0] @ fitted == pytest.approx(-((-X[0]) @ fitted))

    def test_log_loss_is_log_two_when_scores_are_zero(self):
        assert log_loss(np.zeros(10), np.array([0.0, 1.0] * 5)) == pytest.approx(np.log(2.0))


class TestFitWeights:
    def test_weights_value_a_man_at_one_when_fitted(self):
        true_vector = np.linspace(0.5, 2.0, K)
        weights, report = fit_weights(*synthetic_games(true_vector))
        assert weights.men == pytest.approx(1.0)
        assert weights.scale == 25.0
        assert report["slope_per_man"] == pytest.approx(true_vector[0], rel=0.15)

    def test_fitted_weights_predict_held_out_games_better_than_the_defaults(self):
        _, report = fit_weights(*synthetic_games(np.linspace(1.5, -1.0, K)))  # a man is worth something
        losses = report["held_out_log_loss"]
        assert losses["fitted_weights"] < losses["default_weights"]
        assert losses["fitted_weights"] < losses["coin"]

    def test_fit_raises_when_a_man_comes_out_worthless(self):
        true_vector = np.ones(K)
        true_vector[EVAL_FEATURES.index("men")] = -1.0
        with pytest.raises(ValueError):
            fit_weights(*synthetic_games(true_vector))


# ── Games and files ────────────────────────────────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_play_game_records_one_feature_row_per_ply(self):
        game = play_game(seed=3, depth=1, epsilon=1.0, opening_plies=2, max_plies=12)
        assert game["result_white"] in (0.0, 0.5, 1.0)
        assert len(game["positions"]) == game["plies"]
        assert all(len(row) == 3 + K for row in game["positions"])
        assert [row[0] for row in game["positions"]] == list(range(game["plies"]))

    def test_play_game_repeats_when_seed_repeats(self):
        first = play_game(seed=4, depth=1, epsilon=0.5, opening_plies=2, max_plies=10)
        second = play_game(seed=4, depth=1, epsilon=0.5, opening_plies=2, max_plies=10)
        assert first == second

    def test_generate_skips_seeds_already_in_the_file(self, tmp_path):
        out = tmp_path / "games.jsonl"
        assert generate(out, games=2, workers=1, seed=1, depth=1, epsilon=1.0, opening_plies=2) == 2
        assert generate(out, games=3, workers=1, seed=1, depth=1, epsilon=1.0, opening_plies=2) == 1
        assert sorted(json.loads(line)["seed"] for line in out.read_text().splitlines()) == [1, 2, 3]

    def test_generate_reports_and_skips_a_game_when_it_raises(self, tmp_path, monkeypatch, capsys):
        import training.fit_evaluation as module

        real = module.play_game

        def fails_on_seed_two(seed, **kwargs):
            if seed == 2:
                raise RuntimeError("boom")
            return real(seed, **kwargs)

        monkeypatch.setattr(module, "play_game", fails_on_seed_two)
        out = tmp_path / "games.jsonl"
        assert generate(out, games=3, workers=1, seed=1, depth=1, epsilon=1.0, opening_plies=2) == 2
        assert sorted(json.loads(line)["seed"] for line in out.read_text().splitlines()) == [1, 3]
        assert "seed 2 failed: RuntimeError: boom" in capsys.readouterr().err

    def test_load_positions_leaves_out_opening_plies_and_pending_captures(self, tmp_path):
        row = [0.0] * K
        game = {"seed": 1, "result_white": 1.0, "plies": 4,
                "positions": [[0, 1, 0, *row], [4, 1, 0, *row], [5, 0, 1, *row], [6, 1, 0, *row]]}
        path = tmp_path / "one.jsonl"
        path.write_text(json.dumps(game) + "\n")
        X, y, game_of = load_positions([path], min_ply=4, skip_captures=True)
        assert X.shape == (2, K) and y.tolist() == [1.0, 1.0] and game_of.tolist() == [0, 0]
        assert load_positions([path], min_ply=0, skip_captures=False)[0].shape == (4, K)


class TestDefaultsAreData:
    def test_default_weights_load_from_their_own_dictionary(self):
        assert EvalWeights.from_dict(EvalWeights().to_dict()) == EvalWeights()


# ── Arena games as data, and scoring weights on any data ───────────────────────────────────────────────────────


class TestArenaGamesAsData:
    def arena_games(self) -> list[dict]:
        from pathlib import Path

        fixture = Path(__file__).parents[1] / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl"
        return [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()][:4]

    def test_arena_game_becomes_one_feature_row_per_ply_with_its_result(self):
        from arena.arena_records import replay
        from training.fit_evaluation import arena_game_record

        for game in self.arena_games():
            record = arena_game_record(game)
            assert record["plies"] == len(game["moves"]) == len(record["positions"])
            assert record["result_white"] == {"white": 1.0, "black": 0.0, "draw": 0.5}[game["result"]]
            for ply in (0, len(game["moves"]) // 2, len(game["moves"]) - 1):
                env = replay({**game, "moves": game["moves"][:ply]})
                row = record["positions"][ply]
                assert row[0] == ply
                assert row[1] == (1 if game["moves"][ply]["by"] == "white" else 0)
                assert row[2] == (1 if env.capture_pending() else 0)
                assert row[3:] == pytest.approx(list(env.eval_features("white")), abs=1e-6)

    def test_arena_records_load_like_generated_games(self, tmp_path):
        from training.fit_evaluation import arena_game_record

        path = tmp_path / "arena.jsonl"
        path.write_text("".join(json.dumps(arena_game_record(g)) + "\n" for g in self.arena_games()))
        X, y, game = load_positions([path], min_ply=4, skip_captures=True)
        assert X.shape[1] == len(EVAL_FEATURES) and len(X) == len(y) == len(game) > 20
        assert set(np.unique(game)) == {0, 1, 2, 3}


class TestScoreWeights:
    def test_weights_that_generated_the_results_score_better_than_material_alone(self):
        from training.fit_evaluation import score_weights

        true = np.zeros(len(EVAL_FEATURES))
        true[EVAL_FEATURES.index("men")] = 1.0
        true[EVAL_FEATURES.index("queens")] = 2.0
        true[EVAL_FEATURES.index("centre")] = 1.5
        X, y, _ = synthetic_games(true)
        material = np.zeros(len(EVAL_FEATURES))
        material[EVAL_FEATURES.index("men")] = 1.0
        material[EVAL_FEATURES.index("queens")] = 2.0
        best, plain = score_weights(X, y, true), score_weights(X, y, material)
        assert best["log_loss"] < plain["log_loss"] < np.log(2.0)
        assert best["slope"] > 0.0

    def test_score_is_log_two_when_the_weights_say_nothing(self):
        from training.fit_evaluation import score_weights

        X, y, _ = synthetic_games(np.ones(len(EVAL_FEATURES)))
        assert score_weights(X, y, np.zeros(len(EVAL_FEATURES)))["log_loss"] == pytest.approx(np.log(2.0), abs=0.02)

    def test_score_command_prints_a_line_for_the_defaults_and_for_each_weights_file(self, tmp_path, capsys):
        from training.fit_evaluation import arena_game_record, main

        games = TestArenaGamesAsData().arena_games()
        data = tmp_path / "arena.jsonl"
        data.write_text("".join(json.dumps(arena_game_record(g)) + "\n" for g in games))
        weights = tmp_path / "candidate.json"
        weights.write_text(json.dumps(EvalWeights().to_dict()))
        assert main(["score", "--data", str(data), "--weights", str(weights)]) == 0
        printed = capsys.readouterr().out
        assert "material only" in printed and "default weights" in printed and "candidate.json" in printed

