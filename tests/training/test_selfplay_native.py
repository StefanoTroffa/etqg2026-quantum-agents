"""Self-play data from the native engine (src/training/selfplay_native.py)."""

import json
import random

import pytest

from search import native_cheqqers

try:
    native_cheqqers.load()
except Exception as exc:  # the library cannot be built or loaded here
    pytest.skip(f"native library unavailable: {exc}", allow_module_level=True)

from training.selfplay_native import generate, random_opening


def test_random_opening_is_reproducible_and_in_progress():
    a = random_opening(random.Random(7), 4)
    b = random_opening(random.Random(7), 4)
    assert a is not None and not a.done
    assert a.to_tournament_payload() == b.to_tournament_payload()


def test_generate_writes_positions_and_one_result_per_game(tmp_path):
    out = tmp_path / "shard.jsonl"
    positions = generate(out, games=2, seed=3, ms=20.0, depth=3, qp=2, opening_plies=(2, 4), write_state=False)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    results = [line for line in lines if "result" in line]
    assert len(results) == 2
    assert positions == len(lines) - len(results) > 0
    for line in lines:
        if "result" not in line:
            assert line["move"] in line["moves"]


def test_generate_accepts_evaluation_weights(tmp_path):
    from env.cheqqers_official import EvalWeights

    weights = EvalWeights.from_json("config/cheqqers_eval_weights_2026-11-13.json")
    out = tmp_path / "w.jsonl"
    assert generate(out, games=1, seed=4, ms=20.0, depth=3, qp=2, opening_plies=(2, 3), write_state=False,
                    eval_weights=weights) > 0
