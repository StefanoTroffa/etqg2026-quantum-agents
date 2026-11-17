"""The policy's move head and move order (src/models/cheqqers_policy_value.py, track 33 recipe step 1).

The original head sums the squares' vectors, so a->b and b->a (and the permutations of a three-square move) get the
same logit. The "ordered" head projects the slot vectors concatenated in move order: those moves differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]

torch = pytest.importorskip("torch")

from models.cheqqers_policy_value import (  # noqa: E402
    CheqqersPolicyValue,
    batch_moves,
    load_policy_value,
    model_config,
)

MOVES = [[[3, 7], [7, 3], [2, 5, 9], [2, 9, 5], [9, 5, 2]]]


def _logits(model: CheqqersPolicyValue) -> list[float]:
    torch.manual_seed(1)
    sq, gl = torch.randn(1, 18, 7), torch.randn(1, 18)
    idx, lens = batch_moves(MOVES, 5)
    with torch.no_grad():
        logits, _ = model.eval()(sq, gl, idx, lens)
    return logits[0].tolist()


def test_sum_head_cannot_tell_reversed_moves() -> None:
    torch.manual_seed(0)
    lg = _logits(CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64, move_head="sum"))
    assert abs(lg[0] - lg[1]) < 1e-5
    assert abs(lg[2] - lg[3]) < 1e-5 and abs(lg[2] - lg[4]) < 1e-5


def test_ordered_head_tells_every_order_apart() -> None:
    torch.manual_seed(0)
    lg = _logits(CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64, move_head="ordered"))
    assert abs(lg[0] - lg[1]) > 1e-4
    assert min(abs(lg[2] - lg[3]), abs(lg[2] - lg[4]), abs(lg[3] - lg[4])) > 1e-4


def test_config_round_trip_keeps_the_head(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64, move_head="ordered")
    path = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "config": model_config(model)}, path)
    loaded = load_policy_value(path)
    assert loaded.move_head_kind == "ordered" and _logits(loaded) == pytest.approx(_logits(model))


def test_old_checkpoints_load_as_sum(tmp_path: Path) -> None:
    model = CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64)
    path = tmp_path / "old.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"d_model": 32, "layers": 1, "nhead": 4, "ff": 64}}, path)
    assert load_policy_value(path).move_head_kind == "sum"
