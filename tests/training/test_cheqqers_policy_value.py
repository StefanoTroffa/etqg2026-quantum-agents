"""The Cheqqers policy-value transformer (src/models/cheqqers_policy_value.py)."""

import torch

from models.cheqqers_policy_value import CheqqersPolicyValue, batch_moves, encode_position


def _record(turn, pieces, presence):
    return {"turn": turn, "pieces": pieces, "presence": presence, "features": [0.0] * 17, "multi_jump": -1}


def test_encoding_is_from_the_side_to_move():
    pieces = [1, 3] + [0] * 16
    presence = [1.0, 0.5] + [0.0] * 16
    white_sq, _ = encode_position(_record(0, pieces, presence))
    black_sq, _ = encode_position(_record(1, pieces, presence))
    assert white_sq[0][:5] == [0, 1, 0, 0, 0] and black_sq[0][:5] == [0, 0, 0, 1, 0]  # own man / enemy man
    assert white_sq[1][:5] == [0, 0, 0, 1, 0] and white_sq[1][6] == 1.0  # enemy man in superposition
    assert white_sq[2][:5] == [1, 0, 0, 0, 0]


def test_forward_scores_every_legal_move_and_masks_padding():
    model = CheqqersPolicyValue(d_model=32, layers=1, ff=64)
    sq, gl = encode_position(_record(0, [1] * 6 + [0] * 6 + [3] * 6, [1.0] * 6 + [0.0] * 6 + [1.0] * 6))
    moves = [[[3, 6], [4, 6, 7]], [[3, 6]]]
    idx, lens = batch_moves(moves, 2)
    logits, value = model(torch.tensor([sq, sq]), torch.tensor([gl, gl]), idx, lens)
    assert logits.shape == (2, 2) and value.shape == (2, 3)
    assert torch.isfinite(logits[0]).all() and torch.isinf(logits[1, 1])


def test_split_and_merge_on_the_same_squares_score_differently():
    """[a, b, c] as a split and [b, c, a] as a merge use the same squares in another order."""
    torch.manual_seed(0)
    model = CheqqersPolicyValue(d_model=32, layers=1, ff=64)
    sq, gl = encode_position(_record(0, [1] * 6 + [0] * 12, [1.0] * 6 + [0.0] * 12))
    idx, lens = batch_moves([[[4, 6, 7], [6, 7, 4]]], 2)
    logits, _ = model(torch.tensor([sq]), torch.tensor([gl]), idx, lens)
    assert logits[0, 0] != logits[0, 1]
