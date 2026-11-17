"""Policy-value transformer for 6x6 Cheqqers that scores each legal move directly (no fixed action space).

Input, from the side to move's point of view (the self-play records of native_cheqqers.selfplay hold all of it):
* 18 square tokens: piece one-hot (none, own man, own queen, enemy man, enemy queen), occupied probability,
  "in superposition" flag (0 < p < 1), plus a learned embedding per square;
* 1 global token: the 17 evaluation features for the side to move and "a multi-jump is pending".

Output:
* policy: one logit per legal move, from the encoded squares of the move in order ([from, to] or the three squares
  of a split [from, to1, to2] / merge [from1, from2, to]) with an embedding per position in the move; the network
  tells split from merge by the squares' contents;
* value: win / draw / loss logits for the side to move.

The former fixed 288-action map collapses flying-queen moves along one ray into one action; scoring moves from
their squares has no collisions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

SQUARES = 18
N_FEATURES = 17
SQUARE_IN = 5 + 2  # piece one-hot + probability + superposition flag
GLOBAL_IN = N_FEATURES + 1


def encode_position(record: dict) -> tuple[list[list[float]], list[float]]:
    """(18 square vectors, global vector) for one self-play record, from the side to move's point of view."""
    own_white = record["turn"] == 0
    squares = []
    for piece, p in zip(record["pieces"], record["presence"]):
        onehot = [0.0] * 5
        if piece == 0 or p <= 1e-4:
            onehot[0] = 1.0
        else:
            white = piece in (1, 2)
            queen = piece in (2, 4)
            own = white == own_white
            onehot[(1 if own else 3) + (1 if queen else 0)] = 1.0
        squares.append(onehot + [float(p), 1.0 if 1e-4 < p < 1 - 1e-4 else 0.0])
    glob = [float(x) for x in record["features"]] + [1.0 if record.get("multi_jump", -1) >= 0 else 0.0]
    return squares, glob


class CheqqersPolicyValue(nn.Module):
    def __init__(self, d_model: int = 96, nhead: int = 4, layers: int = 3, ff: int = 256, max_move_len: int = 3,
                 move_head: str = "sum"):
        super().__init__()
        # move_head "sum" (original): the move vector is the sum over its squares of (encoding + slot embedding), which
        # erases the order: a->b and b->a, and the permutations of a three-square move, score the same (13.7% of
        # self-play positions have such twins among their legal moves). "ordered": the slot vectors are concatenated
        # in move order (zero where the move is shorter) and projected, so every ordering is a different vector.
        if move_head not in ("sum", "ordered"):
            raise ValueError(f"move_head must be 'sum' or 'ordered', not {move_head!r}")
        self.move_head_kind = move_head
        self.square_in = nn.Linear(SQUARE_IN, d_model)
        self.global_in = nn.Linear(GLOBAL_IN, d_model)
        self.square_embed = nn.Parameter(torch.randn(SQUARES, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.move_pos = nn.Parameter(torch.randn(max_move_len, d_model) * 0.02)
        self.move_len = nn.Embedding(max_move_len + 1, d_model)
        self.move_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.max_move_len = max_move_len
        if move_head == "ordered":
            self.move_proj = nn.Linear(max_move_len * d_model, d_model)

    def encode(self, squares: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """squares (B, 18, 7), glob (B, 18) -> square encodings (B, 18, d), global encoding (B, d)."""
        tokens = torch.cat([self.global_in(glob).unsqueeze(1), self.square_in(squares) + self.square_embed], dim=1)
        h = self.norm(self.encoder(tokens))
        return h[:, 1:], h[:, 0]

    def forward(self, squares: torch.Tensor, glob: torch.Tensor, move_squares: torch.Tensor,
                move_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """move_squares (B, M, 3) square indices (padding any value), move_lens (B, M) with 0 = no move.

        Returns move logits (B, M) with -inf at padding, and value logits (B, 3) = (win, draw, loss)."""
        sq, g = self.encode(squares, glob)
        b, m, k = move_squares.shape
        gathered = torch.gather(sq, 1, move_squares.clamp(0, SQUARES - 1).reshape(b, m * k, 1)
                                .expand(-1, -1, sq.shape[-1])).reshape(b, m, k, -1)
        slot = torch.arange(k, device=move_lens.device).view(1, 1, k) < move_lens.unsqueeze(-1)
        if self.move_head_kind == "ordered":
            per_slot = gathered * slot.unsqueeze(-1)  # (b, m, k, d), zero beyond the move's length
            if k < self.max_move_len:
                per_slot = torch.nn.functional.pad(per_slot, (0, 0, 0, self.max_move_len - k))
            move_vec = self.move_proj(per_slot.reshape(b, m, -1)) + self.move_len(move_lens)
        else:
            move_vec = ((gathered + self.move_pos[:k]) * slot.unsqueeze(-1)).sum(2) + self.move_len(move_lens)
        logits = self.move_head(torch.cat([move_vec, g.unsqueeze(1).expand(-1, m, -1)], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(move_lens == 0, float("-inf"))
        return logits, self.value_head(g)


def batch_moves(moves: Sequence[Sequence[Sequence[int]]], max_moves: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad lists of legal moves to (B, max_moves, 3) indices and (B, max_moves) lengths."""
    idx = torch.zeros(len(moves), max_moves, 3, dtype=torch.long)
    lens = torch.zeros(len(moves), max_moves, dtype=torch.long)
    for i, ms in enumerate(moves):
        for j, mv in enumerate(ms[:max_moves]):
            idx[i, j, :len(mv)] = torch.tensor(mv, dtype=torch.long)
            lens[i, j] = len(mv)
    return idx, lens


# ── Flat inputs (track 33): the C++ encoding (native value_net.h encode_position) = encode_position flattened ──────

FLAT_IN = SQUARES * SQUARE_IN + GLOBAL_IN  # 144


def split_flat(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 144) flat encodings -> squares (B, 18, 7), glob (B, 18)."""
    return x[:, : SQUARES * SQUARE_IN].reshape(-1, SQUARES, SQUARE_IN), x[:, SQUARES * SQUARE_IN:]


def flat_encoding(record: dict) -> list[float]:
    """encode_position(record) flattened: the same 144 numbers the C++ engine writes."""
    sq, gl = encode_position(record)
    return [v for row in sq for v in row] + gl


def model_config(model: "CheqqersPolicyValue") -> dict:
    layer = model.encoder.layers[0]
    return {"d_model": model.square_in.out_features, "layers": len(model.encoder.layers),
            "nhead": layer.self_attn.num_heads, "ff": layer.linear1.out_features,
            "move_head": getattr(model, "move_head_kind", "sum")}


def load_policy_value(path, device: str = "cpu") -> "CheqqersPolicyValue":
    """A trained model.pt (state dict + config; configs without nhead/ff are the 4-head, 256-wide default)."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = dict(blob.get("config", {}))
    model = CheqqersPolicyValue(d_model=int(cfg.get("d_model", 96)), nhead=int(cfg.get("nhead", 4)),
                                layers=int(cfg.get("layers", 3)), ff=int(cfg.get("ff", 256)),
                                move_head=str(cfg.get("move_head", "sum")))
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()
