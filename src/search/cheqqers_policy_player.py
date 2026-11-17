"""The policy-value transformer as the player (track 33): one forward pass per move, the legal move with the highest
policy logit. No search; measurement outcomes are the engine's business (the caller applies the move).

    from search.cheqqers_policy_player import PolicyPlayer
    player = PolicyPlayer("results/cheqqers_pv/run/model.pt")      # CUDA if available, else CPU
    move = player.choose(env, legal_moves)                          # OfficialCheqqersEnv or NativeEnv

The input is the native engine's encoding of the position (NativeEnv.encode, = encode_position of a self-play record
flattened), so the player sees exactly what the network was trained on.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from models.cheqqers_policy_value import batch_moves, load_policy_value, split_flat  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402


def _move_list(move: Any) -> list[int]:
    if isinstance(move, dict):
        raise TypeError("moves must be index arrays")
    return [int(v) for v in move]


class PolicyPlayer:
    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        if device is None:
            device = os.getenv("CHEQQERS_POLICY_DEVICE", "") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.path = str(model_path)
        self.model = load_policy_value(model_path, device)
        self._lock = threading.Lock()

    @torch.no_grad()
    def evaluate(self, envs: Sequence[Any], legal: Sequence[Sequence[Any]]) -> tuple[list[list[float]], list[list[float]]]:
        """Per position: the policy logits of its legal moves (in the given order) and (win, draw, loss)
        probabilities for the side to move."""
        natives = [e if isinstance(e, nc.NativeEnv) else nc.NativeEnv.from_env(e) for e in envs]
        x = torch.tensor([n.encode() for n in natives], dtype=torch.float32)
        moves = [[_move_list(m) for m in ms] for ms in legal]
        idx, lens = batch_moves(moves, max(len(m) for m in moves))
        with self._lock:
            sq, gl = split_flat(x.to(self.device))
            logits, vlog = self.model(sq, gl, idx.to(self.device), lens.to(self.device))
        logits = logits.float().cpu()
        wdl = torch.softmax(vlog.float(), -1).cpu()
        return [logits[i, : len(moves[i])].tolist() for i in range(len(moves))], wdl.tolist()

    def choose(self, env: Any, legal_moves: Sequence[Any]) -> Any:
        """The legal move with the highest policy logit (the first of equals)."""
        if len(legal_moves) == 1:
            return legal_moves[0]
        logits, _ = self.evaluate([env], [legal_moves])
        row = logits[0]
        return legal_moves[max(range(len(row)), key=lambda i: (row[i], -i))]


_players: dict[tuple[str, str], PolicyPlayer] = {}


def get_player(model_path: str | Path, device: Optional[str] = None) -> PolicyPlayer:
    """One loaded player per (path, device) in a process."""
    key = (str(Path(model_path).expanduser().resolve()), device or "")
    if key not in _players:
        _players[key] = PolicyPlayer(key[0], device)
    return _players[key]
