"""Tournament adapter over the exact TiqTaqToe tables in `exact/` (quantumness 0-2, decoherence off).

`ExactTiqTaqToePlayer.get_move(payload)` takes the JSON body the organisers' harness POSTs to `/move` and returns
the optimal move in harness form (`[i]`, `[a, b]`, `[i, "M"]`), or `None` whenever the tables cannot answer, so the
caller falls back to search. It never guesses: `None` covers a configuration outside the model (quantumness 3-4,
decoherence on, whose counters the payload does not carry), tables that are not on disk, a state the model cannot
hold, and a table move that is not in the payload's `legalMoves`.

The tables for quantumness 1 and 2 (decoherence off) are committed under `exact/results/data/`; others are generated (about an hour for the tournament default):

    cd exact && python3 -m tiqtaqtoe.solve --quantumness 2 --decoherence 0

Loading a configuration takes a few seconds, so `preload()` should run at server start, not on the first move.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from pathlib import Path

_EXACT_ROOT = Path(__file__).resolve().parents[2] / "exact"
if str(_EXACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXACT_ROOT))

from tiqtaqtoe import engine as E  # noqa: E402
from tiqtaqtoe.compact import Config  # noqa: E402
from tiqtaqtoe.solve import W, choose  # noqa: E402
from tiqtaqtoe.teacher import OutOfConfiguration, Teacher  # noqa: E402

ZERO_COUNTERS = (0,) * 9
TOURNAMENT_CONFIG = (2, 0)  # quantumness, decoherence of the June tournament and of the harness default


def _normalise_payload(payload: dict) -> dict:
    """The payload with every amplitude as {"real", "imaginary"}, the keys the current harness sends. The June
    tournament server sent {"re", "imaginary"}; "im" is accepted too."""
    state = {}
    for branch, amp in payload["quantumState"].items():
        real = amp.get("real", amp.get("re"))
        imag = amp.get("imaginary", amp.get("im"))
        if real is None or imag is None:
            raise ValueError(f"amplitude {amp!r} of {branch} has no real/imaginary part")
        state[branch] = {"real": float(real), "imaginary": float(imag)}
    return {**payload, "quantumState": state}


class ExactTiqTaqToePlayer:
    """Plays from the solved tables. `w` is what a tie or a draw is worth to us, a win being 1 and a loss 0; 1/2 is
    the zero-sum reading of the match score (a win is a point, a tie a point each, a draw none)."""

    def __init__(self, w=Fraction(1, 2), table_dir: str | os.PathLike | None = None):
        if w not in W:
            raise ValueError(f"draw weight {w!r} is not one of {', '.join(map(str, W))}")
        self.w = w
        self.table_dir = Path(table_dir) if table_dir is not None else None
        self._teachers: dict[tuple[int, int], Teacher | None] = {}
        self._load_errors: dict[tuple[int, int], str] = {}
        self.last_reason = ""  # why the last get_move() returned None, or a line about the move it chose

    def _teacher(self, quantumness: int, decoherence: int):
        """The Teacher of (quantumness, decoherence, manual measurement on), loaded once; None, with the reason kept,
        if the configuration is outside the model or its tables are missing or stale."""
        key = (quantumness, decoherence)
        if key not in self._teachers:
            try:
                cfg = Config(quantumness, decoherence, True)
                kwargs = {} if self.table_dir is None else {"table_dir": self.table_dir}
                self._teachers[key] = Teacher(cfg, **kwargs)
            except (ValueError, RuntimeError) as err:
                self._teachers[key] = None
                self._load_errors[key] = str(err)
        return self._teachers[key]

    def preload(self, quantumness: int = TOURNAMENT_CONFIG[0], decoherence: int = TOURNAMENT_CONFIG[1]) -> bool:
        """Load one configuration now. True if its tables are available."""
        return self._teacher(quantumness, decoherence) is not None

    def get_move(self, payload: dict):
        """The optimal move for the harness payload, or None if the tables cannot answer (see last_reason)."""
        settings = payload.get("settings") or {}
        quantumness = int(settings.get("quantumness", TOURNAMENT_CONFIG[0]))
        decoherence = int(settings.get("decoherence", 0) or 0)
        if decoherence != 0:
            self.last_reason = f"decoherence {decoherence}: the payload carries no counters"
            return None
        teacher = self._teacher(quantumness, decoherence)
        if teacher is None:
            self.last_reason = self._load_errors.get((quantumness, decoherence), "no tables")
            return None
        try:
            state = E.from_round_state(_normalise_payload(payload), counters=ZERO_COUNTERS)
            # the tables are indexed by what a tie or draw is worth to X; to O it is worth 1 - that
            x_weight = self.w if state.turn == 1 else 1 - self.w
            options = teacher.move_values(state, x_weight)
        except (OutOfConfiguration, ValueError, KeyError, TypeError) as err:
            self.last_reason = f"state outside the tables: {err}"
            return None
        if not options:
            self.last_reason = "the tables list no move"
            return None

        legal = [list(m) for m in payload.get("legalMoves", [])]
        cands = [(move, entry) for move, _, entry in options if not legal or E.move_to_harness(move) in legal]
        if not cands:
            self.last_reason = "no table move is among the payload's legalMoves"
            return None
        move, entry = choose(state.turn, cands, x_weight)  # the solver's own policy and tie-breaks
        harness_move = E.move_to_harness(move)
        self.last_reason = (f"exact q{quantumness}: {harness_move} -> P(X wins) {entry.x}, P(O wins) {entry.o}, "
                            f"P(tie) {entry.tie}, P(draw) {entry.draw}")
        return harness_move
