from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from env.base import QuantumGameEnv

# ==============================================================================
# 1. Constants & Rules (Exact Match with local-tiqtaqtoe-server.js)
# ==============================================================================

CHAR_TO_VAL = {"0": 0, "X": 1, "O": 2}
VAL_TO_CHAR = {0: "0", 1: "X", 2: "O"}

POSSIBLE_WINS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

UI_STATES = (
    (None, (1.0, 0.0, 0.0)),
    ("X", (0.0, 1.0, 0.0)),
    ("O", (0.0, 0.0, 1.0)),
    ("half-x", (0.5, 0.5, 0.0)),
    ("half-o", (0.5, 0.0, 0.5)),
    ("xo", (0.0, 0.5, 0.5)),
    ("exO", (0.25, 0.25, 0.5)),
    ("eXo", (0.25, 0.5, 0.25)),
    ("eX", (0.25, 0.75, 0.0)),
    ("eO", (0.25, 0.0, 0.75)),
)

# Precompute all 36 (i, j) pairs with 0 <= i < j < 9 for action indexing
UPPER_TRIANGLE_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(9) for j in range(i + 1, 9)
]
PAIR_TO_ACTION_IDX: dict[tuple[int, int], int] = {
    pair: idx for idx, pair in enumerate(UPPER_TRIANGLE_PAIRS)
}


class TicTacResult:
    UNFINISHED = 0
    X = 1
    O = 2
    BOTH = 3
    DRAW = 4


class TicTacRules:
    CLASSICAL = 0
    QUANTUM_V1 = 1
    QUANTUM_V2 = 2
    QUANTUM_V3 = 3
    QUANTUM_V4 = 4


# ==============================================================================
# 2. D4 Dihedral Symmetry Tables (8 Symmetries of the 3x3 Board)
# ==============================================================================

def _build_d4_tables() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """
    Returns (forward_perms, inverse_perms) for the 8 dihedral symmetries of a 3x3 grid.
    If a piece was at square `s`, under symmetry `sym_idx` it moves to `forward_perms[sym_idx][s]`.
    """
    grid = np.arange(9, dtype=np.int32).reshape(3, 3)
    forward_perms: list[tuple[int, ...]] = []
    inverse_perms: list[tuple[int, ...]] = []

    for reflect in (False, True):
        mat = np.fliplr(grid) if reflect else grid.copy()
        for k in range(4):
            transformed = np.rot90(mat, -k).flatten()
            # transformed[new_pos] = old_pos  =>  old_pos moves to new_pos
            old_to_new = [0] * 9
            new_to_old = [0] * 9
            for new_pos, old_pos in enumerate(transformed.tolist()):
                old_to_new[old_pos] = new_pos
                new_to_old[new_pos] = old_pos
            forward_perms.append(tuple(old_to_new))
            inverse_perms.append(tuple(new_to_old))

    return forward_perms, inverse_perms


D4_FORWARD_PERMS, D4_INVERSE_PERMS = _build_d4_tables()


# ==============================================================================
# 3. Helper Functions Matching local-tiqtaqtoe-server.js & Operations.js
# ==============================================================================

def near(a: float, b: float) -> bool:
    return abs(a - b) < 0.01


def clean_amplitudes(
    state: dict[str, complex], epsilon: float = 1e-10
) -> dict[str, complex]:
    """Matches cleanAmplitudes in Operations.js:3-13."""
    return {
        k: amp
        for k, amp in state.items()
        if abs(amp.real) >= epsilon or abs(amp.imag) >= epsilon
    }


def board_str_to_basis_idx(board_str: str) -> int:
    """Matches boardStringToBasisIndex in local-tiqtaqtoe-server.js:41-48."""
    idx = 0
    for ch in board_str:
        idx = idx * 3 + CHAR_TO_VAL.get(ch, 0)
    return idx


def serialize_engine_state(state: dict[str, complex]) -> dict[str, complex]:
    """
    Matches serializeEngineState in local-tiqtaqtoe-server.js:71-81.
    Preserves V8 ascending numeric basis-index key ordering.
    """
    out: dict[str, complex] = {}
    for k, amp in sorted(state.items(), key=lambda kv: board_str_to_basis_idx(kv[0])):
        mag_sq = amp.real * amp.real + amp.imag * amp.imag
        if mag_sq >= 0.0001:
            out[k] = amp
    return out


def apply_shift_dict(
    state: dict[str, complex], sq: int, amount: int
) -> dict[str, complex]:
    """Matches system.applyLocal(sq, Operations.Shift(amount)) in Operations.js:16-32."""
    new_state: dict[str, complex] = {}
    for board_str, amp in state.items():
        val = CHAR_TO_VAL[board_str[sq]]
        new_val = (val + amount) % 3
        new_char = VAL_TO_CHAR[new_val]
        new_board = board_str[:sq] + new_char + board_str[sq + 1 :]
        new_state[new_board] = new_state.get(new_board, 0.0j) + amp
    return clean_amplitudes(new_state)


def apply_sqrtswap_dict(
    state: dict[str, complex], sq1: int, sq2: int
) -> dict[str, complex]:
    """Matches EntanglementEngine.interactMany([e[sq1], e[sq2]], Operations.SqrtSwap())."""
    new_state: dict[str, complex] = {}
    c_stay = 0.5 + 0.5j
    c_swap = 0.5 - 0.5j
    s_lo, s_hi = (sq1, sq2) if sq1 < sq2 else (sq2, sq1)

    for board_str, amp in state.items():
        c1 = board_str[s_lo]
        c2 = board_str[s_hi]
        if c1 == c2:
            new_state[board_str] = new_state.get(board_str, 0.0j) + amp
        else:
            new_state[board_str] = new_state.get(board_str, 0.0j) + amp * c_stay
            swapped = (
                board_str[:s_lo]
                + c2
                + board_str[s_lo + 1 : s_hi]
                + c1
                + board_str[s_hi + 1 :]
            )
            new_state[swapped] = new_state.get(swapped, 0.0j) + amp * c_swap

    return clean_amplitudes(new_state)


def get_target_probabilities(
    state: dict[str, complex], sq: int
) -> dict[str, float]:
    """
    Matches JointQuantumSystem.getTargetProbabilities in EntanglementEngine.js:203-216.
    Note: V8 iterates numeric object keys ("0", "1", "2") in ascending integer order!
    """
    raw_probs: dict[str, float] = {}
    for board_str, amp in state.items():
        ch = board_str[sq]
        p = amp.real * amp.real + amp.imag * amp.imag
        raw_probs[ch] = raw_probs.get(ch, 0.0) + p
    return {
        ch: raw_probs[ch]
        for ch in ("0", "X", "O")
        if ch in raw_probs
    }


def measure_target_dict(
    state: dict[str, complex],
    sq: int,
    forced_outcome: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[str, float, dict[str, complex]]:
    """
    Matches JointQuantumSystem.measureTarget in EntanglementEngine.js:236-310.
    Returns (selected_char, outcome_prob, collapsed_state).
    """
    marginal_probs = get_target_probabilities(state, sq)
    if forced_outcome is not None:
        selected_char = forced_outcome
    else:
        r = float(rng.random()) if rng is not None else float(np.random.random())
        cum = 0.0
        selected_char = ""
        for ch, prob in marginal_probs.items():
            cum += prob
            if r <= cum:
                selected_char = ch
                break
        if not selected_char:
            keys = list(marginal_probs.keys())
            selected_char = keys[-1] if keys else "0"

    prob = marginal_probs.get(selected_char, 0.0)
    collapsed: dict[str, complex] = {}
    if prob > 1e-12:
        norm = 1.0 / math.sqrt(prob)
        for board_str, amp in state.items():
            if board_str[sq] == selected_char:
                collapsed[board_str] = amp * norm
    else:
        # Edge case fallback matching EntanglementEngine.js:280-304
        max_prob = -1.0
        max_board = "000000000"
        for board_str, amp in state.items():
            mag = amp.real * amp.real + amp.imag * amp.imag
            if mag > max_prob:
                max_prob = mag
                max_board = board_str
        new_board = max_board[:sq] + selected_char + max_board[sq + 1 :]
        collapsed = {new_board: 1.0 + 0.0j}

    return selected_char, prob, clean_amplitudes(collapsed)


def parse_quantum_state(
    raw_state: dict[str, Any] | str | None,
    current_turn: str = "X",
) -> dict[str, complex]:
    """
    Parses quantumState from:
    1. Official TiqTaqToe JSON dict: {"00XOOOO0O": {"real": 0.7071, "imaginary": 0.0}, ...}
    2. Direct Python dict[str, complex]: {"00XOOOO0O": 0.7071+0j, ...}
    3. Dirac ket string format: "+0.71|010000000⟩ + 0.71|100000000⟩" or "|00X000000⟩"
    """
    if not raw_state:
        return {"000000000": 1.0 + 0.0j}

    if isinstance(raw_state, dict):
        parsed: dict[str, complex] = {}
        for k, v in raw_state.items():
            if isinstance(v, complex):
                parsed[str(k)] = v
            elif isinstance(v, (int, float)):
                parsed[str(k)] = complex(float(v), 0.0)
            elif isinstance(v, dict):
                re_val = float(v.get("real", v.get("re", 0.0)))
                im_val = float(v.get("imaginary", v.get("im", 0.0)))
                parsed[str(k)] = complex(re_val, im_val)
        return parsed if parsed else {"000000000": 1.0 + 0.0j}

    if isinstance(raw_state, str):
        parsed = {}
        # Match terms like "+0.71|010000000⟩", "- 0.5i|00X000000>", "(0.5+0.5i)|X00000000⟩"
        pattern = re.compile(r"([+\-]?\s*[0-9.eE+\-iIjJ()]+)?\s*\|([012XO]{9})[⟩>]")
        for match in pattern.finditer(raw_state):
            coeff_str, ket_str = match.group(1), match.group(2)
            # Normalize ket string ('1' -> 'X' or current_turn, '2' -> 'O')
            norm_ket = (
                ket_str.replace("1", "X").replace("2", "O")
                if ("X" not in ket_str and "O" not in ket_str)
                else ket_str
            )
            if not coeff_str or coeff_str.strip() in ("+", ""):
                amp = 1.0 + 0.0j
            elif coeff_str.strip() == "-":
                amp = -1.0 + 0.0j
            else:
                c_clean = (
                    coeff_str.replace(" ", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("i", "j")
                    .replace("I", "j")
                )
                try:
                    amp = complex(c_clean)
                except ValueError:
                    amp = 1.0 + 0.0j
            parsed[norm_ket] = parsed.get(norm_ket, 0.0j) + amp
        if parsed:
            # Normalize if ket string used rounded 2-decimal amplitudes like 0.71
            tot = sum(abs(a) ** 2 for a in parsed.values())
            if tot > 1e-12 and abs(tot - 1.0) > 1e-3:
                scale = 1.0 / math.sqrt(tot)
                parsed = {k: v * scale for k, v in parsed.items()}
            return parsed

    return {"000000000": 1.0 + 0.0j}


def derive_entanglements(
    quantum_state: dict[str, complex],
    previous_pairs: list[list[int] | None] | None = None,
) -> tuple[list[list[int] | None], dict[int, int]]:
    """
    Exact port of deriveEntanglements(quantumState, previousPairs)
    from local-tiqtaqtoe-server.js:96-230.
    """
    if previous_pairs is None:
        previous_pairs = [None, None, None, None]

    marginals: list[dict[str, float]] = [
        {"0": 0.0, "X": 0.0, "O": 0.0} for _ in range(9)
    ]
    total_amp_sq = sum(
        amp.real * amp.real + amp.imag * amp.imag for amp in quantum_state.values()
    )
    if total_amp_sq == 0.0:
        total_amp_sq = 1.0

    state_probs: list[tuple[str, float]] = []
    for board_str, amp in quantum_state.items():
        prob = (amp.real * amp.real + amp.imag * amp.imag) / total_amp_sq
        state_probs.append((board_str, prob))
        for i in range(9):
            ch = board_str[i]
            if ch in marginals[i]:
                marginals[i][ch] += prob

    adj: list[list[int]] = [[] for _ in range(9)]
    alphabet = ("0", "X", "O")

    for i in range(9):
        for j in range(i + 1, 9):
            joint: dict[str, float] = {
                "00": 0.0,
                "0X": 0.0,
                "0O": 0.0,
                "X0": 0.0,
                "XX": 0.0,
                "XO": 0.0,
                "O0": 0.0,
                "OX": 0.0,
                "OO": 0.0,
            }
            for board_str, prob in state_probs:
                key = board_str[i] + board_str[j]
                if key in joint:
                    joint[key] += prob

            is_dependent = False
            for x in alphabet:
                for y in alphabet:
                    p_joint = joint[x + y]
                    p_prod = marginals[i][x] * marginals[j][y]
                    if abs(p_joint - p_prod) > 0.01:
                        is_dependent = True
                        break
                if is_dependent:
                    break

            if is_dependent:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * 9
    raw_components: list[list[int]] = []

    for i in range(9):
        if not visited[i]:
            comp: list[int] = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nb in adj[curr]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) >= 2:
                comp.sort()
                raw_components.append(comp)

    pairs: list[list[int] | None] = [None, None, None, None]
    assigned_slots: set[int] = set()
    component_to_slot: list[dict[str, Any]] = []

    # 1. Inherit slots from previousPairs if there is overlap
    for comp in raw_components:
        best_slot = -1
        max_overlap = 0
        for slot, prev_comp in enumerate(previous_pairs):
            if prev_comp is not None and slot not in assigned_slots:
                overlap = sum(1 for x in comp if x in prev_comp)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_slot = slot
        if best_slot != -1:
            component_to_slot.append({"component": comp, "slot": best_slot})
            assigned_slots.add(best_slot)
        else:
            component_to_slot.append({"component": comp, "slot": None})

    # 2. Assign remaining components to first vacant slot
    for item in component_to_slot:
        if item["slot"] is None:
            slot = 0
            while slot in assigned_slots:
                slot += 1
            item["slot"] = slot
            assigned_slots.add(slot)

    # 3. Build squareToPair and pairs array
    square_to_pair: dict[int, int] = {}
    for item in component_to_slot:
        slot = item["slot"]
        comp = item["component"]
        while slot >= len(pairs):
            pairs.append(None)
        pairs[slot] = comp
        for idx in comp:
            square_to_pair[idx] = slot

    return pairs, square_to_pair


def compute_winner(
    classical_state: str, measured_squares: list[bool]
) -> int:
    """
    Exact port of winner(classicalState, measuredSquares)
    from local-tiqtaqtoe-server.js:237-252.
    """
    x_win_lines = 0
    o_win_lines = 0
    for line in POSSIBLE_WINS:
        if all(classical_state[idx] == "X" and measured_squares[idx] for idx in line):
            x_win_lines += 1
        if all(classical_state[idx] == "O" and measured_squares[idx] for idx in line):
            o_win_lines += 1

    if x_win_lines > o_win_lines and x_win_lines > 0:
        return TicTacResult.X
    if o_win_lines > x_win_lines and o_win_lines > 0:
        return TicTacResult.O
    if x_win_lines > 0 and o_win_lines > 0 and x_win_lines == o_win_lines:
        return TicTacResult.BOTH
    if x_win_lines == 0 and o_win_lines == 0:
        if "0" in classical_state:
            return TicTacResult.UNFINISHED
        return TicTacResult.DRAW
    return TicTacResult.UNFINISHED


def get_closest_ui_state(p: list[float] | tuple[float, float, float]) -> str | None:
    """Exact port of getClosestUiState(p) from local-tiqtaqtoe-server.js:295-306."""
    best_name: str | None = None
    min_dist = float("inf")
    for name, ref_p in UI_STATES:
        dist = abs(p[0] - ref_p[0]) + abs(p[1] - ref_p[1]) + abs(p[2] - ref_p[2])
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


# ==============================================================================
# 4. OfficialTiqTaqToeEnv (100% Parity with local-tiqtaqtoe-server.js)
# ==============================================================================

class OfficialTiqTaqToeEnv(QuantumGameEnv):
    """
    Zero-Cirq, exact 9-qutrit statevector simulator for TiqTaqToe (quantumness=0..4,
    default quantumness=3) with 100% mathematical, logical, and numerical parity
    with local-tiqtaqtoe-server.js and @quantumplayed/quantum-game-engine.
    """

    def __init__(
        self,
        quantumness: int = 3,
        decoherence: int = 0,
        round_number: int = 1,
    ) -> None:
        self.quantumness = int(quantumness)
        self.decoherence = int(decoherence)
        self.round_number = int(round_number)

        # Internal state fields matching roundState in local-tiqtaqtoe-server.js:645-666
        self.status: str = "ongoing"
        self.current_turn: str = "X" if (self.round_number % 2 == 1) else "O"
        self.winner_code: int = TicTacResult.UNFINISHED
        self.just_projected: bool = False
        self.quantum_state: dict[str, complex] = {"000000000": 1.0 + 0.0j}
        self.measured_squares: list[bool] = [False] * 9
        self.empty_squares: list[bool] = [True] * 9
        self.probabilities: list[list[float]] = [[1.0, 0.0, 0.0] for _ in range(9)]
        self.pairs: list[list[int] | None] = [None, None, None, None]
        self.square_to_pair: dict[int, int] = {}
        self.board_ui: list[str | None] = [None] * 9
        self.decoherence_counters: list[int] = [0] * 9
        self.statistics: dict[str, float | int] = {
            "numStates": 1,
            "xWinCount": 0,
            "oWinCount": 0,
            "drawCount": 0,
            "bothCount": 0,
            "xWinProb": 0.0,
            "oWinProb": 0.0,
            "drawProb": 0.0,
            "bothProb": 0.0,
        }
        self.move_count: int = 0
        self.square_turns: list[int] = [0] * 9
        self.pair_metadata: list[dict[str, Any]] = []
        self.error_message: str = ""

    # ------------------------------------------------------------------
    # QuantumGameEnv Abstract Properties & Compatibility Interface
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return 1 if self.current_turn == "X" else -1

    @current_player.setter
    def current_player(self, val: int | str) -> None:
        if isinstance(val, str):
            self.current_turn = "X" if val.upper() == "X" else "O"
        else:
            self.current_turn = "X" if int(val) == 1 else "O"

    @property
    def winner(self) -> int | None:
        if self.winner_code == TicTacResult.X:
            return 1
        if self.winner_code == TicTacResult.O:
            return -1
        if self.winner_code in (TicTacResult.BOTH, TicTacResult.DRAW):
            return 0
        return None

    @winner.setter
    def winner(self, val: int | None) -> None:
        if val == 1:
            self.winner_code = TicTacResult.X
        elif val == -1:
            self.winner_code = TicTacResult.O
        elif val == 0:
            self.winner_code = TicTacResult.DRAW
        else:
            self.winner_code = TicTacResult.UNFINISHED

    @property
    def done(self) -> bool:
        return self.winner_code != TicTacResult.UNFINISHED

    @done.setter
    def done(self, val: bool) -> None:
        if not val:
            self.winner_code = TicTacResult.UNFINISHED
            self.status = "ongoing"
        elif self.winner_code == TicTacResult.UNFINISHED:
            self.winner_code = TicTacResult.DRAW
            self.status = "completed"

    @property
    def board(self) -> list[str]:
        """
        Classical board representation: '' for unmeasured/empty, 'X' for measured X,
        'O' for measured O.
        """
        out: list[str] = [""] * 9
        first_state = next(iter(self.quantum_state.keys()), "000000000")
        for i in range(9):
            if self.measured_squares[i]:
                ch = first_state[i]
                if ch in ("X", "O"):
                    out[i] = ch
        return out

    # ------------------------------------------------------------------
    # Reset, Cloning & Tournament Payload Reconstruction
    # ------------------------------------------------------------------

    def reset(self, round_number: int | None = None) -> np.ndarray:
        if round_number is not None:
            self.round_number = int(round_number)
        self.status = "ongoing"
        self.current_turn = "X" if (self.round_number % 2 == 1) else "O"
        self.winner_code = TicTacResult.UNFINISHED
        self.just_projected = False
        self.quantum_state = {"000000000": 1.0 + 0.0j}
        self.measured_squares = [False] * 9
        self.empty_squares = [True] * 9
        self.probabilities = [[1.0, 0.0, 0.0] for _ in range(9)]
        self.pairs = [None, None, None, None]
        self.square_to_pair = {}
        self.board_ui = [None] * 9
        self.decoherence_counters = [0] * 9
        self.statistics = {
            "numStates": 1,
            "xWinCount": 0,
            "oWinCount": 0,
            "drawCount": 0,
            "bothCount": 0,
            "xWinProb": 0.0,
            "oWinProb": 0.0,
            "drawProb": 0.0,
            "bothProb": 0.0,
        }
        self.move_count = 0
        self.square_turns = [0] * 9
        self.pair_metadata = []
        self.error_message = ""
        return self.get_state()

    def clone(self) -> "OfficialTiqTaqToeEnv":
        new_env = self.__class__.__new__(self.__class__)
        new_env.quantumness = self.quantumness
        new_env.decoherence = self.decoherence
        new_env.round_number = self.round_number
        new_env.status = self.status
        new_env.current_turn = self.current_turn
        new_env.winner_code = self.winner_code
        new_env.just_projected = self.just_projected
        new_env.quantum_state = self.quantum_state.copy()
        new_env.measured_squares = self.measured_squares.copy()
        new_env.empty_squares = self.empty_squares.copy()
        new_env.probabilities = [p.copy() for p in self.probabilities]
        new_env.pairs = [p.copy() if p is not None else None for p in self.pairs]
        new_env.square_to_pair = self.square_to_pair.copy()
        new_env.board_ui = self.board_ui.copy()
        new_env.decoherence_counters = self.decoherence_counters.copy()
        new_env.statistics = self.statistics.copy()
        new_env.move_count = self.move_count
        new_env.square_turns = self.square_turns.copy()
        new_env.pair_metadata = [m.copy() for m in self.pair_metadata]
        new_env.error_message = self.error_message
        return new_env

    @classmethod
    def from_tournament_payload(cls, data: dict[str, Any]) -> "OfficialTiqTaqToeEnv":
        """
        Reconstructs an exact OfficialTiqTaqToeEnv from the /move JSON payload
        sent by local-tiqtaqtoe-server.js:683-697.
        """
        settings = data.get("settings") or {}
        quantumness = int(settings.get("quantumness", 3))
        decoherence = int(settings.get("decoherence", 0))
        env = cls(quantumness=quantumness, decoherence=decoherence)

        turn = data.get("currentTurn") or data.get("player") or "X"
        env.current_turn = "X" if str(turn).upper() == "X" else "O"

        if "quantumState" in data and data["quantumState"]:
            env.quantum_state = parse_quantum_state(data["quantumState"], env.current_turn)
        else:
            # Fallback if only board/probabilities were supplied
            env.quantum_state = {"000000000": 1.0 + 0.0j}

        if "measuredSquares" in data and data["measuredSquares"] is not None:
            env.measured_squares = [bool(x) for x in data["measuredSquares"]]
        else:
            board_in = data.get("board", [None] * 9)
            env.measured_squares = [b in ("X", "O") for b in board_in]

        if "emptySquares" in data and data["emptySquares"] is not None:
            env.empty_squares = [bool(x) for x in data["emptySquares"]]
        else:
            env.empty_squares = [
                env._compute_is_empty(i) for i in range(9)
            ]

        if "pairs" in data and data["pairs"] is not None:
            raw_pairs = data["pairs"]
            env.pairs = [
                [int(x) for x in p] if p is not None else None for p in raw_pairs
            ]
            while len(env.pairs) < 4:
                env.pairs.append(None)
        else:
            env.pairs, env.square_to_pair = derive_entanglements(env.quantum_state)

        if "squareToPair" in data and data["squareToPair"] is not None:
            env.square_to_pair = {
                int(k): int(v) for k, v in data["squareToPair"].items()
            }
        else:
            env.square_to_pair = {}
            for slot, comp in enumerate(env.pairs):
                if comp:
                    for sq in comp:
                        env.square_to_pair[sq] = slot

        env.probabilities = env._compute_square_probabilities()
        env.board_ui = env._compute_board_ui()
        env.statistics = env._compute_many_world_statistics()
        env.winner_code = env._compute_result()
        return env

    def to_tournament_payload(self) -> dict[str, Any]:
        """Serializes current state to the exact JSON payload schema of local-tiqtaqtoe-server.js."""
        return {
            "gameId": "local-test-game",
            "roundId": f"round-{self.round_number}",
            "player": self.current_turn,
            "currentTurn": self.current_turn,
            "board": self.board_ui.copy(),
            "probabilities": [p.copy() for p in self.probabilities],
            "measuredSquares": self.measured_squares.copy(),
            "emptySquares": self.empty_squares.copy(),
            "quantumState": {
                k: {"real": float(v.real), "imaginary": float(v.imag)}
                for k, v in self.quantum_state.items()
            },
            "pairs": [p.copy() if p is not None else None for p in self.pairs],
            "squareToPair": {str(k): v for k, v in self.square_to_pair.items()},
            "settings": {
                "quantumness": self.quantumness,
                "decoherence": self.decoherence,
                "timerEnabled": True,
                "moveTime": 5,
            },
            "legalMoves": self.get_available_moves(deduplicate_measurements=False),
        }

    # ------------------------------------------------------------------
    # Exact Internal Rules Matching local-tiqtaqtoe-server.js
    # ------------------------------------------------------------------

    def _compute_is_empty(self, sq: int) -> bool:
        """Matches computeIsEmpty(quantumState, square) at line 233."""
        return all(board_str[sq] == "0" for board_str in self.quantum_state.keys())

    def _is_board_full(self) -> bool:
        """Matches isBoardFull(roundState) at line 234."""
        return all(not s for s in self.empty_squares)

    def _compute_square_probabilities(self) -> list[list[float]]:
        """Matches computeSquareProbabilities(roundState) at lines 254-269."""
        probs = [[0.0, 0.0, 0.0] for _ in range(9)]
        for board_str, amp in self.quantum_state.items():
            p = amp.real * amp.real + amp.imag * amp.imag
            for j in range(9):
                ch = board_str[j]
                if ch == "0":
                    probs[j][0] += p
                elif ch == "X":
                    probs[j][1] += p
                elif ch == "O":
                    probs[j][2] += p
        return probs

    def _resolve_deterministic_squares(self) -> None:
        """Matches resolveDeterministicSquares(roundState) at lines 271-280."""
        for i in range(9):
            if not self.empty_squares[i]:
                p = self.probabilities[i]
                if near(p[1], 1.0) or near(p[2], 1.0):
                    self.measured_squares[i] = True

    def _compute_board_ui(self) -> list[str | None]:
        """Matches computeBoard(roundState) at lines 308-328."""
        board: list[str | None] = [None] * 9
        for i in range(9):
            p = self.probabilities[i]
            ui = get_closest_ui_state(p)
            if ui is not None:
                pair_idx = self.square_to_pair.get(i)
                if pair_idx is not None and pair_idx < len(self.pairs):
                    pair = self.pairs[pair_idx]
                    if pair is not None and i in pair:
                        pos = pair.index(i)
                        ui = f"{ui}{pos + 1}"
            board[i] = ui
        return board

    def _compute_result(self) -> int:
        """Matches result(roundState) at lines 349-351."""
        first_state = next(iter(self.quantum_state.keys()), "000000000")
        return compute_winner(first_state, self.measured_squares)

    def _compute_many_world_statistics(self) -> dict[str, float | int]:
        """Matches computeManyWorldStatistics(roundState) at lines 330-347."""
        all_True = [True] * 9
        stats: dict[str, float | int] = {
            "numStates": len(self.quantum_state),
            "xWinCount": 0,
            "oWinCount": 0,
            "drawCount": 0,
            "bothCount": 0,
            "xWinProb": 0.0,
            "oWinProb": 0.0,
            "drawProb": 0.0,
            "bothProb": 0.0,
        }
        for board_str, amp in self.quantum_state.items():
            prob = amp.real * amp.real + amp.imag * amp.imag
            w = compute_winner(board_str, all_True)
            if w == TicTacResult.X:
                stats["xWinCount"] = int(stats["xWinCount"]) + 1
                stats["xWinProb"] = float(stats["xWinProb"]) + prob
            elif w == TicTacResult.O:
                stats["oWinCount"] = int(stats["oWinCount"]) + 1
                stats["oWinProb"] = float(stats["oWinProb"]) + prob
            elif w == TicTacResult.BOTH:
                stats["bothCount"] = int(stats["bothCount"]) + 1
                stats["bothProb"] = float(stats["bothProb"]) + prob
            elif w == TicTacResult.DRAW:
                stats["drawCount"] = int(stats["drawCount"]) + 1
                stats["drawProb"] = float(stats["drawProb"]) + prob
        return stats

    def compute_many_worlds(self) -> dict[str, float]:
        """
        Computes exact Many-Worlds statistics across all basis states with |amp|^2 > 1e-9,
        matching computeManyWorldStatistics() in local-tiqtaqtoe-server.js plus expected
        points and turn-sum tiebreak metrics.
        """
        all_True = [True] * 9
        total_prob = sum(
            amp.real * amp.real + amp.imag * amp.imag
            for amp in self.quantum_state.values()
        )
        if total_prob <= 0.0:
            total_prob = 1.0

        x_win_prob = 0.0
        o_win_prob = 0.0
        both_prob = 0.0
        draw_prob = 0.0
        unfinished_prob = 0.0
        x_tiebreak_prob = 0.0
        o_tiebreak_prob = 0.0
        num_states = 0

        for board_str, amp in self.quantum_state.items():
            raw_p = amp.real * amp.real + amp.imag * amp.imag
            if raw_p <= 1e-9:
                continue
            num_states += 1
            p = raw_p / total_prob
            w = compute_winner(board_str, all_True)
            if w == TicTacResult.X:
                x_win_prob += p
            elif w == TicTacResult.O:
                o_win_prob += p
            elif w == TicTacResult.BOTH:
                both_prob += p
                # Turn-sum tiebreak for simultaneous 3-in-a-row
                best_x_turn = min(
                    (
                        sum(self.square_turns[idx] for idx in line)
                        for line in POSSIBLE_WINS
                        if all(board_str[idx] == "X" for idx in line)
                    ),
                    default=999,
                )
                best_o_turn = min(
                    (
                        sum(self.square_turns[idx] for idx in line)
                        for line in POSSIBLE_WINS
                        if all(board_str[idx] == "O" for idx in line)
                    ),
                    default=999,
                )
                if best_x_turn < best_o_turn:
                    x_tiebreak_prob += p
                elif best_o_turn < best_x_turn:
                    o_tiebreak_prob += p
                else:
                    x_tiebreak_prob += 0.5 * p
                    o_tiebreak_prob += 0.5 * p
            elif w == TicTacResult.DRAW:
                draw_prob += p
            else:
                unfinished_prob += p

        # Official local-tiqtaqtoe-server.js:918-928 awards +1 to X on X or BOTH, +1 to O on O or BOTH
        x_expected_points = x_win_prob + both_prob + 0.5 * draw_prob
        o_expected_points = o_win_prob + both_prob + 0.5 * draw_prob

        return {
            "numStates": float(num_states),
            "xWinProb": x_win_prob,
            "oWinProb": o_win_prob,
            "bothProb": both_prob,
            "drawProb": draw_prob,
            "unfinishedProb": unfinished_prob,
            "X": x_win_prob + x_tiebreak_prob,
            "O": o_win_prob + o_tiebreak_prob,
            "Draw": draw_prob + both_prob,
            "xExpectedPoints": x_expected_points,
            "oExpectedPoints": o_expected_points,
        }

    # ------------------------------------------------------------------
    # Move Legality & Available Moves (Exact Parity with lines 353-456)
    # ------------------------------------------------------------------

    def is_legal_move(self, move: list[Any] | tuple[Any, ...]) -> bool:
        """
        Exact port of isLegalMove(move, settings, roundState)
        from local-tiqtaqtoe-server.js:353-435.
        """
        if len(move) == 2 and (move[1] == "M" or move[1] == "measure"):
            try:
                sq = int(move[0])
            except (ValueError, TypeError):
                self.error_message = "Invalid square index."
                return False
            if sq < 0 or sq > 8:
                self.error_message = "Invalid square index."
                return False
            if self.empty_squares[sq]:
                self.error_message = "Cannot measure an empty square."
                return False
            if self.measured_squares[sq]:
                self.error_message = "Square is already measured."
                return False
            self.error_message = ""
            return True

        if len(move) == 1:
            try:
                sq = int(move[0])
            except (ValueError, TypeError):
                self.error_message = "Invalid square index."
                return False
            if sq < 0 or sq > 8 or not self.empty_squares[sq]:
                self.error_message = "Square is not empty."
                return False
            self.error_message = ""
            return True

        if len(move) == 2:
            try:
                sq1, sq2 = int(move[0]), int(move[1])
            except (ValueError, TypeError):
                self.error_message = "Invalid square index."
                return False
            if sq1 < 0 or sq1 > 8 or sq2 < 0 or sq2 > 8:
                self.error_message = "Invalid square index."
                return False
            if self.quantumness == TicTacRules.CLASSICAL and sq1 != sq2:
                self.error_message = "Quantum moves are not available in classical TTT."
                return False
            if self.measured_squares[sq1] or self.measured_squares[sq2]:
                self.error_message = (
                    "Cannot make a quantum move involving a measured square."
                )
                return False

            e1 = self.empty_squares[sq1]
            e2 = self.empty_squares[sq2]

            if e1 and e2:
                if self.quantumness == TicTacRules.CLASSICAL and sq1 != sq2:
                    self.error_message = (
                        "Quantum moves are not available in classical TTT."
                    )
                    return False

            if not e1 and not e2:
                if self.quantumness < TicTacRules.QUANTUM_V4:
                    self.error_message = (
                        "This move is only allowed at the highest quantumness level."
                    )
                    return False
                own_idx = 1 if self.current_turn == "X" else 2
                if near(self.probabilities[sq1][own_idx], 0.0):
                    self.error_message = "Phase moves must have your own tokens."
                    return False
                if self.square_to_pair.get(sq1) != self.square_to_pair.get(sq2):
                    self.error_message = "Phase moves must involve the same pair."
                    return False

            if (e1 and not e2) or (not e1 and e2):
                if self.quantumness < TicTacRules.QUANTUM_V2:
                    self.error_message = (
                        "Entangling moves are only allowed at quantumness level 2 or higher."
                    )
                    return False
                empty_sq = sq1 if e1 else sq2
                non_empty_sq = sq2 if empty_sq == sq1 else sq1
                own_idx = 1 if self.current_turn == "X" else 2
                opp_idx = 2 if self.current_turn == "X" else 1
                if not near(self.probabilities[non_empty_sq][own_idx], 0.0):
                    self.error_message = "You can't entangle with your own token."
                    return False
                if not near(self.probabilities[non_empty_sq][opp_idx], 1.0):
                    if self.quantumness < TicTacRules.QUANTUM_V3:
                        self.error_message = (
                            "Entangling with a non-full square is only allowed at level 3 or higher."
                        )
                        return False

            self.error_message = ""
            return True

        self.error_message = "Select one or maximally two squares."
        return False

    def get_available_moves(
        self, deduplicate_measurements: bool = False
    ) -> list[list[Any]]:
        """
        Exact port of getAvailableMoves(settings, roundState) from lines 437-456.
        If deduplicate_measurements=True, deduplicates [sq, 'M'] moves that belong to
        the same connected entanglement component (pair), since measuring any square in a
        component measures the entire component.
        """
        if self.done:
            return []

        moves: list[list[Any]] = []
        for i in range(9):
            if self.is_legal_move([i]):
                moves.append([i])

        if self.quantumness > 0:
            for i in range(9):
                for j in range(i + 1, 9):
                    if self.is_legal_move([i, j]):
                        moves.append([i, j])

        seen_pairs: set[int] = set()
        for i in range(9):
            if self.is_legal_move([i, "M"]):
                if deduplicate_measurements:
                    pair_idx = self.square_to_pair.get(i)
                    if pair_idx is not None and self.pairs[pair_idx] is not None:
                        if pair_idx in seen_pairs:
                            continue
                        seen_pairs.add(pair_idx)
                moves.append([i, "M"])

        return moves

    # ------------------------------------------------------------------
    # Move Execution & Branch Enumeration (Exact Parity with lines 458-643)
    # ------------------------------------------------------------------

    def _do_classical_move(
        self, target_index: int, token: str, measure_immediately: bool = False
    ) -> None:
        """Matches doClassicalMove(targetIndex, token, roundState, measureImmediately) at lines 458-471."""
        val = 1 if token == "X" else 2
        self.quantum_state = serialize_engine_state(
            apply_shift_dict(self.quantum_state, target_index, val)
        )
        self.empty_squares[target_index] = False
        if measure_immediately:
            self.measured_squares[target_index] = True
        self.pairs, self.square_to_pair = derive_entanglements(
            self.quantum_state, self.pairs
        )
        self.square_turns[target_index] = self.move_count + 1

    def _do_quantum_move(self, sq1: int, sq2: int, token: str) -> None:
        """Matches doQuantumMove(square1, square2, token, roundState) at lines 473-498."""
        val = 1 if token == "X" else 2
        e1 = self.empty_squares[sq1]
        e2 = self.empty_squares[sq2]

        if e1 and e2:
            state = apply_shift_dict(self.quantum_state, sq1, val)
            state = apply_sqrtswap_dict(state, sq1, sq2)
            self.square_turns[sq1] = self.move_count + 1
            self.square_turns[sq2] = self.move_count + 1
        elif (not e1 and e2) or (e1 and not e2):
            empty_sq = sq1 if e1 else sq2
            non_empty_sq = sq2 if empty_sq == sq1 else sq1
            state = apply_shift_dict(self.quantum_state, empty_sq, val)
            state = apply_sqrtswap_dict(state, non_empty_sq, empty_sq)
            self.square_turns[empty_sq] = self.move_count + 1
        else:
            state = apply_sqrtswap_dict(self.quantum_state, sq1, sq2)

        self.quantum_state = serialize_engine_state(state)
        for i in range(9):
            self.empty_squares[i] = self._compute_is_empty(i)
        self.pairs, self.square_to_pair = derive_entanglements(
            self.quantum_state, self.pairs
        )
        self.pair_metadata.append(
            {
                "id": len(self.pair_metadata),
                "sq1": sq1,
                "sq2": sq2,
                "symbol": token,
                "turn": self.move_count + 1,
            }
        )

    def _measure_squares(
        self,
        squares_to_measure: list[int],
        forced_outcomes: dict[int, str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Matches measureSquares(roundState, squaresToMeasure) at lines 500-526.
        Returns the product of Born probabilities for the selected outcomes.
        """
        state = self.quantum_state.copy()
        branch_prob = 1.0
        for idx in squares_to_measure:
            forced = forced_outcomes.get(idx) if forced_outcomes else None
            _, prob, state = measure_target_dict(
                state, idx, forced_outcome=forced, rng=rng
            )
            branch_prob *= prob

        self.quantum_state = serialize_engine_state(state)
        for idx in squares_to_measure:
            self.measured_squares[idx] = True

        for i in range(9):
            self.empty_squares[i] = self._compute_is_empty(i)
            if self.empty_squares[i]:
                self.measured_squares[i] = False

        self.pairs, self.square_to_pair = derive_entanglements(
            self.quantum_state, self.pairs
        )
        self.winner_code = self._compute_result()
        self.statistics = self._compute_many_world_statistics()
        self.probabilities = self._compute_square_probabilities()
        self.board_ui = self._compute_board_ui()
        return branch_prob

    def _enumerate_measure_branches(
        self, squares_to_measure: list[int]
    ) -> list[tuple[float, "OfficialTiqTaqToeEnv"]]:
        """
        Enumerates all non-zero Born probability outcomes of measuring `squares_to_measure`
        sequentially via `measureTarget`, matching `_measure_squares` exactly.
        """
        # Each partial branch is (cum_prob, current_state_dict)
        partials: list[tuple[float, dict[str, complex]]] = [
            (1.0, self.quantum_state.copy())
        ]
        for idx in squares_to_measure:
            next_partials: list[tuple[float, dict[str, complex]]] = []
            for cum_p, st in partials:
                marginals = get_target_probabilities(st, idx)
                for ch, p in marginals.items():
                    if p > 1e-9:
                        _, _, collapsed_st = measure_target_dict(
                            st, idx, forced_outcome=ch
                        )
                        next_partials.append((cum_p * p, collapsed_st))
            partials = next_partials

        # Deduplicate identical resulting serialized quantum states
        merged: dict[tuple[tuple[str, complex], ...], tuple[float, dict[str, complex]]] = {}
        for prob, st in partials:
            ser_st = serialize_engine_state(st)
            if not ser_st:
                continue
            key = tuple(sorted(ser_st.items()))
            if key in merged:
                prev_p, prev_st = merged[key]
                merged[key] = (prev_p + prob, prev_st)
            else:
                merged[key] = (prob, ser_st)

        outcomes: list[tuple[float, "OfficialTiqTaqToeEnv"]] = []
        total_p = sum(p for p, _ in merged.values())
        if total_p <= 0.0:
            total_p = 1.0

        for prob, ser_st in merged.values():
            branch_env = self.clone()
            branch_env.quantum_state = ser_st
            for idx in squares_to_measure:
                branch_env.measured_squares[idx] = True
            for i in range(9):
                branch_env.empty_squares[i] = branch_env._compute_is_empty(i)
                if branch_env.empty_squares[i]:
                    branch_env.measured_squares[i] = False
            branch_env.pairs, branch_env.square_to_pair = derive_entanglements(
                branch_env.quantum_state, branch_env.pairs
            )
            branch_env.winner_code = branch_env._compute_result()
            branch_env.statistics = branch_env._compute_many_world_statistics()
            branch_env.probabilities = branch_env._compute_square_probabilities()
            branch_env.board_ui = branch_env._compute_board_ui()
            outcomes.append((prob / total_p, branch_env))

        return outcomes

    def apply_move(
        self,
        move: list[Any] | tuple[Any, ...],
        project: bool = True,
        rng: np.random.Generator | None = None,
    ) -> "OfficialTiqTaqToeEnv":
        """
        Exact port of doMove(settings, roundState, move, token, project)
        from local-tiqtaqtoe-server.js:570-643. Mutates self and returns self.
        """
        if not self.is_legal_move(move):
            raise ValueError(self.error_message)

        token = self.current_turn
        measure_immediately = self.quantumness in (
            TicTacRules.CLASSICAL,
            TicTacRules.QUANTUM_V1,
        )

        if len(move) == 1:
            self._do_classical_move(int(move[0]), token, measure_immediately)
        elif len(move) == 2:
            if move[1] == "M" or move[1] == "measure":
                sq_idx = int(move[0])
                squares_to_measure = [sq_idx]
                pair_idx = self.square_to_pair.get(sq_idx)
                if pair_idx is not None and pair_idx < len(self.pairs):
                    pair = self.pairs[pair_idx]
                    if pair is not None:
                        for idx in pair:
                            if idx != sq_idx and not self.measured_squares[idx]:
                                squares_to_measure.append(idx)
                self._measure_squares(squares_to_measure, rng=rng)
            else:
                sq1, sq2 = int(move[0]), int(move[1])
                if sq1 == sq2:
                    self._do_classical_move(sq1, token, measure_immediately)
                else:
                    self._do_quantum_move(sq1, sq2, token)

        self.just_projected = False
        has_unmeasured = any(
            (not self.measured_squares[i]) and (not self.empty_squares[i])
            for i in range(9)
        )
        if self._is_board_full() and project and has_unmeasured:
            # Note: local-tiqtaqtoe-server.js:600 calls resolveDeterministicSquares(roundState)
            # BEFORE recomputing roundState.probabilities for the current move!
            self._resolve_deterministic_squares()
            for i in range(9):
                if not self.measured_squares[i] and not self.empty_squares[i]:
                    squares_to_measure = [i]
                    pair_idx = self.square_to_pair.get(i)
                    if pair_idx is not None and pair_idx < len(self.pairs):
                        pair = self.pairs[pair_idx]
                        if pair is not None:
                            for idx in pair:
                                if idx != i and not self.measured_squares[idx]:
                                    squares_to_measure.append(idx)
                    self._measure_squares(squares_to_measure, rng=rng)
            self.statistics = self._compute_many_world_statistics()
            self.probabilities = self._compute_square_probabilities()
            self.board_ui = self._compute_board_ui()
            self.winner_code = self._compute_result()
            self.just_projected = True
            self.move_count += 1
            self.current_turn = "O" if self.current_turn == "X" else "X"
            if self.winner_code != TicTacResult.UNFINISHED:
                self.status = "completed"
            return self

        self.statistics = self._compute_many_world_statistics()
        self.probabilities = self._compute_square_probabilities()
        self.board_ui = self._compute_board_ui()

        if self.decoherence > 0:
            touched = [
                int(m)
                for m in move
                if isinstance(m, int) or (isinstance(m, str) and m.isdigit())
            ]
            if self._compute_decoherence(self.decoherence, touched, rng=rng):
                self.statistics = self._compute_many_world_statistics()
                self.probabilities = self._compute_square_probabilities()
                self.board_ui = self._compute_board_ui()
                self.winner_code = self._compute_result()
                self.just_projected = True
                self.move_count += 1
                self.current_turn = "O" if self.current_turn == "X" else "X"
                if self.winner_code != TicTacResult.UNFINISHED:
                    self.status = "completed"
                return self

        self.winner_code = self._compute_result()
        self.move_count += 1
        self.current_turn = "O" if self.current_turn == "X" else "X"
        if self.winner_code != TicTacResult.UNFINISHED:
            self.status = "completed"
        return self

    def _compute_decoherence(
        self,
        decoherence_limit: int,
        touched_squares: list[int],
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Matches computeDecoherence at lines 528-568."""
        if decoherence_limit <= 0:
            return False
        for idx in touched_squares:
            self.decoherence_counters[idx] = 0

        squares_to_decohere: list[int] = []
        for i in range(9):
            if not self.measured_squares[i] and not self.empty_squares[i]:
                if i not in touched_squares:
                    self.decoherence_counters[i] += 1
                if self.decoherence_counters[i] >= decoherence_limit:
                    squares_to_decohere.append(i)

        if squares_to_decohere:
            unique_to_measure = list(squares_to_decohere)
            for sq in squares_to_decohere:
                pair_idx = self.square_to_pair.get(sq)
                if pair_idx is not None and pair_idx < len(self.pairs):
                    pair = self.pairs[pair_idx]
                    if pair is not None:
                        for idx in pair:
                            if (
                                idx not in unique_to_measure
                                and not self.measured_squares[idx]
                            ):
                                unique_to_measure.append(idx)
            self._measure_squares(unique_to_measure, rng=rng)
            for idx in unique_to_measure:
                self.decoherence_counters[idx] = 0
            return True
        return False

    def get_move_outcomes(
        self, move: list[Any] | tuple[Any, ...], project: bool = True
    ) -> list[tuple[float, "OfficialTiqTaqToeEnv"]]:
        """
        Exact stochastic outcome enumerator for expectimax / PUCT search:
        - For deterministic moves (classical move [sq] or quantum move [sq1, sq2] that
          does not trigger full-board projection), returns [(1.0, next_env)].
        - For measurement moves ([sq, 'M']) or moves that fill the board and trigger
          full-board collapse, enumerates ALL possible post-collapse basis branches
          with their exact Born probabilities [(prob_k, collapsed_env_k), ...].
        """
        if not self.is_legal_move(move):
            raise ValueError(self.error_message)

        base_env = self.clone()
        token = base_env.current_turn
        measure_immediately = base_env.quantumness in (
            TicTacRules.CLASSICAL,
            TicTacRules.QUANTUM_V1,
        )

        # Step 1: Apply the primary action (which is either a gate or an explicit measurement)
        if len(move) == 2 and (move[1] == "M" or move[1] == "measure"):
            sq_idx = int(move[0])
            squares_to_measure = [sq_idx]
            pair_idx = base_env.square_to_pair.get(sq_idx)
            if pair_idx is not None and pair_idx < len(base_env.pairs):
                pair = base_env.pairs[pair_idx]
                if pair is not None:
                    for idx in pair:
                        if idx != sq_idx and not base_env.measured_squares[idx]:
                            squares_to_measure.append(idx)
            branches = base_env._enumerate_measure_branches(squares_to_measure)
        else:
            if len(move) == 1:
                base_env._do_classical_move(int(move[0]), token, measure_immediately)
            else:
                sq1, sq2 = int(move[0]), int(move[1])
                if sq1 == sq2:
                    base_env._do_classical_move(sq1, token, measure_immediately)
                else:
                    base_env._do_quantum_move(sq1, sq2, token)
            branches = [(1.0, base_env)]

        # Step 2: Check full-board collapse on each branch (lines 598-620)
        final_outcomes: list[tuple[float, "OfficialTiqTaqToeEnv"]] = []
        for branch_prob, env_b in branches:
            env_b.just_projected = False
            has_unmeasured = any(
                (not env_b.measured_squares[i]) and (not env_b.empty_squares[i])
                for i in range(9)
            )
            if env_b._is_board_full() and project and has_unmeasured:
                # Match local-tiqtaqtoe-server.js:600 (uses pre-move roundState.probabilities)
                env_b._resolve_deterministic_squares()

                # Sequentially collapse remaining unmeasured squares & their pairs (lines 601-612)
                sub_branches: list[tuple[float, "OfficialTiqTaqToeEnv"]] = [
                    (1.0, env_b)
                ]
                for i in range(9):
                    next_sub: list[tuple[float, "OfficialTiqTaqToeEnv"]] = []
                    for sub_p, sub_env in sub_branches:
                        if (
                            not sub_env.measured_squares[i]
                            and not sub_env.empty_squares[i]
                        ):
                            sq_to_m = [i]
                            p_idx = sub_env.square_to_pair.get(i)
                            if p_idx is not None and p_idx < len(sub_env.pairs):
                                pair = sub_env.pairs[p_idx]
                                if pair is not None:
                                    for idx in pair:
                                        if (
                                            idx != i
                                            and not sub_env.measured_squares[idx]
                                        ):
                                            sq_to_m.append(idx)
                            for m_p, m_env in sub_env._enumerate_measure_branches(
                                sq_to_m
                            ):
                                next_sub.append((sub_p * m_p, m_env))
                        else:
                            next_sub.append((sub_p, sub_env))
                    sub_branches = next_sub

                for sub_p, sub_env in sub_branches:
                    sub_env.statistics = sub_env._compute_many_world_statistics()
                    sub_env.probabilities = sub_env._compute_square_probabilities()
                    sub_env.board_ui = sub_env._compute_board_ui()
                    sub_env.winner_code = sub_env._compute_result()
                    sub_env.just_projected = True
                    sub_env.move_count += 1
                    sub_env.current_turn = (
                        "O" if sub_env.current_turn == "X" else "X"
                    )
                    if sub_env.winner_code != TicTacResult.UNFINISHED:
                        sub_env.status = "completed"
                    final_outcomes.append((branch_prob * sub_p, sub_env))
            else:
                env_b.statistics = env_b._compute_many_world_statistics()
                env_b.probabilities = env_b._compute_square_probabilities()
                env_b.board_ui = env_b._compute_board_ui()
                env_b.winner_code = env_b._compute_result()
                env_b.move_count += 1
                env_b.current_turn = "O" if env_b.current_turn == "X" else "X"
                if env_b.winner_code != TicTacResult.UNFINISHED:
                    env_b.status = "completed"
                final_outcomes.append((branch_prob, env_b))

        return final_outcomes

    # ------------------------------------------------------------------
    # Action Space Mapping & 111-Dim State Vector (RL / QMCTS Interface)
    # ------------------------------------------------------------------

    @staticmethod
    def move_to_action_idx(move: list[Any] | tuple[Any, ...]) -> int:
        """
        Maps a tournament move to the 46-action index space (0..45):
        - 0..35: [sq1, sq2] quantum move (0 <= sq1 < sq2 < 9)
        - 36..44: [sq] classical move (0 <= sq < 9)
        - 45: [sq, 'M'] measurement move
        """
        if len(move) == 1:
            return 36 + int(move[0])
        if len(move) == 2:
            if move[1] == "M" or move[1] == "measure":
                return 45
            s1, s2 = int(move[0]), int(move[1])
            if s1 == s2:
                return 36 + s1
            pair = (s1, s2) if s1 < s2 else (s2, s1)
            return PAIR_TO_ACTION_IDX[pair]
        raise ValueError(f"Unsupported move format: {move}")

    def action_idx_to_move(self, action: int) -> list[Any]:
        """Maps a 0..45 action index back to a legal tournament move [sq], [sq1, sq2], or [sq, 'M']."""
        if 0 <= action <= 35:
            s1, s2 = UPPER_TRIANGLE_PAIRS[action]
            return [s1, s2]
        if 36 <= action <= 44:
            return [action - 36]
        if action == 45:
            # Pick the highest-leverage measurement move (largest entangled component first)
            meas_moves = [
                m
                for m in self.get_available_moves(deduplicate_measurements=True)
                if len(m) == 2 and m[1] == "M"
            ]
            if not meas_moves:
                raise ValueError("No legal measurement move available.")
            best_move = max(
                meas_moves,
                key=lambda m: len(
                    self.pairs[self.square_to_pair[m[0]]]
                    if m[0] in self.square_to_pair
                    and self.pairs[self.square_to_pair[m[0]]] is not None
                    else [m[0]]
                ),
            )
            return best_move
        raise ValueError(f"Invalid action index: {action}")

    def get_valid_actions(self) -> list[int]:
        """Returns valid action indices in 0..45 matching get_available_moves()."""
        if self.done:
            return []
        valid_set: set[int] = set()
        for move in self.get_available_moves(deduplicate_measurements=True):
            valid_set.add(self.move_to_action_idx(move))
        return sorted(valid_set)

    def step(
        self, action: int | list[Any] | tuple[Any, ...], return_state: bool = True
    ) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        """
        Executes `action` (either an integer index 0..45 or a tournament move [sq], [sq1, sq2], [sq, 'M']).
        Returns (next_state, reward, done, info).
        """
        if self.done:
            raise ValueError("Game is already finished.")

        acting_player = self.current_player
        if isinstance(action, (int, np.integer)):
            act_int = int(action)
            if act_int not in self.get_valid_actions():
                st = self.get_state() if return_state else None
                return st, -15.0, True, {"msg": "Invalid action choice."}
            move = self.action_idx_to_move(act_int)
        else:
            move = list(action)
            if not self.is_legal_move(move):
                st = self.get_state() if return_state else None
                return st, -15.0, True, {"msg": self.error_message}

        self.apply_move(move, project=True)
        collapsed = self.just_projected or (len(move) == 2 and move[1] == "M")

        reward = 0.0
        if self.done:
            if self.winner_code in (TicTacResult.BOTH, TicTacResult.DRAW):
                reward = 0.5
            elif self.winner == acting_player:
                reward = 1.0
            else:
                reward = -1.5

        next_state = self.get_state() if return_state else None
        return (
            next_state,
            reward,
            self.done,
            {
                "winner": self.winner,
                "winner_code": self.winner_code,
                "collapsed": collapsed,
                "move": move,
            },
        )

    def get_state(self) -> np.ndarray:
        """
        Returns the 111-dimensional state vector from the current player's perspective,
        preserving exact feature semantics for the trained 111-dim networks while fixing
        the unmeasured vs. measured classical square distinction:
        1. Dims 0..53 (54 dims): Per-square 6-tuple:
           (classical_self, classical_opp, provisional_self, provisional_opp, spooky_self, spooky_opp)
           - classical_self/opp: 1.0 iff measuredSquares[i] is True and square holds self/opp token
           - provisional_self/opp: 1.0 iff measuredSquares[i] is False and p_self/opp >= 0.99
           - spooky_self/opp: 1.0 iff measuredSquares[i] is False and 0.01 <= p_self/opp < 0.99
        2. Dims 54..89 (36 dims): Upper-triangle entanglement adjacency matrix in [-1, 1].
        3. Dims 90..107 (18 dims): Exact marginal collapse probabilities (prob_self, prob_opp) for i=0..8.
        4. Dims 108..110 (3 dims): Global context (active_edges / 8.0, cycle_active, active_player).
        """
        is_x = self.current_turn == "X"
        self_idx = 1 if is_x else 2
        opp_idx = 2 if is_x else 1

        state: list[float] = []

        # 1. Local board state (54 dimensions)
        for i in range(9):
            p_self = self.probabilities[i][self_idx]
            p_opp = self.probabilities[i][opp_idx]
            if self.measured_squares[i]:
                classical_self = 1.0 if p_self > 0.5 else 0.0
                classical_opp = 1.0 if p_opp > 0.5 else 0.0
                provisional_self = 0.0
                provisional_opp = 0.0
                spooky_self = 0.0
                spooky_opp = 0.0
            else:
                classical_self = 0.0
                classical_opp = 0.0
                provisional_self = 1.0 if near(p_self, 1.0) else 0.0
                provisional_opp = 1.0 if near(p_opp, 1.0) else 0.0
                spooky_self = (
                    1.0 if (not near(p_self, 0.0) and not near(p_self, 1.0)) else 0.0
                )
                spooky_opp = (
                    1.0 if (not near(p_opp, 0.0) and not near(p_opp, 1.0)) else 0.0
                )
            state.extend(
                [
                    classical_self,
                    classical_opp,
                    provisional_self,
                    provisional_opp,
                    spooky_self,
                    spooky_opp,
                ]
            )

        # 2. Spooky Adjacency Matrix (36 dimensions)
        adj_val: dict[tuple[int, int], float] = {}
        active_edges = 0
        for comp in self.pairs:
            if comp is not None and len(comp) >= 2:
                for idx_a in range(len(comp)):
                    for idx_b in range(idx_a + 1, len(comp)):
                        u, v = comp[idx_a], comp[idx_b]
                        pair_key = (u, v) if u < v else (v, u)
                        # Weight connection by net token ownership across the pair
                        p_self_uv = 0.5 * (
                            self.probabilities[u][self_idx]
                            + self.probabilities[v][self_idx]
                        )
                        p_opp_uv = 0.5 * (
                            self.probabilities[u][opp_idx]
                            + self.probabilities[v][opp_idx]
                        )
                        val = 1.0 if p_self_uv >= p_opp_uv else -1.0
                        adj_val[pair_key] = val
                        active_edges += 1

        for pair_key in UPPER_TRIANGLE_PAIRS:
            state.append(adj_val.get(pair_key, 0.0))

        # 3. Exact Quantum Collapse Probabilities (18 dimensions)
        for i in range(9):
            state.append(float(self.probabilities[i][self_idx]))
            state.append(float(self.probabilities[i][opp_idx]))

        # 4. Global Context (3 dimensions)
        edges_norm = min(1.0, active_edges / 8.0)
        cycle_active = (
            1.0
            if any(comp is not None and len(comp) >= 3 for comp in self.pairs)
            else 0.0
        )
        active_player = 1.0 if is_x else -1.0
        state.extend([edges_norm, cycle_active, active_player])

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # D4 Dihedral Symmetry Helper (8 Symmetries of 3x3 Board)
    # ------------------------------------------------------------------

    @staticmethod
    def transform_move_d4(
        move: list[Any] | tuple[Any, ...], sym_idx: int, inverse: bool = False
    ) -> list[Any]:
        """Transforms a tournament move ([sq], [sq1, sq2], or [sq, 'M']) under D4 symmetry `sym_idx` (0..7)."""
        perm = D4_INVERSE_PERMS[sym_idx] if inverse else D4_FORWARD_PERMS[sym_idx]
        if len(move) == 1:
            return [perm[int(move[0])]]
        if len(move) == 2:
            if move[1] == "M" or move[1] == "measure":
                return [perm[int(move[0])], "M"]
            s1, s2 = perm[int(move[0])], perm[int(move[1])]
            return [s1, s2] if s1 <= s2 else [s2, s1]
        raise ValueError(f"Invalid move for D4 symmetry: {move}")

    @staticmethod
    def transform_action_d4(
        action: int, sym_idx: int, inverse: bool = False
    ) -> int:
        """Transforms a 46-dim action index (0..45) under D4 symmetry `sym_idx` (0..7)."""
        perm = D4_INVERSE_PERMS[sym_idx] if inverse else D4_FORWARD_PERMS[sym_idx]
        if 0 <= action <= 35:
            s1, s2 = UPPER_TRIANGLE_PAIRS[action]
            ns1, ns2 = perm[s1], perm[s2]
            pair = (ns1, ns2) if ns1 < ns2 else (ns2, ns1)
            return PAIR_TO_ACTION_IDX[pair]
        if 36 <= action <= 44:
            return 36 + perm[action - 36]
        if action == 45:
            return 45
        raise ValueError(f"Invalid action index: {action}")

    @staticmethod
    def transform_state_111_d4(
        state: np.ndarray, sym_idx: int, inverse: bool = False
    ) -> np.ndarray:
        """Transforms a 111-dim state vector under D4 symmetry `sym_idx` (0..7)."""
        perm = D4_INVERSE_PERMS[sym_idx] if inverse else D4_FORWARD_PERMS[sym_idx]
        out = np.empty_like(state)

        # 1. Local square features (54 dims: 9 blocks of 6)
        for old_sq in range(9):
            new_sq = perm[old_sq]
            out[new_sq * 6 : (new_sq + 1) * 6] = state[old_sq * 6 : (old_sq + 1) * 6]

        # 2. Upper-triangle adjacency (36 dims: indices 54..89)
        for old_idx, (i, j) in enumerate(UPPER_TRIANGLE_PAIRS):
            ni, nj = perm[i], perm[j]
            new_pair = (ni, nj) if ni < nj else (nj, ni)
            new_idx = PAIR_TO_ACTION_IDX[new_pair]
            out[54 + new_idx] = state[54 + old_idx]

        # 3. Collapse probabilities (18 dims: indices 90..107, 9 blocks of 2)
        for old_sq in range(9):
            new_sq = perm[old_sq]
            out[90 + new_sq * 2 : 90 + (new_sq + 1) * 2] = state[
                90 + old_sq * 2 : 90 + (old_sq + 1) * 2
            ]

        # 4. Global context (3 dims: indices 108..110)
        out[108:111] = state[108:111]
        return out

    def apply_d4_symmetry(
        self, sym_idx: int, inverse: bool = False
    ) -> "OfficialTiqTaqToeEnv":
        """
        Returns a new OfficialTiqTaqToeEnv with the board, quantum statevector,
        pairs, and probabilities transformed by D4 dihedral symmetry `sym_idx` (0..7).
        """
        if not (0 <= sym_idx < 8):
            raise ValueError(f"sym_idx must be in 0..7, got {sym_idx}")
        if sym_idx == 0:
            return self.clone()

        perm = D4_INVERSE_PERMS[sym_idx] if inverse else D4_FORWARD_PERMS[sym_idx]
        # Build pull permutation: new_board_str[new_sq] = old_board_str[old_sq]
        # where perm[old_sq] = new_sq  =>  old_sq = inv_perm[new_sq]
        inv_perm = D4_FORWARD_PERMS[sym_idx] if inverse else D4_INVERSE_PERMS[sym_idx]

        new_env = self.clone()
        new_qstate: dict[str, complex] = {}
        for board_str, amp in self.quantum_state.items():
            transformed_str = "".join(board_str[inv_perm[new_sq]] for new_sq in range(9))
            new_qstate[transformed_str] = amp
        new_env.quantum_state = new_qstate

        new_env.measured_squares = [
            self.measured_squares[inv_perm[new_sq]] for new_sq in range(9)
        ]
        new_env.empty_squares = [
            self.empty_squares[inv_perm[new_sq]] for new_sq in range(9)
        ]
        new_env.decoherence_counters = [
            self.decoherence_counters[inv_perm[new_sq]] for new_sq in range(9)
        ]
        new_env.square_turns = [
            self.square_turns[inv_perm[new_sq]] for new_sq in range(9)
        ]

        new_prev_pairs: list[list[int] | None] = []
        for comp in self.pairs:
            if comp is not None:
                new_prev_pairs.append(sorted(perm[sq] for sq in comp))
            else:
                new_prev_pairs.append(None)

        new_env.pairs, new_env.square_to_pair = derive_entanglements(
            new_env.quantum_state, new_prev_pairs
        )
        new_env.probabilities = new_env._compute_square_probabilities()
        new_env.board_ui = new_env._compute_board_ui()
        new_env.statistics = new_env._compute_many_world_statistics()
        new_env.winner_code = new_env._compute_result()
        return new_env
