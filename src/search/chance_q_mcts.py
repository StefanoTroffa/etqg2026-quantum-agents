"""
Chance-Node Gumbel AlphaZero / PUCT MCTS with D4 Dihedral Equivariance
and Exact Many-Worlds Endgame Evaluation for Quantum TiqTaqToe.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 1. D4 Dihedral Symmetry Group on the 3x3 Board & 111-Dim State / 46-Dim Action
# ============================================================================

def _build_d4_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Precomputes exact index permutation tables for all 8 elements of the
    dihedral group D4 acting on:
      - 9 board squares (0..8)
      - 111-dimensional state vector:
          [0..53]:   9 squares * 6 local features
          [54..89]:  36 upper-triangular pairs (i, j) with 0 <= i < j < 9
          [90..107]: 9 squares * 2 collapse probabilities
          [108..110]: 3 global context features (invariant)
      - 46-dimensional action space:
          [0..35]:   quantum moves (i, j) with 0 <= i < j < 9
          [36..44]:  classical moves [i] for i in 0..8
          [45]:      measure action
    """
    grid = np.arange(9, dtype=np.int64).reshape(3, 3)
    sq_perms: List[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(grid, k=k)
        sq_perms.append(rot.flatten())
        sq_perms.append(np.fliplr(rot).flatten())

    # Map upper-triangle pair (i, j) -> 0..35
    pair_to_idx: Dict[Tuple[int, int], int] = {}
    idx_to_pair: List[Tuple[int, int]] = []
    p_idx = 0
    for i in range(9):
        for j in range(i + 1, 9):
            pair_to_idx[(i, j)] = p_idx
            idx_to_pair.append((i, j))
            p_idx += 1

    state_perms = np.zeros((8, 111), dtype=np.int64)
    action_perms = np.zeros((8, 46), dtype=np.int64)
    sq_perms_arr = np.stack(sq_perms, axis=0)  # [8, 9]

    for g in range(8):
        perm = sq_perms_arr[g]
        # Permute [g] maps new_sq = perm[old_sq], or if we transform state x'(new_sq) = x(old_sq):
        # Let inv_perm[new_sq] = old_sq so that transformed_state[i] pulls from inv_perm.
        inv_perm = np.argsort(perm)

        # 1. Local board features (54 dims: 9 * 6)
        for new_sq in range(9):
            old_sq = int(inv_perm[new_sq])
            for f in range(6):
                state_perms[g, new_sq * 6 + f] = old_sq * 6 + f

        # 2. Spooky adjacency upper triangle (36 dims: offset 54)
        for new_p_idx, (u, v) in enumerate(idx_to_pair):
            old_u = int(inv_perm[u])
            old_v = int(inv_perm[v])
            pair_key = (min(old_u, old_v), max(old_u, old_v))
            old_p_idx = pair_to_idx[pair_key]
            state_perms[g, 54 + new_p_idx] = 54 + old_p_idx

        # 3. Exact quantum collapse probabilities (18 dims: offset 90)
        for new_sq in range(9):
            old_sq = int(inv_perm[new_sq])
            state_perms[g, 90 + new_sq * 2] = 90 + old_sq * 2
            state_perms[g, 90 + new_sq * 2 + 1] = 90 + old_sq * 2 + 1

        # 4. Global context (3 dims: offset 108)
        state_perms[g, 108] = 108
        state_perms[g, 109] = 109
        state_perms[g, 110] = 110

        # 5. Action permutation: maps original action index -> transformed action index
        for orig_p_idx, (u, v) in enumerate(idx_to_pair):
            tu, tv = int(perm[u]), int(perm[v])
            t_pair = (min(tu, tv), max(tu, tv))
            action_perms[g, orig_p_idx] = pair_to_idx[t_pair]

        for sq in range(9):
            action_perms[g, 36 + sq] = 36 + int(perm[sq])

        action_perms[g, 45] = 45

    return sq_perms_arr, state_perms, action_perms, pair_to_idx


_D4_SQ_PERMS, _D4_STATE_PERMS, _D4_ACTION_PERMS, _PAIR_TO_ACTION_IDX = _build_d4_tables()

_WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


# ============================================================================
# 2. Environment Adapter & Canonical Move Utilities
# ============================================================================

MoveType = Union[List[Any], Tuple[Any, ...], int]
MoveKey = Tuple[Any, ...]


def canonicalize_move(move: MoveType) -> MoveKey:
    """
    Converts a tournament move ([sq], [sq1, sq2], [sq, 'M']) or legacy int action
    into a hashable tuple key while preserving semantics.
    """
    if isinstance(move, (int, np.integer)):
        return ("ACT_INT", int(move))
    if isinstance(move, (list, tuple)):
        if len(move) == 1:
            return ("C", int(move[0]))
        if len(move) == 2:
            if move[1] == "M" or move[1] == "measure":
                return ("M", int(move[0]))
            s1, s2 = int(move[0]), int(move[1])
            if s1 == s2:
                return ("C", s1)
            return ("Q", min(s1, s2), max(s1, s2))
    return tuple(move) if isinstance(move, (list, tuple)) else (move,)


def move_to_action_46(move: MoveType) -> int:
    """
    Maps a tournament move ([sq], [sq1, sq2], [sq, 'M']) or integer action (0..45)
    to the 46-dimensional neural policy index.
    """
    if isinstance(move, (int, np.integer)):
        return int(move)
    if isinstance(move, (list, tuple)):
        if len(move) == 1:
            return 36 + int(move[0])
        if len(move) == 2:
            if move[1] == "M" or move[1] == "measure":
                return 45
            s1, s2 = int(move[0]), int(move[1])
            if s1 == s2:
                return 36 + s1
            return _PAIR_TO_ACTION_IDX.get((min(s1, s2), max(s1, s2)), 0)
    return 0


def is_measurement_move(move: MoveType) -> bool:
    """Returns True if `move` is an explicit projective measurement move."""
    if isinstance(move, (int, np.integer)):
        return int(move) == 45
    if isinstance(move, (list, tuple)) and len(move) == 2:
        return move[1] in ("M", "measure")
    return False


def get_env_turn_symbol(env: Any) -> str:
    """Returns 'X' or 'O' for the active player in `env`."""
    if hasattr(env, "current_turn"):
        ct = env.current_turn
        if ct in ("X", "O"):
            return str(ct)
        if ct == 1:
            return "X"
        if ct in (-1, 2):
            return "O"
    if hasattr(env, "current_player"):
        cp = env.current_player
        if cp in ("X", "O"):
            return str(cp)
        return "X" if cp == 1 else "O"
    return "X"


def is_env_terminal(env: Any) -> bool:
    """Checks whether `env` is in a terminal state."""
    if hasattr(env, "is_terminal") and callable(env.is_terminal):
        return bool(env.is_terminal())
    if hasattr(env, "done"):
        return bool(env.done)
    return False


def get_env_available_moves(env: Any, deduplicate_measurements: bool = True) -> List[Any]:
    """Retrieves legal moves from `env` in its native format."""
    if hasattr(env, "get_available_moves") and callable(env.get_available_moves):
        try:
            return list(env.get_available_moves(deduplicate_measurements=deduplicate_measurements))
        except TypeError:
            return list(env.get_available_moves())
    if hasattr(env, "get_legal_moves") and callable(env.get_legal_moves):
        return list(env.get_legal_moves())
    if hasattr(env, "get_valid_actions") and callable(env.get_valid_actions):
        return list(env.get_valid_actions())
    return []


def count_empty_squares(env: Any) -> int:
    """Returns the number of completely empty squares on the 3x3 board."""
    if hasattr(env, "empty_squares") and env.empty_squares is not None:
        return int(sum(1 for e in env.empty_squares if e))
    if hasattr(env, "emptySquares") and env.emptySquares is not None:
        return int(sum(1 for e in env.emptySquares if e))
    if hasattr(env, "board") and hasattr(env, "provisional_board") and hasattr(env, "spooky_marks"):
        return int(
            sum(
                1
                for i in range(9)
                if env.board[i] == 0
                and env.provisional_board[i] == 0
                and len(env.spooky_marks[i]) == 0
            )
        )
    if hasattr(env, "probabilities") and env.probabilities is not None:
        return int(sum(1 for p in env.probabilities if p[0] > 0.999))
    return 9


def get_terminal_value(env: Any, perspective: str) -> float:
    """
    Returns exact terminal utility in [-1.0, +1.0] from `perspective` ('X' or 'O').
    Handles all winner conventions:
      - 'X', 'O', 'BOTH', 'DRAW'
      - TicTacResult: 1 (X), 2 (O), 3 (BOTH), 4 (DRAW)
      - Legacy RL: 1 (X), -1 (O), 0 (DRAW/BOTH)
    """
    w = getattr(env, "winner", None)
    if w is None and hasattr(env, "get_winner") and callable(env.get_winner):
        w = env.get_winner()

    win_sym: Optional[str] = None
    if isinstance(w, str):
        w_up = w.upper()
        if w_up in ("X", "O", "DRAW", "BOTH"):
            win_sym = w_up
    elif isinstance(w, (int, np.integer)):
        # Distinguish TicTacResult (1=X, 2=O, 3=BOTH, 4=DRAW) from Legacy (1=X, -1=O, 0=DRAW)
        if w == -1:
            win_sym = "O"
        elif w == 0:
            win_sym = "DRAW"
        elif w == 1:
            win_sym = "X"
        elif w == 2:
            win_sym = "O"
        elif w in (3, 4):
            win_sym = "DRAW"

    if win_sym is None or win_sym in ("DRAW", "BOTH"):
        return 0.0
    return 1.0 if win_sym == perspective else -1.0


def evaluate_many_worlds(env: Any, perspective: str) -> Optional[Dict[str, float]]:
    """
    Computes exact Many-Worlds statistics from `env` if available or derivable
    from `quantum_state` / `quantum_circuit`.
    Returns dict with keys: xWinProb, oWinProb, drawProb, bothProb, value.
    """
    stats: Optional[Dict[str, Any]] = None
    if hasattr(env, "compute_many_worlds") and callable(env.compute_many_worlds):
        stats = env.compute_many_worlds()
    elif hasattr(env, "statistics") and isinstance(env.statistics, dict) and "xWinProb" in env.statistics:
        stats = env.statistics
    else:
        # Fallback: compute directly from quantum_state or quantum_circuit.state
        q_state = getattr(env, "quantum_state", None) or getattr(env, "quantumState", None)
        if q_state is None and hasattr(env, "quantum_circuit") and hasattr(env.quantum_circuit, "state"):
            q_state = env.quantum_circuit.state
        if isinstance(q_state, dict) and len(q_state) > 0:
            x_prob = 0.0
            o_prob = 0.0
            draw_prob = 0.0
            both_prob = 0.0
            total_p = 0.0
            for b_str, amp in q_state.items():
                if isinstance(amp, complex):
                    p = float(amp.real * amp.real + amp.imag * amp.imag)
                elif isinstance(amp, dict):
                    re = float(amp.get("real", amp.get("re", 0.0)))
                    im = float(amp.get("imaginary", amp.get("im", 0.0)))
                    p = re * re + im * im
                else:
                    re = float(getattr(amp, "real", 0.0))
                    im = float(getattr(amp, "imaginary", getattr(amp, "im", 0.0)))
                    p = re * re + im * im
                if p <= 1e-12:
                    continue
                total_p += p
                x_lines = sum(1 for a, b, c in _WIN_LINES if b_str[a] == "X" and b_str[b] == "X" and b_str[c] == "X")
                o_lines = sum(1 for a, b, c in _WIN_LINES if b_str[a] == "O" and b_str[b] == "O" and b_str[c] == "O")
                if x_lines > o_lines and x_lines > 0:
                    x_prob += p
                elif o_lines > x_lines and o_lines > 0:
                    o_prob += p
                elif x_lines > 0 and x_lines == o_lines:
                    both_prob += p
                else:
                    draw_prob += p
            if total_p > 0:
                x_prob /= total_p
                o_prob /= total_p
                draw_prob /= total_p
                both_prob /= total_p
            stats = {
                "xWinProb": x_prob,
                "oWinProb": o_prob,
                "drawProb": draw_prob,
                "bothProb": both_prob,
                "numStates": len(q_state),
            }

    if stats is None:
        return None

    x_win = float(stats.get("X", stats.get("xWinProb", 0.0)))
    o_win = float(stats.get("O", stats.get("oWinProb", 0.0)))
    draw_p = float(stats.get("drawProb", 0.0))
    both_p = float(stats.get("bothProb", 0.0))
    val = (x_win - o_win) if perspective == "X" else (o_win - x_win)
    return {
        "xWinProb": x_win,
        "oWinProb": o_win,
        "drawProb": draw_p,
        "bothProb": both_p,
        "value": float(np.clip(val, -1.0, 1.0)),
    }


def extract_state_111(env: Any) -> np.ndarray:
    """
    Extracts the 111-dimensional state vector from `env` from the perspective
    of `env`'s current player. Supports `get_state_111()`, `get_state()`, or
    direct reconstruction from `OfficialTiqTaqToeEnv` attributes.
    """
    if hasattr(env, "get_state_111") and callable(env.get_state_111):
        arr = np.asarray(env.get_state_111(), dtype=np.float32)
        if arr.shape == (111,):
            return arr
    if hasattr(env, "get_state") and callable(env.get_state):
        arr = np.asarray(env.get_state(), dtype=np.float32)
        if arr.shape == (111,):
            return arr

    # Reconstruct 111-dim vector from OfficialTiqTaqToeEnv attributes
    turn_sym = get_env_turn_symbol(env)
    curr_val = 1 if turn_sym == "X" else -1
    probs = getattr(env, "probabilities", None)
    if probs is None:
        probs = [[1.0, 0.0, 0.0] for _ in range(9)]
    measured = getattr(env, "measured_squares", getattr(env, "measuredSquares", [False] * 9))
    pairs = getattr(env, "pairs", []) or []

    classical_board = np.zeros(9, dtype=np.int32)
    provisional_board = np.zeros(9, dtype=np.int32)
    for i in range(9):
        p0, px, po = probs[i]
        if measured[i]:
            if px > 0.99:
                classical_board[i] = 1
            elif po > 0.99:
                classical_board[i] = -1
        else:
            if px > 0.99:
                provisional_board[i] = 1
            elif po > 0.99:
                provisional_board[i] = -1

    spooky_adj: Dict[Tuple[int, int], float] = {}
    spooky_self = np.zeros(9, dtype=np.float32)
    spooky_opp = np.zeros(9, dtype=np.float32)
    active_edges = 0

    for comp in pairs:
        if not comp or len(comp) < 2:
            continue
        for idx in comp:
            _, px, po = probs[idx]
            if turn_sym == "X":
                if px > 0.01 and not measured[idx] and provisional_board[idx] == 0:
                    spooky_self[idx] = 1.0
                if po > 0.01 and not measured[idx] and provisional_board[idx] == 0:
                    spooky_opp[idx] = 1.0
            else:
                if po > 0.01 and not measured[idx] and provisional_board[idx] == 0:
                    spooky_self[idx] = 1.0
                if px > 0.01 and not measured[idx] and provisional_board[idx] == 0:
                    spooky_opp[idx] = 1.0
        for a_i in range(len(comp)):
            for b_i in range(a_i + 1, len(comp)):
                u, v = min(comp[a_i], comp[b_i]), max(comp[a_i], comp[b_i])
                owner_val = 1.0 if (spooky_self[u] > 0 or spooky_self[v] > 0) else -1.0
                spooky_adj[(u, v)] = owner_val
                active_edges += 1

    state = np.zeros(111, dtype=np.float32)
    for i in range(9):
        base = i * 6
        state[base + 0] = 1.0 if classical_board[i] == curr_val else 0.0
        state[base + 1] = 1.0 if classical_board[i] == -curr_val else 0.0
        state[base + 2] = 1.0 if provisional_board[i] == curr_val else 0.0
        state[base + 3] = 1.0 if provisional_board[i] == -curr_val else 0.0
        state[base + 4] = spooky_self[i]
        state[base + 5] = spooky_opp[i]

    p_idx = 0
    for i in range(9):
        for j in range(i + 1, 9):
            state[54 + p_idx] = spooky_adj.get((i, j), 0.0)
            p_idx += 1

    for i in range(9):
        _, px, po = probs[i]
        if curr_val == 1:
            state[90 + 2 * i] = float(px)
            state[90 + 2 * i + 1] = float(po)
        else:
            state[90 + 2 * i] = float(po)
            state[90 + 2 * i + 1] = float(px)

    state[108] = min(1.0, active_edges / 8.0)
    state[109] = 0.0
    state[110] = 1.0 if curr_val == 1 else -1.0
    return state


def get_move_outcomes_safe(env: Any, move: MoveType) -> List[Tuple[float, Any]]:
    """
    Queries `env.get_move_outcomes(move)` to obtain exact Born-weighted child environments
    `[(prob_k, next_env_k), ...]`. Falls back to deterministic/single-sample `.step(move)`
    only if `get_move_outcomes` is not implemented on `env`.
    """
    if hasattr(env, "get_move_outcomes") and callable(env.get_move_outcomes):
        outcomes = env.get_move_outcomes(move)
        if outcomes:
            # Normalize probabilities defensively
            total_p = sum(float(p) for p, _ in outcomes)
            if total_p > 0:
                return [(float(p) / total_p, child_env) for p, child_env in outcomes if float(p) > 1e-9]

    # Fallback for legacy environments without get_move_outcomes
    next_env = env.clone()
    if hasattr(next_env, "apply_move") and callable(next_env.apply_move):
        next_env.apply_move(move)
    else:
        act = move_to_action_46(move) if not isinstance(move, (int, np.integer)) else int(move)
        try:
            next_env.step(act, return_state=False)
        except TypeError:
            next_env.step(act)
    return [(1.0, next_env)]


# ============================================================================
# 3. DecisionNode and ChanceNode Data Structures
# ============================================================================

class DecisionNode:
    """
    Represents a player decision point s in the quantum stochastic game tree.
    All values `value_sum` and `q_value` are stored in [-1.0, +1.0] from
    `self.player`'s perspective ('X' or 'O').
    """
    __slots__ = (
        "env",
        "parent_chance",
        "outcome_prob",
        "player",
        "children",
        "priors",
        "logits",
        "visit_count",
        "value_sum",
        "initial_value",
        "is_expanded",
    )

    def __init__(
        self,
        env: Any,
        parent_chance: Optional["ChanceNode"] = None,
        outcome_prob: float = 1.0,
    ):
        self.env = env
        self.parent_chance = parent_chance
        self.outcome_prob = float(outcome_prob)
        self.player: str = get_env_turn_symbol(env)
        self.children: Dict[MoveKey, "ChanceNode"] = {}
        self.priors: Dict[MoveKey, float] = {}
        self.logits: Dict[MoveKey, float] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.initial_value: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return self.initial_value
        return self.value_sum / self.visit_count


class ChanceNode:
    """
    Represents a quantum stochastic transition (s, a) whose children are
    DecisionNodes weighted by exact Born-rule probabilities p_k = |psi_k|^2.
    All values `value_sum` and `q_value` are stored in [-1.0, +1.0] from
    `self.parent.player`'s perspective.
    """
    __slots__ = (
        "parent",
        "move_key",
        "raw_move",
        "prior",
        "logit",
        "outcomes",
        "visit_count",
        "value_sum",
        "exact_value",
    )

    def __init__(
        self,
        parent: DecisionNode,
        move_key: MoveKey,
        raw_move: MoveType,
        prior: float,
        logit: float,
    ):
        self.parent = parent
        self.move_key = move_key
        self.raw_move = raw_move
        self.prior = float(prior)
        self.logit = float(logit)
        self.outcomes: Optional[List[Tuple[float, DecisionNode]]] = None
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.exact_value: Optional[float] = None

    @property
    def q_value(self) -> float:
        if self.exact_value is not None and self.visit_count <= 1:
            return self.exact_value
        if self.visit_count == 0:
            return self.parent.initial_value
        return self.value_sum / self.visit_count


# ============================================================================
# 4. ChanceQMCTSPlayer (Gumbel Sequential Halving + PUCT + Exact Many-Worlds)
# ============================================================================

class ChanceQMCTSPlayer:
    """
    Exact Chance-Node MCTS for Quantum TiqTaqToe (`quantumness=3`) featuring:
      - Explicit DecisionNode & ChanceNode separation for Born-rule collapses
      - D4 Dihedral group averaging (8 symmetries) for neural policy & value priors
      - Exact Many-Worlds value evaluation for measurement moves `[sq, 'M']` and endgames
      - Gumbel Sequential Halving at the root & FPU-PUCT at interior decision nodes
      - Strict wall-clock timeout enforcement (`time_limit_ms`)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        c_puct: float = 1.5,
        num_searches: int = 160,
        time_limit_ms: float = 1500.0,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        gumbel_TopM: int = 12,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        use_d4_symmetry: bool = True,
        many_worlds_blend: float = 0.65,
        endgame_empty_threshold: int = 3,
        max_exact_outcomes: int = 8,
    ):
        self.model = model
        self.c_puct = float(c_puct)
        self.num_searches = int(num_searches)
        self.time_limit_ms = float(time_limit_ms)
        self.temperature = float(temperature)
        self.use_gumbel = bool(use_gumbel)
        self.gumbel_top_m = int(gumbel_TopM)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.use_d4_symmetry = bool(use_d4_symmetry)
        self.many_worlds_blend = float(many_worlds_blend)
        self.endgame_empty_threshold = int(endgame_empty_threshold)
        self.max_exact_outcomes = int(max_exact_outcomes)

    # ------------------------------------------------------------------
    # Public Entrypoints
    # ------------------------------------------------------------------

    def search(
        self,
        env: Any,
        available_moves: Optional[Sequence[MoveType]] = None,
        return_probs: bool = False,
        temp: float = 0.0,
    ) -> Union[MoveType, Tuple[MoveType, np.ndarray]]:
        """
        Executes Chance-Node MCTS from `env` within `self.time_limit_ms` and
        returns a move guaranteed to be in `available_moves` (or `env.get_available_moves()`).
        """
        return self.get_move(
            env=env,
            available_moves=available_moves,
            return_probs=return_probs,
            temp=temp,
        )

    def get_move(
        self,
        env: Any,
        available_moves: Optional[Sequence[MoveType]] = None,
        return_probs: bool = False,
        temp: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ) -> Union[MoveType, Tuple[MoveType, np.ndarray]]:
        start_time = time.perf_counter()
        deadline = start_time + (self.time_limit_ms / 1000.0) * 0.92

        legal_moves = list(available_moves) if available_moves is not None else get_env_available_moves(env)
        if len(legal_moves) == 0:
            probs = np.zeros(46, dtype=np.float32)
            return (None, probs) if return_probs else None

        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            probs = np.zeros(46, dtype=np.float32)
            probs[move_to_action_46(only_move)] = 1.0
            return (only_move, probs) if return_probs else only_move

        root = DecisionNode(env=env.clone())
        self._expand_decision_node(root, override_moves=legal_moves)

        # 1. Tactical Pre-Screen: Check for immediate 1-step deterministic or high-prob measurement wins
        immediate_win_move = self._check_immediate_winning_move(root, deadline)
        if immediate_win_move is not None:
            probs = np.zeros(46, dtype=np.float32)
            probs[move_to_action_46(immediate_win_move)] = 1.0
            return (immediate_win_move, probs) if return_probs else immediate_win_move

        # 2. Run MCTS simulations (Gumbel Sequential Halving or PUCT)
        if self.use_gumbel and len(root.children) > 1:
            best_key = self._run_gumbel_sequential_halving(
                root=root,
                deadline=deadline,
                sample_gumbel=(temp > 0.0),
            )
        else:
            if temp > 0.0 and len(root.children) > 0:
                keys = list(root.children.keys())
                noise = np.random.dirichlet([dirichlet_alpha] * len(keys))
                for idx, k in enumerate(keys):
                    ch = root.children[k]
                    ch.prior = (1.0 - dirichlet_eps) * ch.prior + dirichlet_eps * float(noise[idx])

            for _ in range(self.num_searches):
                if time.perf_counter() >= deadline:
                    break
                self._simulate_once(root, deadline)

            best_key = self._select_final_root_key(root, temp=temp)

        probs = self._compute_improved_policy_46(root)
        chosen_move = root.children[best_key].raw_move

        # Strict guarantee: chosen_move is an exact member of legal_moves
        if chosen_move not in legal_moves:
            chosen_move = legal_moves[0]

        if return_probs:
            return chosen_move, probs
        return chosen_move

    # ------------------------------------------------------------------
    # Tactical 1-Ply Exact Check (Win-Now / Safe Measurement Collapse)
    # ------------------------------------------------------------------

    def _check_immediate_winning_move(
        self,
        root: DecisionNode,
        deadline: float,
    ) -> Optional[MoveType]:
        """
        Inspects root moves to see if any move guarantees an immediate terminal win
        (with probability 1.0) or if a measurement move `[sq, 'M']` collapses to a
        strictly winning expectation with zero loss probability when ahead.
        """
        root_player = root.player
        best_measure_move: Optional[MoveType] = None
        best_measure_val: float = 0.65  # Only auto-trigger measurement if E[V] >= 0.65 and P(loss) == 0

        for m_key, chance_node in root.children.items():
            if time.perf_counter() >= deadline:
                break
            if is_measurement_move(chance_node.raw_move):
                outcomes = self._ensure_chance_outcomes(chance_node)
                exp_val = 0.0
                loss_prob = 0.0
                win_prob = 0.0
                all_terminal = True
                for prob, child_dec in outcomes:
                    if is_env_terminal(child_dec.env):
                        v = get_terminal_value(child_dec.env, root_player)
                        exp_val += prob * v
                        if v > 0.5:
                            win_prob += prob
                        elif v < -0.5:
                            loss_prob += prob
                    else:
                        all_terminal = False
                if win_prob >= 0.999 and loss_prob <= 1e-6:
                    return chance_node.raw_move
                if all_terminal and loss_prob <= 1e-6 and exp_val > best_measure_val:
                    best_measure_val = exp_val
                    best_measure_move = chance_node.raw_move
            elif len(chance_node.raw_move) == 1 if isinstance(chance_node.raw_move, (list, tuple)) else (chance_node.raw_move >= 36):
                # Classical move: check if it wins immediately
                outcomes = self._ensure_chance_outcomes(chance_node)
                if len(outcomes) == 1:
                    prob, child_dec = outcomes[0]
                    if is_env_terminal(child_dec.env) and get_terminal_value(child_dec.env, root_player) > 0.99:
                        return chance_node.raw_move

        return best_measure_move

    # ------------------------------------------------------------------
    # Gumbel Sequential Halving at Root
    # ------------------------------------------------------------------

    def _run_gumbel_sequential_halving(
        self,
        root: DecisionNode,
        deadline: float,
        sample_gumbel: bool = False,
    ) -> MoveKey:
        all_keys = list(root.children.keys())
        num_actions = len(all_keys)
        if num_actions == 1:
            return all_keys[0]

        gumbel_noise: Dict[MoveKey, float] = {}
        for k in all_keys:
            if sample_gumbel:
                u = np.clip(np.random.uniform(1e-7, 1.0 - 1e-7), 1e-7, 1.0 - 1e-7)
                gumbel_noise[k] = float(-math.log(-math.log(u)))
            else:
                gumbel_noise[k] = 0.0

        # Select initial Top-M candidates by g(a) + logit(a)
        m = min(num_actions, self.gumbel_top_m)
        candidates = sorted(
            all_keys,
            key=lambda k: gumbel_noise[k] + root.children[k].logit,
            reverse=True,
        )[:m]

        num_phases = max(1, int(math.ceil(math.log2(m))))
        sims_remaining = self.num_searches

        for phase in range(num_phases):
            if time.perf_counter() >= deadline or len(candidates) <= 1:
                break
            phases_left = num_phases - phase
            budget_this_phase = max(len(candidates), sims_remaining // phases_left)
            visits_per_cand = max(1, budget_this_phase // len(candidates))

            for _ in range(visits_per_cand):
                if time.perf_counter() >= deadline:
                    break
                for k in candidates:
                    if time.perf_counter() >= deadline:
                        break
                    chance_node = root.children[k]
                    val = self._traverse_chance_node(chance_node, deadline)
                    root.visit_count += 1
                    root.value_sum += val
                    sims_remaining -= 1

            # Rank candidates by g(a) + logit(a) + sigma(q_hat(a))
            max_n = max(root.children[k].visit_count for k in all_keys)
            scale = (self.c_visit + max_n) * self.c_scale
            candidates.sort(
                key=lambda k: (
                    gumbel_noise[k]
                    + root.children[k].logit
                    + scale * root.children[k].q_value
                ),
                reverse=True,
            )
            keep = max(1, len(candidates) // 2)
            candidates = candidates[:keep]

        # Final selection among surviving candidates
        max_n = max(root.children[k].visit_count for k in all_keys)
        scale = (self.c_visit + max_n) * self.c_scale
        best_key = max(
            candidates,
            key=lambda k: (
                gumbel_noise[k]
                + root.children[k].logit
                + scale * root.children[k].q_value
            ),
        )
        return best_key

    # ------------------------------------------------------------------
    # Tree Traversal: DecisionNode <-> ChanceNode
    # ------------------------------------------------------------------

    def _simulate_once(self, root: DecisionNode, deadline: float) -> float:
        chance_node = self._select_puct_child(root)
        val = self._traverse_chance_node(chance_node, deadline)
        root.visit_count += 1
        root.value_sum += val
        return val

    def _select_puct_child(self, node: DecisionNode) -> ChanceNode:
        total_visits = sum(ch.visit_count for ch in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))
        best_score = -1e9
        best_child: Optional[ChanceNode] = None

        for ch in node.children.values():
            # Q(s, a) is already stored in `node.player`'s perspective!
            q_val = ch.q_value
            u_val = self.c_puct * ch.prior * (sqrt_total / (1 + ch.visit_count))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = ch

        assert best_child is not None
        return best_child

    def _ensure_chance_outcomes(self, chance_node: ChanceNode) -> List[Tuple[float, DecisionNode]]:
        if chance_node.outcomes is None:
            raw_outcomes = get_move_outcomes_safe(chance_node.parent.env, chance_node.raw_move)
            chance_node.outcomes = [
                (prob, DecisionNode(env=next_env, parent_chance=chance_node, outcome_prob=prob))
                for prob, next_env in raw_outcomes
            ]
        return chance_node.outcomes

    def _traverse_chance_node(self, chance_node: ChanceNode, deadline: float) -> float:
        """
        Evaluates or traverses `chance_node` and returns the backed-up value in [-1, +1]
        from `chance_node.parent.player`'s perspective.
        """
        parent_player = chance_node.parent.player
        outcomes = self._ensure_chance_outcomes(chance_node)

        # First visit to a multi-outcome ChanceNode (e.g. measurement or forced full-board collapse)
        # with <= max_exact_outcomes branches: compute exact Born expectation over all outcomes!
        if (
            chance_node.visit_count == 0
            and len(outcomes) > 1
            and len(outcomes) <= self.max_exact_outcomes
        ):
            exact_exp = 0.0
            for prob, child_dec in outcomes:
                v_child = self._evaluate_or_expand_decision_node(child_dec)
                # Convert from child_dec.player perspective to parent_player perspective
                v_parent = v_child if child_dec.player == parent_player else -v_child
                exact_exp += prob * v_parent
            chance_node.exact_value = exact_exp
            chance_node.visit_count += 1
            chance_node.value_sum += exact_exp
            return exact_exp

        # Select outcome branch with largest deficit from its target Born probability p_k
        if len(outcomes) == 1:
            chosen_prob, chosen_dec = outcomes[0]
        else:
            total_c_visits = sum(dec.visit_count for _, dec in outcomes)
            best_deficit = -1e9
            chosen_prob, chosen_dec = outcomes[0]
            for prob, dec in outcomes:
                deficit = prob - (dec.visit_count / (1.0 + total_c_visits))
                if deficit > best_deficit:
                    best_deficit = deficit
                    chosen_prob, chosen_dec = prob, dec

        if not chosen_dec.is_expanded or is_env_terminal(chosen_dec.env) or time.perf_counter() >= deadline:
            v_dec = self._evaluate_or_expand_decision_node(chosen_dec)
        else:
            next_chance = self._select_puct_child(chosen_dec)
            v_dec = self._traverse_chance_node(next_chance, deadline)
            chosen_dec.visit_count += 1
            chosen_dec.value_sum += v_dec

        v_parent = v_dec if chosen_dec.player == parent_player else -v_dec
        chance_node.visit_count += 1
        chance_node.value_sum += v_parent
        return v_parent

    def _evaluate_or_expand_decision_node(self, node: DecisionNode) -> float:
        """
        Expands `node` if non-terminal and returns its value in [-1, +1]
        from `node.player`'s perspective.
        """
        if is_env_terminal(node.env):
            val = get_terminal_value(node.env, node.player)
            node.initial_value = val
            node.visit_count += 1
            node.value_sum += val
            node.is_expanded = True
            return val

        if not node.is_expanded:
            val = self._expand_decision_node(node)
            node.visit_count += 1
            node.value_sum += val
            return val

        return node.q_value

    # ------------------------------------------------------------------
    # Node Expansion, D4 Equivariant Inference & Many-Worlds Blending
    # ------------------------------------------------------------------

    def _expand_decision_node(
        self,
        node: DecisionNode,
        override_moves: Optional[Sequence[MoveType]] = None,
    ) -> float:
        moves = list(override_moves) if override_moves is not None else get_env_available_moves(node.env)
        if len(moves) == 0:
            val = get_terminal_value(node.env, node.player)
            node.initial_value = val
            node.is_expanded = True
            return val

        # 1. Evaluate neural policy logits (46-dim) & value in [-1, 1] with D4 equivariance
        logits_46, nn_value = self._infer_d4_policy_value(node.env)

        # 2. Evaluate exact Many-Worlds statistics
        mw_stats = evaluate_many_worlds(node.env, node.player)
        empty_count = count_empty_squares(node.env)

        if mw_stats is not None:
            mw_value = mw_stats["value"]
            if self.model is None:
                leaf_value = mw_value
            elif empty_count <= self.endgame_empty_threshold:
                # Near-endgame: trust exact Many-Worlds statevector expectation heavily
                leaf_value = (1.0 - self.many_worlds_blend) * nn_value + self.many_worlds_blend * mw_value
            else:
                leaf_value = 0.70 * nn_value + 0.30 * mw_value
        else:
            leaf_value = nn_value

        node.initial_value = float(np.clip(leaf_value, -1.0, 1.0))

        # 3. Build tactical move logits and priors over legal moves
        raw_logits: List[float] = []
        move_keys: List[MoveKey] = []

        probs_grid = getattr(node.env, "probabilities", None)
        own_idx = 1 if node.player == "X" else 2
        opp_idx = 2 if node.player == "X" else 1

        for mv in moves:
            m_key = canonicalize_move(mv)
            act_46 = move_to_action_46(mv)
            base_logit = float(logits_46[act_46])

            # Tactical prior shaping using exact marginal probabilities & Many-Worlds
            bonus = 0.0
            if is_measurement_move(mv):
                if mw_stats is not None:
                    # Encourage measuring when Many-Worlds expectation is strongly positive,
                    # penalize measuring when collapsing now has negative expectation!
                    if mw_stats["value"] > 0.25:
                        bonus += 1.25 * mw_stats["value"]
                    elif mw_stats["value"] < -0.05:
                        bonus -= 1.50 * abs(mw_stats["value"])
                    else:
                        bonus -= 0.35
            elif isinstance(mv, (list, tuple)) and len(mv) == 1:
                sq = int(mv[0])
                if sq == 4:
                    bonus += 0.20  # Center square priority
                elif sq in (0, 2, 6, 8):
                    bonus += 0.10  # Corner priority
                if probs_grid is not None:
                    bonus += 0.45 * self._line_completion_potential(probs_grid, sq, own_idx, opp_idx)
            elif isinstance(mv, (list, tuple)) and len(mv) == 2:
                sq1, sq2 = int(mv[0]), int(mv[1])
                if 4 in (sq1, sq2):
                    bonus += 0.18
                if probs_grid is not None:
                    # Entangling with an opponent's unmeasured token at quantumness=3
                    # steals 50% amplitude from their line threat!
                    if probs_grid[sq1][opp_idx] > 0.45 or probs_grid[sq2][opp_idx] > 0.45:
                        t1 = self._line_completion_potential(probs_grid, sq1, own_idx, opp_idx)
                        t2 = self._line_completion_potential(probs_grid, sq2, own_idx, opp_idx)
                        bonus += 0.35 * max(t1, t2)

            move_keys.append(m_key)
            raw_logits.append((base_logit / max(1e-3, self.temperature)) + bonus)

        logits_arr = np.asarray(raw_logits, dtype=np.float64)
        logits_arr = logits_arr - np.max(logits_arr)
        exp_arr = np.exp(logits_arr)
        priors_arr = exp_arr / np.sum(exp_arr)

        for idx, mv in enumerate(moves):
            m_key = move_keys[idx]
            if m_key not in node.children:
                prior = float(priors_arr[idx])
                logit = float(logits_arr[idx])
                node.priors[m_key] = prior
                node.logits[m_key] = logit
                node.children[m_key] = ChanceNode(
                    parent=node,
                    move_key=m_key,
                    raw_move=mv,
                    prior=prior,
                    logit=logit,
                )

        node.is_expanded = True
        return node.initial_value

    @staticmethod
    def _line_completion_potential(
        probs_grid: Sequence[Sequence[float]],
        sq: int,
        own_idx: int,
        opp_idx: int,
    ) -> float:
        """
        Computes how strongly square `sq` completes own lines or blocks opponent lines
        based on marginal square probabilities.
        """
        best = 0.0
        for line in _WIN_LINES:
            if sq not in line:
                continue
            others = [s for s in line if s != sq]
            own_prod = float(probs_grid[others[0]][own_idx]) * float(probs_grid[others[1]][own_idx])
            opp_prod = float(probs_grid[others[0]][opp_idx]) * float(probs_grid[others[1]][opp_idx])
            best = max(best, own_prod, 0.9 * opp_prod)
        return best

    def _infer_d4_policy_value(self, env: Any) -> Tuple[np.ndarray, float]:
        """
        Evaluates `self.model` across all 8 dihedral symmetries D4 in a single
        batched forward pass `[8, 111]` and inverse-permutes the 46 policy logits.
        """
        if self.model is None:
            return np.zeros(46, dtype=np.float32), 0.0

        state_111 = extract_state_111(env)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        if self.use_d4_symmetry:
            # Batch of 8 dihedral transforms: shape [8, 111]
            sym_states = state_111[_D4_STATE_PERMS]
            state_t = torch.from_numpy(sym_states).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                policy_logits_t, value_t = self.model(state_t)
                logits_np = policy_logits_t.detach().cpu().numpy()  # [8, 46]
                val_mean = float(value_t.detach().mean().cpu().item())

            # Inverse-permute each symmetry's logits back to canonical board frame
            avg_logits = np.zeros(46, dtype=np.float32)
            for g in range(8):
                # _D4_ACTION_PERMS[g, orig_a] = transformed_a
                avg_logits += logits_np[g, _D4_ACTION_PERMS[g]]
            avg_logits /= 8.0
            return avg_logits, float(np.clip(val_mean, -1.0, 1.0))
        else:
            state_t = torch.from_numpy(state_111).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                policy_logits_t, value_t = self.model(state_t)
                logits_np = policy_logits_t.squeeze(0).detach().cpu().numpy()
                val_scalar = float(value_t.squeeze().detach().cpu().item())
            return logits_np.astype(np.float32), float(np.clip(val_scalar, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Root Move & Policy Extraction
    # ------------------------------------------------------------------

    def _select_final_root_key(self, root: DecisionNode, temp: float = 0.0) -> MoveKey:
        keys = list(root.children.keys())
        if temp <= 1e-5:
            # Break visit ties using completed Q-value and prior
            return max(
                keys,
                key=lambda k: (
                    root.children[k].visit_count,
                    root.children[k].q_value,
                    root.children[k].prior,
                ),
            )
        visits = np.array([root.children[k].visit_count for k in keys], dtype=np.float64)
        if np.sum(visits) <= 0:
            return keys[int(np.random.randint(len(keys)))]
        scaled = visits ** (1.0 / temp)
        probs = scaled / np.sum(scaled)
        idx = int(np.random.choice(len(keys), p=probs))
        return keys[idx]

    def _compute_improved_policy_46(self, root: DecisionNode) -> np.ndarray:
        """
        Computes the 46-dimensional Gumbel Completed-Q improved policy target
        pi_improved(a | s) = Softmax(logit(a) + sigma(q_hat(a))).
        """
        probs_46 = np.zeros(46, dtype=np.float32)
        if len(root.children) == 0:
            return probs_46

        keys = list(root.children.keys())
        max_n = max(root.children[k].visit_count for k in keys)
        scale = (self.c_visit + max_n) * self.c_scale

        # Compute mixed value estimate v_mix for unvisited actions
        total_n = sum(root.children[k].visit_count for k in keys)
        if total_n > 0:
            visited_p = sum(root.children[k].prior for k in keys if root.children[k].visit_count > 0)
            visited_q = sum(
                root.children[k].prior * root.children[k].q_value
                for k in keys
                if root.children[k].visit_count > 0
            ) / max(1e-8, visited_p)
            v_mix = (root.initial_value + (total_n / max(1e-8, visited_p)) * visited_q) / (1.0 + total_n)
        else:
            v_mix = root.initial_value

        completed_logits = np.zeros(len(keys), dtype=np.float64)
        for idx, k in enumerate(keys):
            ch = root.children[k]
            q_hat = ch.q_value if ch.visit_count > 0 else v_mix
            completed_logits[idx] = ch.logit + scale * q_hat

        completed_logits -= np.max(completed_logits)
        exp_l = np.exp(completed_logits)
        pi_k = exp_l / np.sum(exp_l)

        for idx, k in enumerate(keys):
            act_46 = move_to_action_46(root.children[k].raw_move)
            probs_46[act_46] += float(pi_k[idx])

        norm = np.sum(probs_46)
        if norm > 0:
            probs_46 /= norm
        return probs_46
