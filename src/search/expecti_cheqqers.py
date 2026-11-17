"""
Iterative-Deepening Expecti-AlphaBeta Search with Star1 Chance-Node Pruning,
Zobrist/State-Hash Transposition Table, Tactical Move Ordering, and Hybrid
Neural + Quantum Heuristic Leaf Evaluation for Quantum Cheqqers (6x6, quantumness=2).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 1. Constants & Transposition Table Types
# ============================================================================

TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

V_MAX = 1000.0  # Bounded evaluation range [-V_MAX, +V_MAX] for Star1 chance pruning


class _SearchTimeout(Exception):
    """Raised internally when the wall-clock search budget is exhausted."""


@dataclass
class TTEntry:
    depth: int
    flag: int
    value: float
    best_move_key: Optional[Tuple[Any, ...]]


# ============================================================================
# 2. Environment Inspection & Move Canonicalization Helpers
# ============================================================================

MoveType = Union[List[int], Tuple[int, ...], Dict[str, Any], int]
MoveKey = Tuple[Any, ...]


def move_to_array(move: Any) -> Optional[List[int]]:
    """
    Converts any Cheqqers move representation into the tournament array format:
      - Classical / Take move: [from_index, to_index]
      - Split move: [from_index, to_index1, to_index2]
      - Merge move: [from_index1, from_index2, to_index]
    Returns None if `move` is an integer action ID that cannot be converted without a board.
    """
    if isinstance(move, list):
        return [int(x) for x in move]
    if isinstance(move, tuple):
        return [int(x) for x in move]
    if isinstance(move, dict):
        if "to_index1" in move and "to_index2" in move:
            return [int(move["from_index"]), int(move["to_index1"]), int(move["to_index2"])]
        if "from_index1" in move and "from_index2" in move:
            return [int(move["from_index1"]), int(move["from_index2"]), int(move["to_index"])]
        if "from_index" in move and "to_index" in move:
            return [int(move["from_index"]), int(move["to_index"])]
    if hasattr(move, "from_index") and hasattr(move, "to_index1") and hasattr(move, "to_index2"):
        return [int(move.from_index), int(move.to_index1), int(move.to_index2)]
    if hasattr(move, "from_index1") and hasattr(move, "from_index2") and hasattr(move, "to_index"):
        return [int(move.from_index1), int(move.from_index2), int(move.to_index)]
    if hasattr(move, "from_index") and hasattr(move, "to_index"):
        return [int(move.from_index), int(move.to_index)]
    return None


def canonicalize_cheqqers_move(move: Any) -> MoveKey:
    """Returns a hashable canonical key for any Cheqqers move."""
    arr = move_to_array(move)
    if arr is not None:
        return tuple(arr)
    if isinstance(move, (int, np.integer)):
        return ("INT", int(move))
    return (str(move),)


def normalize_color(color_or_player: Any) -> str:
    """
    Normalizes any player/color representation to 'white' or 'black':
      - 'white' / 'W' / 0 (PieceColor.WHITE in JS) -> 'white'
      - 'black' / 'B' / 1 (PieceColor.BLACK in JS) -> 'black'
      - -1 (Legacy RL Black) -> 'black'
    """
    if isinstance(color_or_player, str):
        c_low = color_or_player.strip().lower()
        if c_low.startswith("w"):
            return "white"
        if c_low.startswith("b"):
            return "black"
    if hasattr(color_or_player, "name"):
        name = str(color_or_player.name).upper()
        if "WHITE" in name:
            return "white"
        if "BLACK" in name:
            return "black"
    if isinstance(color_or_player, (int, np.integer)):
        if color_or_player == 0:
            return "white"
        if color_or_player == -1:
            return "black"
        if color_or_player == 1:
            return "black"
    return "white"


def get_active_color(env: Any) -> str:
    """Determines whether 'white' or 'black' is to move in `env`."""
    if hasattr(env, "current_turn") and env.current_turn is not None:
        ct = env.current_turn
        if isinstance(ct, str):
            return normalize_color(ct)
        if ct == 0:
            return "white"
        if ct == 1:
            return "black"
    if hasattr(env, "turn") and env.turn is not None:
        t = env.turn
        if isinstance(t, str):
            return normalize_color(t)
        if hasattr(t, "name"):
            return normalize_color(t)
        if t == 0:
            return "white"
        if t == 1:
            return "black"
    if hasattr(env, "current_player"):
        cp = env.current_player
        return "white" if cp == 1 else "black"
    return "white"


def is_cheqqers_terminal(env: Any) -> bool:
    """Returns True if `env` is in a finished game state."""
    if hasattr(env, "is_terminal") and callable(env.is_terminal):
        return bool(env.is_terminal())
    if hasattr(env, "done"):
        return bool(env.done)
    if hasattr(env, "state"):
        st = env.state
        if isinstance(st, (int, np.integer)):
            return int(st) != 0  # GameState.IN_PROGRESS == 0
        if hasattr(st, "name"):
            return "IN_PROGRESS" not in str(st.name).upper()
    return False


def get_cheqqers_legal_moves(env: Any) -> List[Any]:
    """Retrieves legal moves from `env` in its native format."""
    if hasattr(env, "get_legal_moves") and callable(env.get_legal_moves):
        return list(env.get_legal_moves())
    if hasattr(env, "get_available_moves") and callable(env.get_available_moves):
        return list(env.get_available_moves())
    if hasattr(env, "get_possible_moves") and callable(env.get_possible_moves):
        return list(env.get_possible_moves())
    if hasattr(env, "get_valid_actions") and callable(env.get_valid_actions):
        return list(env.get_valid_actions())
    return []


def compute_state_hash(env: Any) -> str:
    """Computes a fast, deterministic transposition hash for `env`."""
    if hasattr(env, "get_state_hash") and callable(env.get_state_hash):
        return str(env.get_state_hash())
    if hasattr(env, "get_board_state_hash") and callable(env.get_board_state_hash):
        mj = getattr(env, "multi_jump_square", getattr(env, "multiJumpSquare", None))
        return f"{env.get_board_state_hash()}|MJ:{mj}"

    # Fallback hash from state vector + turn + multi_jump_square
    turn = get_active_color(env)
    mj = getattr(env, "multi_jump_square", getattr(env, "multiJumpSquare", None))
    if hasattr(env, "get_state") and callable(env.get_state):
        vec = np.asarray(env.get_state(), dtype=np.float32)
        quantized = np.round(vec * 1000.0).astype(np.int16).tobytes()
        return f"{turn}:{mj}:{ hash(quantized) }"
    return f"{turn}:{mj}:{id(env)}"


def get_cheqqers_outcomes(env: Any, move: Any) -> List[Tuple[float, Any]]:
    """
    Queries `env.get_move_outcomes(move)` for exact Born-weighted child states
    `[(prob_k, next_env_k), ...]`.
    """
    if hasattr(env, "get_move_outcomes") and callable(env.get_move_outcomes):
        outcomes = env.get_move_outcomes(move)
        if outcomes:
            total_p = sum(float(p) for p, _ in outcomes)
            if total_p > 0:
                normed = [
                    (float(p) / total_p, child_env)
                    for p, child_env in outcomes
                    if float(p) > 1e-9
                ]
                # Sort outcomes in descending probability order for Star1 pruning efficiency
                normed.sort(key=lambda item: item[0], reverse=True)
                return normed

    # Fallback for environments with deterministic/single-sample apply_move or step
    next_env = env.clone()
    if hasattr(next_env, "apply_move") and callable(next_env.apply_move):
        next_env.apply_move(move)
    elif hasattr(next_env, "step") and callable(next_env.step):
        try:
            next_env.step(move, return_state=False)
        except TypeError:
            next_env.step(move)
    return [(1.0, next_env)]


# ============================================================================
# 3. Quantum Cheqqers Heuristic & Neural Leaf Evaluator
# ============================================================================

def _row_col_from_idx(idx: int, board_size: int = 6) -> Tuple[int, int]:
    """
    Matches `CheqqersGame.getRowColFromIndex(index)` in `local-cheqqers-server.js:196-205`:
      squaresPerRow = boardSize // 2
      visualRow = index // squaresPerRow
      row = (boardSize - 1) - visualRow
      col = (index % squaresPerRow) * 2 + (0 if row % 2 == 0 else 1)
    """
    squares_per_row = board_size // 2
    visual_row = idx // squares_per_row
    row = (board_size - 1) - visual_row
    col_in_row = idx % squares_per_row
    col = (col_in_row * 2) if (row % 2 == 0) else (col_in_row * 2 + 1)
    return row, col


def evaluate_cheqqers_Fallback(env: Any, perspective_color: str) -> float:
    """
    Fallback quantum heuristic evaluator in [-V_MAX, +V_MAX] when `env.evaluate_state`
    is not directly provided or for blending.
    """
    board_size = int(getattr(env, "board_size", getattr(env, "boardSize", getattr(env, "size", 6))))
    total_squares = (board_size * board_size) // 2

    if is_cheqqers_terminal(env):
        w = getattr(env, "winner", None)
        if w is not None:
            if isinstance(w, str):
                w_norm = normalize_color(w)
                if "draw" in w.lower():
                    return 0.0
                return V_MAX if w_norm == perspective_color else -V_MAX
            if isinstance(w, (int, np.integer)):
                if w == 0:
                    return 0.0
                w_color = "white" if w == 1 else "black"
                return V_MAX if w_color == perspective_color else -V_MAX
        st = getattr(env, "state", None)
        if st == 1:  # WHITE_WON
            return V_MAX if perspective_color == "white" else -V_MAX
        if st == 2:  # BLACK_WON
            return V_MAX if perspective_color == "black" else -V_MAX
        if st == 3:  # DRAW
            return 0.0

    # Extract per-square piece color, king status, and occupation probability
    score = 0.0
    self_prob_sum = 0.0
    opp_prob_sum = 0.0
    center = (board_size - 1) / 2.0

    piece_props = getattr(env, "piece_properties", getattr(env, "pieceProperties", None))
    chances = getattr(env, "chances", None)
    board_list = getattr(env, "board", None)

    for i in range(total_squares):
        p_color: Optional[str] = None
        crowned: bool = False
        prob: float = 0.0

        if isinstance(board_list, list) and i < len(board_list) and isinstance(board_list[i], dict):
            sq = board_list[i]
            if sq.get("piece") is not None:
                p_color = normalize_color(sq["piece"])
                crowned = bool(sq.get("crowned", False))
                prob = float(sq.get("probability", 1.0))
        elif piece_props is not None:
            prop = piece_props.get(i) if isinstance(piece_props, dict) else (piece_props[i] if i < len(piece_props) else None)
            if prop is not None:
                c_raw = prop.get("color") if isinstance(prop, dict) else getattr(prop, "color", 0)
                p_color = normalize_color(c_raw)
                crowned = bool(prop.get("crowned", False) if isinstance(prop, dict) else getattr(prop, "crowned", False))
                if isinstance(chances, (dict, list)):
                    prob = float(chances[i] if i in chances or isinstance(chances, list) else 1.0)
                elif hasattr(env, "get_square_probability"):
                    prob = float(env.get_square_probability(i))
                else:
                    prob = 1.0

        if p_color is None or prob <= 1e-4:
            continue

        row, col = _row_col_from_idx(i, board_size)
        base_val = 2.25 if crowned else 1.0

        # Advancement bonus (White moves row 0 -> board_size-1, Black moves board_size-1 -> 0)
        if not crowned:
            adv = (row / (board_size - 1)) if p_color == "white" else ((board_size - 1 - row) / (board_size - 1))
            adv_bonus = 0.30 * adv
            # Back-row defense bonus
            back_bonus = 0.18 if ((p_color == "white" and row == 0) or (p_color == "black" and row == board_size - 1)) else 0.0
        else:
            adv_bonus = 0.0
            back_bonus = 0.0

        # Central control bonus
        dist_center = abs(col - center) + abs(row - center)
        center_bonus = 0.15 * (board_size - dist_center) / board_size

        # Coherent superposition survival bonus (having two 50% branches on safe squares)
        quantum_bonus = 0.06 if (0.15 < prob < 0.85) else 0.0

        sq_score = prob * (base_val + adv_bonus + back_bonus + center_bonus + quantum_bonus)
        if p_color == perspective_color:
            score += sq_score
            self_prob_sum += prob
        else:
            score -= sq_score
            opp_prob_sum += prob

    if self_prob_sum <= 1e-4 and opp_prob_sum <= 1e-4:
        return 0.0
    if self_prob_sum <= 1e-4:
        return -V_MAX
    if opp_prob_sum <= 1e-4:
        return V_MAX

    # Scale tactical score into [-900, +900] while reserving +-1000 for terminal wins
    return float(np.clip(score * 25.0, -900.0, 900.0))


_WEIGHTS_BY_PATH: Dict[str, Any] = {}


def eval_weights_from_environment() -> Optional[Any]:
    """
    The evaluation weights the server was told to use: `EvalWeights` from the JSON file named by the environment
    variable CHEQQERS_EVAL_WEIGHTS, read once per path; None (the simulator's default evaluation) when the variable
    is unset or empty. A file that is missing or is not a set of weights raises: call this once at start-up, so
    that a server told to play fitted weights never quietly plays others.
    """
    path = os.getenv("CHEQQERS_EVAL_WEIGHTS", "").strip()
    if not path:
        return None
    if path not in _WEIGHTS_BY_PATH:
        from env.cheqqers_official import EvalWeights

        _WEIGHTS_BY_PATH[path] = EvalWeights.from_json(path)
    return _WEIGHTS_BY_PATH[path]


# ============================================================================
# 4. Iterative-Deepening Expecti-AlphaBeta Player
# ============================================================================

class ExpectiCheqqersPlayer:
    """
    Iterative-Deepening Expecti-AlphaBeta search engine for Quantum Cheqqers (6x6)
    with:
      - Exact Born-weighted ChanceNode evaluation and Star1 Expectimax pruning
      - Turn-aware Min/Max recursion (preserving active player across multi-jump chains)
      - Capture extensions at horizon (d == 0) when inside a multi-jump sequence
      - Zobrist / State-Hash Transposition Table (EXACT, LOWERBOUND, UPPERBOUND)
      - Tactical Move Ordering (PV/TT -> Captures -> Promotions -> Killers -> Quantum Splits -> History)
      - Optional PyTorch Value Network leaf blending
      - Strict wall-clock time budget (`time_limit_ms`)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        max_depth: int = 8,
        time_limit_ms: float = 1500.0,
        nn_blend: float = 0.25,
        max_tt_size: int = 65536,
        max_capture_extension: int = 4,
        star1_pruning: bool = True,
        eval_weights: Optional[Any] = None,
        use_cache: bool = True,
        opening_book: Optional[Any] = None,
        quiescence: bool = False,
        quantum_plies: Optional[int] = None,
    ):
        self.model = model
        self.max_depth = int(max_depth)
        self.time_limit_ms = float(time_limit_ms)
        self.nn_blend = float(nn_blend)
        self.max_tt_size = int(max_tt_size)
        self.max_capture_extension = int(max_capture_extension)
        self.star1_pruning = bool(star1_pruning)
        # `env.cheqqers_official.EvalWeights`, or None for the simulator's default evaluation
        self.eval_weights = eval_weights
        # Search on a copy of the position that caches its square probabilities and legal moves (Track-W)
        self.use_cache = bool(use_cache)
        # `search.opening_book.OpeningBook`, or None: positions in it are answered from it, without a search
        self.opening_book = opening_book
        # Quiescence (Track-Q): at the horizon, a position in which the side to move has a capture to make is
        # searched on instead of evaluated, for as long as `max_capture_extension` allows along the line.
        # Captures are compulsory, so the continuation is narrow. Off: the search as it was before 13 Nov 2026.
        self.quiescence = bool(quiescence)
        # Three-square moves (splits, merges) are tried only by positions fewer than this many plies below the
        # root: 1 is our own move, 2 adds the reply. Deeper positions try two-square moves only (Track-Q step 2).
        # None: every move everywhere, the search as it was before 13 Nov 2026.
        self.quantum_plies = None if quantum_plies is None else int(quantum_plies)

        self.tt: Dict[str, TTEntry] = {}
        self.killers: Dict[int, List[MoveKey]] = {}
        self.history_table: Dict[MoveKey, int] = {}
        self._nodes_visited: int = 0
        self._deadline: float = 0.0
        self.unexpanded_branches: int = 0  # moves of the last get_move() the simulator could not expand
        # What the last get_move() did: whether the move came from the opening book, and otherwise the deepest
        # iteration it finished and the score of the chosen move there (for the side to move, +-1000 a decided game)
        self.from_book: bool = False
        self.last_depth: int = 0
        self.last_score: Optional[float] = None

    # ------------------------------------------------------------------
    # Public Entrypoints
    # ------------------------------------------------------------------

    def search(
        self,
        env: Any,
        available_moves: Optional[Sequence[Any]] = None,
    ) -> Any:
        """
        Runs Iterative-Deepening Expecti-AlphaBeta search and returns a move
        guaranteed to be an exact member of `available_moves` (or `env.get_legal_moves()`).
        """
        return self.get_move(env=env, available_moves=available_moves)

    def get_move(
        self,
        env: Any,
        available_moves: Optional[Sequence[Any]] = None,
        valid_actions: Optional[Sequence[Any]] = None,
    ) -> Any:
        env = self._begin(env)

        # Resolve legal moves list
        raw_legal = (
            list(available_moves)
            if available_moves is not None
            else (list(valid_actions) if valid_actions is not None else get_cheqqers_legal_moves(env))
        )
        if len(raw_legal) == 0:
            return None
        if len(raw_legal) == 1:
            return raw_legal[0]

        if self.opening_book is not None:
            booked = self.opening_book.lookup(env, raw_legal)
            if booked is not None:
                self.from_book = True
                return booked

        root_color = get_active_color(env)

        # Map canonical key -> exact tournament move object from `raw_legal`
        key_to_tournament_move: Dict[MoveKey, Any] = {}
        for mv in raw_legal:
            k = canonicalize_cheqqers_move(mv)
            if k not in key_to_tournament_move:
                key_to_tournament_move[k] = mv

        # Match env-native legal moves with `raw_legal`
        env_moves = get_cheqqers_legal_moves(env)
        search_moves: List[Any] = []
        if len(env_moves) > 0:
            for emv in env_moves:
                k = canonicalize_cheqqers_move(emv)
                if k in key_to_tournament_move:
                    search_moves.append(emv)
        if len(search_moves) == 0:
            search_moves = list(raw_legal)

        # Initial ordering at root
        ordered_root_moves = self._order_moves(env, search_moves, ply=0, tt_best_key=None)
        best_move_key: MoveKey = canonicalize_cheqqers_move(ordered_root_moves[0])
        best_score: float = -math.inf

        # The table lives for one get_move() only. Its values are scores for this call's root colour, so an entry
        # kept for the next call is read with the wrong sign whenever the player then moves for the other side,
        # as it does in every colour-balanced series (tests/search/test_expecti_cheqqers.py::TestPlayerReuse).
        # server.py builds a new player per request and never had the carry-over.
        self.tt.clear()
        self.killers.clear()

        # Iterative Deepening loop
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= self._deadline:
                break

            try:
                alpha = -math.inf
                beta = math.inf
                iter_best_key: Optional[MoveKey] = None
                iter_best_val: float = -math.inf
                move_scores: List[Tuple[float, Any]] = []

                for mv in ordered_root_moves:
                    self._check_timeout()
                    val = self._evaluate_chance_node(
                        env=env,
                        move=mv,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_color=root_color,
                        ply=1,
                        ext_remaining=self.max_capture_extension,
                    )
                    m_key = canonicalize_cheqqers_move(mv)
                    move_scores.append((val, mv))

                    if val > iter_best_val:
                        iter_best_val = val
                        iter_best_key = m_key
                    if val > alpha:
                        alpha = val

                if iter_best_key is not None:
                    best_move_key = iter_best_key
                    best_score = iter_best_val
                    self.last_depth = depth
                    self.last_score = iter_best_val

                # Reorder root moves by exact Expecti-AlphaBeta scores from this completed depth
                move_scores.sort(key=lambda item: item[0], reverse=True)
                ordered_root_moves = [mv for _, mv in move_scores]

                # Early exit if forced mate / win found
                if best_score >= V_MAX * 0.98:
                    break

            except _SearchTimeout:
                break

        chosen = key_to_tournament_move.get(best_move_key, raw_legal[0])
        if chosen not in raw_legal:
            chosen = raw_legal[0]
        return chosen

    def _begin(self, env: Any) -> Any:
        """Start the clock and the counters of one call and return the position the search works on."""
        self._deadline = time.perf_counter() + (self.time_limit_ms / 1000.0) * 0.90
        self._nodes_visited = 0
        self.unexpanded_branches = 0
        self.from_book = False
        self.last_depth = 0
        self.last_score = None
        # The position carries the weights its evaluation uses, and every position cloned from it in the search
        # inherits them. Always set, also to None: in a series both players are handed the same position object.
        if hasattr(env, "eval_weights"):
            env.eval_weights = self.eval_weights
        # The search reads each position many times and changes it only through the simulator's own methods, so
        # it works on a copy with the per-position cache on; the caller's position is left as it was.
        if self.use_cache and hasattr(env, "cache_enabled") and hasattr(env, "clone"):
            env = env.clone()
            env.cache_enabled = True
        return env

    def root_values(self, env: Any, moves: Optional[Sequence[Any]] = None) -> Tuple[int, Dict[MoveKey, float]]:
        """The value of every legal move for the side to move, each searched with a full window, at the deepest
        iteration finished within `max_depth` and `time_limit_ms`: (that depth, {move: value}). Depth 0 and no
        values if not even the first iteration finished. Slower than get_move(), which only needs the best move;
        this is the judge for comparing two moves (opening book, review of played games). With `moves`, only
        those of them that are legal are valued: the same values, for a fraction of the time."""
        env = self._begin(env)
        legal = get_cheqqers_legal_moves(env)
        if moves is not None:
            from search.opening_book import same_move  # no import cycle: that module imports this one lazily

            wanted = [list(canonicalize_cheqqers_move(m)) for m in moves]
            legal = [mv for mv in legal if any(same_move(canonicalize_cheqqers_move(mv), w) for w in wanted)]
        root_color = get_active_color(env)
        self.tt.clear()
        self.killers.clear()
        finished = 0
        values: Dict[MoveKey, float] = {}
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= self._deadline:
                break
            try:
                current: Dict[MoveKey, float] = {}
                for mv in legal:
                    self._check_timeout()
                    current[canonicalize_cheqqers_move(mv)] = self._evaluate_chance_node(
                        env=env, move=mv, depth=depth - 1, alpha=-math.inf, beta=math.inf,
                        root_color=root_color, ply=1, ext_remaining=self.max_capture_extension,
                    )
            except _SearchTimeout:
                break
            finished, values = depth, current
        self.last_depth = finished
        return finished, values

    # ------------------------------------------------------------------
    # Expecti-AlphaBeta Core Recursion (DecisionNode <-> ChanceNode)
    # ------------------------------------------------------------------

    def _check_timeout(self) -> None:
        self._nodes_visited += 1
        if (self._nodes_visited & 31) == 0:
            if time.perf_counter() >= self._deadline:
                raise _SearchTimeout()

    def _search_decision_node(
        self,
        env: Any,
        depth: int,
        alpha: float,
        beta: float,
        root_color: str,
        ply: int,
        ext_remaining: int,
    ) -> float:
        self._check_timeout()

        if is_cheqqers_terminal(env):
            return self._evaluate_leaf(env, root_color, ply)

        # Capture / Multi-Jump Horizon Extension:
        # Never cut off at depth == 0 if a piece is in the middle of a multi-jump!
        in_multi_jump = getattr(env, "multi_jump_square", getattr(env, "multiJumpSquare", None)) is not None
        quiescence_ply = False
        if depth <= 0:
            if in_multi_jump and ext_remaining > 0:
                depth = 1
                ext_remaining -= 1
            elif self.quiescence and ext_remaining > 0 and self._capture_pending(env):
                depth = 1
                ext_remaining -= 1
                quiescence_ply = True
            else:
                return self._evaluate_leaf(env, root_color, ply)

        state_hash = compute_state_hash(env)
        alpha_orig = alpha
        beta_orig = beta
        # The depth this node was asked for is what its table entry is worth to a later probe. The forced-capture
        # extension below searches one ply more, but a probe asking for one ply more gets that extension too, so
        # an entry stored under the extended depth would answer it with a search that is one ply short (it made
        # the agent split its last piece next to an enemy on 10 Nov 2026: tests/search/test_expecti_cheqqers.py).
        # A quiescence ply was asked for at depth 0 and is stored and probed as that.
        requested_depth = 0 if quiescence_ply else depth

        # A position that still tries three-square moves and the same position below the limit are different
        # searches under one table key: the few positions above the limit stay out of the table.
        use_tt = self.quantum_plies is None or ply >= self.quantum_plies
        tt_entry = self.tt.get(state_hash) if use_tt else None
        tt_best_key: Optional[MoveKey] = None
        if tt_entry is not None:
            tt_best_key = tt_entry.best_move_key
            if tt_entry.depth >= requested_depth:
                if tt_entry.flag == TT_EXACT:
                    return tt_entry.value
                elif tt_entry.flag == TT_LOWERBOUND:
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.flag == TT_UPPERBOUND:
                    beta = min(beta, tt_entry.value)
                if alpha >= beta:
                    return tt_entry.value

        legal_moves = get_cheqqers_legal_moves(env)
        if len(legal_moves) == 0:
            return self._evaluate_leaf(env, root_color, ply)
        if self.quantum_plies is not None and ply >= self.quantum_plies:
            two_square = [mv for mv in legal_moves if len(canonicalize_cheqqers_move(mv)) == 2]
            if two_square:
                legal_moves = two_square

        # Extend search by 1 ply if there is a forced single capture response
        if (len(legal_moves) == 1 and ext_remaining > 0 and not quiescence_ply
                and self._is_capture_move(env, legal_moves[0])):
            depth += 1
            ext_remaining -= 1

        ordered_moves = self._order_moves(env, legal_moves, ply=ply, tt_best_key=tt_best_key)
        is_maximizing = (get_active_color(env) == root_color)

        best_val = -math.inf if is_maximizing else math.inf
        best_key: Optional[MoveKey] = None

        if is_maximizing:
            for mv in ordered_moves:
                val = self._evaluate_chance_node(
                    env=env,
                    move=mv,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_color=root_color,
                    ply=ply + 1,
                    ext_remaining=ext_remaining,
                )
                if val > best_val:
                    best_val = val
                    best_key = canonicalize_cheqqers_move(mv)
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    self._record_cutoff(env, mv, depth, ply)
                    break
        else:
            for mv in ordered_moves:
                val = self._evaluate_chance_node(
                    env=env,
                    move=mv,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_color=root_color,
                    ply=ply + 1,
                    ext_remaining=ext_remaining,
                )
                if val < best_val:
                    best_val = val
                    best_key = canonicalize_cheqqers_move(mv)
                if best_val < beta:
                    beta = best_val
                if alpha >= beta:
                    self._record_cutoff(env, mv, depth, ply)
                    break

        # Store in Transposition Table
        if best_val <= alpha_orig:
            flag = TT_UPPERBOUND
        elif best_val >= beta_orig:
            flag = TT_LOWERBOUND
        else:
            flag = TT_EXACT

        if use_tt and len(self.tt) < self.max_tt_size:
            self.tt[state_hash] = TTEntry(
                depth=requested_depth,
                flag=flag,
                value=best_val,
                best_move_key=best_key,
            )

        return best_val

    def _evaluate_chance_node(
        self,
        env: Any,
        move: Any,
        depth: int,
        alpha: float,
        beta: float,
        root_color: str,
        ply: int,
        ext_remaining: int,
    ) -> float:
        """
        Computes the exact Born-weighted expectation over `env.get_move_outcomes(move)`
        using Ballard's Star1 Expectimax pruning on bounded values [-V_MAX, +V_MAX].
        """
        try:
            outcomes = get_cheqqers_outcomes(env, move)
        except RuntimeError as err:
            # The simulator refuses joint states above its dimension limit (many entangled pieces). The move is
            # legal in the organisers' engine, so the search must go on: it scores the move as the position it
            # starts from and counts it. Raising here used to abort get_move() and, in server.py, sent the
            # request to the June code path (tests/search/test_expecti_cheqqers.py::TestSimulatorLimit).
            if "QuantumLimitExceeded" not in str(err):
                raise
            self.unexpanded_branches += 1
            return self._evaluate_leaf(env, root_color, ply)
        if len(outcomes) == 1:
            _, next_env = outcomes[0]
            return self._search_decision_node(
                env=next_env,
                depth=depth,
                alpha=alpha,
                beta=beta,
                root_color=root_color,
                ply=ply,
                ext_remaining=ext_remaining,
            )

        accum_exp = 0.0
        rem_prob = 1.0

        for prob, next_env in outcomes:
            rem_prob = max(0.0, rem_prob - prob)

            if self.star1_pruning and prob > 1e-6:
                # Narrow child search window [child_alpha, child_beta]
                if alpha <= -V_MAX:
                    child_alpha = -math.inf
                else:
                    child_alpha = max(-V_MAX, (alpha - accum_exp - rem_prob * V_MAX) / prob)

                if beta >= V_MAX:
                    child_beta = math.inf
                else:
                    child_beta = min(V_MAX, (beta - accum_exp + rem_prob * V_MAX) / prob)
            else:
                child_alpha = -math.inf
                child_beta = math.inf

            if child_alpha >= child_beta:
                # Star1 Cutoff: even extreme values on remaining branches cannot bring
                # the expectation back inside [alpha, beta].
                bound_val = child_alpha if child_alpha >= beta else child_beta
                return accum_exp + prob * bound_val + rem_prob * (V_MAX if child_alpha >= beta else -V_MAX)

            child_val = self._search_decision_node(
                env=next_env,
                depth=depth,
                alpha=child_alpha,
                beta=child_beta,
                root_color=root_color,
                ply=ply,
                ext_remaining=ext_remaining,
            )
            accum_exp += prob * child_val

        return accum_exp

    # ------------------------------------------------------------------
    # Tactical Move Ordering (PV/TT, Captures, Promotions, Killers, Splits)
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_pending(env: Any) -> bool:
        """Whether the side to move has a capture to make (and so, captures being compulsory, must make one)."""
        pending = getattr(env, "capture_pending", None)
        return bool(pending()) if callable(pending) else False

    def _is_capture_move(self, env: Any, move: Any) -> bool:
        if isinstance(move, dict) and move.get("is_take_move"):
            return True
        if hasattr(move, "is_take_move") and getattr(move, "is_take_move"):
            return True
        # The simulator knows: captures are compulsory, so a legal move is a capture exactly when one is pending.
        # (The distance rule below called every long slide of a queen a capture: searched first, never a killer,
        # and extended when it was the only move. Kept for positions that are not the official simulator's.)
        pending = getattr(env, "capture_pending", None)
        if callable(pending):
            return bool(pending())
        arr = move_to_array(move)
        if arr is not None and len(arr) == 2:
            board_size = int(getattr(env, "board_size", getattr(env, "boardSize", getattr(env, "size", 6))))
            r1, c1 = _row_col_from_idx(arr[0], board_size)
            r2, c2 = _row_col_from_idx(arr[1], board_size)
            return abs(r2 - r1) >= 2
        return False

    def _score_move_for_ordering(
        self,
        env: Any,
        move: Any,
        ply: int,
        tt_best_key: Optional[MoveKey],
        chances: Optional[Any] = None,
    ) -> float:
        m_key = canonicalize_cheqqers_move(move)
        if tt_best_key is not None and m_key == tt_best_key:
            return 1_000_000.0

        board_size = int(getattr(env, "board_size", getattr(env, "boardSize", getattr(env, "size", 6))))
        arr = move_to_array(move)
        if chances is None:
            chances = getattr(env, "chances", None)

        def get_occ_prob(sq_idx: int) -> float:
            if isinstance(chances, dict):
                return float(chances.get(sq_idx, 0.0))
            if isinstance(chances, list) and sq_idx < len(chances):
                return float(chances[sq_idx])
            if hasattr(env, "get_square_probability"):
                return float(env.get_square_probability(sq_idx))
            return 0.0

        # 1. Capture Moves (highest tactical priority after TT move)
        if self._is_capture_move(env, move):
            score = 500_000.0
            if arr is not None and len(arr) == 2:
                from_sq, to_sq = arr[0], arr[1]
                r1, _ = _row_col_from_idx(from_sq, board_size)
                r2, _ = _row_col_from_idx(to_sq, board_size)
                jump_dist = abs(r2 - r1)
                # Prefer captures into deterministically empty landing squares (avoids moveFailed!)
                landing_occ = get_occ_prob(to_sq)
                source_occ = max(0.5, get_occ_prob(from_sq))
                score += 10_000.0 * (1.0 - landing_occ) * source_occ
                score += 500.0 * jump_dist
            return score

        score = 0.0

        # 2. Killer Moves at this ply
        ply_killers = self.killers.get(ply, [])
        if m_key in ply_killers:
            score += 100_000.0

        # 3. Promotion / Crowning & Quantum Split / Merge Evaluation
        active_color = get_active_color(env)
        promo_row = (board_size - 1) if active_color == "white" else 0
        center = (board_size - 1) / 2.0

        if arr is not None:
            if len(arr) == 2:
                from_sq, to_sq = arr[0], arr[1]
                r_to, c_to = _row_col_from_idx(to_sq, board_size)
                landing_occ = get_occ_prob(to_sq)
                # Penalize quiet moves onto occupied superposition squares (risk of moveFailed)
                if landing_occ > 0.01:
                    score -= 15_000.0 * landing_occ
                if r_to == promo_row:
                    score += 80_000.0
                # Center control and advancement
                adv = r_to if active_color == "white" else (board_size - 1 - r_to)
                score += 300.0 * adv - 100.0 * abs(c_to - center)
            elif len(arr) == 3:
                # Distinguish Split [from, to1, to2] vs Merge [from1, from2, to]
                a, b, c = arr[0], arr[1], arr[2]
                occ_b = get_occ_prob(b)
                is_merge = occ_b > 0.05
                if not is_merge:
                    # Split move: [from, to1, to2]
                    r1, c1 = _row_col_from_idx(b, board_size)
                    r2, c2 = _row_col_from_idx(c, board_size)
                    occ_to1 = get_occ_prob(b)
                    occ_to2 = get_occ_prob(c)
                    if occ_to1 < 0.01 and occ_to2 < 0.01:
                        # Clean quantum split into two empty squares: high strategic value in Entanglement!
                        score += 12_000.0
                        if r1 == promo_row or r2 == promo_row:
                            score += 45_000.0
                        score -= 80.0 * (abs(c1 - center) + abs(c2 - center))
                    else:
                        score -= 10_000.0 * (occ_to1 + occ_to2)
                else:
                    # Merge move: [from1, from2, to]
                    r_to, _ = _row_col_from_idx(c, board_size)
                    occ_to = get_occ_prob(c)
                    if occ_to < 0.01:
                        score += 15_000.0
                        if r_to == promo_row:
                            score += 55_000.0

        # 4. History Heuristic
        score += float(min(20_000, self.history_table.get(m_key, 0)))
        return score

    def _order_moves(
        self,
        env: Any,
        moves: Sequence[Any],
        ply: int,
        tt_best_key: Optional[MoveKey],
    ) -> List[Any]:
        chances = getattr(env, "chances", None)  # the squares' presences, read once for all the moves
        return sorted(
            moves,
            key=lambda mv: self._score_move_for_ordering(env, mv, ply, tt_best_key, chances),
            reverse=True,
        )

    def _record_cutoff(self, env: Any, move: Any, depth: int, ply: int) -> None:
        if self._is_capture_move(env, move):
            return
        m_key = canonicalize_cheqqers_move(move)
        ply_list = self.killers.setdefault(ply, [])
        if m_key not in ply_list:
            ply_list.insert(0, m_key)
            if len(ply_list) > 2:
                ply_list.pop()
        self.history_table[m_key] = self.history_table.get(m_key, 0) + depth * depth

    # ------------------------------------------------------------------
    # Hybrid Leaf Evaluation (Exact Quantum Heuristic + Optional Neural Net)
    # ------------------------------------------------------------------

    def _evaluate_leaf(self, env: Any, root_color: str, ply: int) -> float:
        if hasattr(env, "evaluate_state") and callable(env.evaluate_state):
            heur_val = float(env.evaluate_state(root_color))
        else:
            heur_val = evaluate_cheqqers_Fallback(env, root_color)

        # Prefer faster mates / wins and slower losses
        if heur_val >= V_MAX * 0.99:
            return V_MAX - float(ply)
        if heur_val <= -V_MAX * 0.99:
            return -V_MAX + float(ply)

        if self.model is not None and self.nn_blend > 0.0:
            nn_val = self._evaluate_neural_value(env, root_color)
            if nn_val is not None:
                # Scale neural value in [-1, 1] to heuristic score units [-250, +250]
                blended = (1.0 - self.nn_blend) * heur_val + self.nn_blend * (nn_val * 250.0)
                return float(np.clip(blended, -V_MAX * 0.95, V_MAX * 0.95))

        return float(np.clip(heur_val, -V_MAX * 0.95, V_MAX * 0.95))

    def _evaluate_neural_value(self, env: Any, root_color: str) -> Optional[float]:
        try:
            state_vec = None
            if hasattr(env, "get_state_72") and callable(env.get_state_72):
                state_vec = np.asarray(env.get_state_72(), dtype=np.float32)
            elif hasattr(env, "get_state") and callable(env.get_state):
                state_vec = np.asarray(env.get_state(), dtype=np.float32)
            if state_vec is None:
                return None

            expected_in = getattr(self.model, "in_dim", state_vec.shape[0])
            if state_vec.shape[0] != expected_in:
                return None

            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                out = self.model(state_t)
                if isinstance(out, tuple) and len(out) >= 2:
                    val_scalar = float(out[1].squeeze().detach().cpu().item())
                else:
                    return None

            val_quant = round(val_scalar * 64.0) / 64.0
            active_color = get_active_color(env)
            return val_quant if active_color == root_color else -val_quant
        except Exception:
            return None


@dataclass
class _CheqqersMCTSChanceBranch:
    prob: float
    next_env: Any
    child_node: Optional["_CheqqersMCTSDecisionNode"] = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass
class _CheqqersMCTSActionEdge:
    move: Any
    move_key: MoveKey
    prior_logit: float
    prior_prob: float
    visits: int = 0
    value_sum: float = 0.0
    branches: Optional[List[_CheqqersMCTSChanceBranch]] = None

    @property
    def q_value(self) -> float:
        if self.branches is not None and self.visits > 0:
            # Exact Born-rule expectation over visited branches
            total_w = 0.0
            weighted_q = 0.0
            for b in self.branches:
                if b.visits > 0:
                    weighted_q += b.prob * b.q_value
                    total_w += b.prob
            if total_w > 1e-6:
                return weighted_q / total_w
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class _CheqqersMCTSDecisionNode:
    def __init__(self, env: Any, active_color: str, ply: int) -> None:
        self.env = env
        self.active_color = active_color
        self.ply = ply
        self.is_terminal: bool = is_cheqqers_terminal(env)
        self.edges: List[_CheqqersMCTSActionEdge] = []
        self.expanded: bool = False
        self.visits: int = 0


class GumbelQMCTSCheqqersPlayer:
    """
    Smart Quantum MCTS for Cheqqers (`boardSize=6, quantumness=2`):
      1. Gumbel AlphaZero / MuZero Sequential Halving at the root (Danihelka et al., 2022)
         for provable policy improvement under small simulation budgets (32-160 sims).
      2. Explicit Born-Rule Chance Nodes (`get_cheqqers_outcomes`) with Deterministic
         Proportional Allocation (selecting the branch maximizing `b.prob * (edge.visits + 1) - b.visits`
         so zero Monte Carlo variance is introduced on quantum collapses).
      3. DAG Transposition Value Cache (`compute_state_hash`) + Multi-Jump Quiescence
         rollout extension (`ExpectiCheqqersPlayer._evaluate_leaf`).
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        num_simulations: int = 96,
        max_candidates: int = 12,
        c_puct: float = 1.35,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        time_limit_ms: float = 650.0,
        nn_blend: float = 0.35,
        rollout_depth: int = 1,
    ) -> None:
        self.model = model
        self.num_simulations = max(8, int(num_simulations))
        self.max_candidates = max(2, int(max_candidates))
        self.c_puct = float(c_puct)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.time_limit_ms = float(time_limit_ms)
        self.nn_blend = float(nn_blend)
        self.rollout_depth = max(0, int(rollout_depth))
        self._evaluator = ExpectiCheqqersPlayer(
            model=model,
            max_depth=max(1, self.rollout_depth),
            time_limit_ms=max(150.0, time_limit_ms * 0.5),
            nn_blend=nn_blend,
        )
        self._leaf_cache: Dict[str, float] = {}
        self.last_search_stats: Dict[str, Any] = {}

    def _eval_normalized(self, env: Any, root_color: str, ply: int) -> float:
        """
        Returns value in [-1.0, +1.0] from `root_color` perspective, extending
        automatically if `env` is in the middle of a multi-jump sequence.
        """
        if is_cheqqers_terminal(env):
            raw = self._evaluator._evaluate_leaf(env, root_color, ply)
            return float(np.clip(raw / V_MAX, -1.0, 1.0))

        state_key = f"{compute_state_hash(env)}|{root_color}"
        cached = self._leaf_cache.get(state_key)
        if cached is not None:
            return cached

        in_multi_jump = getattr(env, "multi_jump_square", getattr(env, "multiJumpSquare", None)) is not None
        if in_multi_jump or self.rollout_depth >= 2:
            # Resolve immediate tactical / multi-jump horizon via 1-2 ply Expectimax quiescence
            try:
                self._evaluator._deadline = time.perf_counter() + 0.08
                self._evaluator._nodes_visited = 0
                raw_score = self._evaluator._search_decision_node(
                    env=env,
                    depth=1 if not in_multi_jump else 2,
                    alpha=-math.inf,
                    beta=math.inf,
                    root_color=root_color,
                    ply=ply,
                    ext_remaining=2,
                )
            except Exception:
                raw_score = self._evaluator._evaluate_leaf(env, root_color, ply)
        else:
            raw_score = self._evaluator._evaluate_leaf(env, root_color, ply)

        # Normalize heuristic units (~[-250, +250]) into [-1.0, +1.0] via smooth tanh scaling
        if abs(raw_score) >= V_MAX * 0.9:
            val = 1.0 if raw_score > 0 else -1.0
        else:
            val = math.tanh(raw_score / 180.0)

        if len(self._leaf_cache) < 50_000:
            self._leaf_cache[state_key] = val
        return val

    def _expand_node(self, node: _CheqqersMCTSDecisionNode) -> None:
        if node.expanded or node.is_terminal:
            node.expanded = True
            return

        legal_moves = get_cheqqers_legal_moves(node.env)
        if not legal_moves:
            node.is_terminal = True
            node.expanded = True
            return

        # Compute tactical ordering scores and convert to prior logits/probabilities
        raw_scores: List[float] = []
        for mv in legal_moves:
            sc = self._evaluator._score_move_for_ordering(node.env, mv, ply=node.ply, tt_best_key=None)
            raw_scores.append(sc)

        max_sc = max(raw_scores)
        # Scale tactical scores into well-conditioned logits in [-4, 0]
        logits = [float(np.clip((sc - max_sc) / 25_000.0, -5.0, 0.0)) for sc in raw_scores]
        exp_l = [math.exp(l) for l in logits]
        sum_exp = sum(exp_l) or 1.0
        probs = [e / sum_exp for e in exp_l]

        # Sort edges by prior_logit descending
        order = sorted(range(len(legal_moves)), key=lambda i: logits[i], reverse=True)
        node.edges = [
            _CheqqersMCTSActionEdge(
                move=legal_moves[i],
                move_key=canonicalize_cheqqers_move(legal_moves[i]),
                prior_logit=logits[i],
                prior_prob=probs[i],
            )
            for i in order
        ]
        node.expanded = True

    def _simulate_from_edge(
        self,
        parent_node: _CheqqersMCTSDecisionNode,
        edge: _CheqqersMCTSActionEdge,
        root_color: str,
        deadline: float,
        max_tree_depth: int = 6,
    ) -> float:
        """
        Simulates one trajectory down `edge` (expanding its Born-rule Chance Node if needed)
        and returns the value in [-1.0, +1.0] from `root_color`'s perspective.
        """
        if edge.branches is None:
            outcomes = get_cheqqers_outcomes(parent_node.env, edge.move)
            edge.branches = [
                _CheqqersMCTSChanceBranch(prob=float(p), next_env=n_env)
                for p, n_env in outcomes
            ]

        if not edge.branches:
            return 0.0

        # Deterministic Proportional Chance Allocation:
        # Pick the branch whose empirical visit fraction lags furthest behind its Born probability `b.prob`
        if len(edge.branches) == 1:
            branch = edge.branches[0]
        else:
            next_total = edge.visits + 1
            branch = max(
                edge.branches,
                key=lambda b: b.prob * next_total - b.visits,
            )

        if branch.child_node is None:
            branch.child_node = _CheqqersMCTSDecisionNode(
                env=branch.next_env,
                active_color=get_active_color(branch.next_env),
                ply=parent_node.ply + 1,
            )
            val = self._eval_normalized(branch.next_env, root_color, parent_node.ply + 1)
        else:
            val = self._traverse_node(
                node=branch.child_node,
                root_color=root_color,
                deadline=deadline,
                remaining_depth=max_tree_depth - 1,
            )

        branch.visits += 1
        branch.value_sum += val
        edge.visits += 1
        edge.value_sum += val
        parent_node.visits += 1
        return val

    def _traverse_node(
        self,
        node: _CheqqersMCTSDecisionNode,
        root_color: str,
        deadline: float,
        remaining_depth: int,
    ) -> float:
        if node.is_terminal or remaining_depth <= 0 or time.perf_counter() >= deadline:
            return self._eval_normalized(node.env, root_color, node.ply)

        if not node.expanded:
            self._expand_node(node)
            if node.is_terminal or not node.edges:
                return self._eval_normalized(node.env, root_color, node.ply)

        is_root_turn = (node.active_color == root_color)
        sign = 1.0 if is_root_turn else -1.0
        sqrt_total = math.sqrt(max(1, node.visits))

        best_edge = node.edges[0]
        best_puct = -math.inf
        for ed in node.edges:
            q_player = sign * ed.q_value if ed.visits > 0 else 0.0
            u_score = self.c_puct * ed.prior_prob * (sqrt_total / (1.0 + ed.visits))
            score = q_player + u_score
            if score > best_puct:
                best_puct = score
                best_edge = ed

        return self._simulate_from_edge(
            parent_node=node,
            edge=best_edge,
            root_color=root_color,
            deadline=deadline,
            max_tree_depth=remaining_depth,
        )

    def _completed_q_sigma(
        self,
        edge: _CheqqersMCTSActionEdge,
        max_visits: int,
        v_mix: float,
    ) -> float:
        q_hat = edge.q_value if edge.visits > 0 else v_mix
        return (self.c_visit + float(max_visits)) * self.c_scale * q_hat

    def get_move(
        self,
        env: Any,
        available_moves: Optional[Sequence[Any]] = None,
    ) -> Any:
        start_t = time.perf_counter()
        deadline = start_t + (self.time_limit_ms / 1000.0) * 0.90

        raw_legal = list(available_moves) if available_moves is not None else get_cheqqers_legal_moves(env)
        if not raw_legal:
            return None
        if len(raw_legal) == 1:
            return raw_legal[0]

        key_to_tournament_move: Dict[MoveKey, Any] = {}
        for m in raw_legal:
            key_to_tournament_move[canonicalize_cheqqers_move(m)] = m

        root_color = get_active_color(env)
        root = _CheqqersMCTSDecisionNode(env=env, active_color=root_color, ply=0)
        self._expand_node(root)
        if not root.edges:
            return raw_legal[0]

        v_root = self._eval_normalized(env, root_color, 0)

        # Gumbel Sequential Halving over Top-K candidates at the root
        k_init = min(len(root.edges), self.max_candidates)
        active_edges: List[_CheqqersMCTSActionEdge] = list(root.edges[:k_init])
        num_phases = max(1, int(math.ceil(math.log2(max(2, k_init)))))
        sims_remaining = self.num_simulations
        sims_run = 0

        for phase in range(num_phases):
            if len(active_edges) <= 1 or time.perf_counter() >= deadline:
                break
            phases_left = num_phases - phase
            budget_this_phase = max(len(active_edges), sims_remaining // phases_left)
            sims_per_cand = max(1, budget_this_phase // len(active_edges))

            for _ in range(sims_per_cand):
                if time.perf_counter() >= deadline:
                    break
                for ed in active_edges:
                    if time.perf_counter() >= deadline:
                        break
                    self._simulate_from_edge(
                        parent_node=root,
                        edge=ed,
                        root_color=root_color,
                        deadline=deadline,
                        max_tree_depth=6,
                    )
                    sims_run += 1
                    sims_remaining = max(0, sims_remaining - 1)

            max_n = max((ed.visits for ed in active_edges), default=1)
            active_edges.sort(
                key=lambda ed: ed.prior_logit + self._completed_q_sigma(ed, max_n, v_root),
                reverse=True,
            )
            keep = max(1, len(active_edges) // 2)
            if phase == num_phases - 1:
                keep = 1
            active_edges = active_edges[:keep]

        # Final selection by Gumbel completed Q-value
        max_n_final = max((ed.visits for ed in root.edges[:k_init]), default=1)
        best_edge = max(
            root.edges[:k_init],
            key=lambda ed: (
                ed.visits > 0,
                ed.prior_logit + self._completed_q_sigma(ed, max_n_final, v_root),
            ),
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.last_search_stats = {
            "simulations": sims_run,
            "best_q": round(best_edge.q_value, 4),
            "best_visits": best_edge.visits,
            "elapsed_ms": round(elapsed_ms, 2),
        }

        chosen = key_to_tournament_move.get(best_edge.move_key, raw_legal[0])
        if chosen not in raw_legal:
            chosen = raw_legal[0]
        return chosen
