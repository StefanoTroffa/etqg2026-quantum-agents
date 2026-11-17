"""
Leiden Cross-Team Sparring Referee & Blind-Spot Harvester Hub.

Designed for joint sparring between Leiden University ETQG / IEEE CoG teams:
  - Team 5: Chris Chalkias & Stefano Troffa
  - Team 1: Marien Raat & Tommaso Zendron

Supports:
  1. Stateless HTTP `POST /move` agents (`HttpEndpointAgent`) for any local/remote
     server (Node.js, Python, Rust, C++, etc. over localhost, SSH tunnel, or Tailscale).
  2. In-process `ChanceQMCTSPlayer` (`InProcessTiqTaqToeAgent`) for Quantum TiqTaqToe (Q=3).
  3. In-process `ExpectiCheqqersPlayer` (`InProcessCheqqersAgent`) for Quantum Cheqqers (6x6, Q=2).
  4. Color-balanced Head-to-Head & Round-Robin series with Elo + 95% CI, SPRT LLR, and
     latency percentiles (`p50`, `p95`, `max`, timeouts, illegal move tracking).
  5. Automatic Blind-Spot & Move-Disagreement Harvester (`results/sparring_hard_cases.jsonl`)
     and `solve_hard_cases` Teacher Distillation generator (`results/sparring_distilled_targets.jsonl`).
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Ensure `src` is importable whether invoked as `python src/arena/leiden_sparring_hub.py`
# or imported as `src.arena.leiden_sparring_hub`.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.cheqqers_official import EvalWeights, GameState, OfficialCheqqersEnv, PieceColor
from env.tiqtaqtoe_official import OfficialTiqTaqToeEnv, TicTacResult
from models.kan import KANNet
from models.mlp import MLPNet
from models.snn import SpikingNet
from models.transformer import QuantumBoardTransformerNet
from search.chance_q_mcts import (
    ChanceQMCTSPlayer,
    canonicalize_move as canonicalize_ttt_move,
    move_to_action_46,
)
from search.expecti_cheqqers import (
    ExpectiCheqqersPlayer,
    GumbelQMCTSCheqqersPlayer,
    canonicalize_cheqqers_move,
    move_to_array as cheqqers_move_to_array,
)


# ============================================================================
# 1. Statistical Helpers: Elo + 95% Confidence Intervals & SPRT LLR
# ============================================================================

def _score_to_elo(score: float) -> float:
    """Maps win rate mu in (0, 1) to logistic Elo difference: -400 * log10(1/mu - 1)."""
    clamped = max(1e-4, min(1.0 - 1e-4, float(score)))
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def compute_elo_diff_and_ci(
    wins: int,
    draws: int,
    losses: int,
    z_value: float = 1.959963984540054,
) -> Tuple[float, float, float]:
    """
    Computes the empirical Elo difference (Agent A vs Agent B) and 95% confidence
    interval (elo_diff, elo_ci_low, elo_ci_high) from trinomial match counts (W, D, L).
    Uses Bayesian Laplace smoothing (0.5 pseudocount split) when empirical score is 0% or 100%
    so Elo bounds remain finite and informative even on small sample sizes.
    """
    n = wins + draws + losses
    if n <= 0:
        return 0.0, 0.0, 0.0

    raw_score = (wins + 0.5 * draws) / float(n)
    # Regularized mean for finite point estimate at extreme 0% / 100% scores
    reg_score = (wins + 0.5 * draws + 0.5) / float(n + 1.0)
    point_score = raw_score if (0.001 < raw_score < 0.999) else reg_score

    # Sample variance of game outcomes X_i in {1.0, 0.5, 0.0}
    mean_x = raw_score
    var_x = (
        wins * ((1.0 - mean_x) ** 2)
        + draws * ((0.5 - mean_x) ** 2)
        + losses * ((0.0 - mean_x) ** 2)
    ) / max(1.0, float(n - 1))
    # Floor standard error with Wilson-like 1/(2*(n+1)) when variance is zero (all W or all L)
    se_x = max(math.sqrt(var_x / float(n)), 0.5 / float(n + 1))

    low_score = max(1e-3, min(1.0 - 1e-3, point_score - z_value * se_x))
    high_score = max(1e-3, min(1.0 - 1e-3, point_score + z_value * se_x))

    elo_diff = _score_to_elo(point_score)
    elo_low = _score_to_elo(low_score)
    elo_high = _score_to_elo(high_score)
    return float(elo_diff), float(elo_low), float(elo_high)


def compute_sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = 0.0,
    elo1: float = 15.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes the Sequential Probability Ratio Test (SPRT) log-likelihood ratio (LLR)
    for testing H0: Delta_Elo = elo0 vs H1: Delta_Elo = elo1 under a trinomial W/D/L model
    (Michel Van den Bergh / Fishtest trinomial approximation).
    """
    n = wins + draws + losses
    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)

    if n <= 0:
        return {
            "llr": 0.0,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "elo0": float(elo0),
            "elo1": float(elo1),
            "status": "CONTINUE",
        }

    def _expected_score(elo: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))

    s0 = _expected_score(elo0)
    s1 = _expected_score(elo1)

    # Empirical draw ratio regularized away from 0 and 1
    d_prob = max(0.01, min(0.90, (draws + 0.5) / float(n + 1.0)))

    def _trinomial_probs(expected_s: float, d_ratio: float) -> Tuple[float, float, float]:
        # Maintain w - l = 2*expected_s - 1 and w + d + l = 1
        d = min(d_ratio, 2.0 * min(expected_s, 1.0 - expected_s) - 1e-4)
        d = max(1e-4, d)
        w = max(1e-4, expected_s - 0.5 * d)
        l = max(1e-4, 1.0 - w - d)
        total = w + d + l
        return w / total, d / total, l / total

    w0, d0, l0 = _trinomial_probs(s0, d_prob)
    w1, d1, l1 = _trinomial_probs(s1, d_prob)

    llr = (
        wins * math.log(w1 / w0)
        + draws * math.log(d1 / d0)
        + losses * math.log(l1 / l0)
    )

    if llr >= upper_bound:
        status = "H1_ACCEPTED"
    elif llr <= lower_bound:
        status = "H0_ACCEPTED"
    else:
        status = "CONTINUE"

    return {
        "llr": float(llr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "elo0": float(elo0),
        "elo1": float(elo1),
        "status": status,
    }


def compute_latency_summary(latencies_ms: Sequence[float], timeouts: int, illegal_moves: int) -> Dict[str, Any]:
    """Computes p50, p95, max, mean latency in milliseconds plus error counts."""
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "timeouts": int(timeouts),
            "illegal_moves": int(illegal_moves),
        }
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "timeouts": int(timeouts),
        "illegal_moves": int(illegal_moves),
    }


# ============================================================================
# 2. Agent Adapters (HTTP Endpoint, In-Process TiqTaqToe, In-Process Cheqqers)
# ============================================================================

class SparringAgent:
    """Base interface for all sparring agents (HTTP or in-process)."""

    def __init__(self, name: str):
        self.name = str(name)
        self.latencies_ms: List[float] = []
        self.timeouts: int = 0
        self.illegal_moves: int = 0

    def reset_telemetry(self) -> None:
        self.latencies_ms.clear()
        self.timeouts = 0
        self.illegal_moves = 0

    def get_telemetry(self) -> Dict[str, Any]:
        return compute_latency_summary(self.latencies_ms, self.timeouts, self.illegal_moves)

    def select_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Tuple[Any, float, bool]:
        """
        Returns (chosen_move, elapsed_ms, was_fallback).
        Subclasses implement `_compute_move(env, payload, legal_moves)`.
        """
        if not legal_moves:
            return None, 0.0, False

        t0 = time.perf_counter()
        was_fallback = False
        chosen: Any = None
        try:
            chosen = self._compute_move(env, payload, legal_moves)
        except TimeoutError:
            self.timeouts += 1
            was_fallback = True
        except Exception:
            self.illegal_moves += 1
            was_fallback = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(elapsed_ms)

        matched = _match_legal_move(chosen, legal_moves, env)
        if matched is None:
            if not was_fallback:
                self.illegal_moves += 1
                was_fallback = True
            matched = legal_moves[0]

        return matched, elapsed_ms, was_fallback

    def _compute_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Any:
        raise NotImplementedError


def _match_legal_move(candidate: Any, legal_moves: Sequence[Any], env: Any) -> Optional[Any]:
    """
    Verifies whether `candidate` matches one of `legal_moves` in the tournament format
    and returns the exact matching move object from `legal_moves`.
    """
    if candidate is None or not legal_moves:
        return None
    if candidate in legal_moves:
        return candidate

    # Check if TiqTaqToe or Cheqqers
    is_ttt = isinstance(env, OfficialTiqTaqToeEnv)
    if is_ttt:
        try:
            cand_key = canonicalize_ttt_move(candidate)
        except Exception:
            return None
        for lm in legal_moves:
            if canonicalize_ttt_move(lm) == cand_key:
                return lm
        return None
    else:
        cand_arr = cheqqers_move_to_array(candidate)
        if cand_arr is None:
            return None
        cand_tuple = tuple(cand_arr)
        for lm in legal_moves:
            lm_arr = cheqqers_move_to_array(lm)
            if lm_arr is not None and tuple(lm_arr) == cand_tuple:
                return lm
        return None


class HttpEndpointAgent(SparringAgent):
    """
    Calls any remote or local stateless HTTP `POST /move` server
    (e.g., Chris Chalkias's agent, Marien Raat's agent, or our `server.py` on `stefano-pc`).
    """

    def __init__(self, name: str, url: str, timeout_ms: int = 3000):
        super().__init__(name=name)
        cleaned = url.strip().rstrip("/")
        if not cleaned.endswith("/move"):
            cleaned = f"{cleaned}/move"
        self.url = cleaned
        self.timeout_ms = int(timeout_ms)

    def _compute_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Any:
        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        timeout_sec = max(0.1, float(self.timeout_ms) / 1000.0)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw_resp = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            if "timed out" in str(exc).lower() or isinstance(getattr(exc, "reason", None), TimeoutError):
                raise TimeoutError(f"HTTP /move timeout after {self.timeout_ms}ms: {exc}") from exc
            raise RuntimeError(f"HTTP /move request failed for {self.url}: {exc}") from exc
        except TimeoutError as exc:
            raise TimeoutError(f"HTTP /move timeout after {self.timeout_ms}ms: {exc}") from exc

        parsed = json.loads(raw_resp)
        if isinstance(parsed, dict) and "move" in parsed:
            return parsed["move"]
        return parsed


def _load_tiqtaqtoe_model(model_path: Optional[str]) -> Optional[torch.nn.Module]:
    """Loads a 111-dim -> 46-dim TiqTaqToe model from `model_path` if provided."""
    if not model_path or model_path.lower() in ("none", "null", "pure"):
        return None

    resolved_path: Optional[Path] = None
    if model_path.lower() == "auto":
        candidates = [
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_tictactoe_mlp_p1_best_joint.pth",
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_tictactoe_mlp_p1_calibrated_joint.pth",
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_tictactoe_mlp_p1_best_global.pth",
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_mlp_calibrated.pth",
        ]
        for c in candidates:
            if c.exists():
                resolved_path = c
                break
    else:
        p = Path(model_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.exists():
            resolved_path = p

    if resolved_path is None or not resolved_path.exists():
        return None

    name_low = resolved_path.name.lower()
    try:
        state_dict = torch.load(resolved_path, map_location="cpu")
        if "transformer" in name_low or any(k.startswith("encoder.layers.") for k in state_dict.keys()):
            model: torch.nn.Module = QuantumBoardTransformerNet(
                in_dim=111, hidden_dim=64, out_dim=46, nhead=4, num_layers=2
            )
        elif "kan" in name_low:
            model = KANNet(
                in_dim=111, hidden_dims=[128, 128, 128], out_dim=46, decoupled=True, wdl=True
            )
        elif "snn" in name_low:
            model = SpikingNet(in_dim=111, hidden_dim=64, out_dim=46, steps=20)
        else:
            # Inspect state_dict to see if decoupled 128x3 or legacy 64x2
            is_decoupled = any(k.startswith("policy_backbone.") for k in state_dict.keys())
            first_w = state_dict.get("policy_backbone.0.weight", state_dict.get("backbone.0.weight", state_dict.get("network.0.weight")))
            h_dim = int(first_w.shape[0]) if first_w is not None else 128
            if is_decoupled or h_dim == 128:
                model = MLPNet(
                    in_dim=111, hidden_dims=[h_dim, h_dim, h_dim], out_dim=46, decoupled=True, wdl=True
                )
            else:
                model = MLPNet(in_dim=111, hidden_dim=h_dim, out_dim=46)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception:
        return None


def _load_cheqqers_model(model_path: Optional[str], board_size: int = 6) -> Optional[torch.nn.Module]:
    """Loads a 72-dim -> 288-dim Cheqqers model from `model_path` if provided."""
    if not model_path or model_path.lower() in ("none", "null", "heuristic"):
        return None

    resolved_path: Optional[Path] = None
    if model_path.lower() == "auto":
        candidates = [
            _PROJECT_ROOT / "results" / "q_net_official_cheqqers_size6_q2_best.pth",
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_checkers_mlp_ep1500_joint.pth",
            _PROJECT_ROOT / "results" / "q_net_quantum_entanglement_checkers_mlp_ep900_joint.pth",
            _PROJECT_ROOT / "results" / "cheqqers" / "checkers_trained_mlp.pth",
        ]
        for c in candidates:
            if c.exists():
                resolved_path = c
                break
    else:
        p = Path(model_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.exists():
            resolved_path = p

    if resolved_path is None or not resolved_path.exists():
        return None

    name_low = resolved_path.name.lower()
    num_active = (board_size * board_size) // 2
    in_dim = num_active * 4
    out_dim = num_active * 16
    try:
        state_dict = torch.load(resolved_path, map_location="cpu")
        if "transformer" in name_low or any(k.startswith("encoder.layers.") for k in state_dict.keys()):
            model: torch.nn.Module = QuantumBoardTransformerNet(
                in_dim=in_dim, hidden_dim=64, out_dim=out_dim, nhead=4, num_layers=2
            )
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        first_w = state_dict.get("policy_backbone.0.weight", state_dict.get("backbone.0.weight", state_dict.get("network.0.weight")))
        h_dim = int(first_w.shape[0]) if first_w is not None else 128
        is_decoupled = any(k.startswith("policy_backbone.") for k in state_dict.keys())
        is_wdl = (
            "value_head.weight" in state_dict
            and int(state_dict["value_head.weight"].shape[0]) == 3
        )
        if is_decoupled:
            model = MLPNet(
                in_dim=in_dim,
                hidden_dims=[h_dim, h_dim, h_dim],
                out_dim=out_dim,
                decoupled=True,
                wdl=is_wdl,
            )
        else:
            model = MLPNet(in_dim=in_dim, hidden_dim=h_dim, out_dim=out_dim, wdl=is_wdl)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception:
        return None


class InProcessTiqTaqToeAgent(SparringAgent):
    """
    Runs our `ChanceQMCTSPlayer` in-process for ultra-fast TiqTaqToe sparring.
    Modes:
      - `"chance_mcts"`: Chance-Node Gumbel/PUCT MCTS with neural priors + Many-Worlds solver.
      - `"pure_mcts"`: Chance-Node MCTS using pure exact Many-Worlds + tactical priors (`model=None`).
      - `"random"`: Uniform random legal move baseline.
    """

    def __init__(
        self,
        name: str,
        mode: str = "chance_mcts",
        model_path: Optional[str] = "auto",
        num_searches: int = 96,
        time_limit_ms: float = 800.0,
        use_d4_symmetry: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name)
        mode_norm = mode.strip().lower()
        if mode_norm not in ("chance_mcts", "pure_mcts", "random"):
            raise ValueError(f"Unsupported InProcessTiqTaqToeAgent mode: {mode}")
        self.mode = mode_norm
        self.model_path = model_path
        self.num_searches = int(num_searches)
        self.time_limit_ms = float(time_limit_ms)
        self.use_d4_symmetry = bool(use_d4_symmetry)
        self.rng = random.Random(seed)

        self.model = (
            _load_tiqtaqtoe_model(model_path)
            if self.mode == "chance_mcts"
            else None
        )
        self.player = (
            ChanceQMCTSPlayer(
                model=self.model,
                num_searches=self.num_searches,
                time_limit_ms=self.time_limit_ms,
                use_d4_symmetry=self.use_d4_symmetry,
            )
            if self.mode in ("chance_mcts", "pure_mcts")
            else None
        )

    def _compute_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Any:
        if self.mode == "random" or self.player is None:
            return self.rng.choice(legal_moves)
        return self.player.get_move(env=env, available_moves=legal_moves, temp=0.0)


def _log_move(agent: str, env: Any, payload: Dict[str, Any], legal_moves: List[Any], move: Any,
              info: Dict[str, Any]) -> None:
    """T33_GAME_LOG=<file>: one JSON line per Cheqqers move with the position (flat native state), the legal moves, the
    move and the agent's own estimate, for analysing lost games (track 33). Unset: nothing is written."""
    path = os.getenv("T33_GAME_LOG", "")
    if not path:
        return
    try:
        from search import native_cheqqers

        ints, dbl = native_cheqqers.NativeEnv.from_env(env).export()
        rec = {"agent": agent, "pid": os.getpid(), "move_count": (payload or {}).get("moveCount"),
               "turn": int(getattr(env, "current_player", 0) or 0), "legal": [list(m) for m in legal_moves],
               "move": list(move) if move is not None else None, "state": [ints, dbl], **info}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as err:  # logging must never change a move
        print(f"T33_GAME_LOG failed: {err}", flush=True)


class InProcessCheqqersAgent(SparringAgent):
    """
    Runs our `ExpectiCheqqersPlayer` in-process for Quantum Cheqqers (6x6, quantumness=2).
    Modes:
      - `"expecti_neural"`: Iterative-Deepening Expecti-AlphaBeta with Star1 pruning + Neural leaf blend.
      - `"expecti_heuristic"`: Iterative-Deepening Expecti-AlphaBeta with exact quantum heuristic evaluation.
      - `"random"`: Uniform random legal move baseline.
    """

    def __init__(
        self,
        name: str,
        mode: str = "expecti_heuristic",
        max_depth: int = 4,
        model_path: Optional[str] = "auto",
        time_limit_ms: float = 800.0,
        nn_blend: float = 0.25,
        seed: Optional[int] = None,
        eval_weights: Optional[EvalWeights] = None,
        opening_book: Optional[Any] = None,
        quiescence: bool = False,
        max_capture_extension: int = 4,
        quantum_plies: Optional[int] = None,
        engine: str = "python",
        value_net: Optional[str] = None,
        policy_model: Optional[str] = None,
        policy_topk: int = 0,
        mcts_model: Optional[str] = None,
        mcts_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(name=name)
        # policy_model: the policy-value transformer as the player (search/cheqqers_policy_player.py): argmax of its
        # policy over the legal moves, no search (track 33)
        self.policy_model = policy_model or ""
        # policy_topk > 0 (with engine="cpp"): the C++ search plays, restricted at the root to the policy's k best
        # moves; 0: the policy's argmax alone
        self.policy_topk = int(policy_topk)
        # mcts_model: MCTS with chance nodes guided by this policy-value network (search/gpu_mcts_cheqqers.py), on
        # CUDA if available, for time_limit_ms a move (track 33)
        self.mcts_model = mcts_model or ""
        # mcts_params: batch (leaves per round), c_puct, fpu (spec keys mb=, cpuct=, fpu=); defaults = the server's
        self.mcts_params = dict(mcts_params or {})
        if self.mcts_model:
            # load and warm up now (CUDA initialisation, graph capture): before any game clock, as server.py's
            # warm-up runs before the public port opens; a first move must not pay it
            from search.gpu_mcts_cheqqers import get_mcts_player

            get_mcts_player(self.mcts_model, **self.mcts_params)
        if self.policy_topk > 0 and (not self.policy_model or engine != "cpp"):
            raise ValueError("topk needs pv=<model.pt> and cpp=1")
        mode_norm = mode.strip().lower()
        if mode_norm not in ("expecti_neural", "expecti_heuristic", "random"):
            raise ValueError(f"Unsupported InProcessCheqqersAgent mode: {mode}")
        # engine="cpp": the same search with the same options in C++ (search/native_cheqqers.py), as server.py plays
        if engine not in ("python", "cpp"):
            raise ValueError(f"engine must be 'python' or 'cpp', not {engine!r}")
        if engine == "cpp" and mode_norm == "expecti_neural":
            raise ValueError("the native search has no value network: engine='cpp' needs expecti_heuristic")
        # value_net: a value_mlp.json at the C++ search's leaves instead of the linear evaluation (cpp only)
        if value_net and engine != "cpp":
            raise ValueError("value_net needs engine='cpp'")
        self.value_net = value_net or ""
        self.engine = engine
        self.last_depth: Optional[int] = None  # depth the last search finished (both engines)
        self.native_errors = 0  # native searches that raised and answered with their last finished iteration
        self.mode = mode_norm
        self.max_depth = int(max_depth)
        self.model_path = model_path
        self.time_limit_ms = float(time_limit_ms)
        self.nn_blend = float(nn_blend)
        self.rng = random.Random(seed)

        self.model = (
            _load_cheqqers_model(model_path)
            if self.mode == "expecti_neural"
            else None
        )
        self.player = (
            ExpectiCheqqersPlayer(
                model=self.model,
                max_depth=self.max_depth,
                time_limit_ms=self.time_limit_ms,
                nn_blend=self.nn_blend if self.model is not None else 0.0,
                eval_weights=eval_weights,
                opening_book=opening_book,
                quiescence=quiescence,
                max_capture_extension=max_capture_extension,
                quantum_plies=quantum_plies,
            )
            if self.mode in ("expecti_neural", "expecti_heuristic")
            else None
        )

    def _compute_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Any:
        if self.mcts_model:
            from search.gpu_mcts_cheqqers import get_mcts_player

            res = get_mcts_player(self.mcts_model, **self.mcts_params).search(env, legal_moves,
                                                                               time_ms=self.time_limit_ms)
            self.last_depth = res.max_depth
            _log_move(self.name, env, payload, legal_moves, res.move,
                      {"value": res.root_value, "evals": res.evals, "ms": round(res.elapsed_ms, 1)})
            log = os.getenv("T33_MCTS_LOG", "")
            if log:
                with open(log, "a") as fh:
                    fh.write(json.dumps({"agent": self.name, "evals": res.evals, "sims": res.sims,
                                         "ms": round(res.elapsed_ms, 1), "moves": len(legal_moves)}) + "\n")
            return res.move
        if self.policy_model and self.policy_topk <= 0:
            from search.cheqqers_policy_player import get_player

            self.last_depth = None
            return get_player(self.policy_model).choose(env, legal_moves)
        if self.policy_model and len(legal_moves) > self.policy_topk:
            from search.cheqqers_policy_player import get_player

            logits, _ = get_player(self.policy_model).evaluate([env], [legal_moves])
            order = sorted(range(len(legal_moves)), key=lambda i: (-logits[0][i], i))
            legal_moves = [legal_moves[i] for i in sorted(order[: self.policy_topk])]
        if self.mode == "random" or self.player is None:
            return self.rng.choice(legal_moves)
        if self.engine == "cpp":
            from search import native_cheqqers

            p = self.player
            result = native_cheqqers.best_move(
                env.clone(), p.time_limit_ms, p.max_depth, available_moves=legal_moves,
                eval_weights=p.eval_weights, quiescence=p.quiescence, quantum_plies=p.quantum_plies,
                max_capture_extension=p.max_capture_extension, opening_book=p.opening_book,
                value_net=self.value_net)  # "" = none: an agent spec never picks up CHEQQERS_VALUE_NET
            self.last_depth = result.last_depth
            if result.error:
                self.native_errors += 1
            _log_move(self.name, env, payload, legal_moves, result.move,
                      {"score": result.last_score, "depth": result.last_depth})
            return result.move
        mv = self.player.get_move(env=env, available_moves=legal_moves)
        # The Python player returns a forced move before resetting its depth: no search, no depth to report
        self.last_depth = getattr(self.player, "last_depth", None) if len(legal_moves) > 1 else None
        arr = cheqqers_move_to_array(mv)
        return arr if arr is not None else mv


# ============================================================================
# 3. Series Result & Blind-Spot / Disagreement Harvester Data Structures
# ============================================================================

@dataclass
class SeriesResult:
    game: str
    agent_a: str
    agent_b: str
    total_games: int
    wins_a: int
    draws: int
    losses_a: int
    score_a: float
    score_pct_a: float
    elo_diff_a: float
    elo_ci95_low: float
    elo_ci95_high: float
    sprt: Dict[str, Any]
    telemetry_a: Dict[str, Any]
    telemetry_b: Dict[str, Any]
    harvested_cases: int
    hard_cases_path: str
    p1_record_a: Dict[str, int] = field(default_factory=dict)
    p2_record_a: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _append_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> int:
    if not records:
        return 0
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


# ============================================================================
# 4. Head-to-Head Series Runners (TiqTaqToe Q=3 & Cheqqers 6x6 Q=2)
# ============================================================================

def run_tiqtaqtoe_series(
    agent_a: SparringAgent,
    agent_b: SparringAgent,
    num_pairs: int = 10,
    quantumness: int = 3,
    decoherence: int = 0,
    max_moves: int = 60,
    harvest_losses: bool = True,
    harvest_disagreements: bool = False,
    sparring_partner: Optional[SparringAgent] = None,
    hard_cases_path: Union[str, Path] = "results/sparring_hard_cases.jsonl",
    seed: int = 42,
    num_games: Optional[int] = None,
    start_with_a_as_p1: bool = True,
) -> SeriesResult:
    """
    Referees a color-balanced Quantum TiqTaqToe series (`num_pairs` games with `agent_a` as X
    and `num_pairs` games with `agent_a` as O, or `num_games` alternating games when specified,
    e.g., 5-game challenger gate) using `OfficialTiqTaqToeEnv(quantumness=quantumness)`.
    """
    agent_a.reset_telemetry()
    agent_b.reset_telemetry()

    wins_a = 0
    draws = 0
    losses_a = 0
    p1_rec = {"W": 0, "D": 0, "L": 0}
    p2_rec = {"W": 0, "D": 0, "L": 0}
    harvested_records: List[Dict[str, Any]] = []

    total_games = int(num_games) if num_games is not None else 2 * int(num_pairs)
    series_id = f"ttt_q{quantumness}_{int(time.time())}"

    for game_idx in range(total_games):
        a_is_x = (game_idx % 2 == 0) if start_with_a_as_p1 else (game_idx % 2 == 1)
        x_agent = agent_a if a_is_x else agent_b
        o_agent = agent_b if a_is_x else agent_a

        # round_number=1 starts with X
        env = OfficialTiqTaqToeEnv(quantumness=quantumness, decoherence=decoherence, round_number=1)
        np_rng = np.random.default_rng(seed + game_idx * 997)

        trajectory: List[Dict[str, Any]] = []
        ply = 0

        while not env.done and ply < max_moves:
            active_turn = env.current_turn  # 'X' or 'O'
            active_agent = x_agent if active_turn == "X" else o_agent
            other_agent = o_agent if active_turn == "X" else x_agent

            payload = env.to_tournament_payload()
            legal_moves = payload["legalMoves"]
            if not legal_moves:
                break

            chosen_move, latency_ms, was_fallback = active_agent.select_move(
                env=env, payload=payload, legal_moves=legal_moves
            )

            partner_move = None
            disagreed = False
            if harvest_disagreements:
                ref_agent = sparring_partner if sparring_partner is not None else other_agent
                ref_move, _, _ = ref_agent.select_move(
                    env=env.clone(), payload=payload, legal_moves=legal_moves
                )
                partner_move = ref_move
                if canonicalize_ttt_move(chosen_move) != canonicalize_ttt_move(ref_move):
                    disagreed = True

            step_record = {
                "game": "tiqtaqtoe",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "match_id": f"{series_id}_g{game_idx}",
                "game_index": game_idx,
                "ply": ply,
                "active_player": active_turn,
                "agent_name": active_agent.name,
                "opponent_name": other_agent.name,
                "chosen_move": chosen_move,
                "partner_move": partner_move,
                "disagreed": disagreed,
                "latency_ms": round(latency_ms, 3),
                "was_fallback": was_fallback,
                "tournament_payload": payload,
                "game_state_object": {
                    "quantumness": env.quantumness,
                    "current_turn": env.current_turn,
                    "measured_squares": list(env.measured_squares),
                    "empty_squares": list(env.empty_squares),
                    "many_worlds": env.compute_many_worlds(),
                },
            }
            trajectory.append(step_record)

            env.apply_move(chosen_move, project=True, rng=np_rng)
            ply += 1

        # Determine winner
        # TicTacResult: X=1, O=2, BOTH=3, DRAW=4, UNFINISHED=0
        w_code = env.winner_code
        if w_code == TicTacResult.X:
            winner_str = "X"
        elif w_code == TicTacResult.O:
            winner_str = "O"
        elif w_code == TicTacResult.BOTH:
            winner_str = "BOTH"
        else:
            winner_str = "DRAW"

        if winner_str == "X":
            if a_is_x:
                wins_a += 1
                p1_rec["W"] += 1
            else:
                losses_a += 1
                p2_rec["L"] += 1
        elif winner_str == "O":
            if not a_is_x:
                wins_a += 1
                p2_rec["W"] += 1
            else:
                losses_a += 1
                p1_rec["L"] += 1
        else:
            draws += 1
            if a_is_x:
                p1_rec["D"] += 1
            else:
                p2_rec["D"] += 1

        # Harvest losses and/or move disagreements
        for step_rec in trajectory:
            act_p = step_rec["active_player"]
            if winner_str in ("DRAW", "BOTH"):
                outcome_val = 0.0
            elif winner_str == act_p:
                outcome_val = 1.0
            else:
                outcome_val = -1.0

            should_harvest = False
            reasons: List[str] = []
            if harvest_losses and outcome_val < 0.0:
                should_harvest = True
                reasons.append("agent_loss")
            if harvest_disagreements and step_rec["disagreed"]:
                should_harvest = True
                reasons.append("move_disagreement")

            if should_harvest:
                harvested_records.append(
                    {
                        **step_rec,
                        "harvest_reason": "+".join(reasons),
                        "final_winner": winner_str,
                        "outcome_for_active_player": outcome_val,
                    }
                )

    saved_count = _append_jsonl(hard_cases_path, harvested_records)
    score_a = wins_a + 0.5 * draws
    score_pct_a = (score_a / float(total_games) * 100.0) if total_games > 0 else 0.0
    elo_diff, elo_low, elo_high = compute_elo_diff_and_ci(wins_a, draws, losses_a)
    sprt_info = compute_sprt_llr(wins_a, draws, losses_a)

    return SeriesResult(
        game="tiqtaqtoe",
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        total_games=total_games,
        wins_a=wins_a,
        draws=draws,
        losses_a=losses_a,
        score_a=score_a,
        score_pct_a=score_pct_a,
        elo_diff_a=elo_diff,
        elo_ci95_low=elo_low,
        elo_ci95_high=elo_high,
        sprt=sprt_info,
        telemetry_a=agent_a.get_telemetry(),
        telemetry_b=agent_b.get_telemetry(),
        harvested_cases=saved_count,
        hard_cases_path=str(hard_cases_path),
        p1_record_a=p1_rec,
        p2_record_a=p2_rec,
    )


def run_cheqqers_series(
    agent_a: SparringAgent,
    agent_b: SparringAgent,
    num_pairs: int = 10,
    board_size: int = 6,
    start_rows: int = 2,
    game_type: int = 2,
    max_moves: int = 160,
    harvest_losses: bool = True,
    harvest_disagreements: bool = False,
    sparring_partner: Optional[SparringAgent] = None,
    hard_cases_path: Union[str, Path] = "results/sparring_hard_cases.jsonl",
    seed: int = 42,
    num_games: Optional[int] = None,
    start_with_a_as_p1: bool = True,
    opening_plies: int = 0,
    out_path: Optional[Union[str, Path]] = None,
    pair_start: int = 0,
) -> SeriesResult:
    """
    `pair_start` plays pairs `pair_start .. pair_start + num_pairs - 1` of the series this seed defines: each pair
    has the seed, opening and colours it has in the whole series, so processes given disjoint ranges play disjoint
    parts of one series and their `out_path` files merge (`read_series_files` refuses overlapping ones).

    With `opening_plies > 0` every pair of games starts from the same random opening of that many plies, played
    once with each colour, and both games meet the same chance stream after it: what differs inside a pair is
    the agents, not the luck. With `out_path` one JSON line per finished game is appended there (pair, opening,
    colours, winner, plies, slowest move and fallbacks of each agent), so a stopped series keeps what it played.

    Referees a color-balanced Quantum Cheqqers series (`num_pairs` games with `agent_a` as White
    and `num_pairs` games with `agent_a` as Black, or `num_games` alternating games when specified,
    e.g., 5-game challenger gate) using
    `OfficialCheqqersEnv(board_size=board_size, start_rows=start_rows, game_type=game_type)`.
    """
    agent_a.reset_telemetry()
    agent_b.reset_telemetry()

    wins_a = 0
    draws = 0
    losses_a = 0
    p1_rec = {"W": 0, "D": 0, "L": 0}
    p2_rec = {"W": 0, "D": 0, "L": 0}
    harvested_records: List[Dict[str, Any]] = []

    total_games = int(num_games) if num_games is not None else 2 * int(num_pairs)
    series_id = f"cheq_s{board_size}_q{game_type}_{int(time.time())}"
    native_errors_seen = (getattr(agent_a, "native_errors", 0), getattr(agent_b, "native_errors", 0))

    for game_idx in range(2 * int(pair_start), 2 * int(pair_start) + total_games):
        a_is_white = (game_idx % 2 == 0) if start_with_a_as_p1 else (game_idx % 2 == 1)
        white_agent = agent_a if a_is_white else agent_b
        black_agent = agent_b if a_is_white else agent_a

        pair_idx = game_idx // 2
        # Without openings every game has its own seed, as before. With them the two games of a pair share one.
        game_seed = (seed + pair_idx * 1009) if opening_plies > 0 else (seed + game_idx * 1009)
        random.seed(game_seed)
        env = OfficialCheqqersEnv(
            board_size=board_size,
            start_rows=start_rows,
            game_type=game_type,
        )
        opening: List[List[int]] = []
        if opening_plies > 0:
            opening_rng = random.Random(game_seed * 7919 + 17)
            for _ in range(int(opening_plies)):
                opening_legal = env.to_tournament_payload()["legalMoves"]
                if env.done or not opening_legal:
                    break
                opening_move = opening_rng.choice(opening_legal)
                env.step(opening_move, return_state=False)
                opening.append([int(x) for x in opening_move])
            random.seed(game_seed + 1)  # the same chance stream for both games of the pair from here on

        trajectory: List[Dict[str, Any]] = []
        ply = 0

        while not env.done and ply < max_moves:
            active_color = "white" if env.turn == PieceColor.WHITE else "black"
            active_agent = white_agent if active_color == "white" else black_agent
            other_agent = black_agent if active_color == "white" else white_agent

            payload = env.to_tournament_payload()
            legal_moves = payload["legalMoves"]
            if not legal_moves:
                break

            chosen_move, latency_ms, was_fallback = active_agent.select_move(
                env=env, payload=payload, legal_moves=legal_moves
            )
            depth_done = getattr(active_agent, "last_depth", None)

            partner_move = None
            disagreed = False
            if harvest_disagreements:
                ref_agent = sparring_partner if sparring_partner is not None else other_agent
                ref_move, _, _ = ref_agent.select_move(
                    env=env.clone(), payload=payload, legal_moves=legal_moves
                )
                partner_move = ref_move
                if canonicalize_cheqqers_move(chosen_move) != canonicalize_cheqqers_move(ref_move):
                    disagreed = True

            step_record = {
                "game": "cheqqers",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "match_id": f"{series_id}_g{game_idx}",
                "game_index": game_idx,
                "ply": ply,
                "active_player": active_color,
                "agent_name": active_agent.name,
                "opponent_name": other_agent.name,
                "chosen_move": chosen_move,
                "partner_move": partner_move,
                "disagreed": disagreed,
                "latency_ms": round(latency_ms, 3),
                "was_fallback": was_fallback,
                "depth": depth_done,
                "tournament_payload": payload,
                "game_state_object": env.to_game_state_object(skip_moves=True),
            }
            trajectory.append(step_record)

            env.step(chosen_move, return_state=False)
            ply += 1

        # Determine winner
        if env.state == GameState.WHITE_WON:
            winner_str = "white"
        elif env.state == GameState.BLACK_WON:
            winner_str = "black"
        elif env.state == GameState.DRAW:
            winner_str = "draw"
        else:
            # Max moves reached: adjudicate by official terminal state or material/draw
            winner_str = "draw"

        if winner_str == "white":
            if a_is_white:
                wins_a += 1
                p1_rec["W"] += 1
            else:
                losses_a += 1
                p2_rec["L"] += 1
        elif winner_str == "black":
            if not a_is_white:
                wins_a += 1
                p2_rec["W"] += 1
            else:
                losses_a += 1
                p1_rec["L"] += 1
        else:
            draws += 1
            if a_is_white:
                p1_rec["D"] += 1
            else:
                p2_rec["D"] += 1

        if out_path is not None:
            a_colour = "white" if a_is_white else "black"
            steps_a = [s for s in trajectory if s["active_player"] == a_colour]
            steps_b = [s for s in trajectory if s["active_player"] != a_colour]

            def mean_depth(steps: List[Dict[str, Any]]) -> Optional[float]:
                depths = [s["depth"] for s in steps if s["depth"]]
                return round(sum(depths) / len(depths), 3) if depths else None

            native_errors_now = (getattr(agent_a, "native_errors", 0), getattr(agent_b, "native_errors", 0))
            _append_jsonl(out_path, [{
                "series_id": series_id,
                "seed": seed,
                "game_index": game_idx,
                "pair": pair_idx,
                "opening": opening,
                "agent_a": agent_a.name,
                "agent_b": agent_b.name,
                "a_is_white": a_is_white,
                "winner": winner_str,
                "result_for_a": 0.5 if winner_str == "draw" else (1.0 if winner_str == a_colour else 0.0),
                "plies": ply,
                "slowest_ms_a": max((s["latency_ms"] for s in steps_a), default=0.0),
                "slowest_ms_b": max((s["latency_ms"] for s in steps_b), default=0.0),
                "fallbacks_a": sum(1 for s in steps_a if s["was_fallback"]),
                "fallbacks_b": sum(1 for s in steps_b if s["was_fallback"]),
                "mean_depth_a": mean_depth(steps_a),
                "mean_depth_b": mean_depth(steps_b),
                "native_errors_a": native_errors_now[0] - native_errors_seen[0],
                "native_errors_b": native_errors_now[1] - native_errors_seen[1],
            }])
            native_errors_seen = native_errors_now
            if os.getenv("T33_GAME_LOG", ""):  # the game's end, for the move log (_log_move; track 33)
                with open(os.environ["T33_GAME_LOG"], "a") as fh:
                    fh.write(json.dumps({"game_end": True, "pid": os.getpid(), "seed": seed, "pair": pair_idx,
                                         "game_index": game_idx, "agent_a": agent_a.name, "agent_b": agent_b.name,
                                         "a_is_white": a_is_white, "winner": winner_str, "plies": ply}) + "\n")

        # Harvest losses and/or disagreements
        for step_rec in trajectory:
            act_p = step_rec["active_player"]
            if winner_str == "draw":
                outcome_val = 0.0
            elif winner_str == act_p:
                outcome_val = 1.0
            else:
                outcome_val = -1.0

            should_harvest = False
            reasons: List[str] = []
            if harvest_losses and outcome_val < 0.0:
                should_harvest = True
                reasons.append("agent_loss")
            if harvest_disagreements and step_rec["disagreed"]:
                should_harvest = True
                reasons.append("move_disagreement")

            if should_harvest:
                harvested_records.append(
                    {
                        **step_rec,
                        "harvest_reason": "+".join(reasons),
                        "final_winner": winner_str,
                        "outcome_for_active_player": outcome_val,
                    }
                )

    saved_count = _append_jsonl(hard_cases_path, harvested_records)
    score_a = wins_a + 0.5 * draws
    score_pct_a = (score_a / float(total_games) * 100.0) if total_games > 0 else 0.0
    elo_diff, elo_low, elo_high = compute_elo_diff_and_ci(wins_a, draws, losses_a)
    sprt_info = compute_sprt_llr(wins_a, draws, losses_a)

    return SeriesResult(
        game="cheqqers",
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        total_games=total_games,
        wins_a=wins_a,
        draws=draws,
        losses_a=losses_a,
        score_a=score_a,
        score_pct_a=score_pct_a,
        elo_diff_a=elo_diff,
        elo_ci95_low=elo_low,
        elo_ci95_high=elo_high,
        sprt=sprt_info,
        telemetry_a=agent_a.get_telemetry(),
        telemetry_b=agent_b.get_telemetry(),
        harvested_cases=saved_count,
        hard_cases_path=str(hard_cases_path),
        p1_record_a=p1_rec,
        p2_record_a=p2_rec,
    )


# ============================================================================
# 5. Multi-Agent Round-Robin Runner & Standings Table
# ============================================================================

def run_round_robin(
    agents: Sequence[SparringAgent],
    game: str = "tiqtaqtoe",
    num_pairs: int = 5,
    harvest_losses: bool = True,
    harvest_disagreements: bool = False,
    hard_cases_path: Union[str, Path] = "results/sparring_hard_cases.jsonl",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs a full all-play-all Round-Robin tournament across `agents` (with color-balanced
    pairings for every unordered pair `(i, j)`), computes overall W/D/L, points,
    anchored BayesElo/Bradley-Terry ratings (mean 1500.0), and pairwise SPRT LLR matrix.
    """
    if len(agents) < 2:
        raise ValueError("run_round_robin requires at least 2 agents.")

    game_norm = game.strip().lower()
    standings: Dict[str, Dict[str, Any]] = {
        ag.name: {
            "agent": ag.name,
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0.0,
            "score_pct": 0.0,
            "elo": 1500.0,
            "elo_ci95": 0.0,
            "timeouts": 0,
            "illegal_moves": 0,
            "p95_ms": 0.0,
        }
        for ag in agents
    }

    pairwise_results: List[Dict[str, Any]] = []
    all_latencies: Dict[str, List[float]] = {ag.name: [] for ag in agents}

    pair_idx = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            ag_a = agents[i]
            ag_b = agents[j]
            pair_seed = seed + pair_idx * 10007
            pair_idx += 1

            if game_norm in ("tiqtaqtoe", "ttt"):
                res = run_tiqtaqtoe_series(
                    agent_a=ag_a,
                    agent_b=ag_b,
                    num_pairs=num_pairs,
                    harvest_losses=harvest_losses,
                    harvest_disagreements=harvest_disagreements,
                    hard_cases_path=hard_cases_path,
                    seed=pair_seed,
                )
            elif game_norm in ("cheqqers", "checkers"):
                res = run_cheqqers_series(
                    agent_a=ag_a,
                    agent_b=ag_b,
                    num_pairs=num_pairs,
                    harvest_losses=harvest_losses,
                    harvest_disagreements=harvest_disagreements,
                    hard_cases_path=hard_cases_path,
                    seed=pair_seed,
                )
            else:
                raise ValueError(f"Unsupported game for round-robin: {game}")

            pairwise_results.append(res.to_dict())

            # Accumulate A
            st_a = standings[ag_a.name]
            st_a["games"] += res.total_games
            st_a["wins"] += res.wins_a
            st_a["draws"] += res.draws
            st_a["losses"] += res.losses_a
            st_a["points"] += res.score_a
            st_a["timeouts"] += res.telemetry_a["timeouts"]
            st_a["illegal_moves"] += res.telemetry_a["illegal_moves"]
            all_latencies[ag_a.name].extend(ag_a.latencies_ms)

            # Accumulate B
            st_b = standings[ag_b.name]
            st_b["games"] += res.total_games
            st_b["wins"] += res.losses_a
            st_b["draws"] += res.draws
            st_b["losses"] += res.wins_a
            st_b["points"] += (res.total_games - res.score_a)
            st_b["timeouts"] += res.telemetry_b["timeouts"]
            st_b["illegal_moves"] += res.telemetry_b["illegal_moves"]
            all_latencies[ag_b.name].extend(ag_b.latencies_ms)

    # Fit iterative Bradley-Terry / logistic Elo ratings centered at 1500.0
    names = [ag.name for ag in agents]
    elos = {name: 1500.0 for name in names}
    for _ in range(200):
        for name in names:
            st = standings[name]
            if st["games"] == 0:
                continue
            actual_score = st["points"]
            expected_score = 0.0
            for match in pairwise_results:
                if match["agent_a"] == name:
                    opp = match["agent_b"]
                    n_g = match["total_games"]
                    expected_score += n_g / (1.0 + math.pow(10.0, (elos[opp] - elos[name]) / 400.0))
                elif match["agent_b"] == name:
                    opp = match["agent_a"]
                    n_g = match["total_games"]
                    expected_score += n_g / (1.0 + math.pow(10.0, (elos[opp] - elos[name]) / 400.0))
            # Regularized Newton / Minorization-Maximization update step
            reg_actual = actual_score + 0.5
            reg_expected = expected_score + 0.5
            elos[name] += 40.0 * math.log10(reg_actual / reg_expected)
        mean_elo = sum(elos.values()) / float(len(names))
        for name in names:
            elos[name] -= (mean_elo - 1500.0)

    for name in names:
        st = standings[name]
        g = st["games"]
        st["score_pct"] = round((st["points"] / float(g) * 100.0) if g > 0 else 0.0, 2)
        st["elo"] = round(elos[name], 1)
        _, ci_l, ci_h = compute_elo_diff_and_ci(st["wins"], st["draws"], st["losses"])
        st["elo_ci95"] = round(0.5 * abs(ci_h - ci_l), 1)
        lats = all_latencies[name]
        st["p95_ms"] = round(float(np.percentile(lats, 95)), 2) if lats else 0.0

    leaderboard = sorted(
        standings.values(),
        key=lambda row: (row["points"], row["elo"]),
        reverse=True,
    )

    return {
        "game": game_norm,
        "num_pairs_per_matchup": int(num_pairs),
        "leaderboard": leaderboard,
        "pairwise_series": pairwise_results,
        "hard_cases_path": str(hard_cases_path),
    }


# ============================================================================
# 6. Teacher Distillation Solver (`solve_hard_cases`)
# ============================================================================

def solve_hard_cases(
    hard_cases_path: Union[str, Path] = "results/sparring_hard_cases.jsonl",
    output_path: Union[str, Path] = "results/sparring_distilled_targets.jsonl",
    ttt_teacher_searches: int = 600,
    ttt_teacher_time_ms: float = 2500.0,
    cheqqers_teacher_depth: int = 6,
    cheqqers_teacher_time_ms: float = 2500.0,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Loads harvested blind-spot / disagreement positions from `hard_cases_path`,
    reconstructs the exact quantum state via `OfficialTiqTaqToeEnv.from_tournament_payload`
    or `OfficialCheqqersEnv.from_tournament_json`, and executes deep teacher search:
      - TiqTaqToe: `ChanceQMCTSPlayer(num_searches=ttt_teacher_searches)` (default 600)
      - Cheqqers: `ExpectiCheqqersPlayer(max_depth=cheqqers_teacher_depth)` (default 6)
    Writes supervised policy + WDL value distillation targets to `output_path`.
    """
    in_path = Path(hard_cases_path)
    if not in_path.is_absolute():
        in_path = _PROJECT_ROOT / in_path
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path

    if not in_path.exists():
        return {
            "solved_count": 0,
            "tiqtaqtoe_solved": 0,
            "cheqqers_solved": 0,
            "teacher_corrected_moves": 0,
            "input_path": str(in_path),
            "output_path": str(out_path),
        }

    ttt_model = _load_tiqtaqtoe_model("auto")
    ttt_teacher = ChanceQMCTSPlayer(
        model=ttt_model,
        num_searches=int(ttt_teacher_searches),
        time_limit_ms=float(ttt_teacher_time_ms),
        use_d4_symmetry=True,
    )

    cheq_model = _load_cheqqers_model("auto")
    cheq_teacher = ExpectiCheqqersPlayer(
        model=cheq_model,
        max_depth=int(cheqqers_teacher_depth),
        time_limit_ms=float(cheqqers_teacher_time_ms),
        nn_blend=0.25 if cheq_model is not None else 0.0,
    )

    raw_lines: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                raw_lines.append(json.loads(line_s))
            except json.JSONDecodeError:
                continue

    if max_cases is not None and max_cases > 0 and len(raw_lines) > max_cases:
        ttt_lines = [r for r in raw_lines if str(r.get("game", "")).lower() == "tiqtaqtoe"]
        cheq_lines = [r for r in raw_lines if str(r.get("game", "")).lower() == "cheqqers"]
        if ttt_lines and cheq_lines:
            half = max(1, int(max_cases) // 2)
            raw_lines = ttt_lines[:half] + cheq_lines[: int(max_cases) - half]
        else:
            raw_lines = raw_lines[: int(max_cases)]

    distilled_records: List[Dict[str, Any]] = []
    ttt_count = 0
    cheq_count = 0
    corrected_count = 0

    for rec in raw_lines:
        game = str(rec.get("game", "")).lower()
        payload = rec.get("tournament_payload")
        if not isinstance(payload, dict):
            continue

        orig_move = rec.get("chosen_move")

        if game == "tiqtaqtoe":
            env = OfficialTiqTaqToeEnv.from_tournament_payload(payload)
            legal_moves = payload.get("legalMoves") or env.get_available_moves()
            if not legal_moves:
                continue

            teacher_move, policy_46 = ttt_teacher.get_move(
                env=env,
                available_moves=legal_moves,
                return_probs=True,
                temp=0.0,
            )
            mw = env.compute_many_worlds()
            is_x = (env.current_turn == "X")
            mw_val = float(mw["X"] - mw["O"]) if is_x else float(mw["O"] - mw["X"])

            corrected = (
                orig_move is not None
                and canonicalize_ttt_move(teacher_move) != canonicalize_ttt_move(orig_move)
            )
            if corrected:
                corrected_count += 1
            ttt_count += 1

            distilled_records.append(
                {
                    "game": "tiqtaqtoe",
                    "match_id": rec.get("match_id"),
                    "ply": rec.get("ply"),
                    "active_player": env.current_turn,
                    "harvest_reason": rec.get("harvest_reason"),
                    "student_move": orig_move,
                    "teacher_best_move": teacher_move,
                    "teacher_action_46": int(move_to_action_46(teacher_move)),
                    "teacher_corrected": bool(corrected),
                    "teacher_searches": int(ttt_teacher_searches),
                    "teacher_policy_46": [round(float(x), 6) for x in policy_46.tolist()],
                    "state_111": [round(float(x), 6) for x in env.get_state().tolist()],
                    "many_worlds_value": round(mw_val, 6),
                    "game_outcome_z": float(rec.get("outcome_for_active_player", 0.0)),
                    "tournament_payload": payload,
                }
            )

        elif game == "cheqqers":
            env = OfficialCheqqersEnv.from_tournament_json(payload)
            legal_moves = payload.get("legalMoves") or [
                env.move_to_array(m) for m in env.get_possible_moves()
            ]
            if not legal_moves:
                continue

            teacher_raw = cheq_teacher.get_move(env=env, available_moves=legal_moves)
            teacher_move = cheqqers_move_to_array(teacher_raw) or teacher_raw
            heur_val = float(env.evaluate_state(env.turn)) / 1000.0

            try:
                move_dict = env.resolve_move_dict(teacher_move)
                teacher_action_288 = int(env.move_to_canonical_action(move_dict))
            except Exception:
                teacher_action_288 = 0

            corrected = (
                orig_move is not None
                and canonicalize_cheqqers_move(teacher_move) != canonicalize_cheqqers_move(orig_move)
            )
            if corrected:
                corrected_count += 1
            cheq_count += 1

            distilled_records.append(
                {
                    "game": "cheqqers",
                    "match_id": rec.get("match_id"),
                    "ply": rec.get("ply"),
                    "active_player": "white" if env.turn == PieceColor.WHITE else "black",
                    "harvest_reason": rec.get("harvest_reason"),
                    "student_move": orig_move,
                    "teacher_best_move": teacher_move,
                    "teacher_action_288": teacher_action_288,
                    "teacher_corrected": bool(corrected),
                    "teacher_depth": int(cheqqers_teacher_depth),
                    "state_72": [round(float(x), 6) for x in env.get_state_72().tolist()],
                    "teacher_value_norm": round(heur_val, 6),
                    "game_outcome_z": float(rec.get("outcome_for_active_player", 0.0)),
                    "tournament_payload": payload,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d_rec in distilled_records:
            f.write(json.dumps(d_rec) + "\n")

    return {
        "solved_count": len(distilled_records),
        "tiqtaqtoe_solved": ttt_count,
        "cheqqers_solved": cheq_count,
        "teacher_corrected_moves": corrected_count,
        "input_path": str(in_path),
        "output_path": str(out_path),
    }


# ============================================================================
# 7. 24/7 Adversarial Population League Manager (Stefano's Exact Rules)
#    - Model = Weights + Structure (Architecture) + Strategy (`LeagueModelSpec`)
#    - 5-Game Challenger Gate: Admitted iff beats/draws >= 75% (3/4) of pool
#      OR scores in the Top 25% (`pool_percentile >= 75.0`)
#    - Max 25 Models -> Drop Bottom 5 Culling Rule (`25 -> 20`)
# ============================================================================

import re


@dataclass
class LeagueModelSpec:
    """
    Represents a complete League Entrant defined by (Weights + Structure + Strategy),
    together with its live Elo rating, W/D/L record, and pairwise head-to-head ledger.
    """
    model_id: str
    game: str  # "tiqtaqtoe" or "cheqqers"
    architecture: str  # e.g. "MLP-128x3-Decoupled-WDL", "KAN-128x3-Spline", "SNN-LIF-20step", "MLP-Checkers-72x128x288", "PureSearch", "RemoteHTTP"
    strategy: str  # e.g. "TrackedChanceQMCTS(s=120)", "ChanceQMCTS(s=80)", "LegacyQMCTS(s=60)", "DirectPolicyGreedy", "ExpectiAlphaBeta(d=4,w=0.35)"
    weights_path: Optional[str] = None
    endpoint_url: Optional[str] = None
    elo: float = 1500.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games_played: int = 0
    head_to_head: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> float:
        return float(self.wins) + 0.5 * float(self.draws)

    @property
    def score_pct(self) -> float:
        if self.games_played <= 0:
            return 0.0
        return (self.points / float(self.games_played)) * 100.0

    def opponents_beaten_ratio(self) -> float:
        """
        Fraction of opponents played in `head_to_head` against whom this model
        achieved >= 50% score (won or drew the head-to-head series).
        """
        played_opps = [
            rec for rec in self.head_to_head.values() if int(rec.get("games", 0)) > 0
        ]
        if not played_opps:
            return 0.0
        beaten = sum(
            1
            for rec in played_opps
            if float(rec.get("points", 0.0)) >= 0.5 * float(rec.get("games", 1))
        )
        return float(beaten) / float(len(played_opps))

    def composite_rating(self) -> float:
        """
        Composite League Rating combining anchored Bradley-Terry Elo, overall score %,
        and pairwise opponents-beaten ratio for deterministic ranking and bottom-5 culling.
        """
        return (
            float(self.elo)
            + 2.0 * self.score_pct
            + 100.0 * self.opponents_beaten_ratio()
        )

    def record_series(self, opponent_id: str, wins: int, draws: int, losses: int) -> None:
        g = int(wins) + int(draws) + int(losses)
        pts = float(wins) + 0.5 * float(draws)
        self.wins += int(wins)
        self.draws += int(draws)
        self.losses += int(losses)
        self.games_played += g

        h2h = self.head_to_head.setdefault(
            opponent_id,
            {"W": 0, "D": 0, "L": 0, "points": 0.0, "games": 0},
        )
        h2h["W"] = int(h2h.get("W", 0)) + int(wins)
        h2h["D"] = int(h2h.get("D", 0)) + int(draws)
        h2h["L"] = int(h2h.get("L", 0)) + int(losses)
        h2h["points"] = float(h2h.get("points", 0.0)) + pts
        h2h["games"] = int(h2h.get("games", 0)) + g

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = round(self.points, 2)
        d["score_pct"] = round(self.score_pct, 2)
        d["opponents_beaten_ratio"] = round(self.opponents_beaten_ratio(), 4)
        d["composite_rating"] = round(self.composite_rating(), 2)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeagueModelSpec":
        allowed = {
            "model_id",
            "game",
            "architecture",
            "strategy",
            "weights_path",
            "endpoint_url",
            "elo",
            "wins",
            "draws",
            "losses",
            "games_played",
            "head_to_head",
            "created_at",
            "metadata",
        }
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)


def _parse_strategy_param(strategy: str, key: str, default: float) -> float:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*([0-9.]+)", strategy)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


class LeagueConfiguredAgent(SparringAgent):
    """
    Instantiates a runnable `SparringAgent` directly from a `LeagueModelSpec`
    (combining `weights_path` + `architecture` + `strategy`, or `endpoint_url`).
    """

    def __init__(
        self,
        spec: LeagueModelSpec,
        time_limit_ms: float = 600.0,
        seed: Optional[int] = None,
    ):
        super().__init__(name=spec.model_id)
        self.spec = spec
        self.time_limit_ms = float(time_limit_ms)
        self.rng = random.Random(seed)
        self._delegate_http: Optional[HttpEndpointAgent] = None
        self._ttt_player: Optional[ChanceQMCTSPlayer] = None
        self._cheq_player: Optional[ExpectiCheqqersPlayer] = None
        self._model: Optional[torch.nn.Module] = None

        strat_low = spec.strategy.strip().lower()
        arch_low = spec.architecture.strip().lower()

        if spec.endpoint_url or arch_low == "remotehttp" or strat_low.startswith("http"):
            url = spec.endpoint_url or spec.strategy
            self._delegate_http = HttpEndpointAgent(
                name=spec.model_id,
                url=url,
                timeout_ms=max(1000, int(self.time_limit_ms * 2)),
            )
            return

        if spec.game.strip().lower() in ("tiqtaqtoe", "ttt"):
            if "pure" not in arch_low and "random" not in arch_low and "random" not in strat_low:
                self._model = _load_tiqtaqtoe_model(spec.weights_path or "auto")

            searches = int(_parse_strategy_param(spec.strategy, "s", 80.0))
            if "tracked" in strat_low:
                self._ttt_player = ChanceQMCTSPlayer(
                    model=self._model,
                    num_searches=searches,
                    time_limit_ms=self.time_limit_ms,
                    use_gumbel=True,
                    use_d4_symmetry=True,
                    many_worlds_blend=0.75,
                    endgame_empty_threshold=4,
                )
            elif "legacy" in strat_low:
                self._ttt_player = ChanceQMCTSPlayer(
                    model=self._model,
                    num_searches=searches,
                    time_limit_ms=self.time_limit_ms,
                    use_gumbel=False,
                    use_d4_symmetry=False,
                    many_worlds_blend=0.25,
                    endgame_empty_threshold=2,
                )
            elif "pure" in strat_low or ("mcts" in strat_low and self._model is None):
                self._ttt_player = ChanceQMCTSPlayer(
                    model=None,
                    num_searches=searches,
                    time_limit_ms=self.time_limit_ms,
                    use_gumbel=True,
                    use_d4_symmetry=True,
                    many_worlds_blend=0.85,
                    endgame_empty_threshold=4,
                )
            elif "chance" in strat_low or "mcts" in strat_low:
                self._ttt_player = ChanceQMCTSPlayer(
                    model=self._model,
                    num_searches=searches,
                    time_limit_ms=self.time_limit_ms,
                    use_gumbel=True,
                    use_d4_symmetry=True,
                    many_worlds_blend=0.65,
                    endgame_empty_threshold=3,
                )
            elif "direct" in strat_low or "greedy" in strat_low:
                # 1-simulation + D4 equivariant policy prior evaluation
                self._ttt_player = ChanceQMCTSPlayer(
                    model=self._model,
                    num_searches=1,
                    time_limit_ms=self.time_limit_ms,
                    use_gumbel=False,
                    use_d4_symmetry=True,
                    many_worlds_blend=0.50,
                    endgame_empty_threshold=3,
                )
        else:
            if "pure" not in arch_low and "random" not in arch_low and "random" not in strat_low:
                self._model = _load_cheqqers_model(spec.weights_path or "auto")

            depth = int(_parse_strategy_param(spec.strategy, "d", 3.0))
            blend = float(_parse_strategy_param(spec.strategy, "w", 0.35))
            sims = int(_parse_strategy_param(spec.strategy, "s", 96.0))
            if "mcts" in strat_low or "gumbel" in strat_low:
                self._cheq_player = GumbelQMCTSCheqqersPlayer(
                    model=None if "pure" in strat_low else self._model,
                    num_simulations=sims,
                    time_limit_ms=self.time_limit_ms,
                    nn_blend=0.0 if "pure" in strat_low or self._model is None else blend,
                    rollout_depth=max(1, int(_parse_strategy_param(spec.strategy, "d", 1.0))),
                )
            elif "pure" in strat_low:
                self._cheq_player = ExpectiCheqqersPlayer(
                    model=None,
                    max_depth=depth,
                    time_limit_ms=self.time_limit_ms,
                    nn_blend=0.0,
                )
            elif "direct" in strat_low or "greedy" in strat_low:
                self._cheq_player = ExpectiCheqqersPlayer(
                    model=self._model,
                    max_depth=1,
                    time_limit_ms=self.time_limit_ms,
                    nn_blend=0.65 if self._model is not None else 0.0,
                )
            elif "expecti" in strat_low:
                self._cheq_player = ExpectiCheqqersPlayer(
                    model=self._model,
                    max_depth=depth,
                    time_limit_ms=self.time_limit_ms,
                    nn_blend=blend if self._model is not None else 0.0,
                )

    def _compute_move(
        self,
        env: Any,
        payload: Dict[str, Any],
        legal_moves: List[Any],
    ) -> Any:
        if self._delegate_http is not None:
            return self._delegate_http._compute_move(env, payload, legal_moves)
        if self._ttt_player is not None:
            return self._ttt_player.get_move(env=env, available_moves=legal_moves, temp=0.0)
        if self._cheq_player is not None:
            mv = self._cheq_player.get_move(env=env, available_moves=legal_moves)
            arr = cheqqers_move_to_array(mv)
            return arr if arr is not None else mv
        return self.rng.choice(legal_moves)


def build_default_seed_specs(game: str) -> List[LeagueModelSpec]:
    """
    Auto-discovers `.pth` checkpoints in `results/` and constructs a diverse initial seed pool
    of `(weights + architecture + strategy)` models for `tiqtaqtoe` or `cheqqers`.
    """
    g_norm = game.strip().lower()
    specs: List[LeagueModelSpec] = []

    def _rel_if_exists(rel_path: str) -> Optional[str]:
        p = _PROJECT_ROOT / rel_path
        return rel_path if p.exists() else None

    if g_norm in ("tiqtaqtoe", "ttt"):
        candidates = [
            (
                "Team5_OfficialQ3_P1_MLP128x3_TrackedQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "TrackedChanceQMCTS(s=120)",
                "results/q_net_official_tiqtaqtoe_q3_p1_best.pth",
            ),
            (
                "Team5_OfficialQ3_P2_MLP128x3_TrackedQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "TrackedChanceQMCTS(s=120)",
                "results/q_net_official_tiqtaqtoe_q3_p2_best.pth",
            ),
            (
                "Team5_JointBest_P1_MLP128x3_ChanceQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "ChanceQMCTS(s=80)",
                "results/q_net_quantum_entanglement_tictactoe_mlp_p1_best_joint.pth",
            ),
            (
                "Team5_JointBest_P2_MLP128x3_ChanceQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "ChanceQMCTS(s=80)",
                "results/q_net_quantum_entanglement_tictactoe_mlp_p2_best_joint.pth",
            ),
            (
                "Team5_CalibratedJoint_P1_MLP128x3_ChanceQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "ChanceQMCTS(s=80)",
                "results/q_net_quantum_entanglement_tictactoe_mlp_p1_calibrated_joint.pth",
            ),
            (
                "Team5_GlobalBest_P1_MLP128x3_LegacyQMCTS",
                "MLP-128x3-Decoupled-WDL",
                "LegacyQMCTS(s=60)",
                "results/q_net_quantum_entanglement_tictactoe_mlp_p1_best_global.pth",
            ),
            (
                "Team5_KAN128x3_BestFrozen_ChanceQMCTS",
                "KAN-128x3-Spline",
                "ChanceQMCTS(s=80)",
                "results/q_net_quantum_entanglement_tictactoe_kan_best_frozen.pth",
            ),
            (
                "Team5_KAN128x3_CalibratedJoint_TrackedQMCTS",
                "KAN-128x3-Spline",
                "TrackedChanceQMCTS(s=100)",
                "results/q_net_quantum_entanglement_tictactoe_kan_calibrated_joint.pth",
            ),
            (
                "Team5_SNN20_Calibrated_ChanceQMCTS",
                "SNN-LIF-20step",
                "ChanceQMCTS(s=60)",
                "results/q_net_quantum_entanglement_snn_calibrated.pth",
            ),
            (
                "Team5_SNN20_Genetic_LegacyQMCTS",
                "SNN-LIF-20step",
                "LegacyQMCTS(s=50)",
                "results/q_net_quantum_entanglement_snn_genetic.pth",
            ),
            (
                "Team5_DistilledRollouts50_DirectPolicy",
                "MLP-128x3-Decoupled-WDL",
                "DirectPolicyGreedy",
                "results/q_net_quantum_entanglement_mlp_distilled_rollouts50.pth",
            ),
            (
                "Team5_Transformer_ExactMDP_IL_TrackedQMCTS",
                "Transformer-13tok-NonMarkov",
                "TrackedChanceQMCTS(s=100)",
                "results/q_net_official_tiqtaqtoe_q3_transformer_il.pth",
            ),
            (
                "Team5_PureManyWorlds_TrackedQMCTS",
                "PureSearch",
                "PureChanceQMCTS(s=96)",
                None,
            ),
        ]
        for mod_id, arch, strat, w_rel in candidates:
            if w_rel is None or _rel_if_exists(w_rel) is not None:
                specs.append(
                    LeagueModelSpec(
                        model_id=mod_id,
                        game="tiqtaqtoe",
                        architecture=arch,
                        strategy=strat,
                        weights_path=w_rel,
                    )
                )
    else:
        candidates = [
            (
                "Team5_OfficialQ2_MLP_GumbelQMCTS112",
                "MLP-Checkers-72x128x288",
                "GumbelChanceQMCTS(s=112,w=0.35,d=2)",
                "results/q_net_official_cheqqers_size6_q2_best.pth",
            ),
            (
                "Team5_OfficialQ2_MLP_ExpectiD4",
                "MLP-Checkers-72x128x288",
                "ExpectiAlphaBeta(d=4,w=0.35)",
                "results/q_net_official_cheqqers_size6_q2_best.pth",
            ),
            (
                "Team5_OfficialQ2_MLP_ExpectiD3",
                "MLP-Checkers-72x128x288",
                "ExpectiAlphaBeta(d=3,w=0.35)",
                "results/q_net_official_cheqqers_size6_q2_best.pth",
            ),
            (
                "Team5_ExactMDP_IL_MLP_ExpectiD4",
                "MLP-Checkers-72x128x288",
                "ExpectiAlphaBeta(d=4,w=0.35)",
                "results/q_net_official_cheqqers_size6_q2_exact_mdp_mlp.pth",
            ),
            (
                "Team5_Transformer_ExactMDP_IL_ExpectiD4",
                "Transformer-22tok-NonMarkov",
                "ExpectiAlphaBeta(d=4,w=0.35)",
                "results/q_net_official_cheqqers_size6_q2_transformer_il.pth",
            ),
            (
                "Team5_Transformer_ExactMDP_IL_ExpectiD3",
                "Transformer-22tok-NonMarkov",
                "ExpectiAlphaBeta(d=3,w=0.35)",
                "results/q_net_official_cheqqers_size6_q2_transformer_il.pth",
            ),
            (
                "Team5_PureExpecti_Depth4",
                "PureSearch",
                "PureExpecti(d=4)",
                None,
            ),
            (
                "Team5_PureExpecti_Depth3",
                "PureSearch",
                "PureExpecti(d=3)",
                None,
            ),
        ]
        for mod_id, arch, strat, w_rel in candidates:
            if w_rel is None or _rel_if_exists(w_rel) is not None:
                specs.append(
                    LeagueModelSpec(
                        model_id=mod_id,
                        game="cheqqers",
                        architecture=arch,
                        strategy=strat,
                        weights_path=w_rel,
                    )
                )

    return specs


class PopulationLeagueManager:
    """
    Implements Stefano's 24/7 Leiden Live Adversarial League:
      1. Model = Weights + Structure (`architecture`) + Strategy (`strategy`).
      2. Continuous all-play-all matches across the active pool (`run_league_cycle`).
      3. 5-Game Challenger Gate (`evaluate_challenger`):
         A new model is admitted into the league iff across 5 games against EACH active
         model in the pool it either:
           - Wins/draws the 5-game series against >= 75% (3/4) of all active models
             (`opponents_beaten_ratio >= 0.75`), OR
           - Scores in the Top 25% of the pool (`pool_percentile >= 75.0`).
      4. Max 25 Models -> Drop Bottom 5 Culling Rule (`enforce_pool_cap_and_cull`):
         Whenever the active pool reaches `max_models` (25), the bottom `drop_bottom_k` (5)
         models by composite rating are culled and archived, leaving the top 20 models.
    """

    def __init__(
        self,
        game: str = "tiqtaqtoe",
        pool_path: Optional[Union[str, Path]] = None,
        max_models: int = 25,
        drop_bottom_k: int = 5,
        hard_cases_path: Union[str, Path] = "results/sparring_hard_cases.jsonl",
        auto_seed_if_empty: bool = True,
        time_limit_ms: float = 500.0,
    ):
        self.game = "tiqtaqtoe" if game.strip().lower() in ("tiqtaqtoe", "ttt") else "cheqqers"
        self.max_models = int(max_models)
        self.drop_bottom_k = int(drop_bottom_k)
        self.hard_cases_path = Path(hard_cases_path)
        if not self.hard_cases_path.is_absolute():
            self.hard_cases_path = _PROJECT_ROOT / self.hard_cases_path
        self.time_limit_ms = float(time_limit_ms)

        if pool_path is None:
            self.pool_path = _PROJECT_ROOT / "results" / f"leiden_league_pool_{self.game}.json"
        else:
            p = Path(pool_path)
            self.pool_path = p if p.is_absolute() else (_PROJECT_ROOT / p)

        self.active_models: List[LeagueModelSpec] = []
        self.archived_dropped_models: List[Dict[str, Any]] = []
        self.rejected_challengers: List[Dict[str, Any]] = []
        self.cycles_completed: int = 0

        self.load_or_initialize(auto_seed_if_empty=auto_seed_if_empty)

    # ------------------------------------------------------------------
    # Persistence & Standings
    # ------------------------------------------------------------------

    def load_or_initialize(self, auto_seed_if_empty: bool = True) -> None:
        if self.pool_path.exists():
            try:
                data = json.loads(self.pool_path.read_text(encoding="utf-8"))
                self.active_models = [
                    LeagueModelSpec.from_dict(m) for m in data.get("active_models", [])
                ]
                self.archived_dropped_models = list(data.get("archived_dropped_models", []))
                self.rejected_challengers = list(data.get("rejected_challengers", []))
                self.cycles_completed = int(data.get("cycles_completed", 0))
            except Exception:
                self.active_models = []

        if not self.active_models and auto_seed_if_empty:
            self.active_models = build_default_seed_specs(self.game)
            self.save()

    def save(self) -> None:
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.recompute_bradley_terry_elos()
        self.active_models.sort(key=lambda m: m.composite_rating(), reverse=True)
        payload = {
            "game": self.game,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rules": {
                "max_models": self.max_models,
                "drop_bottom_k": self.drop_bottom_k,
                "challenger_games_per_opponent": 5,
                "challenger_min_win_fraction": 0.75,
                "challenger_top_quartile_pct": 75.0,
            },
            "cycles_completed": self.cycles_completed,
            "active_count": len(self.active_models),
            "active_models": [m.to_dict() for m in self.active_models],
            "archived_dropped_models": self.archived_dropped_models,
            "rejected_challengers": self.rejected_challengers[-50:],
        }
        self.pool_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def recompute_bradley_terry_elos(self) -> None:
        """
        Recomputes anchored Bradley-Terry / BayesElo ratings (mean 1500.0) across all
        `active_models` from their pairwise `head_to_head` records.
        """
        if not self.active_models:
            return
        id_to_model = {m.model_id: m for m in self.active_models}
        elos = {m.model_id: float(m.elo) for m in self.active_models}

        for _ in range(150):
            for m_id, mod in id_to_model.items():
                actual_pts = 0.0
                expected_pts = 0.0
                total_g = 0
                for opp_id, rec in mod.head_to_head.items():
                    g = int(rec.get("games", 0))
                    if g <= 0:
                        continue
                    opp_elo = elos.get(opp_id, 1500.0)
                    actual_pts += float(rec.get("points", 0.0))
                    expected_pts += float(g) / (
                        1.0 + math.pow(10.0, (opp_elo - elos[m_id]) / 400.0)
                    )
                    total_g += g
                if total_g > 0:
                    elos[m_id] += 32.0 * math.log10((actual_pts + 0.5) / (expected_pts + 0.5))

            mean_elo = sum(elos.values()) / float(len(elos))
            for m_id in elos:
                elos[m_id] = round(elos[m_id] - (mean_elo - 1500.0), 2)

        for mod in self.active_models:
            mod.elo = elos[mod.model_id]

    def get_status_summary(self) -> Dict[str, Any]:
        self.recompute_bradley_terry_elos()
        self.active_models.sort(key=lambda m: m.composite_rating(), reverse=True)
        return {
            "game": self.game,
            "pool_path": str(self.pool_path),
            "active_count": len(self.active_models),
            "max_models": self.max_models,
            "drop_bottom_k": self.drop_bottom_k,
            "cycles_completed": self.cycles_completed,
            "standings": [
                {
                    "rank": idx + 1,
                    "model_id": m.model_id,
                    "architecture": m.architecture,
                    "strategy": m.strategy,
                    "weights_path": m.weights_path,
                    "endpoint_url": m.endpoint_url,
                    "elo": round(m.elo, 1),
                    "games": m.games_played,
                    "W": m.wins,
                    "D": m.draws,
                    "L": m.losses,
                    "score_pct": round(m.score_pct, 2),
                    "opponents_beaten_ratio": round(m.opponents_beaten_ratio(), 3),
                    "composite_rating": round(m.composite_rating(), 2),
                }
                for idx, m in enumerate(self.active_models)
            ],
            "archived_dropped_count": len(self.archived_dropped_models),
            "archived_dropped_models": self.archived_dropped_models,
        }

    # ------------------------------------------------------------------
    # Match Execution Helper Between Two LeagueModelSpecs
    # ------------------------------------------------------------------

    def _play_match_between_specs(
        self,
        spec_a: LeagueModelSpec,
        spec_b: LeagueModelSpec,
        num_games: int = 5,
        start_with_a_as_p1: bool = True,
        harvest_losses: bool = True,
        harvest_disagreements: bool = False,
        seed: int = 42,
    ) -> SeriesResult:
        ag_a = LeagueConfiguredAgent(
            spec_a, time_limit_ms=self.time_limit_ms, seed=seed
        )
        ag_b = LeagueConfiguredAgent(
            spec_b, time_limit_ms=self.time_limit_ms, seed=seed + 7919
        )
        if self.game == "tiqtaqtoe":
            return run_tiqtaqtoe_series(
                agent_a=ag_a,
                agent_b=ag_b,
                num_pairs=max(1, num_games // 2),
                num_games=num_games,
                start_with_a_as_p1=start_with_a_as_p1,
                harvest_losses=harvest_losses,
                harvest_disagreements=harvest_disagreements,
                hard_cases_path=self.hard_cases_path,
                seed=seed,
            )
        else:
            return run_cheqqers_series(
                agent_a=ag_a,
                agent_b=ag_b,
                num_pairs=max(1, num_games // 2),
                num_games=num_games,
                start_with_a_as_p1=start_with_a_as_p1,
                harvest_losses=harvest_losses,
                harvest_disagreements=harvest_disagreements,
                hard_cases_path=self.hard_cases_path,
                seed=seed,
            )

    # ------------------------------------------------------------------
    # Rule 1: 5-Game Challenger Admission Gate (>= 75% / Top 25%)
    # ------------------------------------------------------------------

    def evaluate_challenger(
        self,
        challenger_spec: LeagueModelSpec,
        games_per_opponent: int = 5,
        min_win_fraction: float = 0.75,
        auto_admit: bool = True,
        harvest_losses: bool = True,
        harvest_disagreements: bool = False,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Evaluates `challenger_spec` by playing `games_per_opponent` (default 5) games
        against EVERY active model in the league pool.

        Stefano's Admission Gate:
          - Plays 5 games against each of the N active models in the pool.
          - Computes:
              1. `opponents_beaten_ratio`: fraction of pool models against which the
                 challenger scores >= 50% (i.e. >= 2.5 / 5.0 points, winning/drawing the
                 5-game series), plus `strict_opponents_beaten_ratio` (> 2.5 / 5.0).
              2. `overall_score_pct`: total points / (5 * N) * 100.
              3. `pool_percentile`: percentile rank (0..100) of the challenger's average
                 score compared against the active pool's standings.
          - Admitted iff:
              `opponents_beaten_ratio >= min_win_fraction` (default 0.75, i.e. 3/4 of pool)
              OR `pool_percentile >= (min_win_fraction * 100.0)` (Top 25% of pool).
          - If admitted and `len(active_models) >= max_models` (default 25), automatically
            triggers `enforce_pool_cap_and_cull(max_models, drop_bottom_k)` to drop the
            bottom 5 models!
        """
        if not self.active_models:
            if auto_admit:
                self.active_models.append(challenger_spec)
                self.save()
            return {
                "challenger_id": challenger_spec.model_id,
                "admitted": True,
                "reason": "Pool was empty; admitted as initial seed.",
                "opponents_tested": 0,
            }

        per_opponent_results: List[Dict[str, Any]] = []
        opponents_beaten = 0
        strict_opponents_beaten = 0
        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_games = 0

        # Play 5 games against each active model in the pool
        for idx, defender in enumerate(self.active_models):
            if defender.model_id == challenger_spec.model_id:
                continue

            # Alternate starting color on game 1 across opponents so overall P1/P2 count is 50/50
            start_as_p1 = (idx % 2 == 0)
            match_seed = seed + idx * 4099
            series = self._play_match_between_specs(
                spec_a=challenger_spec,
                spec_b=defender,
                num_games=int(games_per_opponent),
                start_with_a_as_p1=start_as_p1,
                harvest_losses=harvest_losses,
                harvest_disagreements=harvest_disagreements,
                seed=match_seed,
            )

            # Record on challenger
            challenger_spec.record_series(
                opponent_id=defender.model_id,
                wins=series.wins_a,
                draws=series.draws,
                losses=series.losses_a,
            )

            total_wins += series.wins_a
            total_draws += series.draws
            total_losses += series.losses_a
            total_games += series.total_games

            beaten_or_drawn = series.score_a >= (0.5 * series.total_games)
            strictly_beaten = series.score_a > (0.5 * series.total_games)
            if beaten_or_drawn:
                opponents_beaten += 1
            if strictly_beaten:
                strict_opponents_beaten += 1

            per_opponent_results.append(
                {
                    "opponent_id": defender.model_id,
                    "opponent_architecture": defender.architecture,
                    "opponent_strategy": defender.strategy,
                    "games": series.total_games,
                    "W": series.wins_a,
                    "D": series.draws,
                    "L": series.losses_a,
                    "score": series.score_a,
                    "score_pct": round(series.score_pct_a, 2),
                    "beaten_or_drawn": beaten_or_drawn,
                    "strictly_beaten": strictly_beaten,
                    " series_elo_diff": round(series.elo_diff_a, 1),
                    " defender_record_tuple": (
                        defender.model_id,
                        series.losses_a,
                        series.draws,
                        series.wins_a,
                    ),
                }
            )

        num_opps = len(per_opponent_results)
        opponents_beaten_ratio = (
            float(opponents_beaten) / float(num_opps) if num_opps > 0 else 1.0
        )
        strict_beaten_ratio = (
            float(strict_opponents_beaten) / float(num_opps) if num_opps > 0 else 1.0
        )
        overall_points = float(total_wins) + 0.5 * float(total_draws)
        overall_score_pct = (
            (overall_points / float(total_games)) * 100.0 if total_games > 0 else 0.0
        )

        # Compute challenger's percentile rank against current active pool score_pct distribution
        pool_scores = [
            (m.score_pct if m.games_played > 0 else 50.0) for m in self.active_models
        ]
        beaten_in_standings = sum(1 for s in pool_scores if overall_score_pct > s) + 0.5 * sum(
            1 for s in pool_scores if abs(overall_score_pct - s) <= 1e-6
        )
        pool_percentile = (
            (beaten_in_standings / float(len(pool_scores))) * 100.0
            if pool_scores
            else 100.0
        )

        target_pct = float(min_win_fraction) * 100.0
        passed_3_4_gate = opponents_beaten_ratio >= float(min_win_fraction)
        passed_top_25_pct_gate = (
            pool_percentile >= target_pct and overall_score_pct >= 55.0
        )
        admitted = bool(passed_3_4_gate or passed_top_25_pct_gate)

        cull_report: Optional[Dict[str, Any]] = None
        if admitted and auto_admit:
            # Commit defender head-to-head records now that challenger is admitted
            id_to_def = {m.model_id: m for m in self.active_models}
            for item in per_opponent_results:
                def_id, d_w, d_d, d_l = item[" defender_record_tuple"]
                if def_id in id_to_def:
                    id_to_def[def_id].record_series(
                        opponent_id=challenger_spec.model_id,
                        wins=d_w,
                        draws=d_d,
                        losses=d_l,
                    )

            # Replace if same model_id existed, otherwise append
            self.active_models = [
                m for m in self.active_models if m.model_id != challenger_spec.model_id
            ]
            self.active_models.append(challenger_spec)
            self.recompute_bradley_terry_elos()

            # Check Max 25 Models -> Drop Bottom 5 trigger!
            if len(self.active_models) >= self.max_models:
                cull_report = self.enforce_pool_cap_and_cull(
                    max_models=self.max_models,
                    drop_bottom_k=self.drop_bottom_k,
                    run_consolidation_sweep=False,
                )
            else:
                self.save()
        else:
            rejection_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model_id": challenger_spec.model_id,
                "architecture": challenger_spec.architecture,
                "strategy": challenger_spec.strategy,
                "weights_path": challenger_spec.weights_path,
                "opponents_beaten_ratio": round(opponents_beaten_ratio, 4),
                "overall_score_pct": round(overall_score_pct, 2),
                "pool_percentile": round(pool_percentile, 2),
            }
            self.rejected_challengers.append(rejection_entry)
            self.save()

        # Strip internal tuple key before returning report
        clean_results = [
            {k: v for k, v in r.items() if not k.startswith(" ")}
            for r in per_opponent_results
        ]

        return {
            "challenger_id": challenger_spec.model_id,
            "game": self.game,
            "architecture": challenger_spec.architecture,
            "strategy": challenger_spec.strategy,
            "weights_path": challenger_spec.weights_path,
            "games_per_opponent": int(games_per_opponent),
            "opponents_tested": num_opps,
            "total_games": total_games,
            "W": total_wins,
            "D": total_draws,
            "L": total_losses,
            "overall_score_pct": round(overall_score_pct, 2),
            "opponents_beaten": opponents_beaten,
            "opponents_beaten_ratio": round(opponents_beaten_ratio, 4),
            "strict_opponents_beaten_ratio": round(strict_beaten_ratio, 4),
            "pool_percentile": round(pool_percentile, 2),
            "min_win_fraction_required": float(min_win_fraction),
            "passed_3_of_4_opponents_gate": passed_3_4_gate,
            "passed_top_25_percent_gate": passed_top_25_pct_gate,
            "admitted": admitted,
            "active_pool_size_after": len(self.active_models),
            "cull_event": cull_report,
            "matchup_breakdown": clean_results,
        }

    # ------------------------------------------------------------------
    # Rule 2: Max 25 Models -> Drop Bottom 5 Culling Rule (`25 -> 20`)
    # ------------------------------------------------------------------

    def enforce_pool_cap_and_cull(
        self,
        max_models: Optional[int] = None,
        drop_bottom_k: Optional[int] = None,
        run_consolidation_sweep: bool = False,
        consolidation_games_per_pair: int = 2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Enforces Stefano's `Max 25 Models -> Drop Bottom 5` culling rule:
        Whenever `len(self.active_models) >= cap` (default 25):
          1. Optionally plays a fast consolidation sweep so any unplayed pairs have ratings.
          2. Recomputes Bradley-Terry / BayesElo ratings and sorts all models by
             `composite_rating()` (Elo + score % + head-to-head opponents beaten ratio).
          3. Drops the bottom `drop_k` (default 5) models, archiving them in
             `self.archived_dropped_models` with timestamp and rank, leaving `cap - drop_k`
             (e.g. 25 - 5 = 20) models in the active league!
        """
        cap = int(max_models) if max_models is not None else self.max_models
        drop_k = int(drop_bottom_k) if drop_bottom_k is not None else self.drop_bottom_k

        before_count = len(self.active_models)
        if before_count < cap:
            self.save()
            return {
                "culled": False,
                "before_count": before_count,
                "after_count": before_count,
                "max_models_trigger": cap,
                "dropped_models": [],
            }

        if run_consolidation_sweep:
            self.run_league_cycle(
                games_per_pair=consolidation_games_per_pair,
                only_unplayed_pairs=True,
                auto_cull=False,
                seed=seed,
            )

        self.recompute_bradley_terry_elos()
        self.active_models.sort(key=lambda m: m.composite_rating(), reverse=True)

        target_keep = max(1, before_count - drop_k)
        survivors = self.active_models[:target_keep]
        dropped = self.active_models[target_keep:]

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dropped_summaries: List[Dict[str, Any]] = []
        for idx, dm in enumerate(dropped):
            rank = target_keep + idx + 1
            entry = {
                "dropped_at": now_iso,
                "dropped_rank": rank,
                "pool_size_at_cull": before_count,
                "reason": f"Culled in bottom {drop_k} when pool reached {before_count} >= {cap} models",
                "model": dm.to_dict(),
            }
            self.archived_dropped_models.append(entry)
            dropped_summaries.append(
                {
                    "rank": rank,
                    "model_id": dm.model_id,
                    "architecture": dm.architecture,
                    "strategy": dm.strategy,
                    "elo": round(dm.elo, 1),
                    "score_pct": round(dm.score_pct, 2),
                    "composite_rating": round(dm.composite_rating(), 2),
                }
            )

        self.active_models = survivors
        self.save()

        return {
            "culled": True,
            "before_count": before_count,
            "after_count": len(self.active_models),
            "max_models_trigger": cap,
            "drop_bottom_k": drop_k,
            "dropped_models": dropped_summaries,
            "surviving_top_3": [m.model_id for m in self.active_models[:3]],
        }

    # ------------------------------------------------------------------
    # Rule 3: All-Play-All League Cycle & 24/7 Continuous Daemon
    # ------------------------------------------------------------------

    def run_league_cycle(
        self,
        games_per_pair: int = 2,
        only_unplayed_pairs: bool = False,
        harvest_losses: bool = True,
        harvest_disagreements: bool = False,
        auto_cull: bool = True,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Plays an all-play-all adversarial cycle across `self.active_models`, updating
        pairwise `head_to_head` records, W/D/L, and Bradley-Terry Elo ratings, and
        enforces the `Max 25 -> Drop Bottom 5` culling rule if `auto_cull=True`.
        """
        n = len(self.active_models)
        matches_played = 0
        games_played = 0

        for i in range(n):
            for j in range(i + 1, n):
                mod_a = self.active_models[i]
                mod_b = self.active_models[j]
                if only_unplayed_pairs:
                    prev_g = int(mod_a.head_to_head.get(mod_b.model_id, {}).get("games", 0))
                    if prev_g > 0:
                        continue

                pair_seed = (
                    seed
                    + self.cycles_completed * 100003
                    + (i * 131 + j * 997)
                )
                series = self._play_match_between_specs(
                    spec_a=mod_a,
                    spec_b=mod_b,
                    num_games=int(games_per_pair),
                    start_with_a_as_p1=((i + j + self.cycles_completed) % 2 == 0),
                    harvest_losses=harvest_losses,
                    harvest_disagreements=harvest_disagreements,
                    seed=pair_seed,
                )
                mod_a.record_series(
                    opponent_id=mod_b.model_id,
                    wins=series.wins_a,
                    draws=series.draws,
                    losses=series.losses_a,
                )
                mod_b.record_series(
                    opponent_id=mod_a.model_id,
                    wins=series.losses_a,
                    draws=series.draws,
                    losses=series.wins_a,
                )
                matches_played += 1
                games_played += series.total_games

        self.cycles_completed += 1
        self.recompute_bradley_terry_elos()

        cull_report = None
        if auto_cull and len(self.active_models) >= self.max_models:
            cull_report = self.enforce_pool_cap_and_cull(
                max_models=self.max_models,
                drop_bottom_k=self.drop_bottom_k,
                run_consolidation_sweep=False,
            )
        else:
            self.save()

        return {
            "game": self.game,
            "cycle_number": self.cycles_completed,
            "pairings_played": matches_played,
            "total_games_played": games_played,
            "cull_event": cull_report,
            "standings": self.get_status_summary()["standings"],
        }


# ============================================================================
# 8. Self-Test Verification Suite (`--self-test`)
# ============================================================================

class _EphemeralSparringHttpHandler(http.server.BaseHTTPRequestHandler):
    """Lightweight local HTTP `POST /move` server used by `--self-test` to verify `HttpEndpointAgent`."""

    ttt_agent = InProcessTiqTaqToeAgent(
        name="HTTP_TTT_Search",
        mode="pure_mcts",
        num_searches=24,
        time_limit_ms=300.0,
        seed=7,
    )
    cheq_agent = InProcessCheqqersAgent(
        name="HTTP_Cheq_Expecti",
        mode="expecti_heuristic",
        max_depth=2,
        time_limit_ms=300.0,
        seed=7,
    )

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Keep self-test console output clean

    def do_POST(self) -> None:
        if self.path != "/move":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        legal_moves = data.get("legalMoves", [])
        board = data.get("board", [])

        is_cheqqers = bool(board and isinstance(board[0], dict) and "piece" in board[0])
        if is_cheqqers:
            env = OfficialCheqqersEnv.from_tournament_json(data)
            mv, _, _ = self.cheq_agent.select_move(env, data, legal_moves)
        else:
            env = OfficialTiqTaqToeEnv.from_tournament_payload(data)
            mv, _, _ = self.ttt_agent.select_move(env, data, legal_moves)

        resp_bytes = json.dumps({"move": mv}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


def run_self_test() -> Dict[str, Any]:
    """
    Runs a deterministic end-to-end verification of:
      1. `InProcessTiqTaqToeAgent` + `HttpEndpointAgent` head-to-head series on `OfficialTiqTaqToeEnv(quantumness=3)`
      2. `InProcessCheqqersAgent` + `HttpEndpointAgent` head-to-head series on `OfficialCheqqersEnv(board_size=6, game_type=2)`
      3. Multi-agent `run_round_robin` leaderboard, Elo + 95% CI, and SPRT LLR computation
      4. Blind-spot & disagreement harvesting into `results/sparring_hard_cases.jsonl`
      5. `solve_hard_cases` teacher distillation into `results/sparring_distilled_targets.jsonl`
    """
    hard_cases_file = _PROJECT_ROOT / "results" / "sparring_hard_cases.jsonl"
    distilled_file = _PROJECT_ROOT / "results" / "sparring_distilled_targets.jsonl"
    if hard_cases_file.exists():
        hard_cases_file.unlink()

    # Start ephemeral HTTP server on an OS-assigned free port
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _EphemeralSparringHttpHandler)
    port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        http_agent = HttpEndpointAgent(
            name="Remote_Leiden_HTTP_Agent",
            url=f"http://127.0.0.1:{port}/move",
            timeout_ms=2000,
        )
        ttt_mcts = InProcessTiqTaqToeAgent(
            name="Team5_ChanceQMCTS",
            mode="chance_mcts",
            model_path="auto",
            num_searches=32,
            time_limit_ms=300.0,
            seed=101,
        )
        ttt_random = InProcessTiqTaqToeAgent(
            name="Baseline_Random_TTT",
            mode="random",
            seed=202,
        )

        # 1. TiqTaqToe Head-to-Head + HTTP Endpoint Verification
        ttt_series = run_tiqtaqtoe_series(
            agent_a=ttt_mcts,
            agent_b=http_agent,
            num_pairs=2,
            quantumness=3,
            harvest_losses=True,
            harvest_disagreements=True,
            sparring_partner=ttt_random,
            hard_cases_path=hard_cases_file,
            seed=42,
        )

        # 2. TiqTaqToe Mini Round-Robin
        ttt_rr = run_round_robin(
            agents=[ttt_mcts, http_agent, ttt_random],
            game="tiqtaqtoe",
            num_pairs=1,
            harvest_losses=True,
            harvest_disagreements=False,
            hard_cases_path=hard_cases_file,
            seed=123,
        )

        # 3. Cheqqers Head-to-Head + HTTP Endpoint Verification
        cheq_expecti = InProcessCheqqersAgent(
            name="Team5_ExpectiCheqqers",
            mode="expecti_heuristic",
            max_depth=2,
            time_limit_ms=300.0,
            seed=303,
        )
        cheq_random = InProcessCheqqersAgent(
            name="Baseline_Random_Cheqqers",
            mode="random",
            seed=404,
        )
        cheq_series = run_cheqqers_series(
            agent_a=cheq_expecti,
            agent_b=http_agent,
            num_pairs=1,
            board_size=6,
            start_rows=2,
            game_type=2,
            max_moves=80,
            harvest_losses=True,
            harvest_disagreements=True,
            sparring_partner=cheq_random,
            hard_cases_path=hard_cases_file,
            seed=99,
        )

        # 4. Solve harvested hard cases with Teacher Distillation
        distill_summary = solve_hard_cases(
            hard_cases_path=hard_cases_file,
            output_path=distilled_file,
            ttt_teacher_searches=48,
            ttt_teacher_time_ms=400.0,
            cheqqers_teacher_depth=3,
            cheqqers_teacher_time_ms=400.0,
            max_cases=12,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert ttt_series.telemetry_a["illegal_moves"] == 0, "TiqTaqToe Agent A produced illegal moves!"
    assert ttt_series.telemetry_b["illegal_moves"] == 0, "HTTP Agent produced illegal moves in TiqTaqToe!"
    assert cheq_series.telemetry_a["illegal_moves"] == 0, "Cheqqers Agent A produced illegal moves!"
    assert cheq_series.telemetry_b["illegal_moves"] == 0, "HTTP Agent produced illegal moves in Cheqqers!"
    assert distill_summary["solved_count"] > 0, "Harvester + solve_hard_cases produced 0 distilled targets!"

    # 5. Verify PopulationLeagueManager:
    #    (a) Seed production pools (`results/leiden_league_pool_tiqtaqtoe.json` & `cheqqers.json`)
    prod_ttt_league = PopulationLeagueManager(game="tiqtaqtoe", auto_seed_if_empty=True)
    prod_cheq_league = PopulationLeagueManager(game="cheqqers", auto_seed_if_empty=True)

    #    (b) Live 5-Game Challenger Gate (`games_per_opponent=5, min_win_fraction=0.75`)
    test_pool_path = _PROJECT_ROOT / "results" / "_self_test_league_pool_ttt.json"
    if test_pool_path.exists():
        test_pool_path.unlink()

    test_league = PopulationLeagueManager(
        game="tiqtaqtoe",
        pool_path=test_pool_path,
        max_models=25,
        drop_bottom_k=5,
        auto_seed_if_empty=False,
        time_limit_ms=150.0,
    )
    # Populate 4 baseline defenders so 5 games x 4 opponents = 20 games (3/4 = 75% threshold)
    test_league.active_models = [
        LeagueModelSpec(
            model_id="Pool_Defender_1_Random",
            game="tiqtaqtoe",
            architecture="RandomBaseline",
            strategy="RandomBaseline",
            elo=1350.0,
        ),
        LeagueModelSpec(
            model_id="Pool_Defender_2_Random",
            game="tiqtaqtoe",
            architecture="RandomBaseline",
            strategy="RandomBaseline",
            elo=1360.0,
        ),
        LeagueModelSpec(
            model_id="Pool_Defender_3_DirectPolicy",
            game="tiqtaqtoe",
            architecture="MLP-128x3-Decoupled-WDL",
            strategy="DirectPolicyGreedy",
            weights_path="results/q_net_official_tiqtaqtoe_q3_p1_best.pth",
            elo=1520.0,
        ),
        LeagueModelSpec(
            model_id="Pool_Defender_4_PureMCTS",
            game="tiqtaqtoe",
            architecture="PureSearch",
            strategy="PureChanceQMCTS(s=16)",
            elo=1570.0,
        ),
    ]
    test_league.save()

    # Strong challenger: TrackedChanceQMCTS should beat >= 3/4 (75%) of defenders across 5-game series
    strong_challenger = LeagueModelSpec(
        model_id="Team5_Challenger_TrackedQMCTS",
        game="tiqtaqtoe",
        architecture="MLP-128x3-Decoupled-WDL",
        strategy="TrackedChanceQMCTS(s=36)",
        weights_path="results/q_net_official_tiqtaqtoe_q3_p1_best.pth",
    )
    strong_gate_res = test_league.evaluate_challenger(
        challenger_spec=strong_challenger,
        games_per_opponent=5,
        min_win_fraction=0.75,
        auto_admit=True,
        harvest_losses=True,
        seed=777,
    )
    assert strong_gate_res["admitted"] is True, (
        f"Expected strong TrackedChanceQMCTS challenger to pass >=75% gate, got: {strong_gate_res}"
    )

    # Weak challenger: Random agent should fail the >= 75% gate against the upgraded 5-model pool
    weak_challenger = LeagueModelSpec(
        model_id="Weak_Random_Challenger",
        game="tiqtaqtoe",
        architecture="RandomBaseline",
        strategy="RandomBaseline",
    )
    weak_gate_res = test_league.evaluate_challenger(
        challenger_spec=weak_challenger,
        games_per_opponent=5,
        min_win_fraction=0.75,
        auto_admit=True,
        harvest_losses=False,
        seed=888,
    )
    assert weak_gate_res["admitted"] is False, (
        f"Expected weak random challenger to be rejected by >=75% gate, got: {weak_gate_res}"
    )

    #    (c) Verify `Max 25 Models -> Drop Bottom 5` Culling Rule (`25 -> 20`)
    while len(test_league.active_models) < 25:
        idx = len(test_league.active_models) + 1
        synthetic = LeagueModelSpec(
            model_id=f"Synthetic_Pool_Entrant_{idx:02d}",
            game="tiqtaqtoe",
            architecture="MLP-128x3-Decoupled-WDL",
            strategy=f"ChanceQMCTS(s={16 + idx})",
            weights_path="results/q_net_official_tiqtaqtoe_q3_p1_best.pth",
            elo=1400.0 + idx * 5.0,
            wins=idx,
            draws=2,
            losses=max(0, 25 - idx),
            games_played=27,
        )
        synthetic.head_to_head["Pool_Defender_1_Random"] = {
            "W": idx,
            "D": 2,
            "L": max(0, 25 - idx),
            "points": float(idx) + 1.0,
            "games": 27,
        }
        test_league.active_models.append(synthetic)

    assert len(test_league.active_models) == 25
    cull_res = test_league.enforce_pool_cap_and_cull(
        max_models=25,
        drop_bottom_k=5,
        run_consolidation_sweep=False,
    )
    assert cull_res["culled"] is True
    assert cull_res["before_count"] == 25
    assert cull_res["after_count"] == 20
    assert len(cull_res["dropped_models"]) == 5

    if test_pool_path.exists():
        test_pool_path.unlink()

    report = {
        "status": "SELF_TEST_PASSED",
        "tiqtaqtoe_h2h": ttt_series.to_dict(),
        "tiqtaqtoe_round_robin": ttt_rr["leaderboard"],
        "cheqqers_h2h": cheq_series.to_dict(),
        "distillation_summary": distill_summary,
        "population_league_verification": {
            "seeded_tiqtaqtoe_pool_size": len(prod_ttt_league.active_models),
            "seeded_cheqqers_pool_size": len(prod_cheq_league.active_models),
            "strong_challenger_5game_gate": {
                "challenger_id": strong_gate_res["challenger_id"],
                "games_per_opponent": strong_gate_res["games_per_opponent"],
                "opponents_beaten_ratio": strong_gate_res["opponents_beaten_ratio"],
                "overall_score_pct": strong_gate_res["overall_score_pct"],
                "pool_percentile": strong_gate_res["pool_percentile"],
                "admitted": strong_gate_res["admitted"],
            },
            "weak_challenger_5game_gate": {
                "challenger_id": weak_gate_res["challenger_id"],
                "opponents_beaten_ratio": weak_gate_res["opponents_beaten_ratio"],
                "overall_score_pct": weak_gate_res["overall_score_pct"],
                "admitted": weak_gate_res["admitted"],
            },
            "max25_drop_bottom5_cull": cull_res,
        },
    }
    print(json.dumps(report, indent=2))
    return report


# ============================================================================
# 9. CLI Entrypoint
# ============================================================================

def read_series_files(paths: Sequence[Union[str, Path]]) -> List[Dict[str, Any]]:
    """The per-game records of one or more series files, each record's pair made unique across files so that
    several runs can be read as one series. A record with a seed is keyed by (seed, pair): shards of one seed
    (`--pair-start`) merge, runs with different seeds stay apart; a game seen twice (overlapping shards, a file
    given twice) raises. A record without a seed is keyed by its file."""
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for index, path in enumerate(paths):
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if "seed" in record:
                        record["pair"] = f"s{record['seed']}:{record['pair']}"
                    else:
                        record["pair"] = f"f{index}:{record['pair']}"
                    key = (record["pair"], record.get("a_is_white"))
                    if key in seen:
                        raise ValueError(f"{path}: pair {record['pair']} with a_is_white={key[1]} read twice "
                                         f"(overlapping shards?)")
                    seen.add(key)
                    records.append(record)
    return records


def summarize_pairs(games: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A paired series read pair by pair. Both games of a pair start from one opening with colours swapped and meet
    the same chance stream, so their results are not independent: the unit is the pair. `score_a` is the mean
    pair score of agent A (0..1), `standard_error` its standard error over pairs, and the Elo figures are the
    logistic transform of the score and of score +- 1.96 standard errors (clipped away from 0 and 1). Pairs with
    a game missing are counted and left out. Input: the records `run_cheqqers_series(out_path=...)` writes.
    """
    by_pair: Dict[Any, List[Dict[str, Any]]] = {}
    for record in games:
        by_pair.setdefault(record["pair"], []).append(record)
    complete = [pair for pair in by_pair.values() if len(pair) == 2]
    scores = [(pair[0]["result_for_a"] + pair[1]["result_for_a"]) / 2.0 for pair in complete]
    n = len(scores)
    mean = sum(scores) / n if n else 0.5
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    standard_error = math.sqrt(variance / n) if n else 0.0

    def to_elo(score: float) -> float:
        clipped = min(max(score, 1e-3), 1.0 - 1e-3)
        return 0.0 if abs(clipped - 0.5) < 1e-12 else -400.0 * math.log10(1.0 / clipped - 1.0)

    counts = {"2-0": 0, "1-1": 0, "0-2": 0, "other": 0}
    for score in scores:
        counts["2-0" if score == 1.0 else "0-2" if score == 0.0 else "1-1" if score == 0.5 else "other"] += 1
    played = [g for pair in complete for g in pair]

    def mean_of(key: str) -> Optional[float]:
        values = [g[key] for g in played if g.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    return {
        "pairs": n,
        "games": {"won": sum(1 for g in played if g["result_for_a"] == 1.0),
                  "drawn": sum(1 for g in played if g["result_for_a"] == 0.5),
                  "lost": sum(1 for g in played if g["result_for_a"] == 0.0)},
        "incomplete_pairs": len(by_pair) - n,
        "score_a": mean,
        "standard_error": standard_error,
        "elo": to_elo(mean),
        "elo_low": to_elo(mean - 1.96 * standard_error),
        "elo_high": to_elo(mean + 1.96 * standard_error),
        "pair_counts": counts,
        "slowest_ms_a": max((g["slowest_ms_a"] for g in games), default=0.0),
        "slowest_ms_b": max((g["slowest_ms_b"] for g in games), default=0.0),
        "fallbacks_a": sum(g["fallbacks_a"] for g in games),
        "fallbacks_b": sum(g["fallbacks_b"] for g in games),
        "native_errors_a": sum(g.get("native_errors_a", 0) for g in games),
        "native_errors_b": sum(g.get("native_errors_b", 0) for g in games),
        "mean_depth_a": mean_of("mean_depth_a"),
        "mean_depth_b": mean_of("mean_depth_b"),
    }


def _build_agent_from_spec(spec: str, game: str, timeout_ms: int = 3000) -> SparringAgent:
    """
    Parses an agent specification string:
      - `http://...` or `https://...` or `Name=http://...` -> `HttpEndpointAgent`
      - `chance_mcts` / `pure_mcts` / `random` -> `InProcessTiqTaqToeAgent`
      - `expecti_neural` / `expecti_heuristic` / `random` -> `InProcessCheqqersAgent`
    """
    if "=" in spec and ("http://" in spec or "https://" in spec):
        name, url = spec.split("=", 1)
        return HttpEndpointAgent(name=name.strip(), url=url.strip(), timeout_ms=timeout_ms)
    if spec.startswith("http://") or spec.startswith("https://"):
        return HttpEndpointAgent(name=spec, url=spec, timeout_ms=timeout_ms)

    spec, budget = _split_budget(spec)
    name = spec
    mode = spec.strip().lower()
    if ":" in spec:
        name, mode = spec.split(":", 1)
        mode = mode.strip().lower()

    time_limit_ms = float(budget.get("ms", float(timeout_ms) * 0.8))
    if game.lower() in ("tiqtaqtoe", "ttt"):
        if any(key in budget for key in ("depth", "weights", "book", "q", "ext", "qp", "cpp", "net", "pv", "topk", "mcts", "mb", "cpuct", "fpu")):
            raise ValueError(f"agent spec {spec!r}: only ms applies to TiqTaqToe, the other options are for Cheqqers")
        if mode not in ("chance_mcts", "pure_mcts", "random"):
            mode = "chance_mcts"
        return InProcessTiqTaqToeAgent(name=name, mode=mode, time_limit_ms=time_limit_ms)
    else:
        if mode not in ("expecti_neural", "expecti_heuristic", "random"):
            mode = "expecti_heuristic"
        kwargs: Dict[str, Any] = {"max_depth": int(budget["depth"])} if "depth" in budget else {}
        if "weights" in budget:
            kwargs["eval_weights"] = EvalWeights.from_json(str(budget["weights"]))
        if "book" in budget:
            from search.opening_book import OpeningBook

            kwargs["opening_book"] = OpeningBook.from_json(str(budget["book"]))
        if "q" in budget:
            kwargs["quiescence"] = bool(int(budget["q"]))
        if "ext" in budget:
            kwargs["max_capture_extension"] = int(budget["ext"])
        if "qp" in budget:
            kwargs["quantum_plies"] = int(budget["qp"])
        if "cpp" in budget:
            kwargs["engine"] = "cpp" if int(budget["cpp"]) else "python"
        if "net" in budget:
            if kwargs.get("engine") != "cpp":
                raise ValueError(f"agent spec {spec!r}: net=<file> needs cpp=1")
            kwargs["value_net"] = str(budget["net"])
        if "pv" in budget:
            kwargs["policy_model"] = str(budget["pv"])
        if "topk" in budget:
            kwargs["policy_topk"] = int(budget["topk"])
        if "mcts" in budget:
            kwargs["mcts_model"] = str(budget["mcts"])
        mp = {name: budget[key] for key, name in (("mb", "batch"), ("cpuct", "c_puct"), ("fpu", "fpu")) if key in budget}
        if mp:
            if "mcts" not in budget:
                raise ValueError(f"agent spec {spec!r}: mb=, cpuct=, fpu= need mcts=<model.pt>")
            kwargs["mcts_params"] = {k: (int(v) if k == "batch" else float(v)) for k, v in mp.items()}
        return InProcessCheqqersAgent(name=name, mode=mode, time_limit_ms=time_limit_ms, **kwargs)


def _split_budget(spec: str) -> Tuple[str, Dict[str, Any]]:
    """
    Splits an in-process agent spec `Name:mode@depth=8,ms=2200,weights=config/w.json` into (`Name:mode`,
    {"depth": 8.0, "ms": 2200.0, "weights": "config/w.json"}). The part after `@` is the agent's own search
    budget and, for Cheqqers, its search options: `weights=<file>` (evaluation weights), `book=<file>` (opening
    book), `q=1` (quiescence on pending captures), `ext=<plies>` (extension budget along a line), `qp=<plies>`
    (three-square moves only in the first plies of the search), `cpp=1` (search with the C++ engine, as
    server.py does; same options), `net=<file>` (with cpp=1 only: a value_mlp.json at the C++ search's leaves
    instead of the linear evaluation), `pv=<model.pt>` (the policy-value transformer plays: argmax of its policy, no
    search; with `topk=<k>` and cpp=1 the C++ search plays among the policy's k best moves). Without it
    the series defaults apply. Any other key raises, so a typo never runs with a silent default.
    """
    if "@" not in spec:
        return spec, {}
    head, tail = spec.rsplit("@", 1)
    budget: Dict[str, Any] = {}
    for item in tail.split(","):
        key, sep, raw = item.partition("=")
        key = key.strip().lower()
        if not sep or key not in ("depth", "ms", "weights", "book", "q", "ext", "qp", "cpp", "net", "pv", "topk",
                                  "mcts", "mb", "cpuct", "fpu"):
            raise ValueError(
                f"agent spec {spec!r}: item {item!r} is not depth=<plies>, ms=<milliseconds>, weights=<file>, "
                f"book=<file>, q=<0|1>, ext=<plies>, qp=<plies>, cpp=<0|1>, net=<file>, pv=<model.pt>, topk=<k>, mcts=<model.pt>, mb=, cpuct= or fpu=")
        budget[key] = raw.strip() if key in ("weights", "book", "net", "pv", "mcts") else float(raw)
    return head, budget


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leiden Cross-Team Sparring Referee, 24/7 Population League & Blind-Spot Harvester Hub"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run end-to-end verification of HTTP + in-process sparring, Elo/SPRT, Population League (5-game gate >=75% + Max-25 Drop-Bottom-5), and teacher distillation.",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # 1. tiqtaqtoe head-to-head
    p_ttt = subparsers.add_parser("tiqtaqtoe", help="Run a head-to-head Quantum TiqTaqToe (Q=3) series")
    p_ttt.add_argument("--agent-a", type=str, default="Team5_ChanceQMCTS:chance_mcts")
    p_ttt.add_argument("--agent-b", type=str, default="Team5_PureMCTS:pure_mcts")
    p_ttt.add_argument("--num-pairs", type=int, default=10)
    p_ttt.add_argument("--quantumness", type=int, default=3)
    p_ttt.add_argument("--timeout-ms", type=int, default=3000)
    p_ttt.add_argument("--harvest-disagreements", action="store_true")
    p_ttt.add_argument("--hard-cases-path", type=str, default="results/sparring_hard_cases.jsonl")
    p_ttt.add_argument("--seed", type=int, default=42)

    # 2. cheqqers head-to-head
    p_cheq = subparsers.add_parser("cheqqers", help="Run a head-to-head Quantum Cheqqers (6x6, Q=2) series")
    p_cheq.add_argument("--agent-a", type=str, default="Team5_ExpectiNeural:expecti_neural")
    p_cheq.add_argument("--agent-b", type=str, default="Team5_ExpectiHeuristic:expecti_heuristic")
    p_cheq.add_argument("--num-pairs", type=int, default=10)
    p_cheq.add_argument("--board-size", type=int, default=6)
    p_cheq.add_argument("--quantumness", type=int, default=2)
    p_cheq.add_argument("--timeout-ms", type=int, default=3000)
    p_cheq.add_argument("--harvest-disagreements", action="store_true")
    p_cheq.add_argument("--hard-cases-path", type=str, default="results/sparring_hard_cases.jsonl")
    p_cheq.add_argument("--seed", type=int, default=42)
    p_cheq.add_argument("--opening-plies", type=int, default=0,
                        help="random opening of this many plies, played once with each colour per pair")
    p_cheq.add_argument("--out", type=str, default=None, help="append one JSON line per finished game here")
    p_cheq.add_argument("--pair-start", type=int, default=0,
                        help="first pair to play: shards of one series (same --seed) play disjoint ranges")
    p_cheq.add_argument("--no-harvest", action="store_true", help="do not write the lost games' positions")

    # 3. round_robin
    p_sum = subparsers.add_parser("summarize", help="Read per-game files written with --out, pair by pair")
    p_sum.add_argument("file", type=str, nargs="+", help="one or more files; pairs of different files stay apart")

    p_rr = subparsers.add_parser("round_robin", help="Run a multi-agent Round-Robin tournament")
    p_rr.add_argument("--game", type=str, choices=["tiqtaqtoe", "cheqqers"], default="tiqtaqtoe")
    p_rr.add_argument(
        "--agents",
        nargs="+",
        default=["Team5_ChanceQMCTS:chance_mcts", "Team5_PureMCTS:pure_mcts", "Random:random"],
    )
    p_rr.add_argument("--num-pairs", type=int, default=5)
    p_rr.add_argument("--timeout-ms", type=int, default=3000)
    p_rr.add_argument("--harvest-disagreements", action="store_true")
    p_rr.add_argument("--hard-cases-path", type=str, default="results/sparring_hard_cases.jsonl")
    p_rr.add_argument("--seed", type=int, default=42)

    # 4. solve_hard_cases
    p_solve = subparsers.add_parser(
        "solve_hard_cases",
        help="Run deep teacher analysis (ChanceQMCTS 600 sims / Expecti depth 6) on harvested positions",
    )
    p_solve.add_argument("--hard-cases-path", type=str, default="results/sparring_hard_cases.jsonl")
    p_solve.add_argument("--output-path", type=str, default="results/sparring_distilled_targets.jsonl")
    p_solve.add_argument("--ttt-searches", type=int, default=600)
    p_solve.add_argument("--cheqqers-depth", type=int, default=6)
    p_solve.add_argument("--max-cases", type=int, default=None)

    # 5. league_status
    p_lstat = subparsers.add_parser(
        "league_status",
        help="Show current 24/7 Population League standings (weights + architecture + strategy) and culled history",
    )
    p_lstat.add_argument("--game", type=str, choices=["tiqtaqtoe", "cheqqers"], default="tiqtaqtoe")
    p_lstat.add_argument("--pool-path", type=str, default=None)

    # 6. league_challenge (5-Game Challenger Admission Gate >= 75% / Top 25%)
    p_lchal = subparsers.add_parser(
        "league_challenge",
        help="Evaluate a Challenger Model (5 games vs every pool model; admitted iff >=75% beaten or Top 25%)",
    )
    p_lchal.add_argument("--game", type=str, choices=["tiqtaqtoe", "cheqqers"], default="tiqtaqtoe")
    p_lchal.add_argument("--model-id", type=str, required=True, help="Unique identifier for the challenger model")
    p_lchal.add_argument(
        "--architecture",
        type=str,
        default="MLP-128x3-Decoupled-WDL",
        help="Architecture structure (e.g. MLP-128x3-Decoupled-WDL, KAN-128x3-Spline, SNN-LIF-20step, RemoteHTTP)",
    )
    p_lchal.add_argument(
        "--strategy",
        type=str,
        default="TrackedChanceQMCTS(s=120)",
        help="Search/decision strategy (e.g. TrackedChanceQMCTS(s=120), ExpectiAlphaBeta(d=4,w=0.35), RemoteHTTP)",
    )
    p_lchal.add_argument("--weights-path", type=str, default=None, help="Path to .pth checkpoint")
    p_lchal.add_argument("--endpoint-url", type=str, default=None, help="Optional remote POST /move URL")
    p_lchal.add_argument("--games-per-opponent", type=int, default=5)
    p_lchal.add_argument("--min-win-fraction", type=float, default=0.75)
    p_lchal.add_argument("--max-models", type=int, default=25)
    p_lchal.add_argument("--drop-bottom", type=int, default=5)
    p_lchal.add_argument("--time-limit-ms", type=float, default=600.0)
    p_lchal.add_argument("--pool-path", type=str, default=None)
    p_lchal.add_argument("--harvest-disagreements", action="store_true")
    p_lchal.add_argument("--seed", type=int, default=42)

    # 7. league_cycle (All-play-all cycle + Max-25 -> Drop-Bottom-5 enforcement)
    p_lcyc = subparsers.add_parser(
        "league_cycle",
        help="Run one all-play-all adversarial cycle across the active league pool and cull bottom 5 if size >= 25",
    )
    p_lcyc.add_argument("--game", type=str, choices=["tiqtaqtoe", "cheqqers"], default="tiqtaqtoe")
    p_lcyc.add_argument("--games-per-pair", type=int, default=2)
    p_lcyc.add_argument("--only-unplayed", action="store_true")
    p_lcyc.add_argument("--max-models", type=int, default=25)
    p_lcyc.add_argument("--drop-bottom", type=int, default=5)
    p_lcyc.add_argument("--time-limit-ms", type=float, default=500.0)
    p_lcyc.add_argument("--pool-path", type=str, default=None)
    p_lcyc.add_argument("--harvest-disagreements", action="store_true")
    p_lcyc.add_argument("--seed", type=int, default=42)

    # 8. league_daemon (24/7 Continuous Adversarial Loop + Teacher Distillation)
    p_ldaemon = subparsers.add_parser(
        "league_daemon",
        help="Run the 24/7 continuous adversarial league loop + automated blind-spot teacher distillation",
    )
    p_ldaemon.add_argument("--game", type=str, choices=["tiqtaqtoe", "cheqqers", "both"], default="both")
    p_ldaemon.add_argument("--cycles", type=int, default=0, help="Number of cycles to run (0 = infinite 24/7 loop)")
    p_ldaemon.add_argument("--games-per-pair", type=int, default=2)
    p_ldaemon.add_argument("--max-models", type=int, default=25)
    p_ldaemon.add_argument("--drop-bottom", type=int, default=5)
    p_ldaemon.add_argument("--distill-every", type=int, default=1)
    p_ldaemon.add_argument("--time-limit-ms", type=float, default=500.0)
    p_ldaemon.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if args.subcommand == "tiqtaqtoe":
        ag_a = _build_agent_from_spec(args.agent_a, "tiqtaqtoe", args.timeout_ms)
        ag_b = _build_agent_from_spec(args.agent_b, "tiqtaqtoe", args.timeout_ms)
        res = run_tiqtaqtoe_series(
            agent_a=ag_a,
            agent_b=ag_b,
            num_pairs=args.num_pairs,
            quantumness=args.quantumness,
            harvest_losses=True,
            harvest_disagreements=args.harvest_disagreements,
            hard_cases_path=args.hard_cases_path,
            seed=args.seed,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.subcommand == "cheqqers":
        ag_a = _build_agent_from_spec(args.agent_a, "cheqqers", args.timeout_ms)
        ag_b = _build_agent_from_spec(args.agent_b, "cheqqers", args.timeout_ms)
        res = run_cheqqers_series(
            agent_a=ag_a,
            agent_b=ag_b,
            num_pairs=args.num_pairs,
            board_size=args.board_size,
            game_type=args.quantumness,
            harvest_losses=not args.no_harvest,
            harvest_disagreements=args.harvest_disagreements,
            hard_cases_path=args.hard_cases_path,
            seed=args.seed,
            opening_plies=args.opening_plies,
            out_path=args.out,
            pair_start=args.pair_start,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.subcommand == "summarize":
        print(json.dumps(summarize_pairs(read_series_files(args.file)), indent=2))
        return 0

    if args.subcommand == "round_robin":
        agents = [_build_agent_from_spec(s, args.game, args.timeout_ms) for s in args.agents]
        rr = run_round_robin(
            agents=agents,
            game=args.game,
            num_pairs=args.num_pairs,
            harvest_losses=True,
            harvest_disagreements=args.harvest_disagreements,
            hard_cases_path=args.hard_cases_path,
            seed=args.seed,
        )
        print(json.dumps(rr, indent=2))
        return 0

    if args.subcommand == "solve_hard_cases":
        summary = solve_hard_cases(
            hard_cases_path=args.hard_cases_path,
            output_path=args.output_path,
            ttt_teacher_searches=args.ttt_searches,
            cheqqers_teacher_depth=args.cheqqers_depth,
            max_cases=args.max_cases,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.subcommand == "league_status":
        mgr = PopulationLeagueManager(game=args.game, pool_path=args.pool_path)
        print(json.dumps(mgr.get_status_summary(), indent=2))
        return 0

    if args.subcommand == "league_challenge":
        mgr = PopulationLeagueManager(
            game=args.game,
            pool_path=args.pool_path,
            max_models=args.max_models,
            drop_bottom_k=args.drop_bottom,
            time_limit_ms=args.time_limit_ms,
        )
        challenger = LeagueModelSpec(
            model_id=args.model_id,
            game=mgr.game,
            architecture=args.architecture,
            strategy=args.strategy,
            weights_path=args.weights_path,
            endpoint_url=args.endpoint_url,
        )
        report = mgr.evaluate_challenger(
            challenger_spec=challenger,
            games_per_opponent=args.games_per_opponent,
            min_win_fraction=args.min_win_fraction,
            auto_admit=True,
            harvest_losses=True,
            harvest_disagreements=args.harvest_disagreements,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.subcommand == "league_cycle":
        mgr = PopulationLeagueManager(
            game=args.game,
            pool_path=args.pool_path,
            max_models=args.max_models,
            drop_bottom_k=args.drop_bottom,
            time_limit_ms=args.time_limit_ms,
        )
        report = mgr.run_league_cycle(
            games_per_pair=args.games_per_pair,
            only_unplayed_pairs=args.only_unplayed,
            harvest_losses=True,
            harvest_disagreements=args.harvest_disagreements,
            auto_cull=True,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.subcommand == "league_daemon":
        games = ["tiqtaqtoe", "cheqqers"] if args.game == "both" else [args.game]
        managers = {
            g: PopulationLeagueManager(
                game=g,
                max_models=args.max_models,
                drop_bottom_k=args.drop_bottom,
                time_limit_ms=args.time_limit_ms,
            )
            for g in games
        }
        cycle_idx = 0
        while args.cycles <= 0 or cycle_idx < args.cycles:
            cycle_idx += 1
            for g in games:
                cyc_res = managers[g].run_league_cycle(
                    games_per_pair=args.games_per_pair,
                    harvest_losses=True,
                    auto_cull=True,
                    seed=args.seed + cycle_idx * 1009,
                )
                print(
                    json.dumps(
                        {
                            "daemon_iteration": cycle_idx,
                            "game": g,
                            "cycle_number": cyc_res["cycle_number"],
                            "total_games_played": cyc_res["total_games_played"],
                            "top_3": cyc_res["standings"][:3],
                            "cull_event": cyc_res["cull_event"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            if args.distill_every > 0 and (cycle_idx % args.distill_every == 0):
                distill_res = solve_hard_cases(max_cases=64)
                print(json.dumps({"daemon_distillation": distill_res}, indent=2), flush=True)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
