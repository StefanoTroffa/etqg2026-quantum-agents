"""Fit the Cheqqers evaluation weights to our own games (Track-V2, docs/specs/cheqqers_best_agent.md).

Texel's method, as draughts engines use it: play many fast games, keep every quiet position with the game's
result, and find the weights for which a logistic curve of the evaluation predicts the results best. The
evaluation is linear in `OfficialCheqqersEnv.eval_features`, so the fit is a logistic regression and the fitted
file is a set of `EvalWeights`.

    uv run python src/training/fit_evaluation.py generate --games 4000 --workers 14 --out results/eval_fit/a.jsonl
    uv run python src/training/fit_evaluation.py fit --data results/eval_fit/a.jsonl --out config/cheqqers_eval_weights.json

`generate` plays self-play games at a small fixed depth from random openings, with a share of random moves so
that the games differ, and writes one JSON line per game: its result and, for every position, the features from
White's side. `fit` leaves out the opening plies and the positions where a capture is pending (their value is
decided by the capture, not by the position), holds out a tenth of the games, and writes the weights scaled so
that a man is worth 1, with a report beside them.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import multiprocessing
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.cheqqers_official import (  # noqa: E402
    EVAL_FEATURES,
    EvalWeights,
    GameState,
    OfficialCheqqersEnv,
    PieceColor,
)

RESULT_FOR_WHITE = {GameState.WHITE_WON: 1.0, GameState.BLACK_WON: 0.0, GameState.DRAW: 0.5}


# ── Games ──────────────────────────────────────────────────────────────────────────────────────────────────────


def play_game(seed: int, depth: int = 2, epsilon: float = 0.15, opening_plies: int = 4, max_plies: int = 160,
              weights: Optional[EvalWeights] = None) -> dict[str, Any]:
    """One self-play game. Returns {"seed", "result_white", "plies", "positions"}; a position is
    [ply, 1 if White is to move else 0, 1 if the mover has a capture else 0, *features from White's side].
    Both sides search `depth` plies with `weights` (None: the defaults) and play a random legal move with
    probability `epsilon`; the first `opening_plies` moves are random. A game still running after `max_plies` is
    a draw."""
    from search.expecti_cheqqers import ExpectiCheqqersPlayer  # imports torch: keep it out of module import

    rng = random.Random(seed)
    random.seed(seed)  # the simulator draws measurement outcomes from the global generator
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=600000.0, eval_weights=weights)
    positions: list[list[float]] = []
    ply = 0
    while not env.done and ply < max_plies:
        legal = env.to_tournament_payload()["legalMoves"]
        if not legal:
            break
        capture = bool(env.find_moves(is_take=True, restrict_to_square_index=None, ignore_turn=False))
        positions.append([ply, 1 if env.turn == PieceColor.WHITE else 0, 1 if capture else 0,
                          *(round(f, 6) for f in env.eval_features("white"))])
        if ply < opening_plies or rng.random() < epsilon or len(legal) == 1:
            move = rng.choice(legal)
        else:
            move = player.get_move(env=env, available_moves=legal)
        env.step(move, return_state=False)
        ply += 1
    return {"seed": seed, "result_white": RESULT_FOR_WHITE.get(env.state, 0.5), "plies": ply,
            "positions": positions}


def _play_game_job(job: tuple) -> dict[str, Any]:
    """One game for the pool. A game that raises is returned as {"seed", "error"} instead of killing the run: it
    is reported and not written, so a later run plays that seed again."""
    seed, depth, epsilon, opening_plies, weights = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return play_game(seed, depth=depth, epsilon=epsilon, opening_plies=opening_plies, weights=weights)
    except Exception as err:  # noqa: BLE001 - any failure of one game must not cost the others
        return {"seed": seed, "error": f"{type(err).__name__}: {err}"}


def generate(out: Path, games: int, workers: int, seed: int, depth: int, epsilon: float, opening_plies: int,
             weights: Optional[EvalWeights] = None) -> int:
    """Play `games` games on `workers` processes, appending one JSON line per game to `out`. Seeds are
    seed, seed + 1, ...; a seed already in `out` is not played again, so a stopped run resumes. Returns the number
    of games written by this call."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(line)["seed"] for line in f if line.strip()}
    jobs = [(s, depth, epsilon, opening_plies, weights) for s in range(seed, seed + games) if s not in done]
    written = 0
    with open(out, "a") as f:
        if workers <= 1:
            results: Iterable[dict[str, Any]] = map(_play_game_job, jobs)
            for game in results:
                if "error" in game:
                    print(f"seed {game['seed']} failed: {game['error']}", file=sys.stderr, flush=True)
                    continue
                f.write(json.dumps(game, separators=(",", ":")) + "\n")
                written += 1
        else:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                for game in pool.imap_unordered(_play_game_job, jobs, chunksize=4):
                    if "error" in game:
                        print(f"seed {game['seed']} failed: {game['error']}", file=sys.stderr, flush=True)
                        continue
                    f.write(json.dumps(game, separators=(",", ":")) + "\n")
                    f.flush()
                    written += 1
                    if written % 200 == 0:
                        print(f"{written} of {len(jobs)} games", file=sys.stderr, flush=True)
    return written


# ── Positions ──────────────────────────────────────────────────────────────────────────────────────────────────


def load_positions(paths: Sequence[Path], min_ply: int = 4, skip_captures: bool = True):
    """(X, y, game): the feature rows from White's side, the game's result for White (1, 0.5, 0) and the index
    of the game each row comes from. Positions before `min_ply` and, if `skip_captures`, positions where the
    mover has a capture are left out."""
    rows, results, game_of = [], [], []
    game_index = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                game = json.loads(line)
                for position in game["positions"]:
                    ply, _, capture = position[0], position[1], position[2]
                    if ply < min_ply or (skip_captures and capture):
                        continue
                    rows.append(position[3:])
                    results.append(game["result_white"])
                    game_of.append(game_index)
                game_index += 1
    X = np.asarray(rows, dtype=np.float64).reshape(-1, len(EVAL_FEATURES))
    return X, np.asarray(results, dtype=np.float64), np.asarray(game_of, dtype=np.int64)


# ── Arena games as data ────────────────────────────────────────────────────────────────────────────────────────


def arena_game_record(game: dict[str, Any]) -> dict[str, Any]:
    """A game from the arena's record file in the format `generate` writes: {"seed": its id, "result_white",
    "plies", "positions"}, one position per ply before the move was made. Games between real agents are few
    but are the only positions with results from play at the level we have to meet."""
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
    positions: list[list[float]] = []
    for ply, move in enumerate(game["moves"]):
        positions.append([ply, 1 if env.turn == PieceColor.WHITE else 0, 1 if env.capture_pending() else 0,
                          *(round(f, 6) for f in env.eval_features("white"))])
        env.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
    return {"seed": game.get("id"), "result_white": {"white": 1.0, "black": 0.0}.get(game["result"], 0.5),
            "plies": len(game["moves"]), "positions": positions}


def import_arena(records: Path, out: Path, include_house: bool = False) -> int:
    """Write the arena games of `records` that ended normally as positions to `out`; games against the house
    bot only with `include_house`. Returns the number of games written."""
    written = 0
    with open(records) as source, open(out, "w") as target:
        for line in source:
            if not line.strip():
                continue
            game = json.loads(line)
            if game.get("reason", "normal") != "normal":
                continue
            if not include_house and any(side.get("house") for side in game.get("players", {}).values()):
                continue
            target.write(json.dumps(arena_game_record(game)) + "\n")
            written += 1
    return written


# ── The fit ────────────────────────────────────────────────────────────────────────────────────────────────────


def log_loss(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of sigmoid(logits) against results y in [0, 1]."""
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iterations: int = 200) -> np.ndarray:
    """The vector v minimising the mean cross-entropy of sigmoid(X v) against y plus l2 |v|^2 / 2, by Newton's
    method. There is no intercept: every feature changes sign with the colour, so a position and its mirror must
    get opposite scores."""
    n, k = X.shape
    v = np.zeros(k)
    for _ in range(iterations):
        z = X @ v
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = X.T @ (p - y) / n + l2 * v
        hessian = (X.T * (p * (1.0 - p))) @ X / n + l2 * np.eye(k)
        step = np.linalg.solve(hessian, gradient)
        v = v - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return v


def best_slope(scores: np.ndarray, y: np.ndarray) -> float:
    """The single factor a for which sigmoid(a * scores) predicts y best: how a fixed evaluation is put on the
    probability scale before its predictions are compared with a fitted one's."""
    return float(fit_logistic(scores.reshape(-1, 1), y, l2=0.0)[0])


def score_weights(X: np.ndarray, y: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    """How well a fixed weight vector predicts the results of the positions: {"slope", "log_loss"}, the score
    w.x put on the probability scale by its own best slope (so that evaluations in different units compare)."""
    scores = X @ np.asarray(vector, dtype=np.float64)
    if not np.any(scores):
        return {"slope": 0.0, "log_loss": log_loss(np.zeros(len(y)), y)}
    slope = best_slope(scores, y)
    return {"slope": slope, "log_loss": log_loss(slope * scores, y)}


def fit_weights(X: np.ndarray, y: np.ndarray, game: np.ndarray, l2: float = 1e-3, holdout: float = 0.1,
                seed: int = 0, squash: bool = False) -> tuple[EvalWeights, dict[str, Any]]:
    """Fit on all games but a held-out share, and report on the held-out ones. The weights are scaled so that a
    man is worth 1 (the unit the search's constants assume); the slope that turns the evaluation in men into a
    logit is in the report and, with `squash`, in the weights (`logit_per_man`: the evaluation then returns the
    expected result instead of points). ValueError if the fit makes a man worth nothing or less."""
    games = np.unique(game)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=max(1, int(len(games) * holdout)), replace=False).tolist())
    test = np.fromiter((g in held for g in game), dtype=bool, count=len(game))
    train = ~test

    v = fit_logistic(X[train], y[train], l2=l2)
    men = v[EVAL_FEATURES.index("men")]
    if men <= 0:
        raise ValueError(f"the fit values a man at {men:.4f}: not usable (too few games, or broken features)")
    weights = EvalWeights(**{name: float(v[i] / men) for i, name in enumerate(EVAL_FEATURES)},
                          logit_per_man=float(men) if squash else 0.0)

    default = np.asarray(EvalWeights().as_vector())
    default_scores = X @ default
    slope_default = best_slope(default_scores[train], y[train])
    material_only = X[:, [EVAL_FEATURES.index("men"), EVAL_FEATURES.index("queens")]] @ np.array([1.0, 2.0])
    slope_material = best_slope(material_only[train], y[train])
    report = {
        "date": datetime.date.today().isoformat(),
        "positions": int(len(y)), "games": int(len(games)), "held_out_games": int(len(held)),
        "l2": l2,
        "slope_per_man": float(men),
        "held_out_log_loss": {
            "coin": float(math.log(2.0)),
            "material_only": log_loss(slope_material * material_only[test], y[test]),
            "default_weights": log_loss(slope_default * default_scores[test], y[test]),
            "fitted_weights": log_loss(X[test] @ v, y[test]),
        },
        "train_log_loss": {
            "default_weights": log_loss(slope_default * default_scores[train], y[train]),
            "fitted_weights": log_loss(X[train] @ v, y[train]),
        },
        "weights": weights.to_dict(),
        "default": EvalWeights().to_dict(),
    }
    return weights, report


# ── Command line ───────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="play self-play games and write their positions")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--games", type=int, default=1000)
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--depth", type=int, default=2)
    g.add_argument("--epsilon", type=float, default=0.15)
    g.add_argument("--opening-plies", type=int, default=4)
    g.add_argument("--weights", type=Path, default=None, help="play with these weights instead of the defaults")

    f = sub.add_parser("fit", help="fit the weights to generated games")
    f.add_argument("--data", nargs="+", required=True, help="files or globs written by generate")
    f.add_argument("--out", type=Path, required=True, help="the weights as JSON; a report is written beside it")
    f.add_argument("--l2", type=float, default=1e-3)
    f.add_argument("--min-ply", type=int, default=4)
    f.add_argument("--keep-captures", action="store_true", help="also use positions with a capture pending")
    f.add_argument("--squash", action="store_true",
                   help="write logit_per_man, so that the evaluation returns the expected result, not points")

    i = sub.add_parser("import-arena", help="turn the arena's record file into positions like generate's")
    i.add_argument("records", type=Path)
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--include-house", action="store_true", help="also the games against the house bot")

    s = sub.add_parser("score", help="how well weights predict the results of the positions in the data")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--weights", nargs="*", default=[], type=Path, help="weight files to score beside the defaults")
    s.add_argument("--min-ply", type=int, default=4)
    s.add_argument("--keep-captures", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "import-arena":
        print(f"{import_arena(args.records, args.out, args.include_house)} games written to {args.out}")
        return 0
    if args.command == "score":
        paths = sorted({Path(p) for pattern in args.data for p in glob.glob(pattern)})
        if not paths:
            parser.error(f"no file matches {args.data}")
        X, y, game = load_positions(paths, min_ply=args.min_ply, skip_captures=not args.keep_captures)
        material = np.zeros(len(EVAL_FEATURES))
        material[EVAL_FEATURES.index("men")] = 1.0
        defaults = EvalWeights()
        material[EVAL_FEATURES.index("queens")] = defaults.queens
        rows = [("material only", material), ("default weights", np.asarray(defaults.as_vector()))]
        rows += [(path.name, np.asarray(EvalWeights.from_json(path).as_vector())) for path in args.weights]
        print(f"{len(y)} positions of {len(np.unique(game))} games; mean result for White {float(np.mean(y)):.3f}")
        for name, vector in rows:
            result = score_weights(X, y, vector)
            print(f"  {name:>28}: log-loss {result['log_loss']:.4f} (slope {result['slope']:.3f})")
        return 0
    if args.command == "generate":
        weights = EvalWeights.from_json(args.weights) if args.weights else None
        written = generate(args.out, args.games, args.workers, args.seed, args.depth, args.epsilon,
                           args.opening_plies, weights)
        print(f"{written} games written to {args.out}")
        return 0

    paths = sorted({Path(p) for pattern in args.data for p in glob.glob(pattern)})
    if not paths:
        parser.error(f"no file matches {args.data}")
    X, y, game = load_positions(paths, min_ply=args.min_ply, skip_captures=not args.keep_captures)
    weights, report = fit_weights(X, y, game, l2=args.l2, squash=args.squash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights.to_dict(), indent=2) + "\n")
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    losses = report["held_out_log_loss"]
    print(f"{report['positions']} positions of {report['games']} games; held-out log-loss: "
          f"material only {losses['material_only']:.4f}, default {losses['default_weights']:.4f}, "
          f"fitted {losses['fitted_weights']:.4f}")
    print(f"weights -> {args.out}; report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
