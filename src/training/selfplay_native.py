"""Training data for the Cheqqers network: whole self-play games in the C++ engine, from varied openings.

    uv run python src/training/selfplay_native.py --out results/selfplay/shard_00.jsonl --games 200 --seed 1000 \
        --ms 300 --depth 12 --qp 2 --opening-plies 2:6

Each game starts from the tournament position after a random number of random plies (uniform legal move, measurement
outcome drawn with its probability, both from `--seed`), then both sides play the native search
(`native_cheqqers.selfplay`). The engine writes one JSON line per searched position (features, pieces, legal
moves, root scores, the move played, depth, nodes, and with --state the flat state) and one line per game with its
result. Root scores are alpha-beta bounds for moves searched after a better one: the training targets are the move
played and the game's result, not the scores.

Run one process per core with disjoint seeds; shards are independent files.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.cheqqers_official import EvalWeights, OfficialCheqqersEnv  # noqa: E402
from search import native_cheqqers  # noqa: E402


def random_opening(rng: random.Random, plies: int) -> Optional[OfficialCheqqersEnv]:
    """The tournament start position after `plies` random plies; None if the game ended on the way."""
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
    for _ in range(plies):
        moves = env.get_legal_moves()
        if env.done or not moves:
            return None
        outcomes = env.get_move_outcomes(list(rng.choice(moves)))
        draw, total = rng.random(), 0.0
        child = outcomes[-1][1]
        for probability, candidate in outcomes:
            total += probability
            if draw < total:
                child = candidate
                break
        env = child
    return None if env.done else env


def generate(out: Path, games: int, seed: int, ms: float, depth: int, qp: Optional[int],
             opening_plies: tuple[int, int], write_state: bool, eval_weights: Optional[EvalWeights] = None,
             leaf_rate: float = 0.0) -> int:
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    positions = 0
    for k in range(games):
        start = None
        while start is None:
            start = random_opening(rng, rng.randint(*opening_plies))
        positions += native_cheqqers.selfplay(out, games=1, seed=seed * 1_000_003 + k, start=start,
                                              time_limit_ms=ms, max_depth=depth, quantum_plies=qp,
                                              write_state=write_state, eval_weights=eval_weights,
                                              leaf_rate=leaf_rate)
    return positions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ms", type=float, default=300.0, help="search time per move")
    parser.add_argument("--depth", type=int, default=12, help="depth cap")
    parser.add_argument("--qp", type=int, default=2, help="three-square moves in the first plies only; -1 = everywhere")
    parser.add_argument("--opening-plies", type=str, default="2:6", help="min:max random plies before self-play")
    parser.add_argument("--state", action="store_true", help="write the flat state of every position")
    parser.add_argument("--weights", type=Path, default=None, help="evaluation weights (JSON); default: built-in")
    parser.add_argument("--leaf-rate", type=float, default=0.0,
                        help="also write each searched leaf with this probability (network input; for value training)")
    args = parser.parse_args(argv)
    low, high = (int(x) for x in args.opening_plies.split(":"))
    started = time.time()
    positions = generate(args.out, args.games, args.seed, args.ms, args.depth, None if args.qp < 0 else args.qp,
                         (low, high), args.state, EvalWeights.from_json(args.weights) if args.weights else None,
                         args.leaf_rate)
    print(f"{args.games} games, {positions} positions in {time.time() - started:.0f} s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
