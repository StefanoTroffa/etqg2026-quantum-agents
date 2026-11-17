"""The native Cheqqers engine against the Python one on many positions, in parallel (identity gates 1 and 2).

    CUDA_VISIBLE_DEVICES= uv run python scripts/native_identity_sweep.py rules  <games.jsonl>... [--random 200]
    CUDA_VISIBLE_DEVICES= uv run python scripts/native_identity_sweep.py search <games.jsonl>... [--every 9] [--depth 4]

`rules`: every position of every game (replayed with the recorded measurements) and of `--random` seeded random games
(quantumness 0-3, a few 8x8), with the per-position cache off and on: legal moves, outcome probabilities (bits) and
child positions, state, winner, eval_features and evaluate_state (tests/native/native_identity.py). Also the native
replay of each game after every move.
`search`: the six benchmark positions and every `--every`-th position of the games, searched by
ExpectiCheqqersPlayer and by the native search at a fixed depth, default options and quantum_plies=2 (and with
--options, the other option sets): move, nodes, last depth, score bits.
Prints counts and every mismatch; exit status 1 if there is one.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "native")]

from native_identity import (  # noqa: E402
    BENCH,
    ODD_WEIGHTS,
    compare_position,
    load_games,
    random_positions,
    replay_positions,
    same_export,
    with_cache,
)
from search import native_cheqqers as nc  # noqa: E402


def rules_of_game(game: dict) -> tuple[int, int, list[str]]:
    positions = checks = 0
    problems: list[str] = []
    native = None
    try:
        for ply, env, move in replay_positions(game):
            positions += 1
            for label, position in (("", env), (" cache", with_cache(env))):
                problems += [f"{game['id']} ply {ply}{label}: {p}" for p in compare_position(position)]
                checks += 1 + len(position.get_legal_moves())
            if native is None:
                native = nc.NativeEnv.from_env(env)
            if not same_export(nc.export_env(env), native.export()):
                problems.append(f"{game['id']} ply {ply}: native replay differs")
                native = nc.NativeEnv.from_env(env)
            if move is not None:
                native.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
    except Exception as err:  # a replay the Python itself rejects is reported, not counted
        problems.append(f"{game['id']}: {type(err).__name__}: {err}")
    return positions, checks, problems


def rules_of_random(spec: tuple[int, int, int]) -> tuple[int, int, list[str]]:
    seed, size, q = spec
    positions = checks = 0
    problems: list[str] = []
    for k, env in enumerate(random_positions(seed, size, q, max_plies=150)):
        positions += 1
        for label, position in (("", env), (" cache", with_cache(env))):
            problems += [f"random {seed}/{size}x{size}/q{q} position {k}{label}: {p}" for p in compare_position(position)]
            checks += 1 + len(position.get_legal_moves())
    return positions, checks, problems


def search_one(job: tuple[str, int, dict, list[int], list[float]]) -> tuple[str, list[str], int, float, float]:
    from search.expecti_cheqqers import ExpectiCheqqersPlayer

    label, depth, options, ints, dbl = job
    env = nc.import_env(ints, dbl)
    opts = dict(options)
    if "eval_weights" in opts:
        opts["eval_weights"] = ODD_WEIGHTS[opts["eval_weights"]]
    t0 = time.perf_counter()
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=1e9, **opts)
    py_move = player.get_move(env.clone())
    t1 = time.perf_counter()
    res = nc.best_move(env.clone(), time_limit_ms=1e9, max_depth=depth, **opts)
    t2 = time.perf_counter()
    problems = []
    if list(py_move) != list(res.move):
        problems.append(f"move {py_move} != {res.move}")
    if player._nodes_visited != res.nodes:
        problems.append(f"nodes {player._nodes_visited} != {res.nodes}")
    if player.last_depth != res.last_depth:
        problems.append(f"depth {player.last_depth} != {res.last_depth}")
    pa = None if player.last_score is None else struct.pack("d", player.last_score)
    pb = None if res.last_score is None else struct.pack("d", res.last_score)
    if pa != pb:
        problems.append(f"score {player.last_score!r} != {res.last_score!r}")
    return label, [f"{label}: {p}" for p in problems], player._nodes_visited, t1 - t0, t2 - t1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["rules", "search"])
    ap.add_argument("games", nargs="+")
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--every", type=int, default=9)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--options", action="store_true", help="also quiescence, other weights, small extension budget")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    nc.load()
    games = [g for path in args.games for g in load_games(Path(path))]
    start = time.time()
    problems: list[str] = []
    with multiprocessing.Pool(args.workers) as pool:
        if args.gate == "rules":
            specs = [(seed, 6, seed % 4) for seed in range(args.random)] + [(10_000 + s, 8, 2) for s in range(args.random // 20)]
            positions = checks = 0
            for p, c, probs in pool.imap_unordered(rules_of_game, games, chunksize=4):
                positions, checks, problems = positions + p, checks + c, problems + probs
            game_positions, game_checks = positions, checks
            for p, c, probs in pool.imap_unordered(rules_of_random, specs, chunksize=2):
                positions, checks, problems = positions + p, checks + c, problems + probs
            print(f"rules: {len(games)} games, {game_positions} game positions; {len(specs)} random games, "
                  f"{positions - game_positions} random positions; each with the cache off and on: "
                  f"{checks} position checks + move-outcome checks; {len(problems)} mismatches; {time.time() - start:.0f} s")
        else:
            jobs = []
            bench = load_games(BENCH)
            option_sets: list[tuple[str, dict]] = [("q2", {"quantum_plies": 2}), ("default", {})]
            if args.options:
                option_sets += [("quiescence", {"quiescence": True, "quantum_plies": 2}),
                                ("weights", {"eval_weights": 0, "quantum_plies": 2}),
                                ("logit", {"eval_weights": 1, "quantum_plies": 2}),
                                ("ext1", {"max_capture_extension": 1, "quiescence": True})]
            positions = []
            for game in bench:
                for ply, env, move in replay_positions(game, upto=game["benchPly"]):
                    if move is None:
                        positions.append((f"bench {game['id']}@{ply}", nc.export_env(env)))
            for game in games:
                for ply, env, move in replay_positions(game):
                    if ply % args.every == args.every // 2 and not env.is_terminal() and len(env.get_legal_moves()) > 1:
                        positions.append((f"{game['id']}@{ply}", nc.export_env(env)))
            for name, opts in option_sets:
                for label, (ints, dbl) in positions:
                    jobs.append((f"{label} {name} d{args.depth}", args.depth, opts, ints, dbl))
            py_time = c_time = 0.0
            nodes = 0
            for label, probs, n, tp, tc in pool.imap_unordered(search_one, jobs, chunksize=1):
                problems += probs
                nodes, py_time, c_time = nodes + n, py_time + tp, c_time + tc
            print(f"search: {len(positions)} positions (6 benchmark) x {len(option_sets)} option sets at depth "
                  f"{args.depth} = {len(jobs)} searches, {nodes} nodes; {len(problems)} mismatches; Python "
                  f"{py_time:.0f} s, native {c_time:.1f} s of worker time; {time.time() - start:.0f} s wall")
    for p in problems[:200]:
        print("MISMATCH", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
