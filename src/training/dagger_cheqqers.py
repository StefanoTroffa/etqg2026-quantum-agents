"""Expert iteration for the Cheqqers policy (track 33): the transformer plays, the C++ search labels.

    # 1. the policy plays many games at once on the GPU (one batched forward pass per ply for all live games);
    #    every position with >= 2 legal moves is written with its flat state
    uv run python src/training/dagger_cheqqers.py play --model M.pt --games 2000 --seed 1 --out play.jsonl
    # 2. the FW search labels them (CPU pool): best move at a longer budget
    uv run python src/training/dagger_cheqqers.py label play.jsonl --out labels.npz --ms 1000 --workers 30 \
        --weights config/cheqqers_eval_weights_2026-11-13.json
    # 3. retrain with the labelled positions added: train_cheqqers_transformer.py ... --extra-npz labels.npz

play: games start from the tournament position after --opening random plies (default 2-6; uniform move, outcome
drawn with its probability). Each side picks the policy's argmax, except with probability --explore (and always in
the first --sample-plies plies) a move sampled from softmax(logits / --temperature) (so the games reach positions the argmax player would reach after a mistake, as
DAgger needs). Measurement outcomes are drawn with their probabilities. Optionally one side is the FW search
(--fw-side-ms > 0: FW plays one colour in half the games, at that budget, in a CPU pool): the positions the policy
reaches against FW are the ones the series tests.
label: one search per position (native best_move, depth cap 20); the label is its move. The npz has the trainer's
cache layout (train_cheqqers_transformer.shard_arrays) with value = -1 (no game result: the value head is not trained
on these rows).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from search import native_cheqqers as nc  # noqa: E402


def _sample_child(outcomes, rng: random.Random):
    u, acc = rng.random(), 0.0
    for p, child in outcomes:
        acc += p
        if u <= acc:
            return child
    return outcomes[-1][1]


def random_start(rng: random.Random, low: int, high: int) -> Optional[nc.NativeEnv]:
    env = nc.NativeEnv.new(6, 2, 2)
    for _ in range(rng.randint(low, high)):
        if env.terminal:
            return None
        legal = env.legal_moves()
        if not legal:
            return None
        env = _sample_child(env.move_outcomes(rng.choice(legal)), rng)
    return None if env.terminal else env


_FW: dict = {}


def _fw_init(weights: Optional[str], ms: float) -> None:
    from env.cheqqers_official import EvalWeights

    _FW["w"] = EvalWeights.from_json(weights) if weights else None
    _FW["ms"] = ms


def _fw_move(item):
    key, ints, dbl, legal = item
    env = nc.NativeEnv.from_flat(ints, dbl)
    res = nc.best_move(env, time_limit_ms=_FW["ms"], max_depth=20, available_moves=legal, eval_weights=_FW["w"])
    return key, res.move, res.last_depth, res.root_scores


def play(args) -> int:
    import torch

    from search.cheqqers_policy_player import PolicyPlayer

    if args.gpu_mem_frac > 0 and torch.cuda.is_available():  # share of a GPU used by other jobs too
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    player = PolicyPlayer(args.model)
    rng = random.Random(args.seed)
    games = []
    while len(games) < args.games:
        env = random_start(rng, args.open_low, args.open_high)
        if env is not None:
            fw_colour = -1
            if args.fw_side_ms > 0 and len(games) % 2 == 0:
                fw_colour = rng.randint(0, 1)
            games.append({"env": env, "ply": 0, "id": len(games), "fw": fw_colour})
    pool = Pool(args.workers, initializer=_fw_init, initargs=(args.weights, args.fw_side_ms)) \
        if args.fw_side_ms > 0 else None
    out = open(args.out, "a")
    written, finished, started = 0, 0, time.time()
    results = {"1": 0, "-1": 0, "0": 0, "2": 0}
    live = games
    while live:
        legal_all = [g["env"].legal_moves() for g in live]
        nxt = []
        pol_idx = [i for i, g in enumerate(live) if g["env"].turn.value != g["fw"] and len(legal_all[i]) > 0]
        fw_idx = [i for i, g in enumerate(live) if g["env"].turn.value == g["fw"] and len(legal_all[i]) > 0]
        chosen: dict[int, list[int]] = {}
        argmax: dict[int, list[int]] = {}  # the policy's own move (the DAgger question: is it the teacher's?)
        for s in range(0, len(pol_idx), args.batch):
            part = pol_idx[s:s + args.batch]
            logits, _ = player.evaluate([live[i]["env"] for i in part], [legal_all[i] for i in part])
            for i, lg in zip(part, logits):
                legal = legal_all[i]
                argmax[i] = legal[max(range(len(lg)), key=lambda k: (lg[k], -k))]
                if len(legal) > 1 and (live[i]["ply"] < args.sample_plies or rng.random() < args.explore):
                    p = torch.softmax(torch.tensor(lg) / args.temperature, 0).tolist()
                    chosen[i] = rng.choices(legal, weights=p)[0]
                else:
                    chosen[i] = legal[max(range(len(lg)), key=lambda k: (lg[k], -k))]
        if fw_idx:
            items = [(i, *live[i]["env"].export(), legal_all[i]) for i in fw_idx]
            for i, mv, _, _ in pool.imap_unordered(_fw_move, items):
                chosen[i] = mv
        for i, g in enumerate(live):
            env, legal = g["env"], legal_all[i]
            if not legal or g["ply"] >= args.max_plies:
                results["2"] += 1
                finished += 1
                continue
            mv = chosen[i]
            if len(legal) > 1 and env.turn.value != g["fw"]:
                ints, dbl = env.export()
                out.write(json.dumps({"game": args.seed * 100000 + g["id"], "ply": g["ply"], "moves": legal,
                                      "played": mv, "argmax": argmax.get(i), "state": [ints, dbl]}) + "\n")
                written += 1
            g["env"] = _sample_child(env.move_outcomes(mv), rng)
            g["ply"] += 1
            if g["env"].terminal:
                results[str(g["env"].winner if g["env"].winner is not None else 0)] += 1
                finished += 1
            else:
                nxt.append(g)
        live = nxt
    out.close()
    if pool:
        pool.close()
    print(f"{args.games} games ({results}), {written} positions in {time.time() - started:.0f} s -> {args.out}",
          flush=True)
    return 0


def _quick(item):
    i, rec = item
    env = nc.NativeEnv.from_flat(*rec["state"])
    res = nc.best_move(env, time_limit_ms=_FW["ms"], max_depth=20, available_moves=rec["moves"], eval_weights=_FW["w"])
    return i, res.move


def filter_disagreements(args) -> int:
    """Keep the positions where a quick FW search (--ms) disagrees with the policy's own move ("argmax"), plus a
    random --keep-agree share of the others; written in the input format for `label` (the 1 s teacher)."""
    recs = []
    for path in args.inputs:
        for line in open(path):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(args.seed)
    started = time.time()
    keep = [False] * len(recs)
    dis = 0
    with Pool(args.workers, initializer=_fw_init, initargs=(args.weights, args.ms)) as pool:
        for i, mv in pool.imap_unordered(_quick, list(enumerate(recs)), chunksize=8):
            am = recs[i].get("argmax")
            disagree = am is None or list(mv) != list(am)
            dis += disagree
            keep[i] = disagree or rng.random() < args.keep_agree
    with open(args.out, "w") as fh:
        for r, k in zip(recs, keep):
            if k:
                fh.write(json.dumps(r) + "\n")
    print(f"{len(recs)} positions, quick FW ({args.ms:.0f} ms) disagrees with the policy in {dis} "
          f"({dis / max(len(recs), 1):.3f}); kept {sum(keep)} -> {args.out} in {time.time() - started:.0f} s",
          flush=True)
    return 0


def _label(item):
    i, rec = item
    env = nc.NativeEnv.from_flat(*rec["state"])
    res = nc.best_move(env, time_limit_ms=_FW["ms"], max_depth=20, available_moves=rec["moves"], eval_weights=_FW["w"])
    return i, env.encode(), res.move, res.last_depth


def label(args) -> int:
    recs = []
    for path in args.inputs:
        for line in open(path):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit and len(recs) > args.limit:
        recs = random.Random(args.seed).sample(recs, args.limit)
    started = time.time()
    xs = [None] * len(recs)
    tgt = [0] * len(recs)
    depth = 0
    with Pool(args.workers, initializer=_fw_init, initargs=(args.weights, args.ms)) as pool:
        for k, (i, x, mv, d) in enumerate(pool.imap_unordered(_label, list(enumerate(recs)), chunksize=4)):
            xs[i] = x
            tgt[i] = [list(m) for m in recs[i]["moves"]].index(list(mv))
            depth += d
            if (k + 1) % 5000 == 0:
                print(f"{k + 1}/{len(recs)} labelled, {time.time() - started:.0f} s", flush=True)
    mst, msq, mlen = [0], [], []
    for r in recs:
        for m in r["moves"]:
            msq.append(list(m) + [0] * (3 - len(m)))
            mlen.append(len(m))
        mst.append(len(mlen))
    agree = sum(1 for i, r in enumerate(recs) if list(r["moves"][tgt[i]]) == list(r.get("played", [])))
    stats = {"games": len({r.get("game") for r in recs}), "excluded_positions": 0, "unfinished": 0, "ply_limit": 0,
             "labelled": len(recs), "ms": args.ms, "mean_depth": depth / max(len(recs), 1),
             "played_equals_label": agree}
    n = len(recs)
    tmp = str(args.out) + ".tmp.npz"
    np.savez(tmp, x=np.asarray(xs, dtype=np.float32).reshape(n, -1), mstart=np.asarray(mst, dtype=np.int64),
             msq=np.asarray(msq, dtype=np.int8).reshape(-1, 3), mlen=np.asarray(mlen, dtype=np.int8),
             target=np.asarray(tgt, dtype=np.int32), value=np.full(n, -1, dtype=np.int8),
             game=np.asarray([r.get("game", 0) for r in recs], dtype=np.int32), stats=np.asarray(json.dumps(stats)))
    os.replace(tmp, args.out)
    print(f"{n} positions labelled at {args.ms:.0f} ms (mean depth {stats['mean_depth']:.1f}), label = move played "
          f"in {agree / max(n, 1):.3f}, {time.time() - started:.0f} s -> {args.out}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--model", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--explore", type=float, default=0.1, help="probability of a sampled (not argmax) move")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-plies", type=int, default=0,
                   help="every policy move of the first plies is sampled from softmax(logits / T) (diversity)")
    p.add_argument("--opening", default="2:6", help="min:max uniformly random plies before the game")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--gpu-mem-frac", type=float, default=0.08, help="cap on this process's share of GPU memory")
    p.add_argument("--fw-side-ms", type=float, default=0.0, help="> 0: FW plays one side in half the games")
    p.add_argument("--weights", default="config/cheqqers_eval_weights_2026-11-13.json")
    p.add_argument("--workers", type=int, default=16)
    q = sub.add_parser("label")
    q.add_argument("inputs", nargs="+")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--ms", type=float, default=1000.0)
    q.add_argument("--weights", default="config/cheqqers_eval_weights_2026-11-13.json")
    q.add_argument("--workers", type=int, default=30)
    q.add_argument("--limit", type=int, default=0, help="label a random subset of this size")
    q.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("filter")
    f.add_argument("inputs", nargs="+")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--ms", type=float, default=100.0)
    f.add_argument("--keep-agree", type=float, default=0.2, help="share of agreeing positions kept anyway")
    f.add_argument("--weights", default="config/cheqqers_eval_weights_2026-11-13.json")
    f.add_argument("--workers", type=int, default=15)
    f.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "filter":
        return filter_disagreements(args)
    if args.cmd == "play":
        args.open_low, args.open_high = (int(v) for v in args.opening.split(":"))
    return play(args) if args.cmd == "play" else label(args)


if __name__ == "__main__":
    raise SystemExit(main())
