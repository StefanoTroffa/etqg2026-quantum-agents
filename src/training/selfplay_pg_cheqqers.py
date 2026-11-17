"""Pure self-play reinforcement learning of the Cheqqers policy (track 33, side experiment): no search labels.

    uv run python src/training/selfplay_pg_cheqqers.py --init M.pt --out runs/rl --games 1024 --iters 1000

Each iteration plays `--games` games at once (batched forward passes on the GPU), both sides sampling moves from
softmax(logits / T), from the tournament position after 2-6 random plies; measurement outcomes drawn with their
probabilities. Then one update on all the decisions of those games: REINFORCE with the value head as baseline,
    policy loss  = -(z - v(s)) * log pi(a | s)        (z = the game's result for the mover: +1 / 0 / -1; v detached)
    value loss   = cross-entropy of the (win, draw, loss) head against z
    entropy bonus  --entropy * H(pi(. | s))
Positions with one legal move carry no decision and train the value only. Checkpoints every --save-every iterations
(<out>/iter_XXXX.pt, <out>/last.pt) and one JSON line per iteration in <out>/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.cheqqers_policy_value import batch_moves, load_policy_value, model_config, split_flat  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402
from training.dagger_cheqqers import _sample_child, random_start  # noqa: E402


def play_batch(model, n_games: int, rng: random.Random, temperature: float, device: str, max_plies: int = 300):
    """Self-play games; returns per decision (x, moves, chosen index, mover) and per game its result (white's view)."""
    games = []
    while len(games) < n_games:
        env = random_start(rng, 2, 6)
        if env is not None:
            games.append({"env": env, "ply": 0, "id": len(games), "steps": []})
    results = [0] * n_games
    live = games
    model.eval()
    while live:
        legal = [g["env"].legal_moves() for g in live]
        xs = [g["env"].encode() for g in live]
        idx, lens = batch_moves(legal, max(max(len(m) for m in legal), 1))
        with torch.no_grad():
            x = torch.tensor(xs, dtype=torch.float32, device=device)
            sq, gl = split_flat(x)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, _ = model(sq, gl, idx.to(device), lens.to(device))
            probs = torch.softmax(logits.float() / temperature, -1).cpu().numpy()
        nxt = []
        for i, g in enumerate(live):
            env, moves = g["env"], legal[i]
            if not moves or g["ply"] >= max_plies:
                results[g["id"]] = 0
                continue
            p = probs[i, : len(moves)]
            p = p / p.sum()
            k = int(np.searchsorted(np.cumsum(p), rng.random() * p.sum() * 0.999999)) if len(moves) > 1 else 0
            g["steps"].append((xs[i], moves, k, int(env.turn)))
            g["env"] = _sample_child(env.move_outcomes(moves[k]), rng)
            g["ply"] += 1
            if g["env"].terminal:
                results[g["id"]] = g["env"].winner or 0
            else:
                nxt.append(g)
        live = nxt
    model.train()
    return games, results


class NativeBatch:
    """Many self-play games stepped in C++ (cq_sp_*): one encode + one apply call per ply for all games, on `threads`
    threads. Trajectories stay on the GPU."""

    def __init__(self, threads: int = 3, max_moves: int = 256):
        import ctypes

        self.ct = ctypes
        self.lib = nc.load()
        if not hasattr(self.lib, "cq_sp_new"):
            raise nc.NativeUnavailable(f"{nc.LIB_PATH} has no batched self-play (rebuild it)")
        self.threads, self.M = threads, max_moves

    def play(self, model, n_games: int, seed: int, temperature: float, device: str, open_lo: int = 2,
             open_hi: int = 6, max_plies: int = 300):
        ct, lib, M = self.ct, self.lib, self.M
        err = nc._err()
        h = lib.cq_sp_new(n_games, seed & 0xFFFFFFFFFFFFFFFF, open_lo, open_hi, max_plies, M, self.threads, err, nc._ERR)
        if not h:
            nc._raise(err, "self-play batch")
        h = ct.c_void_p(h)
        pin = device.startswith("cuda")
        x = torch.zeros((n_games, 144), dtype=torch.float32, pin_memory=pin)
        mv = torch.zeros((n_games, M, 3), dtype=torch.int32, pin_memory=pin)
        ln = torch.zeros((n_games, M), dtype=torch.int32, pin_memory=pin)
        ids = torch.zeros(n_games, dtype=torch.int32)
        turn = torch.zeros(n_games, dtype=torch.int32)
        chosen = torch.zeros(n_games, dtype=torch.int32)
        P = lambda t, c: ct.cast(t.data_ptr(), ct.POINTER(c))  # noqa: E731
        mm = ct.c_int32()
        traj = []
        model.eval()
        try:
            while True:
                n = lib.cq_sp_encode(h, P(x, ct.c_float), P(mv, ct.c_int32), P(ln, ct.c_int32), P(ids, ct.c_int32),
                                     P(turn, ct.c_int32), ct.byref(mm), self.threads, err, nc._ERR)
                if n < 0:
                    nc._raise(err, "self-play encode")
                if n == 0:
                    break
                m = max(1, mm.value)
                xg = x[:n].to(device, non_blocking=True)
                mg = mv[:n, :m].to(device, non_blocking=True).long()
                lg = ln[:n, :m].to(device, non_blocking=True).long()
                with torch.no_grad():
                    sq, gl = split_flat(xg)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                        logits, _ = model(sq, gl, mg, lg)
                    probs = torch.softmax(logits.float() / temperature, -1)
                    k = torch.multinomial(probs, 1).squeeze(1)
                chosen[:n] = k.to(torch.int32).cpu()
                traj.append((xg, mg.to(torch.int16), lg.to(torch.int8), k, ids[:n].to(device).long(),
                             turn[:n].to(device).long()))
                if lib.cq_sp_apply(h, P(chosen, ct.c_int32), n, self.threads, err, nc._ERR) < 0:
                    nc._raise(err, "self-play apply")
            res = torch.zeros(n_games, dtype=torch.int32)
            plies = torch.zeros(n_games, dtype=torch.int32)
            lib.cq_sp_results(h, P(res, ct.c_int32), P(plies, ct.c_int32))
        finally:
            lib.cq_sp_free(h)
        model.train()
        return traj, res, plies


def update_gpu(model, opt, traj, results, device: str, entropy: float, batch: int = 4096) -> dict:
    """The update of `update` on trajectories kept on the GPU (NativeBatch.play)."""
    mmax = max(t[1].shape[1] for t in traj)
    xs = torch.cat([t[0] for t in traj])
    mvs = torch.cat([F.pad(t[1], (0, 0, 0, mmax - t[1].shape[1])) for t in traj])
    lns = torch.cat([F.pad(t[2], (0, mmax - t[2].shape[1])) for t in traj])
    ks = torch.cat([t[3] for t in traj])
    gid = torch.cat([t[4] for t in traj])
    trn = torch.cat([t[5] for t in traj])
    zw = results.to(device).float()[gid]
    z = torch.where(trn == 0, zw, -zw)
    perm = torch.randperm(len(ks), device=device)
    stats = {"pol": 0.0, "val": 0.0, "ent": 0.0, "n": 0}
    for s in range(0, len(perm), batch):
        b = perm[s:s + batch]
        cnt = (lns[b] > 0).sum(-1)
        m = int(cnt.max())
        sq, gl = split_flat(xs[b])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, vlog = model(sq, gl, mvs[b, :m].long(), lns[b, :m].long())
        logits, vlog = logits.float(), vlog.float()
        wdl = torch.softmax(vlog, -1)
        v = (wdl[:, 0] - wdl[:, 2]).detach()
        zb = z[b]
        vt = torch.where(zb > 0, 0, torch.where(zb < 0, 2, 1))
        multi = cnt > 1
        logp = torch.log_softmax(logits.masked_fill(torch.isinf(logits), -1e9), -1)
        chosen = logp.gather(1, ks[b].view(-1, 1)).squeeze(1)
        p = logp.exp()
        ent = -(p * logp.masked_fill(p == 0, 0)).sum(-1)
        pol = -((zb - v) * chosen)[multi].mean() if multi.any() else logits.new_zeros(())
        ent_m = ent[multi].mean() if multi.any() else logits.new_zeros(())
        val = F.cross_entropy(vlog, vt)
        loss = pol + val - entropy * ent_m
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        nb = len(b)
        stats["pol"] += pol.item() * nb
        stats["val"] += val.item() * nb
        stats["ent"] += ent_m.item() * nb
        stats["n"] += nb
    n = max(stats["n"], 1)
    return {"policy_loss": stats["pol"] / n, "value_loss": stats["val"] / n, "entropy": stats["ent"] / n,
            "decisions": stats["n"]}


def update(model, opt, games, results, device: str, entropy: float, batch: int = 2048) -> dict:
    rows = []
    for g in games:
        z_white = results[g["id"]]
        for x, moves, k, mover in g["steps"]:
            rows.append((x, moves, k, z_white if mover == 0 else -z_white))
    random.shuffle(rows)
    stats = {"pol": 0.0, "val": 0.0, "ent": 0.0, "n": 0}
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        x = torch.tensor([r[0] for r in part], dtype=torch.float32, device=device)
        idx, lens = batch_moves([r[1] for r in part], max(len(r[1]) for r in part))
        k = torch.tensor([r[2] for r in part], device=device)
        z = torch.tensor([r[3] for r in part], dtype=torch.float32, device=device)
        sq, gl = split_flat(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, vlog = model(sq, gl, idx.to(device), lens.to(device))
        logits, vlog = logits.float(), vlog.float()
        wdl = torch.softmax(vlog, -1)
        v = (wdl[:, 0] - wdl[:, 2]).detach()
        vt = torch.where(z > 0, 0, torch.where(z < 0, 2, 1))
        multi = (lens > 0).sum(-1).to(device) > 1
        logp = torch.log_softmax(logits.masked_fill(torch.isinf(logits), -1e9), -1)
        chosen = logp.gather(1, k.view(-1, 1)).squeeze(1)
        p = logp.exp()
        ent = -(p * logp.masked_fill(p == 0, 0)).sum(-1)
        adv = z - v
        pol = -(adv * chosen)[multi].mean() if multi.any() else logits.new_zeros(())
        ent_m = ent[multi].mean() if multi.any() else logits.new_zeros(())
        val = F.cross_entropy(vlog, vt)
        loss = pol + val - entropy * ent_m
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats["pol"] += pol.item() * len(part)
        stats["val"] += val.item() * len(part)
        stats["ent"] += ent_m.item() * len(part)
        stats["n"] += len(part)
    n = max(stats["n"], 1)
    return {"policy_loss": stats["pol"] / n, "value_loss": stats["val"] / n, "entropy": stats["ent"] / n,
            "decisions": stats["n"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hours", type=float, default=0.0, help="> 0: stop after this many hours")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--native-threads", type=int, default=0,
                        help="> 0: step the games in C++ on this many threads (cq_sp_*), trajectories on the GPU")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = load_policy_value(args.init, args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    args.out.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + args.hours * 3600 if args.hours > 0 else float("inf")
    native = NativeBatch(args.native_threads) if args.native_threads > 0 else None
    for it in range(1, args.iters + 1):
        t0 = time.time()
        if args.native_threads > 0:
            traj, res, plies = native.play(model, args.games, args.seed * 1_000_003 + it, args.temperature, args.device)
            results = res.tolist()
            t1 = time.time()
            st = update_gpu(model, opt, traj, res, args.device, args.entropy)
            st["mean_plies"] = float(plies.float().mean())
        else:
            games, results = play_batch(model, args.games, rng, args.temperature, args.device)
            t1 = time.time()
            st = update(model, opt, games, results, args.device, args.entropy)
        rec = {"iter": it, "play_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1),
               "white_wins": sum(1 for r in results if r == 1), "black_wins": sum(1 for r in results if r == -1),
               "draws": sum(1 for r in results if r == 0), **st}
        with open(args.out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        blob = {"state_dict": model.state_dict(), "config": model_config(model), "iter": it}
        if it % args.save_every == 0:
            torch.save(blob, args.out / f"iter_{it:04d}.pt")
            torch.save(blob, args.out / "last.pt")
        if time.time() >= t_end:
            torch.save(blob, args.out / "last.pt")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
