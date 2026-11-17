"""Train the Cheqqers policy-value transformer on native self-play shards (src/training/selfplay_native.py).

    uv run python src/training/train_cheqqers_transformer.py --train 'results/selfplay/shard_0[0-8].jsonl' \
        --val results/selfplay/shard_09.jsonl --out results/cheqqers_pv/run1 --epochs 10

Targets: the move the search played (cross-entropy over the position's legal moves) and the game's result for the
side to move (win/draw/loss). Positions with a single legal move carry no policy information and train the value
only. Games that hit the ply limit (result 2) are dropped. Validation shards are whole games never seen in training.

Writes <out>/model.pt (state dict + config), <out>/metrics.jsonl (one line per epoch).

Large data (track 33): `--cache-dir DIR` parses every shard once, in parallel, into DIR/<dir>_<shard>_<tag>.npz
(flat encodings, legal moves, played move, result) and trains from arrays; `--exclude-states pos.jsonl` drops every
position equal to one of those (the offline proxy's positions, by flat state; FW self-play is deterministic until a
chance event, so the same position recurs across games and dropping whole games would drop most of them). A game
still being written (no result line) is never used. Model size: --d-model, --layers, --heads, --ff.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from multiprocessing import Pool
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cheqqers_policy_value import (  # noqa: E402
    FLAT_IN,
    CheqqersPolicyValue,
    batch_moves,
    encode_position,
    flat_encoding,
    model_config,
    split_flat,
)



def load_positions(patterns: Sequence[str]) -> list[dict]:
    """Every searched position of every finished game, with its value target (0 win, 1 draw, 2 loss for the mover)."""
    out: list[dict] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            game: list[dict] = []
            with open(path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a shard still being written ends mid-line
                    if "result" not in rec:
                        game.append(rec)
                        continue
                    if rec["result"] in (1, -1, 0):
                        for p in game:
                            mover_white = p["turn"] == 0
                            if rec["result"] == 0:
                                p["value"] = 1
                            else:
                                p["value"] = 0 if (rec["result"] == 1) == mover_white else 2
                            sq, gl = encode_position(p)
                            moves = [list(m) for m in p["moves"]]
                            target = moves.index(list(p["move"])) if list(p["move"]) in moves else -1
                            out.append({"sq": sq, "gl": gl, "moves": moves, "target": target, "value": p["value"]})
                    game = []
    return out


def to_batch(items: list[dict], device: str):
    squares = torch.tensor([it["sq"] for it in items], dtype=torch.float32, device=device)
    glob_ = torch.tensor([it["gl"] for it in items], dtype=torch.float32, device=device)
    # Pad to this batch's longest move list: positions with flying queens and splits reach 166 legal moves.
    idx, lens = batch_moves([it["moves"] for it in items], max(len(it["moves"]) for it in items))
    target = torch.tensor([it["target"] if len(it["moves"]) > 1 and it["target"] >= 0 else -100
                           for it in items], dtype=torch.long, device=device)
    value = torch.tensor([it["value"] for it in items], dtype=torch.long, device=device)
    return squares, glob_, idx.to(device), lens.to(device), target, value


@torch.no_grad()
def evaluate(model: CheqqersPolicyValue, data: list[dict], device: str, batch: int) -> dict:
    model.eval()
    n_pol = correct = 0
    pol_loss = val_loss = 0.0
    val_correct = 0
    for i in range(0, len(data), batch):
        sq, gl, idx, lens, target, value = to_batch(data[i:i + batch], device)
        logits, vlogits = model(sq, gl, idx, lens)
        mask = target >= 0
        if mask.any():
            pol_loss += F.cross_entropy(logits[mask], target[mask], reduction="sum").item()
            correct += (logits[mask].argmax(-1) == target[mask]).sum().item()
            n_pol += int(mask.sum())
        val_loss += F.cross_entropy(vlogits, value, reduction="sum").item()
        val_correct += (vlogits.argmax(-1) == value).sum().item()
    model.train()
    return {"policy_loss": pol_loss / max(n_pol, 1), "policy_top1": correct / max(n_pol, 1),
            "value_loss": val_loss / max(len(data), 1), "value_acc": val_correct / max(len(data), 1),
            "policy_positions": n_pol, "positions": len(data)}


# ── Cached arrays (--cache-dir) ─────────────────────────────────────────────────────────────────────────────────

def state_key(state) -> str:
    return hashlib.sha1(json.dumps(state, separators=(",", ":")).encode()).hexdigest()


def exclude_keys(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    return {state_key(json.loads(line)["state"]) for line in open(path) if line.strip()}


def shard_arrays(path: str, excluded: set[str]) -> dict:
    """One shard -> arrays of every position of every finished game (result 1, -1, 0), minus excluded games."""
    xs, mst, msq, mlen, tgt, val, gid = [], [0], [], [], [], [], []
    stats = {"games": 0, "excluded_positions": 0, "unfinished": 0, "ply_limit": 0}
    game: list[dict] = []
    g = 0
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "leaf" in rec:
                continue
            if "result" not in rec:
                game.append(rec)
                continue
            res = rec["result"]
            if res not in (1, -1, 0):
                stats["ply_limit"] += 1
            else:
                stats["games"] += 1
                for p in game:
                    if excluded and "state" in p and state_key(p["state"]) in excluded:
                        stats["excluded_positions"] += 1
                        continue
                    mover_white = p["turn"] == 0
                    value = 1 if res == 0 else (0 if (res == 1) == mover_white else 2)
                    moves = [list(m) for m in p["moves"]]
                    played = list(p["move"])
                    xs.append(flat_encoding(p))
                    for m in moves:
                        msq.append(m + [0] * (3 - len(m)))
                        mlen.append(len(m))
                    mst.append(len(mlen))
                    tgt.append(moves.index(played) if played in moves else -1)
                    val.append(value)
                    gid.append(g)
                g += 1
            game = []
    if game:
        stats["unfinished"] += 1
    n = len(xs)
    return {"x": np.asarray(xs, dtype=np.float32).reshape(n, FLAT_IN),
            "mstart": np.asarray(mst, dtype=np.int64),
            "msq": np.asarray(msq, dtype=np.int8).reshape(-1, 3), "mlen": np.asarray(mlen, dtype=np.int8),
            "target": np.asarray(tgt, dtype=np.int32), "value": np.asarray(val, dtype=np.int8),
            "game": np.asarray(gid, dtype=np.int32), "stats": np.asarray(json.dumps(stats))}


def _cache_one(job: tuple[str, str, Optional[str]]) -> str:
    path, out, excl = job
    if not os.path.exists(out):
        arrays = shard_arrays(path, exclude_keys(Path(excl)) if excl else set())
        tmp = out + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, out)
    return out


def cached(patterns: Sequence[str], cache_dir: Path, excl: Optional[Path], workers: int) -> list[str]:
    tag = hashlib.sha1((excl.read_bytes() if excl else b"")).hexdigest()[:10]
    jobs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            jobs.append((str(p), str(cache_dir / f"{p.parent.name}_{p.stem}_{tag}.npz"), str(excl) if excl else None))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Pool(max(1, min(workers, len(jobs)))) as pool:
        return pool.map(_cache_one, jobs, chunksize=1)


class Arrays:
    """Concatenated shard arrays with vectorised batching."""

    def __init__(self, files: Sequence[str]):
        parts = [np.load(f) for f in files]
        self.stats = [json.loads(str(p["stats"])) for p in parts]
        self.x = np.concatenate([p["x"] for p in parts])
        self.target = np.concatenate([p["target"] for p in parts]).astype(np.int64)
        self.value = np.concatenate([p["value"] for p in parts]).astype(np.int64)
        starts, sq, ln, base = [], [], [], 0
        for p in parts:
            starts.append(p["mstart"][:-1] + base)
            base += int(p["mstart"][-1])
            sq.append(p["msq"])
            ln.append(p["mlen"])
        self.mstart = np.concatenate(starts + [np.asarray([base])])
        self.msq = np.concatenate(sq).astype(np.int64)
        self.mlen = np.concatenate(ln).astype(np.int64)
        self.count = np.diff(self.mstart)
        self.n = len(self.x)
        self.weight: Optional[np.ndarray] = None
        # soft policy targets per move row (Gumbel self-play shards carry "policy"); other files: one-hot at target
        self.has_soft = any("policy" in p.files for p in parts)
        self._soft_parts = [np.asarray(p["policy"], dtype=np.float32) if "policy" in p.files else None for p in parts]
        self._part_rows = [len(p["msq"]) for p in parts]

    def soft(self) -> np.ndarray:
        """One target probability per move row: the file's "policy" if it has one, else 1 at the target move (0 for
        positions with one move or no target)."""
        out = np.zeros(len(self.msq), dtype=np.float32)
        ok = (self.target >= 0) & (self.count > 1)
        idx = np.flatnonzero(ok)
        out[self.mstart[idx] + self.target[idx]] = 1.0
        if self.has_soft:
            row0 = 0
            for part, rows in zip(self._soft_parts, self._part_rows):
                if part is not None:
                    out[row0:row0 + rows] = part
                row0 += rows
        return out

    def batch(self, idx: np.ndarray, device: str):
        cnt = self.count[idx]
        m = int(cnt.max())
        b = len(idx)
        rows = np.repeat(np.arange(b), cnt)
        cols = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        src = np.repeat(self.mstart[idx], cnt) + cols
        mi = np.zeros((b, m, 3), dtype=np.int64)
        ml = np.zeros((b, m), dtype=np.int64)
        mi[rows, cols] = self.msq[src]
        ml[rows, cols] = self.mlen[src]
        tgt = np.where((cnt > 1) & (self.target[idx] >= 0), self.target[idx], -100)
        t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
        sq, gl = split_flat(t(self.x[idx]))
        val = np.where(self.value[idx] >= 0, self.value[idx], -100)  # -1: no game result (labelled positions)
        return sq, gl, t(mi), t(ml), t(tgt), t(val)


@torch.no_grad()
def evaluate_arrays(model: CheqqersPolicyValue, data: Arrays, device: str, batch: int, limit: int = 0) -> dict:
    model.eval()
    n = data.n if not limit else min(limit, data.n)
    n_pol = correct = n_val = 0
    pol_loss = val_loss = 0.0
    val_correct = 0
    for i in range(0, n, batch):
        sq, gl, idx, lens, target, value = data.batch(np.arange(i, min(i + batch, n)), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, vlogits = model(sq, gl, idx, lens)
        logits, vlogits = logits.float(), vlogits.float()
        mask = target >= 0
        if mask.any():
            pol_loss += F.cross_entropy(logits[mask], target[mask], reduction="sum").item()
            correct += (logits[mask].argmax(-1) == target[mask]).sum().item()
            n_pol += int(mask.sum())
        vmask = value >= 0
        if vmask.any():
            val_loss += F.cross_entropy(vlogits[vmask], value[vmask], reduction="sum").item()
            val_correct += (vlogits[vmask].argmax(-1) == value[vmask]).sum().item()
            n_val += int(vmask.sum())
    model.train()
    return {"policy_loss": pol_loss / max(n_pol, 1), "policy_top1": correct / max(n_pol, 1),
            "value_loss": val_loss / max(n_val, 1), "value_acc": val_correct / max(n_val, 1),
            "policy_positions": n_pol, "positions": n}


def train_arrays(args) -> int:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    train_files = cached(args.train, args.cache_dir, args.exclude_states, args.workers)
    extra = [f for pattern in args.extra_npz for f in sorted(glob.glob(pattern))]
    train = Arrays(train_files + extra)
    if extra and args.extra_weight != 1.0:  # labelled positions' policy loss weighted
        sizes = [len(np.load(f)["target"]) for f in train_files + extra]
        w = np.ones(train.n, dtype=np.float32)
        w[sum(sizes[:len(train_files)]):] = args.extra_weight
        train.weight = w
    print(f"extra labelled files {len(extra)}", flush=True)
    val = Arrays(cached(args.val, args.cache_dir, args.exclude_states, args.workers))
    excl = sum(s["excluded_positions"] for s in train.stats + val.stats)
    games = sum(s["games"] for s in train.stats)
    print(f"train {train.n} positions ({games} games), val {val.n} positions, excluded positions {excl}, "
          f"loaded in {time.time() - t0:.0f} s, device {args.device}", flush=True)
    if args.init:
        from models.cheqqers_policy_value import load_policy_value

        model = load_policy_value(args.init, args.device).train()
    else:
        model = CheqqersPolicyValue(d_model=args.d_model, nhead=args.heads, layers=args.layers,
                                    ff=args.ff).to(args.device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model {model_config(model)} {nparams:,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(train.n / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(args.epochs * steps_per_epoch, 1),
                                                pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    use_amp = args.device.startswith("cuda")
    for epoch in range(args.epochs):
        perm = rng.permutation(train.n)
        started = time.time()
        for i in range(0, train.n, args.batch):
            bidx = perm[i:i + args.batch]
            sq, gl, idx, lens, target, value = train.batch(bidx, args.device)
            weight = None
            if train.weight is not None:
                weight = torch.from_numpy(train.weight[bidx]).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, vlogits = model(sq, gl, idx, lens)
            logits, vlogits = logits.float(), vlogits.float()
            mask = target >= 0
            vmask = value >= 0
            loss = logits.new_zeros(())
            if vmask.any():
                loss = loss + args.value_weight * F.cross_entropy(vlogits[vmask], value[vmask])
            if mask.any():
                pw = weight[mask] if weight is not None else None
                ce = F.cross_entropy(logits[mask], target[mask], reduction="none")
                loss = loss + (ce * pw).sum() / pw.sum() if pw is not None else loss + ce.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        metrics = {"epoch": epoch + 1, "seconds": round(time.time() - started, 1),
                   "train": evaluate_arrays(model, train, args.device, 2048, limit=50000),
                   "val": evaluate_arrays(model, val, args.device, 2048)}
        print(json.dumps(metrics), flush=True)
        with open(args.out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(metrics) + "\n")
        score = metrics["val"]["policy_loss"] + args.value_weight * metrics["val"]["value_loss"]
        blob = {"state_dict": model.state_dict(), "config": model_config(model), "epoch": epoch + 1,
                "val": metrics["val"]}
        torch.save(blob, args.out / "last.pt")
        if score < best:
            best = score
            torch.save(blob, args.out / "model.pt")
    return 0


# ── Deduplicated training (--dedup) ──────────────────────────────────────────────────────────────────────────────

def position_keys(a: Arrays) -> np.ndarray:
    """One key per row: the encoding and the legal moves (identical rows = the same position for the network)."""
    keys = np.empty(a.n, dtype=object)
    for i in range(a.n):
        s, e = a.mstart[i], a.mstart[i + 1]
        keys[i] = hashlib.sha1(a.x[i].tobytes() + a.msq[s:e].tobytes() + a.mlen[s:e].tobytes()).digest()
    return keys


class Dedup:
    """The unique positions of an Arrays with aggregated soft targets.

    Policy target: if any copy of the position carries a search label (value < 0 rows from dagger_cheqqers.py label),
    the distribution of its labels; otherwise the distribution of the moves played there. Row weight: label_weight
    for labelled positions, 1 otherwise (each position counts once, however often the games repeated it). Value
    target: the distribution of the results (win / draw / loss) over the copies that have one."""

    def __init__(self, a: Arrays, keys: np.ndarray, label_weight: float):
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        group = np.empty(a.n, dtype=np.int64)
        group[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, a.n]))
        self.rep = order[starts]  # first copy (in key order) of each unique position
        self.a = a
        self.u = len(self.rep)
        self.keys = keys[self.rep]
        is_label = a.value < 0
        has_label = np.zeros(self.u, dtype=bool)
        np.logical_or.at(has_label, group, is_label)
        # policy counts, aligned with each representative's move rows
        rep_start = a.mstart[self.rep]
        self.pstart = np.r_[0, np.cumsum(a.count[self.rep])]
        self.pol = np.zeros(int(self.pstart[-1]), dtype=np.float32)
        use = (a.count > 1) & (is_label | ~has_label[group])
        rows_used = np.flatnonzero(use)
        cnt = a.count[rows_used]
        local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        src = np.repeat(a.mstart[rows_used], cnt) + local
        dst = np.repeat(self.pstart[group[rows_used]], cnt) + local
        np.add.at(self.pol, dst, a.soft()[src])  # one-hot per played/labelled move, or the Gumbel policy target
        del rep_start
        self.val = np.zeros((self.u, 3), dtype=np.float32)
        vrow = a.value >= 0
        np.add.at(self.val, (group[vrow], a.value[vrow]), 1.0)
        self.weight = np.where(has_label, label_weight, 1.0).astype(np.float32)
        self.has_label = has_label

    def batch(self, uidx: np.ndarray, device: str):
        rows = self.rep[uidx]
        sq, gl, mi, ml, _, _ = self.a.batch(rows, device)
        cnt = self.a.count[rows]
        b, m = len(rows), int(cnt.max())
        r = np.repeat(np.arange(b), cnt)
        c = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pt = np.zeros((b, m), dtype=np.float32)
        pt[r, c] = self.pol[np.repeat(self.pstart[uidx], cnt) + c]
        t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)  # noqa: E731
        return sq, gl, mi, ml, t(pt), t(self.val[uidx]), t(self.weight[uidx])


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per row: cross-entropy against a (not normalised) count target; rows with no count give 0."""
    tot = target.sum(-1, keepdim=True)
    p = target / tot.clamp_min(1e-9)
    logp = torch.log_softmax(logits.masked_fill(torch.isinf(logits), -1e9), -1)
    return -(p * logp).sum(-1)


@torch.no_grad()
def top1_on(model: CheqqersPolicyValue, data: Arrays, rows: np.ndarray, device: str, batch: int = 2048) -> dict:
    """Hard-target policy loss and top-1 on the given rows (>= 2 moves, target known)."""
    model.eval()
    rows = rows[(data.count[rows] > 1) & (data.target[rows] >= 0)]
    loss = hit = 0.0
    for i in range(0, len(rows), batch):
        sq, gl, mi, ml, tgt, _ = data.batch(rows[i:i + batch], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg, _ = model(sq, gl, mi, ml)
        lg = lg.float()
        loss += F.cross_entropy(lg, tgt, reduction="sum").item()
        hit += (lg.argmax(-1) == tgt).sum().item()
    model.train()
    n = max(len(rows), 1)
    return {"positions": int(len(rows)), "policy_loss": loss / n, "top1": hit / n}


def train_dedup(args) -> int:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    train_files = cached(args.train, args.cache_dir, args.exclude_states, args.workers)
    extra = [f for pattern in args.extra_npz for f in sorted(glob.glob(pattern))]
    raw = Arrays(train_files + extra)
    keys = position_keys(raw)
    train = Dedup(raw, keys, args.extra_weight)
    seen = set(train.keys.tolist())
    val = Arrays(cached(args.val, args.cache_dir, args.exclude_states, args.workers))
    vkeys = position_keys(val)
    unseen = np.flatnonzero(np.array([k not in seen for k in vkeys]))
    lval = None
    if args.extra_val_npz:
        lval = Arrays([f for pattern in args.extra_val_npz for f in sorted(glob.glob(pattern))])
        lkeys = position_keys(lval)
        lunseen = np.flatnonzero(np.array([k not in seen for k in lkeys]))
    print(f"train rows {raw.n} -> {train.u} unique positions ({int(train.has_label.sum())} with a search label, "
          f"from {len(extra)} label files); val {val.n} rows, {len(unseen)} unseen in training"
          + (f"; label-val {lval.n} rows, {len(lunseen)} unseen" if lval is not None else "")
          + f"; {time.time() - t0:.0f} s", flush=True)
    if args.init:
        from models.cheqqers_policy_value import load_policy_value

        model = load_policy_value(args.init, args.device).train()
    else:
        model = CheqqersPolicyValue(d_model=args.d_model, nhead=args.heads, layers=args.layers,
                                    ff=args.ff, move_head=args.move_head).to(args.device)
    print(f"model {model_config(model)} {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(train.u / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(args.epochs * steps, 1),
                                                pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    use_amp = args.device.startswith("cuda")
    for epoch in range(args.epochs):
        perm = rng.permutation(train.u)
        started = time.time()
        for i in range(0, train.u, args.batch):
            sq, gl, mi, ml, pt, vt, w = train.batch(perm[i:i + args.batch], args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg, vl = model(sq, gl, mi, ml)
            lg, vl = lg.float(), vl.float()
            pm = pt.sum(-1) > 0
            vm = vt.sum(-1) > 0
            loss = lg.new_zeros(())
            if pm.any():
                loss = loss + (soft_ce(lg[pm], pt[pm]) * w[pm]).sum() / w[pm].sum()
            if vm.any():
                loss = loss + args.value_weight * soft_ce(vl[vm], vt[vm]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        metrics = {"epoch": epoch + 1, "seconds": round(time.time() - started, 1),
                   "val_unseen": top1_on(model, val, unseen, args.device),
                   "val_all": top1_on(model, val, np.arange(val.n), args.device)}
        if lval is not None:
            metrics["label_val_unseen"] = top1_on(model, lval, lunseen, args.device)
        print(json.dumps(metrics), flush=True)
        with open(args.out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(metrics) + "\n")
        sel = metrics["label_val_unseen" if lval is not None else "val_unseen"]["policy_loss"]
        blob = {"state_dict": model.state_dict(), "config": model_config(model), "epoch": epoch + 1,
                "metrics": metrics}
        torch.save(blob, args.out / "last.pt")
        if sel < best:
            best = sel
            torch.save(blob, args.out / "model.pt")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", nargs="+", required=True, help="glob patterns of training shards")
    parser.add_argument("--val", nargs="+", required=True, help="glob patterns of validation shards (whole games)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=None, help="parse shards once into npz arrays here")
    parser.add_argument("--exclude-states", type=Path, default=None,
                        help="JSONL with a 'state' per line: positions equal to any of them are not used")
    parser.add_argument("--workers", type=int, default=16, help="parallel shard parsing (--cache-dir)")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init", type=Path, default=None, help="start from this model.pt (--cache-dir only)")
    parser.add_argument("--extra-npz", nargs="*", default=[],
                        help="labelled positions (dagger_cheqqers.py label), added to training (--cache-dir only)")
    parser.add_argument("--extra-weight", type=float, default=1.0, help="policy-loss weight of the --extra-npz rows")
    parser.add_argument("--move-head", choices=("sum", "ordered"), default="sum",
                        help="ordered: moves with the same squares in another order get different vectors (--dedup)")
    parser.add_argument("--dedup", action="store_true",
                        help="one row per unique position, soft targets; search labels override played moves")
    parser.add_argument("--extra-val-npz", nargs="*", default=[],
                        help="held-out labelled positions: top-1 against the label on those unseen in training")
    args = parser.parse_args(argv)
    frac = float(os.getenv("T33_GPU_FRAC", "0") or 0)
    if frac > 0 and str(args.device).startswith("cuda"):  # a GPU shared with other jobs (track 33 budget)
        torch.cuda.set_per_process_memory_fraction(frac)
    if args.cache_dir is not None:
        return train_dedup(args) if args.dedup else train_arrays(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val = load_positions(args.train), load_positions(args.val)
    print(f"train {len(train)} positions, val {len(val)} positions, device {args.device}", flush=True)
    model = CheqqersPolicyValue(d_model=args.d_model, layers=args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    for epoch in range(args.epochs):
        random.shuffle(train)
        started = time.time()
        for i in range(0, len(train), args.batch):
            sq, gl, idx, lens, target, value = to_batch(train[i:i + args.batch], args.device)
            logits, vlogits = model(sq, gl, idx, lens)
            mask = target >= 0
            vmask = value >= 0
            loss = logits.new_zeros(())
            if vmask.any():
                loss = loss + args.value_weight * F.cross_entropy(vlogits[vmask], value[vmask])
            if mask.any():
                pw = weight[mask] if weight is not None else None
                ce = F.cross_entropy(logits[mask], target[mask], reduction="none")
                loss = loss + (ce * pw).sum() / pw.sum() if pw is not None else loss + ce.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        metrics = {"epoch": epoch + 1, "seconds": round(time.time() - started, 1),
                   "train": evaluate(model, train[:20000], args.device, 512),
                   "val": evaluate(model, val, args.device, 512)}
        print(json.dumps(metrics), flush=True)
        with open(args.out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(metrics) + "\n")
        score = metrics["val"]["policy_loss"] + args.value_weight * metrics["val"]["value_loss"]
        if score < best:
            best = score
            torch.save({"state_dict": model.state_dict(),
                        "config": {"d_model": args.d_model, "layers": args.layers}}, args.out / "model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
