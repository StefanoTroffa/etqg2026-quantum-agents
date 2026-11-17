"""One training step of the Cheqqers expert-iteration loop (recipe docs/specs/cheqqers_rl_recipe.md, section 4).

    uv run python src/training/train_generation.py --init gen/model_003.pt --out gen/model_004.pt \
        --selfplay 'gen/g*.npz' --fw 'labels/*_1000ms.npz' --fw-frac 0.2 --samples 300000

Buffer: the newest self-play shards (gumbel_selfplay.py) up to --buffer positions (sliding window over generations),
duplicates aggregated with counts (Dedup: the policy targets summed, the results counted). FW-labelled positions
(dagger_cheqqers.py label) are a fixed share of every batch (--fw-frac; the caller decays it from 0.2 to 0.05-0.1).
Losses: self-play rows: cross-entropy to the (aggregated) policy target and to the realised result's distribution
(win / draw / loss for the side to move); FW rows: cross-entropy to the label, no value. AdamW, lr 1e-4, weight
decay 1e-4, batch 1,024, gradient clip 1, bf16. --samples rows are drawn (about one pass over the new data).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.cheqqers_policy_value import load_policy_value, model_config  # noqa: E402
from training.train_cheqqers_transformer import Arrays, Dedup, position_keys, soft_ce  # noqa: E402


def buffer_files(patterns: Sequence[str], limit: int) -> list[str]:
    """Newest files first until `limit` rows."""
    files = sorted({f for p in patterns for f in glob.glob(p)}, key=os.path.getmtime, reverse=True)
    out, rows = [], 0
    for f in files:
        if rows >= limit:
            break
        out.append(f)
        rows += len(np.load(f)["value"])
    return out


def train_step(model, opt, pools, rng, batch: int, device: str, stats: dict) -> float:
    """One optimiser step on a batch drawn from (pool, rows, kind) triples: kind "sp" = policy + value, "fw" = policy
    only, "an" = value only. Adds the per-kind losses to `stats`; returns the total loss (float)."""
    use_amp = device.startswith("cuda")
    loss = torch.zeros((), device=device)
    for pool, k, kind in pools:
        if pool is None or k == 0:
            continue
        idx = rng.integers(0, pool.u, size=k)
        sq, gl, mi, ml, pt, vt, _ = pool.batch(idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg, vl = model(sq, gl, mi, ml)
        lg, vl = lg.float(), vl.float()
        pm = pt.sum(-1) > 0
        if kind == "an":  # anchor rows train the value head only
            vm = vt.sum(-1) > 0
            if vm.any():
                vls = soft_ce(vl[vm], vt[vm]).mean()
                loss = loss + vls * (k / batch)
                stats["an_val"] = stats.get("an_val", 0.0) + vls.item()
            continue
        if pm.any():
            pl = soft_ce(lg[pm], pt[pm]).mean()
            loss = loss + pl * (k / batch)
            stats[f"{kind}_pol"] = stats.get(f"{kind}_pol", 0.0) + pl.item()
        vm = vt.sum(-1) > 0
        if kind in ("sp", "fg") and vm.any():  # fg: self-play games against FW (train_concurrent's separate pool)
            vls = soft_ce(vl[vm], vt[vm]).mean()
            loss = loss + vls * (k / batch)
            stats[f"{kind}_val"] = stats.get(f"{kind}_val", 0.0) + vls.item()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selfplay", nargs="+", required=True)
    parser.add_argument("--fw", nargs="*", default=[])
    parser.add_argument("--fw-frac", type=float, default=0.2)
    parser.add_argument("--buffer", type=int, default=1_000_000)
    parser.add_argument("--value-anchor", nargs="*", default=[],
                        help="npz shards with real game results (e.g. the cached FW self-play): value loss only")
    parser.add_argument("--value-anchor-frac", type=float, default=0.0, help="share of each batch from --value-anchor")
    parser.add_argument("--samples", type=int, default=0, help="rows drawn; 0 = the newest shard's rows")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    sp_files = buffer_files(args.selfplay, args.buffer)
    sp_raw = Arrays(sp_files)
    sp = Dedup(sp_raw, position_keys(sp_raw), 1.0)
    fw = None
    fw_files = sorted({f for p in args.fw for f in glob.glob(p)})
    if fw_files and args.fw_frac > 0:
        fw_raw = Arrays(fw_files)
        fw = Dedup(fw_raw, position_keys(fw_raw), 1.0)
    anchor = None
    anchor_files = sorted({f for p in args.value_anchor for f in glob.glob(p)})
    if anchor_files and args.value_anchor_frac > 0:  # positions with real game results: value loss only
        an_raw = Arrays(anchor_files)
        anchor = Dedup(an_raw, position_keys(an_raw), 1.0)
    samples = args.samples or len(np.load(sp_files[0])["value"])
    model = load_policy_value(args.init, args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_fw = int(round(args.batch * args.fw_frac)) if fw is not None else 0
    n_an = int(round(args.batch * args.value_anchor_frac)) if anchor is not None else 0
    n_sp = args.batch - n_fw - n_an
    steps = max(1, samples // args.batch)
    stats = {"sp_pol": 0.0, "sp_val": 0.0, "fw_pol": 0.0, "an_val": 0.0}
    for step in range(steps):
        train_step(model, opt, ((sp, n_sp, "sp"), (fw, n_fw, "fw"), (anchor, n_an, "an")), rng, args.batch,
                   args.device, stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model_config(model), "init": str(args.init)}, args.out)
    rec = {"out": str(args.out), "steps": steps, "samples": steps * args.batch, "buffer_files": len(sp_files),
           "buffer_rows": sp_raw.n, "buffer_unique": sp.u, "fw_unique": fw.u if fw is not None else 0,
           "fw_frac": args.fw_frac if fw is not None else 0.0, "anchor_frac": args.value_anchor_frac if anchor else 0.0,
           "lr": args.lr, "seconds": round(time.time() - t0),
           **{k: v / steps for k, v in stats.items()}}
    print(json.dumps(rec), flush=True)
    with open(str(args.out) + ".json", "w") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
