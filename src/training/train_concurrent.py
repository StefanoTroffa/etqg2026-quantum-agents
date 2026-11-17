"""Concurrent trainer for the Cheqqers expert-iteration loop (recipe docs/specs/cheqqers_rl_recipe.md, section 4).

    uv run python src/training/train_concurrent.py --init gen/models/model_110.pt --dir gen \
        --selfplay 'gen/data/g*.npz' 'gen/data/fw_*.npz' --fw 'labels/*_1000ms.npz' --fw-frac 0.075

Runs beside the self-play generators on the same GPU, instead of a sequential training step per generation:
* buffer: the newest --buffer self-play rows (train_generation.buffer_files), re-scanned every --rescan seconds;
* the same step as train_generation (train_step: AdamW, bf16, clip 1, policy + value / FW policy / value anchor);
* replay ratio capped: rows trained <= --max-reuse x new self-play rows seen since the start (+ --warm-rows), sleeps
  when ahead;
* a checkpoint every --save-every seconds: <dir>/ckpt/ckpt_<n>.pt written atomically, then <dir>/latest.json
  (path, sha256, steps, rows trained, new rows, reuse, mean losses) replaced atomically; one line per checkpoint in
  <dir>/trainer.jsonl. Generators read latest.json at each generation start.
* validity: a non-finite loss skips the step (counted); 20 in a row stop the trainer with exit code 3.
Stop: touch <dir>/STOP_TRAINER (trainer only) or <dir>/STOP (the whole loop, e.g. for a panel).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.cheqqers_policy_value import load_policy_value, model_config  # noqa: E402
from training.train_cheqqers_transformer import Arrays, Dedup, position_keys  # noqa: E402
from training.train_generation import buffer_files, train_step  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(files: Sequence[str]) -> dict:
    return {f: len(np.load(f)["value"]) for f in files}


def atomic_json(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--selfplay", nargs="+", required=True)
    parser.add_argument("--fw", nargs="*", default=[])
    parser.add_argument("--fw-frac", type=float, default=0.075)
    parser.add_argument("--value-anchor", nargs="*", default=[])
    parser.add_argument("--value-anchor-frac", type=float, default=0.0)
    parser.add_argument("--buffer", type=int, default=1_000_000)
    parser.add_argument("--rescan", type=float, default=60.0)
    parser.add_argument("--save-every", type=float, default=150.0)
    parser.add_argument("--max-reuse", type=float, default=6.0, help="rows trained per new self-play row (cap)")
    parser.add_argument("--warm-rows", type=int, default=0, help="new-row credit at the start")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep", type=int, default=12, help="checkpoints kept (plus every 20th)")
    parser.add_argument("--fwgame", nargs="*", default=[], help="self-play games against FW (fw_loop shards): a "
                        "separate pool, never evicted by age, newest --fwgame-max rows, a fixed --fwgame-frac of each batch")
    parser.add_argument("--fwgame-frac", type=float, default=0.05)
    parser.add_argument("--fwgame-max", type=int, default=200_000)
    parser.add_argument("--fwgame-frac-early", type=float, default=0.02,
                        help="share while the pool holds fewer than --fwgame-full-rows rows (a small pool memorises)")
    parser.add_argument("--fwgame-full-rows", type=int, default=100_000)
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="pause after each step (GPU power cap)")
    parser.add_argument("--resume", default="", help="latest.json of a previous run: weights, AdamW state, RNG, "
                        "step and replay counters restored (a pause for a panel does not reset the moments)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many steps in total (tests)")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (0 = until STOP_TRAINER)")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") and args.gpu_mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ck = args.dir / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    resume = json.loads(Path(args.resume).read_text()) if args.resume and Path(args.resume).exists() else None
    model = load_policy_value(resume["path"] if resume else args.init, args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # on CPU: the RNG states must stay CPU ByteTensors; AdamW.load_state_dict moves its moments to the parameters' device
    state = torch.load(resume["state"], map_location="cpu", weights_only=False) if resume else None
    if state is not None:
        opt.load_state_dict(state["opt"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def pool(files):
        raw = Arrays(files)
        return Dedup(raw, position_keys(raw), 1.0), raw.n

    fw = an = None
    fw_files = sorted({f for p in args.fw for f in glob.glob(p)})
    if fw_files and args.fw_frac > 0:
        fw, _ = pool(fw_files)
    an_files = sorted({f for p in args.value_anchor for f in glob.glob(p)})
    if an_files and args.value_anchor_frac > 0:
        an, _ = pool(an_files)
    n_fw = int(round(args.batch * args.fw_frac)) if fw is not None else 0
    n_an = int(round(args.batch * args.value_anchor_frac)) if an is not None else 0
    fg = None
    fg_files: list[str] = []
    fg_rows = 0
    n_fg = 0
    fg_was_full = None  # the first scan logs the starting share
    switches: list[dict] = []
    n_sp_full = args.batch - n_fw - n_an

    seen = set(f for p in args.selfplay for f in glob.glob(p))  # present at the start: not new
    new_rows = 0
    if state is not None:  # the replay counters continue: files seen before the pause stay seen, new ones count
        seen = set(state["seen"])
        new_rows = state["new_rows"]
    sp_files: list[str] = []
    sp = None
    buf_rows = 0
    t_scan = t_save = 0.0
    t_start = time.time()
    steps = skipped = bad_run = n_ck = slept = 0
    if state is not None:
        steps, skipped, n_ck = state["steps"], state["skipped"], state["n_ck"]
    stats = {"sp_pol": 0.0, "sp_val": 0.0, "fw_pol": 0.0, "an_val": 0.0}
    st_steps = 0
    def save(n: int) -> int:
        """Checkpoint n: weights (ckpt_n.pt, what generators load) + trainer state (ckpt_n.state.pt: AdamW moments and
        step counts, RNGs, replay counters), then latest.json; both files written then renamed."""
        out, st = ck / f"ckpt_{n:05d}.pt", ck / f"ckpt_{n:05d}.state.pt"
        for path, obj in ((out, {"state_dict": model.state_dict(), "config": model_config(model), "init": str(args.init),
                                 "steps": steps}),
                          (st, {"opt": opt.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
                                "seen": sorted(seen), "new_rows": new_rows, "steps": steps, "skipped": skipped,
                                "n_ck": n})):
            tmp = path.with_suffix(".tmp")
            torch.save(obj, tmp)
            os.replace(tmp, path)
        rec = {"ckpt": n, "path": str(out), "state": str(st), "sha256": sha256(out), "time": round(time.time()),
               "steps": steps, "rows_trained": steps * args.batch, "new_rows": new_rows,
               "reuse": round(steps * args.batch / max(1, new_rows), 2), "skipped": skipped,
               "buffer_rows": buf_rows, "buffer_files": len(sp_files), "buffer_unique": sp.u if sp is not None else 0,
               "fw_shards_in_buffer": sum(1 for f in sp_files if os.path.basename(f).startswith("fw_")),
               "fwgame_files": len(fg_files), "fwgame_rows": fg_rows, "fwgame_unique": fg.u if fg is not None else 0,
               "fwgame_batch_share": round(n_fg / args.batch, 4) if fg is not None else 0.0,
               "fwgame_share_switches": switches,
               "slept_s": slept, "elapsed_s": round(time.time() - t_start), "resumed_from": resume["ckpt"] if resume else None,
               **{k: round(v / max(1, st_steps), 4) for k, v in stats.items()}}
        atomic_json(args.dir / "latest.json", rec)
        old = n - args.keep  # retention: the last --keep checkpoints and every 20th (generators copy theirs)
        if old > 0 and old % 20:
            (ck / f"ckpt_{old:05d}.pt").unlink(missing_ok=True)
            (ck / f"ckpt_{old:05d}.state.pt").unlink(missing_ok=True)
        with open(args.dir / "trainer.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        return n

    while not any((args.dir / f).exists() for f in ("STOP_TRAINER", "STOP")):  # STOP: panels / series pause all
        now = time.time()
        if (args.max_seconds and now - t_start > args.max_seconds) or (args.max_steps and steps >= args.max_steps):
            break
        if now - t_scan >= args.rescan or sp is None:
            files = sorted({f for p in args.selfplay for f in glob.glob(p)})
            fresh = [f for f in files if f not in seen]
            if fresh:
                time.sleep(2)  # a shard just renamed in: let np.savez finish (it writes then closes)
                try:
                    new_rows += sum(rows_of(fresh).values())
                    seen.update(fresh)
                except Exception as exc:  # a shard still being written: retry at the next scan
                    print(json.dumps({"scan_retry": str(exc)[:120]}), flush=True)
            try:
                nb = buffer_files(args.selfplay, args.buffer)
            except Exception as exc:
                print(json.dumps({"buffer_retry": str(exc)[:120]}), flush=True)
                nb = sp_files
            if nb != sp_files:
                try:
                    sp, buf_rows = pool(nb)
                    sp_files = nb
                except Exception as exc:
                    print(json.dumps({"pool_retry": str(exc)[:120]}), flush=True)
            t_scan = now
            if args.fwgame:
                try:
                    nf = buffer_files(args.fwgame, args.fwgame_max)
                except Exception as exc:
                    print(json.dumps({"fwgame_retry": str(exc)[:120]}), flush=True)
                    nf = fg_files
                if nf and nf != fg_files:
                    try:
                        fg, fg_rows = pool(nf)
                        fg_files = nf
                    except Exception as exc:
                        print(json.dumps({"fwgame_pool_retry": str(exc)[:120]}), flush=True)
        if sp is None:
            time.sleep(5)
            continue
        fg_full = fg is not None and fg_rows >= args.fwgame_full_rows
        if fg is not None and fg_full != fg_was_full:
            print(json.dumps({"fwgame_share_switch": args.fwgame_frac if fg_full else args.fwgame_frac_early,
                              "fwgame_rows": fg_rows, "steps": steps, "time": round(time.time())}), flush=True)
            switches.append({"to": args.fwgame_frac if fg_full else args.fwgame_frac_early, "rows": fg_rows, "steps": steps})
            fg_was_full = fg_full
        n_fg = int(round(args.batch * (args.fwgame_frac if fg_full else args.fwgame_frac_early))) if fg is not None else 0
        n_sp = n_sp_full - n_fg
        if steps * args.batch >= args.max_reuse * (new_rows + args.warm_rows):
            time.sleep(2)
            slept += 2
        else:
            loss = train_step(model, opt, ((sp, n_sp, "sp"), (fg, n_fg, "fg"), (fw, n_fw, "fw"), (an, n_an, "an")), rng, args.batch,
                              args.device, stats)
            if not math.isfinite(loss):
                skipped += 1
                bad_run += 1
                if bad_run >= 20:
                    print(json.dumps({"error": "20 non-finite losses in a row"}), flush=True)
                    return 3
                continue
            bad_run = 0
            steps += 1
            st_steps += 1
            if args.sleep_ms > 0:  # duty cycle (GPU power cap): full-speed bursts drew ~395 W on their own
                time.sleep(args.sleep_ms / 1000.0)
        if time.time() - t_save >= args.save_every and st_steps > 0:
            n_ck = save(n_ck + 1)
            stats = {k: 0.0 for k in stats}
            st_steps = 0
            t_save = time.time()
    if st_steps > 0 or args.max_steps:  # a final checkpoint at the stop: the restart continues from exactly here
        save(n_ck + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
