"""value_drift.py <cache_dir> <model.pt>...: the value head's cross-entropy (win / draw / loss, side to move) and accuracy
on ONE fixed held-out set, the validation shards selfplay_fw/shard_15 + selfplay/shard_09 (every position of finished
games, as the trainer caches them). A loss rising across generations = the value head drifting."""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("T33_TREE", os.path.expanduser("~/qg_t33/tree"))
sys.path.insert(0, os.path.join(REPO, "src"))
from models.cheqqers_policy_value import load_policy_value  # noqa: E402
from training.train_cheqqers_transformer import Arrays  # noqa: E402

cache, *models = sys.argv[1:]
SLEEP = float(os.environ.get("VD_SLEEP_MS", "5")) / 1000.0
files = sorted(f for f in glob.glob(f"{cache}/*.npz") if "selfplay_fw_shard_15" in f or "selfplay_shard_09" in f)
val = Arrays(files)
print(f"held-out: {[os.path.basename(f) for f in files]}, {val.n} positions, value classes "
      f"{np.bincount(val.value, minlength=3).tolist()} (win/draw/loss for the mover)")
for m in models:
    net = load_policy_value(m, "cuda")
    loss = hit = 0.0
    with torch.no_grad():
        for s in range(0, val.n, 512):
            idx = np.arange(s, min(s + 512, val.n))
            sq, gl, mi, ml, _, vt = val.batch(idx, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, vl = net(sq, gl, mi, ml)
            vl = vl.float()
            loss += F.cross_entropy(vl, vt, reduction="sum").item()
            hit += (vl.argmax(-1) == vt).sum().item()
            time.sleep(SLEEP)  # duty cycle (GPU power cap: an unthrottled pass spiked to ~450 W)
    print(f"{os.path.basename(m)}: value CE {loss / val.n:.4f}, accuracy {hit / val.n:.4f}", flush=True)
