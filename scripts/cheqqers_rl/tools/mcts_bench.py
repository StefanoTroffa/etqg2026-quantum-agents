"""mcts_bench.py <model.pt> <pos_all.jsonl> <n_positions> <ms> <batch>...: simulations and network evaluations per
search of `ms` milliseconds (deadline 0.9 x ms, as the server) on proxy positions, per batch size."""
import json
import os
import sys
import time

REPO = os.environ.get("T33_TREE", os.path.expanduser("~/qg_t33/tree"))
sys.path.insert(0, os.path.join(REPO, "src"))
import torch  # noqa: E402

from search import native_cheqqers as nc  # noqa: E402
from search.gpu_mcts_cheqqers import MctsNet, MctsPlayer  # noqa: E402

model, pos_path, n, ms, *batches = sys.argv[1:]
pos = [json.loads(line) for line in open(pos_path)][: int(n)]
net = MctsNet(model)
print(f"device {net.device} {torch.cuda.get_device_name(0) if net.device.startswith('cuda') else ''}")
for b in map(int, batches):
    p = MctsPlayer(net, batch=b)
    p.search(nc.NativeEnv.from_flat(*pos[0]["state"]), pos[0]["moves"], time_ms=300)  # warm-up
    sims = evals = batches_ = 0
    el = 0.0
    for r in pos:
        res = p.search(nc.NativeEnv.from_flat(*r["state"]), r["moves"], time_ms=float(ms))
        sims += res.sims
        evals += res.evals
        batches_ += res.batches
        el += res.elapsed_ms
    k = len(pos)
    print(f"batch {b}: {sims / k:,.0f} sims/move, {evals / k:,.0f} evals/move, {evals / (el / 1000):,.0f} evals/s, "
          f"{batches_ / k:.0f} batches/move, {el / k:.0f} ms/move", flush=True)
