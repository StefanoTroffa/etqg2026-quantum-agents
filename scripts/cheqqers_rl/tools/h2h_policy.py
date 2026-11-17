"""h2h_policy.py <a.pt> <b.pt> <games> <seed>: policy-only head-to-head (no search, argmax), games batched in C++
(cq_sp_*), 2-6 random opening plies, colours alternate (game i: A white iff i even, same opening seed for i, i+1 via
paired seeds). Prints one JSON line: W/D/L for A, score, Elo."""
import ctypes
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("T33_TREE", os.path.expanduser("~/qg_t33/tree"))
sys.path.insert(0, os.path.join(REPO, "src"))
from models.cheqqers_policy_value import load_policy_value, split_flat  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402

pa, pb, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lib = nc.load()
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, B = load_policy_value(pa, dev), load_policy_value(pb, dev)
M = 256
SLEEP = float(os.environ.get("H2H_SLEEP_MS", "15")) / 1000.0
err = nc._err()
h = ctypes.c_void_p(lib.cq_sp_new(n, seed, 2, 6, 300, M, 4, err, nc._ERR))
x = np.zeros((n, 144), np.float32); mv = np.zeros((n, M, 3), np.int32); ln = np.zeros((n, M), np.int32)
ids = np.zeros(n, np.int32); turn = np.zeros(n, np.int32); ch = np.zeros(n, np.int32); mm = ctypes.c_int32()
P = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))  # noqa: E731
while True:
    r = lib.cq_sp_encode(h, P(x, ctypes.c_float), P(mv, ctypes.c_int32), P(ln, ctypes.c_int32), P(ids, ctypes.c_int32),
                         P(turn, ctypes.c_int32), ctypes.byref(mm), 4, err, nc._ERR)
    if r <= 0:
        break
    m = max(1, mm.value)
    a_moves = (ids[:r] % 2 == 0) == (turn[:r] == 0)  # A is white in even games
    for net, mask in ((A, a_moves), (B, ~a_moves)):
        rows = np.flatnonzero(mask)
        if not len(rows):
            continue
        with torch.no_grad():
            sq, gl = split_flat(torch.from_numpy(x[rows]).to(dev))
            lg, _ = net(sq, gl, torch.from_numpy(mv[rows, :m]).to(dev).long(), torch.from_numpy(ln[rows, :m]).to(dev).long())
        ch[rows] = lg.float().argmax(-1).cpu().numpy()
    lib.cq_sp_apply(h, P(ch, ctypes.c_int32), r, 4, err, nc._ERR)
    time.sleep(SLEEP)  # duty cycle (GPU power cap; the tracker drew ~405 W unthrottled)
res = np.zeros(n, np.int32); pl = np.zeros(n, np.int32)
lib.cq_sp_results(h, P(res, ctypes.c_int32), P(pl, ctypes.c_int32))
lib.cq_sp_free(h)
a_res = np.where(np.arange(n) % 2 == 0, res, -res)
w, d, l = int((a_res > 0).sum()), int((a_res == 0).sum()), int((a_res < 0).sum())
s = (w + 0.5 * d) / n
elo = -400 * math.log10(1 / min(max(s, 1e-3), 1 - 1e-3) - 1)
se = math.sqrt(max(s * (1 - s), 1e-6) / n)
lo = -400 * math.log10(1 / min(max(s - 1.96 * se, 1e-3), 1 - 1e-3) - 1)
hi = -400 * math.log10(1 / min(max(s + 1.96 * se, 1e-3), 1 - 1e-3) - 1)
print(json.dumps({"a": os.path.basename(pa), "b": os.path.basename(pb), "games": n, "wdl": [w, d, l],
                  "score": round(s, 4), "elo": round(elo), "ci95": [round(lo), round(hi)]}))
