"""summ.py <out_dir> [<comparison> ...]: per comparison, summary over all complete pairs and over pairs 0..N-1 for N in 50,100,150,200."""
import glob, sys, os
sys.path.insert(0, os.path.join(os.environ.get("T33_TREE", os.path.expanduser("~/qg_t33/tree")), "src"))
from arena.leiden_sparring_hub import read_series_files, summarize_pairs

out = sys.argv[1]
comps = sys.argv[2:] or sorted(os.listdir(out))
for c in comps:
    files = sorted(glob.glob(f"{out}/{c}/*.jsonl"))
    if not files:
        continue
    recs = read_series_files(files)
    def show(label, rs):
        s = summarize_pairs(rs)
        g = s["games"]
        print(f"{c:16s} {label:8s} pairs={s['pairs']:3d} W/D/L={g['won']}/{g['drawn']}/{g['lost']} score={s['score_a']:.3f} "
              f"elo={s['elo']:+.0f} [{s['elo_low']:+.0f},{s['elo_high']:+.0f}] 2-0/1-1/0-2={s['pair_counts']['2-0']}/{s['pair_counts']['1-1']}/{s['pair_counts']['0-2']} "
              f"slow_a={s['slowest_ms_a']:.0f} slow_b={s['slowest_ms_b']:.0f} fb={s['fallbacks_a']}/{s['fallbacks_b']} "
              f"nerr={s['native_errors_a']}/{s['native_errors_b']} depth={s['mean_depth_a']}/{s['mean_depth_b']}")
    show("all", recs)
    for n in (50, 100, 150, 200):
        sub = [r for r in recs if int(r["pair"].split(":")[1]) < n]
        pairs = {r["pair"] for r in sub}
        if len(pairs) == n and len(sub) == 2 * n:
            show(f"0..{n-1}", sub)
