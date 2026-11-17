"""look.py <out_dir> <comparison> <n>: summary of the complete pairs 0..n-1 (both games present), or nothing if any of
those pairs is incomplete. Same summary as summ.py (summarize_pairs), for any n."""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("T33_TREE", os.path.expanduser("~/qg_t33/tree")), "src"))
from arena.leiden_sparring_hub import read_series_files, summarize_pairs  # noqa: E402

out, comp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(f"{out}/{comp}/*.jsonl"))
recs = [r for r in read_series_files(files) if int(r["pair"].split(":")[1]) < n] if files else []
if len({r["pair"] for r in recs}) == n and len(recs) == 2 * n:
    s = summarize_pairs(recs)
    g = s["games"]
    print(f"{comp} 0..{n - 1} pairs={s['pairs']} W/D/L={g['won']}/{g['drawn']}/{g['lost']} score={s['score_a']:.3f} "
          f"elo={s['elo']:+.0f} [{s['elo_low']:+.0f},{s['elo_high']:+.0f}] fb={s['fallbacks_a']}/{s['fallbacks_b']} "
          f"nerr={s['native_errors_a']}/{s['native_errors_b']} slow={s['slowest_ms_a']:.0f}/{s['slowest_ms_b']:.0f}")
