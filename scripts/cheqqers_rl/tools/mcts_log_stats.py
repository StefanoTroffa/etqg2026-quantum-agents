"""mcts_log_stats.py <moves.jsonl> [agent]: evals / simulations / time per searched move (moves with >= 2 options)."""
import json
import statistics as st
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
if len(sys.argv) > 2:
    rows = [r for r in rows if r["agent"] == sys.argv[2]]
rows = [r for r in rows if r["moves"] > 1]
ev = [r["evals"] for r in rows]
print(f"{len(rows)} searched moves: evals/move median {st.median(ev):,.0f}, mean {st.mean(ev):,.0f}, "
      f"p10 {sorted(ev)[len(ev) // 10]:,}, min {min(ev):,}; sims/move median {st.median(r['sims'] for r in rows):,.0f}; "
      f"ms max {max(r['ms'] for r in rows):.0f}, moves over 2,100 ms {sum(r['ms'] > 2100 for r in rows)}")
