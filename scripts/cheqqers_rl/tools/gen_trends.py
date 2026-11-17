"""gen_trends.py [gens.jsonl]: per generation: games, samples, samples/game, evals/sample, games/hour, training losses."""
import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/qg_t33/gen/gens.jsonl")
print("gen games samples s/game evals/s games/h sp_pol sp_val fw_pol")
for line in open(path):
    r = json.loads(line)
    s, t = r["selfplay"], r["train"]
    print(r["gen"], s["games"], s["samples"], round(s["samples"] / s["games"], 1), round(s["evals"] / s["samples"], 1),
          s["games_per_hour"], round(t.get("sp_pol", 0), 3), round(t.get("sp_val", 0), 3), round(t.get("fw_pol", 0), 3))
