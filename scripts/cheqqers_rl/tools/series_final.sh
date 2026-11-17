#!/usr/bin/env bash
# series_final.sh <model.pt> <batch> <c_puct> <fpu> [seed=1117] [pairs=200] [parallel=3]
# Track 33 final series under PLAN_rules (series/PLAN_t33_final_DRAFT.txt): MCTS(net) at 2,200 ms vs FW 2,200 ms.
#  1. pause the generation loop; nothing else on the GPU (nvidia-smi logged at the start and every 30 min);
#  2. freeze: read-only copies of the library and the network in ${QG_T33:-$HOME/qg_t33}/final/, sha256 of them and of the key code
#     files; the completed plan appended (timestamped) to series/PLAN_rules.txt BEFORE the first game;
#  3. pairs 0..N-1 on <seed>, shards of 4 pairs in ascending order, <= <parallel> games at a time;
#  4. looks only at 100, 150 and 200 complete pairs: stop early iff the 95% CI is entirely above or below 0;
#  5. sha256 re-checked at the end; validity (fallbacks, native errors, moves > 3,000 ms) in final/RESULT.txt.
set -u
M=$1; MB=$2; CP=$3; FPU=$4; SEED=${5:-1117}; NP=${6:-200}; PAR=${7:-3}
F=${T33_FINAL_DIR:-${QG_T33:-$HOME/qg_t33}/final}; S=${QG_T33:-$HOME/qg_t33}/series; G=${QG_T33:-$HOME/qg_t33}/gen; TREE=${QG_T33:-$HOME/qg_t33}/tree
PLAN=${T33_PLAN:-$S/PLAN_rules.txt}  # dry runs: T33_FINAL_DIR, T33_PLAN elsewhere and T33_NO_PAUSE=1
mkdir -p "$F"
export PATH="$HOME/.local/bin:$PATH"
[ "${T33_NO_PAUSE:-0}" = "1" ] || touch "$G/STOP"
[ "${T33_NO_PAUSE:-0}" = "1" ] || bash "${QG_T33:-$HOME/qg_t33}/tools/wait_gen_idle.sh"
# 2. freeze
cp "${QG_T33:-$HOME/qg_t33}/lib/libcheqqers_t33_dev10.so" "$F/libcheqqers_final.so"; chmod a-w "$F/libcheqqers_final.so"
cp "$M" "$F/net_final.pt"; chmod a-w "$F/net_final.pt"
( cd "$TREE" && sha256sum "$F/libcheqqers_final.so" "$F/net_final.pt" src/search/gpu_mcts_cheqqers.py \
    src/search/native_cheqqers.py src/models/cheqqers_policy_value.py src/arena/leiden_sparring_hub.py server.py \
    native/cheqqers/*.cpp native/cheqqers/*.h config/cheqqers_eval_weights_2026-11-13.json ) > "$F/sha256_before.txt"
A="T33:expecti_heuristic@ms=2200,mcts=$F/net_final.pt,mb=$MB,cpuct=$CP,fpu=$FPU"
B="FW:expecti_heuristic@depth=12,ms=2200,cpp=1,weights=config/cheqqers_eval_weights_2026-11-13.json"
{
  echo ""
  echo "=== Track 33 FINAL series (MCTS + network vs FW), written $(date -Is) before any game of this series ==="
  echo "Agents: T33 = $A"
  echo "        FW  = $B   (server default since track 30)"
  echo "Frozen: library $F/libcheqqers_final.so sha256 $(sha256sum "$F/libcheqqers_final.so" | cut -c1-64),"
  echo "  network $F/net_final.pt sha256 $(sha256sum "$F/net_final.pt" | cut -c1-64); code sha256 in $F/sha256_before.txt"
  echo "Method: 2,200 ms a move both sides, depth cap 12 for FW, 4-ply random openings, each opening once per colour, same"
  echo "  chance seed in a pair; seed $SEED (fresh: 1114 was used for model selection); pairs 0..$((NP - 1)); unit = pair;"
  echo "  Elo 95% CI = summarize_pairs. At most $PAR games at a time on the RTX 5090, nothing else on the GPU."
  echo "Rule: looks only at 100, 150 and $NP complete pairs; stop early iff the 95% CI lies entirely above or below 0;"
  echo "  otherwise play to $NP. Ship rule (lead amendment 2026-11-14): T33 (the RL agent) is primary and ships UNLESS it"
  echo "  clearly loses: upper end of the pair-level 95% CI below a 50% score (= Elo upper bound < 0) at the stop -> FW stays."
  echo "Validity before the result counts: fallbacks 0 both sides, native errors 0, no forfeits, sha256 unchanged at the end."
} >> "$PLAN"
# 1. GPU log
( while [ ! -e "$F/DONE" ]; do echo "$(date -Is) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> "$F/gpu.log"; sleep 1800; done ) &
# 3. tasks
C=T33_FINAL_vs_FW
T=$S/tasks_final.txt; : > "$T"
K=$(( NP < 4 ? NP : 4 ))
for ((p = 0; p < NP; p += K)); do echo "$C|$A|$B|$p|$K|0" >> "$T"; done
export T33_SEED=$SEED T33_MCTS_LOG=$F/moves.jsonl
(cd "$S" && bash ./run_queue_t33b.sh "$TREE" "$F/libcheqqers_final.so" "$F/out" "$T" "$PAR" > "$F/queue.log" 2>&1) &
QPID=$!
# 4. looks
look() {  # complete pairs 0..n-1: prints the summary line or nothing
  (cd "$TREE" && uv run python "${QG_T33:-$HOME/qg_t33}/tools/look.py" "$F/out" "$C" "$1" 2>/dev/null)
}
verdict=""
for n in $(for x in 100 150 $NP; do [ $x -le $NP ] && echo $x; done | sort -nu); do
  while [ -z "$(look $n)" ] && kill -0 $QPID 2>/dev/null; do sleep 60; done
  line=$(look $n)
  [ -z "$line" ] && break
  echo "$(date -Is) LOOK at $n pairs: $line" | tee -a "$F/looks.txt" >> "$PLAN"
  lo=$(echo "$line" | grep -oE "elo=[-+0-9]+ \[[-+0-9]+" | grep -oE "\[[-+0-9]+" | tr -d "[")
  hi=$(echo "$line" | grep -oE "\[[-+0-9]+,[-+0-9]+\]" | cut -d, -f2 | tr -d "]")
  if [ "$n" = "$NP" ] || [ "${lo:-0}" -gt 0 ] || [ "${hi:-0}" -lt 0 ]; then
    verdict="stop at $n pairs: CI [$lo,$hi]"
    break
  fi
done
kill $QPID 2>/dev/null; pkill -f "run_shard_t33b.sh .*$C" 2>/dev/null; sleep 5
# 5. re-check and report
( cd "$TREE" && sha256sum -c "$F/sha256_before.txt" ) > "$F/sha256_after.txt" 2>&1
all=$(cd "$TREE" && uv run python "$S/summ.py" "$F/out" "$C" 2>/dev/null | grep " all ")
over=$(cd "$TREE" && uv run python - "$F/out" <<'EOF'
import glob, json, sys
g = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*/*.jsonl") for l in open(f)]
print(sum(1 for x in g if max(x["slowest_ms_a"], x["slowest_ms_b"]) > 3000), "games with a move > 3,000 ms")
EOF
)
ev=$(cd "$TREE" && uv run python "${QG_T33:-$HOME/qg_t33}/tools/mcts_log_stats.py" "$F/moves.jsonl" T33 2>/dev/null)
{
  echo "$(date -Is) $verdict"
  bash "${QG_T33:-$HOME/qg_t33}/tools/ship_decision.sh" "$F" "$line"
  echo "complete-pair look: $line"
  echo "all games (incl. shards cut at the stop): $all"
  echo "evals/move: $ev"
  echo "slow moves: $over"
  echo "sha256 re-check: $(grep -c ': OK' "$F/sha256_after.txt") OK, $(grep -vc ': OK' "$F/sha256_after.txt") not OK"
} | tee "$F/RESULT.txt" >> "$PLAN"
touch "$F/DONE"
[ "${T33_NO_PAUSE:-0}" = "1" ] || unlink "$G/STOP"
