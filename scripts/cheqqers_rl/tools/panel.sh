#!/usr/bin/env bash
# panel.sh <tag> <model.pt> <parent.pt> [frozen.pt] [round4.pt]: the fixed evaluation panel (recipe section 5), each agent = MCTS
# with its network at 2,200 ms (PUCT, chance by expectation, CUDA; warmed up before the clock):
#   headline: vs FW 2,200 ms, 100 colour-balanced pairs (seed 1114, 4-ply openings; 40 until panel B);
#   vs the imitation parent (round 1), the DAgger round 4 and a frozen checkpoint, 10 pairs each (if given).
# At most 3 match processes (CPU cap). Appends one line per comparison to ${QG_T33:-$HOME/qg_t33}/gen/panel.txt.
set -u
T=$1; M=$2; PAR=$3; FRZ=${4:-}; R4=${5:-}
S=${QG_T33:-$HOME/qg_t33}/series
export PATH="$HOME/.local/bin:$PATH" T33_MCTS_LOG=$S/panel_${T}_moves.jsonl
LIB=${QG_T33:-$HOME/qg_t33}/lib/libcheqqers_t33_dev10.so
FW="FW:expecti_heuristic@depth=12,ms=2200,cpp=1,weights=config/cheqqers_eval_weights_2026-11-13.json"
A="P$T:expecti_heuristic@ms=2200,mcts=$M"
F=$S/tasks_panel_$T.txt
: > $F
for p in $(seq 0 4 96); do echo "P${T}_vs_FW|$A|$FW|$p|4|0" >> $F; done
for p in $(seq 0 5 5); do echo "P${T}_vs_PARENT|$A|PAR:expecti_heuristic@ms=2200,mcts=$PAR|$p|5|0" >> $F; done
if [ -n "$R4" ]; then
  for p in $(seq 0 5 5); do echo "P${T}_vs_ROUND4|$A|R4:expecti_heuristic@ms=2200,mcts=$R4|$p|5|0" >> $F; done
fi
if [ -n "$FRZ" ]; then
  for p in $(seq 0 5 5); do echo "P${T}_vs_FROZEN|$A|FRZ:expecti_heuristic@ms=2200,mcts=$FRZ|$p|5|0" >> $F; done
fi
(cd $S && flock ${QG_T33:-$HOME/qg_t33}/eval.lock bash ./run_queue_t33.sh ${QG_T33:-$HOME/qg_t33}/tree $LIB $S/out $F 3 > $S/queue_panel_$T.log 2>&1)
cd ${QG_T33:-$HOME/qg_t33}/tree
for c in FW PARENT ROUND4 FROZEN; do
  r=$(uv run python $S/summ.py $S/out P${T}_vs_$c 2>/dev/null | grep " all " | grep -oE "pairs= *[0-9]+ W/D/L=[0-9/]+ score=[0-9.]+ elo=[-+0-9]+ \[[-+0-9,]+\]|fb=[0-9/]+ nerr=[0-9/]+" | paste -sd" ")
  [ -n "$r" ] && echo "$(date -Is) panel $T ($M) vs $c: $r" >> ${QG_T33:-$HOME/qg_t33}/gen/panel.txt
done
echo "$(date -Is) panel $T evals/move: $(uv run python ${QG_T33:-$HOME/qg_t33}/tools/mcts_log_stats.py $T33_MCTS_LOG P$T 2>/dev/null)" >> ${QG_T33:-$HOME/qg_t33}/gen/panel.txt
# validity (lead, 14 Nov): every model's sha256 next to its result; fb (fallback moves) and nerr (native errors) per
# comparison are in the lines above and must be 0/0 for the panel to count
for f in "$M" "$PAR" ${FRZ:+"$FRZ"} ${R4:+"$R4"}; do
  echo "$(date -Is) panel $T model $(sha256sum "$f")" >> ${QG_T33:-$HOME/qg_t33}/gen/panel.txt
done
