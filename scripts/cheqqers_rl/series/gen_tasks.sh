#!/usr/bin/env bash
# gen_tasks.sh <label> <model.pt> <opponent: rnd|fw> <pairs> <pairs_per_shard> [extra agent options] > tasks.txt
# Agent A = the policy player (pv=<model>), B = random or FW at 2,200 ms (the shipped agent).
L=$1; M=$2; O=$3; N=$4; K=$5; X=${6:-}
FW='FW:expecti_heuristic@depth=12,ms=2200,cpp=1,weights=config/cheqqers_eval_weights_2026-11-13.json'
RND='RND:random'
B=$FW; [ "$O" = rnd ] && B=$RND
for ((p = 0; p < N; p += K)); do
  echo "${L}_vs_${O^^}|${L}:expecti_heuristic@pv=${M}${X}|${B}|${p}|${K}|"
done
