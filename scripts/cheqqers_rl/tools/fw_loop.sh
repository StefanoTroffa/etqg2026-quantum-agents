#!/usr/bin/env bash
# fw_loop.sh: FW-opponent self-play as a POOL of small independent processes (lead, 15 Nov 00:10). In one process a
# round waits for every pending FW move (2.2 s), so a process with many games runs them in lockstep (64 in flight:
# 227-276 games/h, 256: 154). Here FWN workers (default 10), each 1 C++ thread, nice 10, GPU memory cap 3%, FWPAR
# games in flight (workers alternate 1 and 2 unless FWPAR is set), FWRUN games per process start (16), the CURRENT
# model (models/model_<gen>.pt, sha256 logged) at each run start; FW at 2,200 ms, depth cap 20. Shards
# data/fw_k<gen>_w<worker>_r<run>_*.npz (only the learner's moves; values from the learner's side).
# One JSON line per run in gen/fw_runs.jsonl. This script = the manager: fw_loop.pid; workers fw_w<i>.pid.
# Stop: touch ${QG_T33:-$HOME/qg_t33}/gen/STOP (each worker stops after its run; the manager waits for all).
set -u
G=${QG_T33:-$HOME/qg_t33}/gen; mkdir -p $G/data
echo $$ > $G/fw_loop.pid
export PATH="$HOME/.local/bin:$PATH" CHEQQERS_NATIVE_LIB=${QG_T33:-$HOME/qg_t33}/lib/libcheqqers_t33_dev10.so CHEQQERS_NATIVE_BUILD=0 \
  OMP_NUM_THREADS=1 UV_NO_SYNC=1
cd ${QG_T33:-$HOME/qg_t33}/tree
worker() {
  local w=$1 par=$2 r=0
  echo $BASHPID > $G/fw_w$w.pid
  while [ ! -e $G/STOP ]; do
    local k=$(cat $G/gen); local M=$G/models/model_$(printf %03d $k).pt
    local sha=$(sha256sum $M | cut -c1-16) t0=$(date +%s)
    nice -n 10 uv run python src/training/gumbel_selfplay.py --model $M --games ${FWRUN:-16} --parallel $par \
      --out $G/data/fw_k$(printf %03d $k)_w$(printf %02d $w)_r$(printf %05d $r)_$(date +%s) --m 16 --n 128 --threads 1 \
      --seed $(( (w * 1000003 + r) * 7 + $(date +%s) )) --frozen-frac 0 --fw-frac 1.0 --fw-ms 2200 --fw-depth 20 \
      --gpu-mem-frac 0.03 > $G/fw_w$w.log 2>&1
    local rc=$?
    echo "{\"fw_worker\": $w, \"run\": $r, \"in_flight\": $par, \"gen\": $k, \"sha256\": \"$sha\", \"rc\": $rc, \"wall_s\": $(( $(date +%s) - t0 )), \"end\": $(date +%s), \"result\": $(tail -1 $G/fw_w$w.log | grep '^{' || echo null)}" >> $G/fw_runs.jsonl
    r=$((r + 1))
    [ $rc -ne 0 ] && sleep 60
  done
  unlink $G/fw_w$w.pid
}
pids=()
for ((w = 0; w < ${FWN:-10}; w++)); do
  par=${FWPAR:-$(( w % 2 + 1 ))}
  worker $w $par &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}"
unlink $G/fw_loop.pid
