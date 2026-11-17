#!/usr/bin/env bash
# start_trainer.sh <init.pt> <seed>: the concurrent trainer (train_concurrent.py) in the background, pidfile gen/trainer.pid.
# Buffer: newest 1M rows of main self-play (data/g*.npz); FW-game pool (data/fw_*.npz, newest 200k rows, never
# age-evicted by main self-play) at 5% of each batch (lead, 15 Nov); FW labels 7.5%; value anchor 10%; replay cap 6;
# checkpoint every 150 s; resumes weights + AdamW + RNG + counters from gen/latest.json if present; TSLEEP ms a step.
G=${QG_T33:-$HOME/qg_t33}/gen
cd ${QG_T33:-$HOME/qg_t33}/tree
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1 UV_NO_SYNC=1
nohup bash -c 'echo $$ > '$G'/trainer.pid; exec uv run python src/training/train_concurrent.py --init '$1' --dir '$G' \
  --selfplay "'$G'/data/g*.npz" --fwgame "'$G'/data/fw_*.npz" --fwgame-frac 0.05 --fwgame-frac-early 0.02 --fwgame-full-rows 100000 --fwgame-max 200000 \
  --fw ${QG_T33:-$HOME/qg_t33}/dagger/it1/split/train_*_1000ms.npz "${QG_T33:-$HOME/qg_t33}/loop/queue/*_1000ms.npz" "${QG_T33:-$HOME/qg_t33}/relabel/*_1000ms.npz" \
  --fw-frac 0.075 --value-anchor "${QG_T33:-$HOME/qg_t33}/cache/selfplay_fw_shard_0*_aa785ebbe6.npz" "${QG_T33:-$HOME/qg_t33}/cache/selfplay_fw_shard_1[0-4]_aa785ebbe6.npz" \
  --value-anchor-frac 0.1 --max-reuse 6 --save-every 150 --rescan 60 --gpu-mem-frac 0.2 --sleep-ms '${TSLEEP:-150}' \
  --resume '$G'/latest.json --seed '$2 >> $G/trainer.out 2>&1 < /dev/null &
