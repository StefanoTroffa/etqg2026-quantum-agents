#!/usr/bin/env bash
# gen_loop10.sh <start_model.pt>: expert iteration WITHOUT generation barriers (lead, 14 Nov night).
#   generator: ONE long-running gumbel_selfplay.py --continuous: every finished game restarts in its slot; every 300 s
#     it checks gen/latest.json and hot-swaps to a new trainer checkpoint for new searches (the shard is cut at the swap;
#     each shard's stats record the model sha256s that produced its rows; the frozen opponent is re-drawn from models/).
#     m 16, n 128, 20% vs frozen; PAR games in flight, TH threads, SLEEP ms duty cycle (GPU power cap); games start after
#     OPEN_LO..OPEN_HI random plies (default 2..6; 0..2 puts a third of the games on the real start position, 16 Nov).
#   trainer: start_trainer.sh (train_concurrent.py; FW-game pool 5%; resume from latest.json; replay cap 6).
#   fw_loop.sh: a pool of FWN small FW-opponent processes (2,200 ms, 1 thread each, nice 10), current model each run.
#   tracker: every TRACK_MIN minutes the newest checkpoint is snapshotted to models/model_<k>.pt (k = gen counter,
#     frozen pool + fw_loop + panels use it) and measured: policy-only h2h vs gen 10 / gen 48, value CE; games/h of the
#     generator over the window. One "tracker" line per snapshot in gens.jsonl.
# CPU peak: generator 6 + fw pool 10 x (1 thread + python) <= 14 + trainer 1 + eval matches 3 <= 24.
# Stop: touch ${QG_T33:-$HOME/qg_t33}/gen/STOP (all stop; wait_gen_idle.sh waits on the pidfiles).
set -u
G=${QG_T33:-$HOME/qg_t33}/gen; mkdir -p $G/data $G/models
PAR=${PAR:-1024}; TH=${TH:-6}; SLEEP=${SLEEP:-10}; TSLEEP=${TSLEEP:-150}; TRACK_MIN=${TRACK_MIN:-25}
export PATH="$HOME/.local/bin:$PATH" CHEQQERS_NATIVE_LIB=${QG_T33:-$HOME/qg_t33}/lib/libcheqqers_t33_dev10.so CHEQQERS_NATIVE_BUILD=0 \
  OMP_NUM_THREADS=1 UV_NO_SYNC=1
cd ${QG_T33:-$HOME/qg_t33}/tree
echo $$ > $G/gen_loop.pid
alive() { [ -s $G/$1 ] && kill -0 "$(cat $G/$1)" 2>/dev/null; }
k=$(cat $G/gen 2>/dev/null || echo 0)
[ -s $G/models/model_$(printf %03d $k).pt ] || cp "$1" $G/models/model_$(printf %03d $k).pt
alive trainer.pid || bash ${QG_T33:-$HOME/qg_t33}/tools/start_trainer.sh $G/models/model_$(printf %03d $k).pt $k
alive fw_loop.pid || (nohup bash ${QG_T33:-$HOME/qg_t33}/tools/fw_loop.sh >> $G/fw_loop.out 2>&1 < /dev/null &)
# the generator starts on the newest checkpoint if there is one
M=$G/models/model_$(printf %03d $k).pt
[ -s $G/latest.json ] && M=$(uv run python -c "import json; print(json.load(open('$G/latest.json'))['path'])")
FRZ=$G/models/model_$(printf %03d $((RANDOM % (k + 1)))).pt
uv run python src/training/gumbel_selfplay.py --continuous $G --model $M --frozen-model $FRZ --frozen-frac 0.2 \
  --games 1000000 --parallel $PAR --out $G/data/gc$(date +%s) --m 16 --n 128 --threads $TH --sleep-ms $SLEEP \
  --swap-every 300 --gpu-mem-frac ${GMEM:-0.45} --seed $(date +%s) --open-lo ${OPEN_LO:-2} --open-hi ${OPEN_HI:-6} > $G/selfplay_cont.log 2>&1 &
GP=$!
echo $GP > $G/generator.pid
games_at() { tail -200 $G/selfplay_cont.log | grep '"games_per_hour"' | tail -1 | grep -oE '"(games|seconds)": [0-9]+' | grep -oE '[0-9]+' | paste -sd' '; }
while kill -0 $GP 2>/dev/null && [ ! -e $G/STOP ]; do
  w0=$(games_at); t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $((TRACK_MIN * 60)) ] && kill -0 $GP 2>/dev/null && [ ! -e $G/STOP ]; do sleep 15; done
  [ -e $G/STOP ] && break
  kill -0 $GP 2>/dev/null || break
  w1=$(games_at)
  k=$((k + 1)); N=$G/models/model_$(printf %03d $k).pt
  rec=$(cat $G/latest.json 2>/dev/null)
  src=$(echo "$rec" | uv run python -c "import json,sys; print(json.load(sys.stdin)['path'])" 2>/dev/null)
  [ -n "$src" ] && [ -s "$src" ] || src=$G/models/model_$(printf %03d $((k - 1))).pt
  cp "$src" $N.tmp && mv $N.tmp $N && echo $k > $G/gen
  h1=$(uv run python ${QG_T33:-$HOME/qg_t33}/tools/h2h_policy.py $N ${QG_T33:-$HOME/qg_t33}/candidates/gen010.pt 1000 $((k * 13)) 2>/dev/null | tail -1)
  h2=$(uv run python ${QG_T33:-$HOME/qg_t33}/tools/h2h_policy.py $N ${QG_T33:-$HOME/qg_t33}/candidates/gen048.pt 1000 $((k * 17)) 2>/dev/null | tail -1)
  vce=$(T33_TREE=${QG_T33:-$HOME/qg_t33}/tree uv run python ${QG_T33:-$HOME/qg_t33}/tools/value_drift.py ${QG_T33:-$HOME/qg_t33}/cache $N 2>/dev/null | grep -o "value CE [0-9.]*" | cut -d" " -f3)
  gph=$(echo "$w0 $w1" | awk '{ if (NF == 4 && $4 > $2) printf "%d", ($3 - $1) / ($4 - $2) * 3600; else print "null" }')
  echo "{\"tracker\": $k, \"model_sha256\": \"$(sha256sum $N | cut -c1-16)\", \"from_ckpt\": \"$(basename "$src")\", \"window_min\": $TRACK_MIN, \"selfplay_games_per_hour\": $gph, \"fw_runs_total\": $(wc -l < $G/fw_runs.jsonl 2>/dev/null || echo 0), \"trainer\": ${rec:-null}, \"vs_gen010\": ${h1:-null}, \"vs_gen048\": ${h2:-null}, \"value_ce\": ${vce:-null}}" >> $G/gens.jsonl
done
[ -e $G/STOP ] && wait $GP
unlink $G/gen_loop.pid
