#!/usr/bin/env bash
# run_shard_t33b.sh (seed from T33_SEED, default 1114) <tree> <lib> <outdir> <comparison> <agent_a> <agent_b> <pair_start> <num_pairs> [cuda_devices]
# One shard of a colour-balanced series (seed 1114, 4-ply openings), like series/run_shard_net.sh, with a given library.
set -u
T=$1; L=$2; O=$3; C=$4; A=$5; B=$6; P0=$7; N=$8; DEV=${9:-}
mkdir -p "$O/$C" "$O/logs/$C"
export PATH="$HOME/.local/bin:$PATH"
export CHEQQERS_NATIVE_LIB="$L" CHEQQERS_NATIVE_BUILD=0 CUDA_VISIBLE_DEVICES="$DEV" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_NO_SYNC=1
unset CHEQQERS_VALUE_NET
cd "$T"
OUT="$O/$C/p$(printf %04d $P0).jsonl"
if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -ge $((2*N)) ]; then echo "skip $C $P0 (done)"; exit 0; fi
rm -f "$OUT"
timeout 4h uv run python src/arena/leiden_sparring_hub.py cheqqers --agent-a "$A" --agent-b "$B" \
  --num-pairs "$N" --pair-start "$P0" --seed ${T33_SEED:-1114} --opening-plies 4 --no-harvest \
  --hard-cases-path "$O/logs/$C/hard_unused.jsonl" --out "$OUT" > "$O/logs/$C/p$(printf %04d $P0).log" 2>&1
echo "done $C $P0 rc=$?"
