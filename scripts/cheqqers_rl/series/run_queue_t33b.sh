#!/usr/bin/env bash
# run_queue_t33b.sh (seed from T33_SEED) <tree> <lib> <outdir> <taskfile> <parallel>
# task line: comparison|agent_a|agent_b|pair_start|num_pairs[|cuda_devices]
T=$1; L=$2; O=$3; F=$4; P=$5
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O/logs"
echo $$ > "$O/logs/queue_$(basename "$F").pid"
tr '\n' '\0' < "$F" | xargs -0 -P "$P" -I{} bash -c 'IFS="|" read -r C A B P0 N D <<< "$1"; "$2/run_shard_t33b.sh" "$3" "$4" "$5" "$C" "$A" "$B" "$P0" "$N" "$D"' _ {} "$S" "$T" "$L" "$O"
echo "queue $F finished $(date -Is)"
