#!/usr/bin/env bash
# setup_layout.sh [root]: build the working folder the Cheqqers training and series scripts expect
# (default root: $QG_T33, else ~/qg_t33). Every script under tools/ and series/ reads QG_T33.
#
#   root/tree        -> this repository (symlink)
#   root/tools       training loop, panels, final series (copied from scripts/cheqqers_rl/tools)
#   root/series      series runners (copied from scripts/cheqqers_rl/series)
#   root/lib         the native library the scripts load (libcheqqers_t33_dev10.so = native/cheqqers/libcheqqers.so)
#   root/candidates  checkpoints to compare (the shipped network is copied in as day_ckpt_00225.pt)
#   root/gen         the running training loop (data/, models/, ckpt/, logs, pidfiles, STOP)
#   root/dagger, loop, relabel, cache   training data: unpack cheqqers_training_data.tar into root
set -eu
ROOT=${1:-${QG_T33:-$HOME/qg_t33}}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT"/{tools,series,lib,candidates,final,gen/data,gen/models,cache,dagger/it1/split,loop/queue,relabel}
cp "$REPO"/scripts/cheqqers_rl/tools/* "$ROOT/tools/"
cp "$REPO"/scripts/cheqqers_rl/series/* "$ROOT/series/"
[ -e "$ROOT/tree" ] || ln -s "$REPO" "$ROOT/tree"
cp "$REPO/native/cheqqers/libcheqqers.so" "$ROOT/lib/libcheqqers_t33_dev10.so"
[ -e "$ROOT/candidates/day_ckpt_00225.pt" ] || cp "$REPO/config/cheqqers_net.pt" "$ROOT/candidates/day_ckpt_00225.pt"
echo "layout ready in $ROOT"
echo "next: tar -xf cheqqers_training_data.tar -C $ROOT   (FW labels, anchor positions, FW games, recent self-play, gen010/gen048)"
echo "then: export QG_T33=$ROOT  (every script reads it)"
