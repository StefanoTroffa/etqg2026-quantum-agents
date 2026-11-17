#!/usr/bin/env bash
# Builds the native Cheqqers engine (native/cheqqers/*.cpp) into native/cheqqers/libcheqqers.so.
#
#   bash scripts/build_native.sh            # portable x86-64 binary (no -march=native: it may run on another CPU)
#   CQ_MARCH=native bash scripts/build_native.sh   # tuned for this CPU, for measurement only
#
# -ffp-contract=off and no -ffast-math are part of the contract: the engine reproduces the Python's floating-point
# operations one by one, and a fused multiply-add would change the last bits.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$here/native/cheqqers"
out="${CQ_OUT:-$src/libcheqqers.so}"
cxx="${CXX:-g++}"
march=()
if [[ -n "${CQ_MARCH:-}" ]]; then march=("-march=${CQ_MARCH}"); fi
tmp="$out.tmp.$$"
"$cxx" -O3 -std=c++17 -shared -fPIC -ffp-contract=off -fno-fast-math -fvisibility=hidden -fvisibility-inlines-hidden -fno-semantic-interposition "${march[@]}" \
    -Wall -Wextra -Wno-unused-parameter \
    "$src/engine.cpp" "$src/search.cpp" "$src/api.cpp" "$src/value_net.cpp" "$src/mcts.cpp" -lpthread -o "$tmp"
mv -f "$tmp" "$out"
echo "built $out"
