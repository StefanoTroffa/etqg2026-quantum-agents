#!/usr/bin/env bash
# ship_decision.sh <final_dir> "<complete-pair look line or empty>": the series' SHIP DECISION line, validity first
# (lead, 14 Nov): INVALID if no complete planned look, fb > 0, nerr > 0, any sha256 re-check not OK, or any move
# > 2,100 ms (either side); otherwise T33 ships unless the Elo (pair-level score) 95% upper bound < 0.
F=$1; line=$2; why=""
[ -z "$line" ] && why="$why no complete planned look;"
fb=$(echo "$line" | grep -oE "fb=[0-9]+/[0-9]+" | cut -d= -f2)
ne=$(echo "$line" | grep -oE "nerr=[0-9]+/[0-9]+" | cut -d= -f2)
sl=$(echo "$line" | grep -oE "slow=[0-9]+/[0-9]+" | cut -d= -f2)
[ -n "$line" ] && [ "$fb" != "0/0" ] && why="$why fallbacks $fb;"
[ -n "$line" ] && [ "$ne" != "0/0" ] && why="$why native errors $ne;"
for v in ${sl/\// }; do [ "$v" -gt 2100 ] && why="$why move ${v} ms > 2,100;"; done
bad=$(grep -vc ': OK' "$F/sha256_after.txt" 2>/dev/null); ok=$(grep -c ': OK' "$F/sha256_after.txt" 2>/dev/null)
{ [ ! -s "$F/sha256_after.txt" ] || [ "${bad:-1}" -gt 0 ] || [ "${ok:-0}" -eq 0 ]; } && why="$why sha256 re-check: ${bad:-?} not OK;"
if [ -n "$why" ]; then echo "SHIP DECISION: INVALID (${why% ;}) - FW stays until a valid series"; exit 0; fi
hi=$(echo "$line" | grep -oE "\[[-+0-9]+,[-+0-9]+\]" | cut -d, -f2 | tr -d "]+")
if [ "$hi" -lt 0 ]; then echo "SHIP DECISION: FW stays (T33 clearly loses: Elo upper bound $hi < 0)"
else echo "SHIP DECISION: T33 ships (valid series; Elo upper bound $hi >= 0; rule: ships unless clearly losing)"; fi
