#!/usr/bin/env bash
# Two move servers against each other through the organisers' Cheqqers harness, both colours.
#
#   scripts/harness_match.sh <harness dir> <url of A> <url of B> <games per colour> [out dir]
#
# <harness dir> holds local-cheqqers-server.js and the engine (npm install @quantumplayed/quantum-game-engine@1.0.2).
# Results are from A's side. A forfeit (timeout, illegal move, bad reply) ends the game in the harness: it is
# counted, and the log of that game says whose it was. Exit 0 always; read the RESULT line.
set -u
harness_dir=$1 url_a=$2 url_b=$3 games=$4 out_dir=${5:-$(mktemp -d)}
mkdir -p "$out_dir"
wins=0 losses=0 draws=0 forfeits=0 unfinished=0 total=0

for colour in white black; do
  for i in $(seq 1 "$games"); do
    log="$out_dir/match_${colour}_${i}.log"
    if [ "$colour" = white ]; then
      (cd "$harness_dir" && node local-cheqqers-server.js --p1-url "$url_a" --p2-url "$url_b") > "$log" 2>&1
      ours="WHITE" theirs="BLACK"
    else
      (cd "$harness_dir" && node local-cheqqers-server.js --p1-url "$url_b" --p2-url "$url_a") > "$log" 2>&1
      ours="BLACK" theirs="WHITE"
    fi
    total=$((total + 1))
    moves=$(grep -a -c "Remote AI returned" "$log" || true)
    grep -a -q "FORFEIT!" "$log" && forfeits=$((forfeits + 1))
    if grep -a -q "PLAYER $ours WINS THE GAME" "$log"; then wins=$((wins + 1)); result=win
    elif grep -a -q "PLAYER $theirs WINS THE GAME" "$log"; then losses=$((losses + 1)); result=loss
    elif grep -a -q "THE MATCH IS A DRAW" "$log"; then draws=$((draws + 1)); result=draw
    else unfinished=$((unfinished + 1)); result=unfinished; fi
    echo "game $total, A as $colour: $result, $moves moves"
  done
done
echo "RESULT for A: $wins wins, $draws draws, $losses losses, $unfinished unfinished of $total; games with a forfeit $forfeits; logs in $out_dir"
