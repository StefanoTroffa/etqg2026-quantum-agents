#!/usr/bin/env bash
# Pre-flight for critical workflow CW1 (docs/architecture/critical-workflows.md): our server against the organisers'
# Cheqqers harness and its random player, in both colours.
#
#   scripts/preflight_cheqqers.sh <harness dir> <move url> <games per colour> <server log> [out dir]
#
# <harness dir> holds local-cheqqers-server.js and the engine (npm install @quantumplayed/quantum-game-engine@1.0.2).
# <server log> is the running server's log: a harness result counts only next to the number of moves our server
# answered, because a mistyped player type makes the harness forfeit one side and report a meaningless win.
# Exit 0 only if every game was won, with no forfeit, and the server answered in every game.
set -u
harness_dir=$1 url=$2 games=$3 server_log=$4 out_dir=${5:-$(mktemp -d)}
mkdir -p "$out_dir"
wins=0 losses=0 draws=0 forfeits=0 silent=0 total=0

answers() { grep -a -c "Cheqqers Official" "$server_log" || true; }

for colour in white black; do
  for i in $(seq 1 "$games"); do
    log="$out_dir/cheqqers_${colour}_${i}.log"
    before=$(answers)
    if [ "$colour" = white ]; then
      (cd "$harness_dir" && node local-cheqqers-server.js --p1-url "$url" --p2 local) > "$log" 2>&1
      ours="WHITE"
    else
      (cd "$harness_dir" && node local-cheqqers-server.js --p1 local --p2-url "$url") > "$log" 2>&1
      ours="BLACK"
    fi
    answered=$(( $(answers) - before ))
    total=$((total + 1))
    grep -a -q "FORFEIT!" "$log" && forfeits=$((forfeits + 1))
    [ "$answered" -eq 0 ] && silent=$((silent + 1))
    if grep -a -q "PLAYER $ours WINS THE GAME" "$log"; then wins=$((wins + 1)); result=win
    elif grep -a -q "THE MATCH IS A DRAW" "$log"; then draws=$((draws + 1)); result=draw
    else losses=$((losses + 1)); result=loss; fi
    echo "game $total as $colour: $result, $answered server answers"
  done
done

slowest=$(grep -a "Cheqqers Official" "$server_log" | sed -n 's/.*latency=\([0-9.]*\)ms.*/\1/p' | sort -n | tail -1)
echo "RESULT: $wins wins, $draws draws, $losses losses of $total; forfeits $forfeits; games without a server answer $silent; slowest move in the log ${slowest:-?} ms; logs in $out_dir"
[ "$wins" -eq "$total" ] && [ "$forfeits" -eq 0 ] && [ "$silent" -eq 0 ]
