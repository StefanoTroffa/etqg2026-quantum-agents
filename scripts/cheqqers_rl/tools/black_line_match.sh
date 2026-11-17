#!/usr/bin/env bash
# black_line_match.sh <harness dir> <white proxy url> <our url> <games> <out dir>
# Our agent as BLACK against a white opponent behind forced_opening_proxy.py (first move forced, then normal play),
# through the organisers' harness. One line per game: forced first move, result for us, forfeit.
set -u
H=$1 W=$2 B=$3 N=$4 O=$5; mkdir -p "$O"
for i in $(seq 1 "$N"); do
  log="$O/game_$i.log"
  (cd "$H" && node local-cheqqers-server.js --p1-url "$W" --p2-url "$B") > "$log" 2>&1
  first=$(grep -a -m1 -o "Remote AI returned[^]]*]" "$log" | grep -o "\[.*")
  if grep -a -q "PLAYER BLACK WINS THE GAME" "$log"; then r=win
  elif grep -a -q "PLAYER WHITE WINS THE GAME" "$log"; then r=loss
  elif grep -a -q "THE MATCH IS A DRAW" "$log"; then r=draw
  else r=unfinished; fi
  f=$(grep -a -q "FORFEIT!" "$log" && echo forfeit || echo ok)
  echo "game $i first=$first result=$r $f"
done
