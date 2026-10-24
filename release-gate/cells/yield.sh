# Suite `yield` (gate.sh --cells, or agent-tools/s1-yield-proof.sh): a game on a consumer GPU beside sdxl, from the
# default home. Uses `log`, $CLI, $M, $R, $HERE. game.py and game_hold.py allocate raw driver memory, so TensorD
# sees it as a game: bytes leaving its books.
#   hold: the game holds the card to high + 1 GiB while sdxl warms to the GPU (an idle executor's preload asks
#         speculatively: free above high only). PASS: free never fell under 1 GiB (min 512 MiB + the warm executor's
#         context), and the game's next 1 GiB came at once.
#   ramp: while an sdxl run is in progress, the game takes free + 4 GiB in 256 MiB blocks, each retried every 50 ms.
#         Recorded: each block's wait, the ramp's total, and that sdxl completed.
ln -sfn "$HERE/cells/game.py" game.py; ln -sfn "$HERE/cells/game_hold.py" game_hold.py
TAG=${YIELD_TAG:-dev}
gpu() { timeout 60 "$CLI" machine show "$M" --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); g=[x for x in (d.get("status") or {}).get("gpus") or d.get("gpus") or [] if x.get("memory")]; print(g[0]["memory"].get(sys.argv[1], 0) if g else 0)' "$1"; }
run_number() { grep -o '"number": *[0-9]*' "$1" | head -1 | grep -o '[0-9]*$'; }
status_of() { [ -n "$1" ] && timeout 60 "$CLI" run show "$1" --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null; }

"$CLI" run sdxl/generate --on "$M" --warm off >/dev/null 2>&1
"$CLI" run sdxl/generate --on "$M" --warm installed >"warm-installed-$TAG.log" 2>&1
"$CLI" run ./game_hold.py --on "$M" --json >"game-hold-$TAG.ndjson" 2>&1 &
game=$!
before=$(gpu foreign_bytes); for _ in $(seq 120); do f=$(gpu foreign_bytes); [ "$f" -gt $((before + (1 << 30))) ] && [ "$f" = "${last:-}" ] && break; last=$f; sleep 2; done
log "yield-$TAG hold: the game holds $(( (f - before) >> 20 )) MiB; warming sdxl to the GPU"
t=$(date +%s); "$CLI" run sdxl/generate --on "$M" --warm gpu >"warm-gpu-$TAG.log" 2>&1; log "yield-$TAG warm gpu took $(( $(date +%s) - t ))s"
wait "$game"
g=$(run_number "game-hold-$TAG.ndjson"); timeout 120 "$CLI" run show "$g" --json >"game-hold-$TAG.json" 2>&1
log "yield-$TAG hold: game run $g $(status_of "$g")"

"$CLI" run sdxl/generate --on "$M" --json --detach prompt="a lighthouse on a cliff at dawn, photo" seed:=$((R + 301)) >"sdxl-ramp-$TAG.ndjson" 2>&1
n=$(run_number "sdxl-ramp-$TAG.ndjson")
[ -n "$n" ] || { log "yield-$TAG ramp: the sdxl run's number is unreadable ($(head -c 200 "sdxl-ramp-$TAG.ndjson" | tr '\n' ' ')); ramp skipped"; return 0; }
for _ in $(seq 300); do s=$(status_of "$n"); [ "$s" = in_progress ] || [ "$s" = completed ] || [ "$s" = failed ] && break; sleep 2; done
sleep 20  # the proof's own pause: well into its denoising loop, its weights bound
log "yield-$TAG ramp: sdxl run $n is $(status_of "$n"); the game ramps"
t=$(date +%s); "$CLI" run ./game.py --on "$M" --json >"game-ramp-$TAG.ndjson" 2>&1; log "yield-$TAG game ramp run took $(( $(date +%s) - t ))s"
# Until sdxl ends; a status unreadable 30 times running (no news in 150 s) ends the wait instead.
blind=0; until s=$(status_of "$n"); [ "$s" = completed ] || [ "$s" = failed ] || [ "$s" = canceled ]; do
  if [ -z "$s" ]; then blind=$((blind + 1)); [ "$blind" -lt 30 ] || { s=unreadable; break; }; else blind=0; fi
  sleep 5
done
g=$(run_number "game-ramp-$TAG.ndjson"); timeout 120 "$CLI" run show "$g" --json >"game-ramp-$TAG.json" 2>&1
timeout 120 "$CLI" run show "$n" --json >"sdxl-ramp-$TAG.json" 2>&1
log "yield-$TAG ramp: sdxl run $n $s; game run $g $(status_of "$g")"
