# Suite `varena` (gate.sh --suite varena): the memory manager through its paces on three representative models, SDXL
# (small), Klein (mid) and H3 (large, 5 s), on the architectures in ./skus. Not every package: the runtime and
# packages suites hold those. Sourced in gate.sh's shell: `cell`, `pkg`, `log`, $CLI, $M, $R, $REF, $SKU, $HERE and
# the scenario's $CHAR $MOTION $OBJECT $SCENE. Every request has its own prompt and seed.
#
#   ceiling          this machine's pinned host-to-device and cold disk-read bandwidth (ceiling.py)
#   cold-*           each model loaded from disk into a fresh process
#   co-*             all three again on one card: what does not fit is evicted and refilled
#   reload-*         after an unload (a restart ends every executor), each model back from page cache or disk
#   s8-* s6-* s3-*   under 8, 6 and 3 GiB software caps; at 6 GiB SDXL and Klein alternate three times
#   (≤ 8 GB card)    cold, co and reload only: the driver's own out-of-memory paths (a capped card never fills)
#   host-*           on a CPU rental, under its cgroup's memory.max (host-cap.py): pages must spill to disk
#
# check.py judges caps, host RAM, alternation and load/refill bandwidth as a share of `ceiling`; analyze.py, time
# and peak VRAM against the last pass. A cell named in known-failing.json's `cells` for this SKU or "*" is skipped.
V=$HERE/suites/varena
parked() {
  python3 -c 'import json, sys
sys.exit(0 if any(sys.argv[3] in k.get("cells", []) and k["sku"] in (sys.argv[2], "*") for k in json.load(open(sys.argv[1]))) else 1)' \
    "$HERE/known-failing.json" "$SKU" "$1"
}
vcell() { if parked "$1"; then log "$1 parked (known-failing.json)"; else cell "$@"; fi; }
cap() {
  "$CLI" machine update "$M" --gpu-memory-limit "$1" -y >"cap-$1.log" 2>&1
  local i
  for i in $(seq 12); do  # the restarted machine writes its ledger line once it is up
    timeout 60 "$CLI" machine logs "$M" memory -n 50 >"cap-$1.memory.log" 2>&1
    [ "$1" = off ] || [ "$(grep -o '"capacity":[0-9]*[^}]*"event":"ledger"' "cap-$1.memory.log" | tail -1 | grep -o '[0-9]*' | head -1)" -le $(( ${1%GiB} << 30 )) ] 2>/dev/null && break
    sleep 5
  done
  log "cap $1 on $M: $(grep -E '^status' "cap-$1.log")"
}
SDXL=$(pkg sdxl generate) KLEIN=$(pkg flux-3-klein generate_turbo) H3=$(pkg minimax-h3 ref2va_turbo)
h3() { vcell "$1" "$H3" --asset ref="$REF" seed:=$((R + $2)) duration_s:="$3" resolution=768p prompt="$MOTION $4"; }
ln -sfn "$V/ceiling.py" ceiling.py; ln -sfn "$V/host-cap.py" host-cap.py

if ! grep -q '^GPU 0' machine.txt; then
  vcell host-cap ./host-cap.py
  vcell host-sdxl "$SDXL" prompt="$SCENE, photo" seed:=$((R + 51))
  vcell host-klein "$KLEIN" prompt="$CHAR, studio portrait" seed:=$((R + 52))
  return 0
fi
vcell ceiling ./ceiling.py
vcell cold-sdxl "$SDXL" prompt="$SCENE, photo" seed:=$((R + 1))
vcell cold-klein "$KLEIN" prompt="$CHAR" seed:=$((R + 2))
h3 cold-h3 3 5 ""
vcell co-sdxl "$SDXL" prompt="$OBJECT on a wooden table, product photo" seed:=$((R + 4))
vcell co-klein "$KLEIN" prompt="$CHAR, close-up portrait, soft window light" seed:=$((R + 5))
h3 co-h3 6 5 "The camera slowly pushes in."
cap off  # unload
vcell reload-sdxl "$SDXL" prompt="$SCENE at dusk, wide shot, photo" seed:=$((R + 7))
vcell reload-klein "$KLEIN" prompt="$OBJECT, macro photo, shallow depth of field" seed:=$((R + 8))
[ "$(grep -oE '· [0-9]+ GiB ·' machine.txt | head -1 | grep -oE '[0-9]+')" -gt 8 ] || return 0
h3 reload-h3 9 5 "A slow pan follows the action."
cap 8GiB
vcell s8-sdxl "$SDXL" prompt="$OBJECT in a museum display case, photo" seed:=$((R + 11))
vcell s8-klein "$KLEIN" prompt="$SCENE, watercolor" seed:=$((R + 12))
h3 s8-h3 13 5 "Handheld, close."
h3 s8-h3-15 14 15 "Wide, steady."
cap 6GiB
for i in 1 2 3; do
  vcell s6-sdxl-$i "$SDXL" prompt="$CHAR in $SCENE, film still $i" seed:=$((R + 20 + i))
  vcell s6-klein-$i "$KLEIN" prompt="$OBJECT, studio photo, angle $i" seed:=$((R + 30 + i))
done
cap 3GiB
vcell s3-sdxl "$SDXL" prompt="$CHAR, oil painting" seed:=$((R + 41))
vcell s3-klein "$KLEIN" prompt="an aerial photo of $SCENE" seed:=$((R + 42))
cap off
