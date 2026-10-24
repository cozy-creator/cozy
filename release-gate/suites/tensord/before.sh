# Suite tensord, tracker cl-650: on the fresh pod, before the candidate goes on (gate.sh sources a suite's before.sh).
# Uses gate.sh's $CLI (cozy 0.11.17 or later), $M, $OUT, $TDF and $BIN (the candidate cozy), expect, log and the scenario.
#  (b) cozy #1925: $CLI updates a real 0.11.15 TensorD. The 0.11.15 CLI moves the pod there first, in one call (its
#      --tensord, --runtime-wheel, --tensorfs-wheel, the pod's own pair); then `$CLI machine update --tensord <candidate>`
#      must leave the machine reporting the candidate's version, or fail `update_unapplied`.
#  (a) A release whose cozy-runtime floor the pod's pair misses refuses with `runtime_floor_unmet`: sdxl 2.13.1 floors
#      cozy-runtime>=0.31 (packages 710a0a1d) and the pod holds the Hub's 0.30 pair. `cozy machine update <m> --runtime
#      0.31.0 --tensorfs 0.14.0` (versions the machine fetches from PyPI) then serves it.
OLDV=0.11.15 FLOORED=sdxl/generate@2.13.1 MEETS=(--runtime 0.31.0 --tensorfs 0.14.0)
pair() { timeout 60 "$CLI" --json machine show "$M" 2>/dev/null | jq -r '.status | "\(.version) \(.runtime) \(.tensorfs)"'; }
# pypi <dist> <version>: its wheel for the pod's CPython 3.12 on manylinux x86_64 (else pure), as URL#sha256=<hex>.
pypi() { curl -sf --retry 5 --retry-all-errors "https://pypi.org/pypi/$1/$2/json" | jq -r '[.urls[] | select(.filename | endswith(".whl"))] as $w
  | ([$w[] | select(.filename | test("manylinux.*x86_64") and test("-cp312-|-abi3-"))] + [$w[] | select(.filename | test("-none-any"))])[0]
  | .url + "#sha256=" + .digests.sha256'; }
number() { jq -r 'select(.run.number != null) | .run.number' "$1" 2>/dev/null | head -1; }  # a run's NDJSON
ended_as() { jq -r 'select(.run.status != null) | .run.status' "$1" 2>/dev/null | tail -1; }
mkdir -p "$OUT/cl650"; C=$OUT/cl650
# cozy 0.11.15's Linux archive, its digest checked: straight from github.com (the API that `gh` uses drops out here).
u=https://github.com/cozy-creator/cozy-creator/releases/download/v$OLDV/cozy-x86_64-unknown-linux-gnu.tar.gz
for _ in 1 2 3; do
  curl -fsSL --retry 5 --retry-all-errors "$u" -o "$C/cozy-x86_64-unknown-linux-gnu.tar.gz" && curl -fsSL --retry 5 --retry-all-errors "$u.sha256" -o "$C/cozy-x86_64-unknown-linux-gnu.tar.gz.sha256" \
    && (cd "$C" && sha256sum -c cozy-x86_64-unknown-linux-gnu.tar.gz.sha256 >/dev/null && tar -xzf cozy-x86_64-unknown-linux-gnu.tar.gz) && break
done
OLD=$C/cozy-x86_64-unknown-linux-gnu/cozy
for _ in $(seq 30); do read -r v rt tfs <<< "$(pair)"; [ -n "$v" ] && break; sleep 5; done  # a fresh pod reports once up
log "cl650: fresh $M runs cozy $v, Runtime $rt, TensorFS $tfs"

if [ -z "$TDF" ]; then expect cl650-b-update-from-0.11.15 FAIL "no --tensord candidate to update a 0.11.15 TensorD to"; else
"$OLD" machine update "$M" --tensord "$OLD" --runtime-wheel "$(pypi cozy-runtime "$rt")" \
  --tensorfs-wheel "$(pypi tensorfs "$tfs")" -y > "$C/old.log" 2>&1
read -r v rt tfs <<< "$(pair)"
[ "$v" = "$OLDV" ] || expect cl650-old-pod FAIL "cozy $OLDV's update left $M on cozy ${v:-?}: $(tail -c 300 "$C/old.log")"
tdf=$TDF; case $tdf in *://*) tdf=$(fetch "$tdf") || tdf=;; esac  # cozy 0.11.16+ sends a file
"$CLI" machine update "$M" --tensord "$tdf" -y > "$C/b.log" 2>&1; code=$?
read -r v2 rt tfs <<< "$(pair)"
if [ "$code" = 0 ] && [ "$v2" = "$BIN" ]; then
  expect "cl650-b-update-from-$v" PASS "cozy $BIN's machine update reached TensorD $v; the machine verified $v2"
elif grep -q update_unapplied "$C/b.log"; then
  expect "cl650-b-update-from-$v" FAIL "update_unapplied, loudly as designed, but $M runs ${v2:-?}, not $BIN"
else
  expect "cl650-b-update-from-$v" FAIL "exit $code, $M runs ${v2:-?}: $(tail -c 300 "$C/b.log")"
fi
fi

"$CLI" --json run "$FLOORED" --on "$M" prompt="$SCENE, a floor check" seed:=$((R + 20)) > "$C/a-refused.ndjson" 2> "$C/a-refused.err"
n=$(number "$C/a-refused.ndjson")
if grep -qs runtime_floor_unmet "$C/a-refused.ndjson" "$C/a-refused.err"; then
  expect cl650-a-floor-refuses PASS "$FLOORED (run ${n:-none}) on Runtime $rt / TensorFS $tfs: runtime_floor_unmet"
else
  expect cl650-a-floor-refuses FAIL "$FLOORED (run ${n:-none}) on Runtime $rt / TensorFS $tfs: $(ended_as "$C/a-refused.ndjson") $(tail -c 200 "$C/a-refused.err")"
fi
"$CLI" machine update "$M" "${MEETS[@]}" -y > "$C/a-update.log" 2>&1
read -r v rt tfs <<< "$(pair)"
"$CLI" --json run "$FLOORED" --on "$M" prompt="$OBJECT, the floor met" seed:=$((R + 21)) > "$C/a-served.ndjson" 2> "$C/a-served.err"
n=$(number "$C/a-served.ndjson")
if [ "$(ended_as "$C/a-served.ndjson")" = completed ]; then
  expect cl650-a-floor-met-serves PASS "after machine update ${MEETS[*]} (Runtime $rt / TensorFS $tfs): run $n completed"
else
  expect cl650-a-floor-met-serves FAIL "after machine update ${MEETS[*]} (Runtime $rt / TensorFS $tfs): run ${n:-none} $(ended_as "$C/a-served.ndjson") $(tail -c 200 "$C/a-served.err")"
fi
