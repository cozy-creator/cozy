#!/bin/bash
# Release regression gate (tracker design ci-v2 §3): before every publish of cozy, the Runtime, TensorFS or a
# first-party package. Any FAIL blocks the publish. Runs by hand through the default ~/.cozy, and from
# .github/workflows/release-gate.yaml on a runner signed in with `cozy login --key`.
#   gate.sh <SKU> --suite name,... [--gpus N] [--out DIR] [--establish] [--publish] [--runtime F|URL] [--tensorfs F|URL]
#           [--tensord BIN|URL] [--packages DIR] [--pins name@release,...] [--only cell,cell] [--before FILE]...
#           [--cells FILE]... [--cli BIN] [--disk GB]
# --suite runs those suites (suites/<name>/: README.md says what each proves and costs); there is no default, the
# release agent chooses. A suite's before.sh runs on the fresh pod before the candidate goes on, its cells.sh after it
# and its check.py in analyze.py; the baseline is gate-<SKU>-<suites>.json.
# A candidate URL is https://…#sha256=<hex>: fetched here, held by its digest, and sent as a file (cozy 0.11.16+
# `machine update --runtime/--tensorfs/--tensord <path>`; an older --cli gets --runtime-wheel/--tensorfs-wheel).
# One FRESH rental. The candidate goes on with `cozy machine update` (a new executor), then each cell runs ONCE
# with a scenario prompt from scenarios.tsv and fresh seeds. A cell runs the root (first-party) package users get;
# --pins runs exact releases of them instead (a yanked candidate installs by its exact pin), --packages a packages
# checkout's dirs. A cell another suite ran, or one known-failing.json parks on the SKU (its "cells"), does not run.
# candidate.json records the set the verdict is for (release-gate/verify.sh reads it before a promotion).
# analyze.py holds each cell against the SKU's baseline, gate-<SKU>.json on the `release-gate-baselines`
# release; a SKU with none needs --establish. --publish uploads a PASS as the SKU's new baseline.
set -uo pipefail
# Run from a private copy: editing this file while a gate runs must never change the running gate.
if [ "${1:-}" != --copy ]; then
  c=$(mktemp "${TMPDIR:-/tmp}/release-gate.XXXXXX.sh"); cp "$0" "$c"
  exec bash "$c" --copy "$(cd "$(dirname "$0")" && pwd)" "$@"
fi
HERE=$2; shift 2
REPO=cozy-creator/cozy-creator TAG=release-gate-baselines
SKU=$1; shift; CAND=() ONLY= EXTRA=() BEFORE=() CLI=cozy DISK=400 OUT= PKGS= ESTABLISH= PUBLISH= PINS= SUITES= K= CLIP= GPUS=
RT= TFS= BIN= RW= TW= TD=
ver() { basename "${1%%#*}" | cut -d- -f2; }
while [ $# -gt 0 ]; do
  case $1 in
    --establish) ESTABLISH=1; shift; continue;;
    --publish) PUBLISH=1; shift; continue;;
    --out) OUT=$2;;
    --only) ONLY=",$2,";;
    --cells) EXTRA+=("$2");;
    --before) BEFORE+=("$2");;
    --suite) SUITES=$2;;
    --gpus) GPUS=$2;;
    --cli) CLI=$2;;
    --disk) DISK=$2;;
    --packages) PKGS=$(cd "$2" && pwd);;
    --pins) PINS=",$2,";;
    --runtime|--runtime-wheel) RW=$2; RT=$(ver "$2");;
    --tensorfs|--tensorfs-wheel) TW=$2; TFS=$(ver "$2");;
    --tensord) TD=$2; if [ -f "$2" ]; then BIN=$("$2" --version | awk '{print $2}'); else BIN=$(ver "$2"); fi;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac; shift 2
done
[ -n "$SUITES" ] || { echo "gate.sh needs --suite: $(ls "$HERE/suites" | paste -sd, -) (release-gate/README.md)" >&2; exit 2; }
for s in ${SUITES//,/ }; do
  [ -f "$HERE/suites/$s/cells.sh" ] || { echo "no suite $s: $(ls "$HERE/suites" | paste -sd, -)" >&2; exit 2; }
done
SW=$(curl -sf https://tensorhub.com/v1/software | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["cozy"], d["runtime"], d["tensorfs"])')
[ -n "$SW" ] || { echo "no /v1/software"; exit 1; }
read -r C0 R0 T0 <<< "$SW"
LABEL=cozy${BIN:-$C0}-rt${RT:-$R0}-tfs${TFS:-$T0}${PKGS:+-packages$(git -C "$PKGS" rev-parse --short HEAD)}${PINS:+-pins$(cksum <<< "$PINS" | cut -d' ' -f1)}
OUT=${OUT:-$PWD/gate-$SKU-$LABEL-$(date -u +%Y%m%dT%H%MZ)}
B=$OUT/baseline
mkdir -p "$B"; cd "$OUT"; : > cells.tsv; echo "$LABEL" > label
digest() { case $1 in *'#sha256='*) echo "${1##*#sha256=}";; '') ;; *) [ -f "$1" ] && sha256sum "$1" | cut -d' ' -f1;; esac; }
python3 - "$LABEL" "$BIN" "$(digest "$TD")" "$(basename "${RW%%#*}")" "$(digest "$RW")" "$(basename "${TW%%#*}")" "$(digest "$TW")" \
  "${PINS//,/ }" "${PKGS:+$(git -C "$PKGS" rev-parse HEAD)}" > candidate.json <<'PY'
import json, sys
label, cozy, cozy_sha, rt, rt_sha, tfs, tfs_sha, pins, packages = sys.argv[1:]
print(json.dumps({"label": label, "cozy": cozy, "cozy_sha256": cozy_sha, "runtime_wheel": rt, "runtime_sha256": rt_sha,
                  "tensorfs_wheel": tfs, "tensorfs_sha256": tfs_sha, "pins": sorted(pins.split()), "packages_ref": packages},
                 indent=1, sort_keys=True))
PY
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a gate.log; }
# expect <check> PASS|FAIL <what happened>: a CLI-behaviour check (cells/*.sh); analyze.py fails the gate on a FAIL.
expect() { printf '%s\t%s\t%s\n' "$1" "$2" "$(tr '\n\t' '  ' <<< "$3" | cut -c1-400)" >> expect.tsv; log "$1: $2 $3"; }
# fetch <file|URL#sha256=hex>: the candidate as a file here, its digest checked.
fetch() {
  local f
  case $1 in
    '') return 0;;
    *://*#sha256=*) mkdir -p "$OUT/candidate"; f=$OUT/candidate/$(basename "${1%%#*}")
      [ -f "$f" ] || curl -fsSL --retry 5 "${1%%#*}" -o "$f" || { log "cannot fetch ${1%%#*}"; return 1; }
      [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${1##*#sha256=}" ] || { log "$f does not hash to ${1##*#sha256=}"; return 1; };;
    *://*) log "$1 names no #sha256="; return 1;;
    *) f=$1; [ -f "$f" ] || { log "no file $f"; return 1; };;
  esac
  echo "$f"
}
RWF=$(fetch "$RW") && TWF=$(fetch "$TW") && TDF=$(fetch "$TD") || exit 1
[ -z "$TDF" ] || { chmod +x "$TDF"; BIN=$("$TDF" --version | awk '{print $2}'); }
W=; "$CLI" machine update --help 2>/dev/null | grep -q -- --runtime-wheel && W=-wheel
[ -z "$RWF" ] || CAND+=("--runtime$W" "$RWF"); [ -z "$TWF" ] || CAND+=("--tensorfs$W" "$TWF"); [ -z "$TDF" ] || CAND+=(--tensord "$TDF")
BK=$SKU${SUITES:+-${SUITES//,/-}}  # the baseline's name
gh release download "$TAG" -R "$REPO" -D "$B" -p "gate-$BK.json" -p "ref-$SKU.*" -p "ref-$SKU-clip.mp4" 2>/dev/null
if [ -f "$B/gate-$BK.json" ]; then
  [ -z "$ESTABLISH" ] || { log "$BK already has a baseline; --establish is only for a SKU without one"; exit 1; }
else
  [ -n "$ESTABLISH" ] || { log "no baseline for $BK on $REPO@$TAG: --establish starts one"; exit 1; }
fi
REF=$(ls "$B/ref-$SKU".* 2>/dev/null | grep -v -- -clip | head -1); REF=${REF:-$HERE/ref/seed.jpg}
mapfile -t ROWS < <(grep -v '^#' "$HERE/scenarios.tsv")
I=$(( $(date +%s) / 60 + $(cksum <<< "$SKU" | cut -d' ' -f1) ))
IFS=$'\t' read -r NAME CHAR MOTION OBJECT SCENE <<< "${ROWS[$((I % ${#ROWS[@]}))]}"
R=$(( $(date +%s%N) / 1000 % 4000000000 ))
M=
end_it() {
  [ -n "$M" ] || return
  # The machine's own logs leave with it: keep them for every gate (a poisoned executor is diagnosed from them).
  timeout 300 "$CLI" machine logs "$M" memory -n 0 >machine-memory.log 2>&1
  timeout 300 "$CLI" machine logs "$M" tensord -n 0 >machine-tensord.log 2>&1
  timeout 120 "$CLI" logs calcifer >calcifer.log 2>&1
  "$CLI" machine end "$M" -y >end.log 2>&1
  for _ in $(seq 1 30); do timeout 30 "$CLI" machine list 2>/dev/null | grep -q "^$M " || { log "ended $M, gone"; M=; return; }; sleep 10; done
  log "WARNING $M still listed"
}
trap end_it EXIT
trap 'exit 143' TERM INT
# cell <name> <target> [args...]: runs on $M, keeps its record as <name>.json
# --only names cells or packages.
# A cell another suite already ran, or one known-failing.json parks on this SKU ("cells"; sku "*" is every SKU), does
# not run.
cell() {
  local name=$1 target=$2 p issue; shift 2
  p=${target%/*}; p=${p##*/}
  [ -z "$ONLY" ] || [[ $ONLY == *",$name,"* || $ONLY == *",$p,"* ]] || return 0
  ! cut -f1 cells.tsv | grep -qx -- "$name" || return 0
  issue=$(jq -r --arg s "$SKU" --arg c "$name" '.[] | select((.sku == $s or .sku == "*") and ((.cells // []) | index($c))) | .issue' \
    "$HERE/known-failing.json" | head -1)
  [ -z "$issue" ] || { log "$name: parked on $SKU ($issue), not run"; return 0; }
  for p in "$@"; do [[ $p != prompt=* ]] || printf '%s\t%s\n' "$name" "${p#prompt=}" >> prompts.tsv; done  # quality.py
  "$CLI" run "$target" --on "$M" --json "$@" >"$name.ndjson" 2>"$name.err"
  local n; n=$(grep -o '"number":[0-9]*' "$name.ndjson" | head -1 | cut -d: -f2)
  [ -n "$n" ] && timeout 120 "$CLI" run show "$n" --json >"$name.json" 2>/dev/null
  printf '%s\t%s\t%s\n' "$name" "$target" "${n:-none}" >> cells.tsv
  log "$name run ${n:-none} $(grep -o '"status":"[a-z_]*"' "$name.ndjson" | tail -1)"
}
# pkg <package> <fn>: the root package's function users get, its --pins release, or the --packages checkout's dir.
pkg() {
  local pin; pin=$(tr , '\n' <<< "$PINS" | grep -m1 "^$1@" | cut -d@ -f2)
  if [ -n "$pin" ]; then echo "$1/$2@$pin"
  elif [ -n "$PKGS" ] && grep -qs cozy.application "$PKGS/$1/pyproject.toml"; then ln -sfn "$PKGS/$1" "$OUT/$1"; echo "./$1/$2"
  else echo "$1/$2"; fi
}
output() { local n; n=$(awk -F'\t' -v c="$1" '$1==c{print $3}' cells.tsv); ls ~/.cozy/outputs/*/"$n"-* 2>/dev/null | grep -E "\.($2)$" | head -1; }
. "$HERE/products.sh"
log "gate $LABEL on $SKU (suites ${SUITES//,/ }): scenario $NAME, seeds from $R, candidate: ${CAND[*]:-released}${PKGS:+ packages $PKGS}${PINS:+ pins ${PINS//,/ }}, cli: $CLI"
"$CLI" machine rent "$SKU" ${GPUS:+--gpus "$GPUS"} --disk-gb "$DISK" -y --json >rent.json 2>rent.err
M=$(python3 -c 'import json,sys
t = open(sys.argv[1]).read(); dec = json.JSONDecoder(); i = 0
while i < len(t):
    i = t.find("{", i)
    if i < 0: break
    d, i = dec.raw_decode(t, i)
    m = d.get("machine")
    if isinstance(m, dict) and m.get("name"): print(m["name"]); break' rent.json)
[ -n "$M" ] || { log "rent failed: $(tail -c 300 rent.err) $(tail -c 300 rent.json)"; exit 1; }
echo "$M" > machine
log "rented $M"
timeout 60 "$CLI" machine show "$M" > machine-fresh.txt 2>&1
# Sourced on the fresh pod, before the candidate goes on: each suite's before.sh, then `--before FILE`s.
for s in ${SUITES//,/ }; do [ ! -f "$HERE/suites/$s/before.sh" ] || . "$HERE/suites/$s/before.sh"; done
for f in "${BEFORE[@]}"; do . "$f"; done
if [ ${#CAND[@]} -gt 0 ]; then
  "$CLI" machine update "$M" "${CAND[@]}" -y >update.log 2>&1; code=$?
  log "candidate on $M: exit $code $(grep -E '^status' update.log)"
  [ "$code" = 0 ] || expect candidate-update FAIL "$(tail -c 400 update.log)"
fi
timeout 60 "$CLI" machine show "$M" > machine.txt 2>&1
if [ -n "$BIN" ]; then
  v=$(timeout 60 "$CLI" --json machine show "$M" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"]["version"])' 2>/dev/null)
  [ "$v" = "$BIN" ] && expect candidate-tensord PASS "$M runs cozy $v" || expect candidate-tensord FAIL "$M runs cozy ${v:-?}, not $BIN"
fi
printf '%s\n' ${SUITES//,/ } > suites
for s in ${SUITES//,/ }; do . "$HERE/suites/$s/cells.sh"; done
# More cells (`--cells FILE`): `name<TAB>target<TAB>args`, args expanded here ($SCENE, $CHAR, $R, $K, $M, $CLI ...).
for f in "${EXTRA[@]}"; do
  [[ $f == *.sh ]] && continue
  while IFS=$'\t' read -r name target args; do
    [ -n "$name" ] && [ "${name:0:1}" != "#" ] && eval "cell \"\$name\" \"\$target\" $args"
  done < "$f"
done
# A --cells FILE ending in .sh is sourced last, for cells that change the machine between them.
for f in "${EXTRA[@]}"; do [[ $f != *.sh ]] || . "$f"; done
end_it
uv run -q --no-project --with pillow --with numpy --with torch --with transformers --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match python -I "$HERE/analyze.py" "$OUT" "$SKU" "$B/gate-$BK.json" | tee verdict.txt
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
# A PASS hands the next gate new references of the same shape: this Klein image and 3 s of this H3 video.
if [ -z "$ONLY" ] && [ -n "$K" ] && [ -n "$CLIP" ]; then
  rm -f "$B/ref-$SKU".* && cp "$K" "$B/ref-$SKU.${K##*.}"
  nice -n 19 ffmpeg -v error -y -i "$CLIP" -t 3 -an -c:v libx264 -pix_fmt yuv420p "$B/ref-$SKU-clip.mp4"
fi
[ -z "$PUBLISH" ] || "$HERE/publish.sh" "$OUT" "$BK" | tee -a gate.log
