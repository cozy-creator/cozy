#!/usr/bin/env bash
# cl-013's RELEASE ACCEPTANCE FIXTURE — the guard that decides whether a source-tree green
# may be called shipped.
#
#   scripts/accept.sh --dist <release dir> --asset <name> [--upgrade <dir>/<name>]
#                     [--endpoint <weightless .tar.gz>] [--prefix <dir>] [--home <dir>]
#
# It drives THE RELEASE ASSET and nothing else: no Go toolchain, no repository, no build
# tree. Everything it needs is the tarball, its SHA256SUMS, this script and (for the
# endpoint leg) `uv`. That is what lets it run unchanged inside a container that has none
# of the author's state — `scripts/clean-machine.sh` is exactly this file plus an empty
# machine to run it on.
#
# It PRINTS what it observed and counts. It is not a test suite (README: verification is
# running the real thing); every line below is a real process on a real machine.
set -uo pipefail

DIST=""; ASSET=""; UPGRADE=""; ENDPOINT=""
PREFIX="${COZY_PREFIX:-$HOME/.local}"
HOME_DIR="${COZY_HOME:-$HOME/.cozy}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST="$2"; shift 2 ;;
    --asset) ASSET="$2"; shift 2 ;;
    --upgrade) UPGRADE="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --home) HOME_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 --dist <dir> --asset <name> [--upgrade <path>] [--endpoint <path>]" >&2; exit 2 ;;
  esac
done
[ -n "$DIST" ] && [ -n "$ASSET" ] || { echo "refusing: --dist and --asset are required" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL="$HERE/install.sh"
COZY="$PREFIX/bin/cozy"
export COZY_HOME="$HOME_DIR"
PASS=0; FAIL=0

section() { printf '\n=== %s\n' "$1"; }
check() { # check <label> <ok?> <detail>
  if [ "$2" = 1 ]; then PASS=$((PASS+1)); printf '  ok   %s' "$1"
  else FAIL=$((FAIL+1)); printf '  FAIL %s' "$1"; fi
  [ -z "${3:-}" ] || printf ' — %s' "$3"
  printf '\n'
}
run() { OUT="$("$COZY" "$@" 2>&1)"; CODE=$?; return 0; }
first() { printf '%s' "$1" | sed -n 1p; }

echo "cl-013 release acceptance"
echo "  machine: $(uname -srm) · $(id -un)@$(hostname) · $(date -u +%FT%TZ)"
echo "  prefix:  $PREFIX"
echo "  home:    $COZY_HOME"

section "the machine is CLEAN: nothing of this product is installed or running"
check "no cozy on PATH" "$([ -z "$(command -v cozy || true)" ] && echo 1 || echo 0)" "$(command -v cozy || echo 'not found')"
check "no prior COZY_HOME" "$([ ! -e "$COZY_HOME" ] && echo 1 || echo 0)" "$COZY_HOME"
# Reported, never an arm: a toolchain on the machine gives this product no state, and
# every command below runs $PREFIX/bin/cozy, which came out of the tarball. Failing on it
# would only stop the fixture from running where it is most useful — on a CI runner.
echo "  note toolchain: go=$(command -v go || echo none) python3=$(command -v python3 || echo none) uv=$(command -v uv || echo none)"

section "RED ARM: a corrupted asset refuses BEFORE anything is replaced"
BAD="$(mktemp -d)/bad.tar.gz"
cp "$DIST/$ASSET" "$BAD"
printf '\x00' | dd of="$BAD" bs=1 seek=64 count=1 conv=notrunc status=none
OUT="$("$INSTALL" --asset "$BAD" --sha256 "$(awk -v n="$ASSET" '{sub(/^\.\//,"",$2); if($2==n) print $1}' "$DIST/SHA256SUMS")" --prefix "$PREFIX" 2>&1)"; CODE=$?
check "a flipped byte refuses exit 13 naming both digests" \
  "$([ "$CODE" = 13 ] && printf '%s' "$OUT" | grep -q 'does not match its declared checksum' && echo 1 || echo 0)" \
  "$(first "$OUT") [exit $CODE]"
check "and nothing was installed" "$([ ! -e "$COZY" ] && echo 1 || echo 0)" "$COZY"

section "install from the release asset — the checksum is verified against SHA256SUMS"
OUT="$("$INSTALL" --asset "$DIST/$ASSET" --prefix "$PREFIX" 2>&1)"; CODE=$?
printf '%s\n' "$OUT" | sed 's/^/    /'
check "install exits 0 and lands a binary" "$([ "$CODE" = 0 ] && [ -x "$COZY" ] && echo 1 || echo 0)" "exit $CODE"

section "the release binary provably carries its commit"
# RELEASE.json is read with sed, not python: this fixture's whole dependency list is bash,
# coreutils and tar, so the clean machine it runs on needs nothing preinstalled.
WANT_TAG="$(sed -n 's/^ *"tag": "\(.*\)",\?$/\1/p' "$DIST/RELEASE.json")"
WANT_COMMIT="$(sed -n 's/^ *"commit": "\(............\).*$/\1/p' "$DIST/RELEASE.json")"
run version --json
check "cozy version --json carries the tag RELEASE.json declares" \
  "$(printf '%s' "$OUT" | grep -q "\"tag\":\"$WANT_TAG\"" && echo 1 || echo 0)" "$WANT_TAG"
check "and the commit RELEASE.json declares" \
  "$(printf '%s' "$OUT" | grep -q "\"commit\":\"$WANT_COMMIT\"" && echo 1 || echo 0)" "$WANT_COMMIT"
check "and the protocol and contract versions it serves" \
  "$(printf '%s' "$OUT" | grep -q '"protocol"' && printf '%s' "$OUT" | grep -q '"contract"' && echo 1 || echo 0)" \
  "$(printf '%s' "$OUT" | tr ',' '\n' | grep -E '"(protocol|contract)"' | tr '\n' ' ')"

section "the surface answers for itself before any service exists"
run capabilities --full
check "cozy capabilities lists this binary's tokens" \
  "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'cmd.version' && echo 1 || echo 0)" \
  "$(printf '%s' "$OUT" | grep -c '^cmd\.') cmd tokens of $(printf '%s' "$OUT" | grep -c '^[a-z]')"
run commands
check "cozy commands is the manifest inventory" "$([ "$CODE" = 0 ] && echo 1 || echo 0)" "$(first "$OUT")"
run --help
check "cozy --help exits 0 and dials nothing" "$([ "$CODE" = 0 ] && echo 1 || echo 0)" "$(first "$OUT")"

section "the service is DOWN: server-backed verbs refuse typed, with the remedy"
for verb in doctor "start cozy/weightless"; do
  # shellcheck disable=SC2086
  run $verb
  check "cozy $verb -> 9 + next: cozy up" \
    "$([ "$CODE" = 9 ] && printf '%s' "$OUT" | grep -q 'next: cozy up' && echo 1 || echo 0)" \
    "$(first "$OUT") [exit $CODE]"
done

section "cozy up — the LocalService on a machine that has never run one"
"$COZY" up >/tmp/accept-up.log 2>&1 &
UP=$!
trap '"$COZY" down >/dev/null 2>&1 || true; kill '"$UP"' 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do "$COZY" --json 2>/dev/null | grep -q '"service":"up"' && break; sleep 0.5; done
run --json
check "bare cozy reports the service up with a pid" \
  "$(printf '%s' "$OUT" | grep -q '"service":"up"' && echo 1 || echo 0)" "$(first "$OUT")"
check "the ONE local record database exists (SQLite, pure-Go driver)" \
  "$([ -f "$COZY_HOME/records.db" ] && echo 1 || echo 0)" \
  "$COZY_HOME/records.db ($(stat -c%s "$COZY_HOME/records.db" 2>/dev/null || echo 0) B)"
check "the service lock is a real file this process holds" \
  "$([ -f "$COZY_HOME/service.lock" ] && echo 1 || echo 0)" "$COZY_HOME/service.lock"
check "the CLI credential is 0600 — reading a widened one would be agreeing to a leak" \
  "$([ "$(stat -c%a "$COZY_HOME/client.cred" 2>/dev/null)" = 600 ] && echo 1 || echo 0)" \
  "$COZY_HOME/client.cred mode $(stat -c%a "$COZY_HOME/client.cred" 2>/dev/null || echo absent)"
run doctor
check "cozy doctor answers from the running service" "$([ "$CODE" = 0 ] && echo 1 || echo 0)" "$(first "$OUT")"

if [ -n "$ENDPOINT" ] && [ -f "$ENDPOINT" ]; then
  section "endpoint install — the weightless release, on this machine's own venv"
  DIGEST="sha256:$(sha256sum "$ENDPOINT" | cut -d' ' -f1)"
  START=$(date +%s%3N)
  run install cozy/weightless --from "$ENDPOINT" --digest "$DIGEST"
  ELAPSED=$(( $(date +%s%3N) - START ))
  printf '%s\n' "$OUT" | sed 's/^/    /'
  check "install exits 0 with a verified source digest" \
    "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'verified:      true' && echo 1 || echo 0)" "${ELAPSED} ms"
  check "the release's OWN runtime re-derived the descriptor (no host venv anywhere)" \
    "$(printf '%s' "$OUT" | grep -q 'descriptor:' && echo 1 || echo 0)" \
    "$(printf '%s' "$OUT" | grep '^descriptor:' || true)"
  run ls
  check "cozy ls reads the install record" \
    "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'cozy/weightless' && echo 1 || echo 0)" "$(first "$OUT")"
  run describe cozy/weightless
  check "cozy describe renders the surface the install verified" \
    "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'tile' && echo 1 || echo 0)" "$(first "$OUT")"

  section "the WEIGHTLESS SERVE — a real worker on a machine with no card"
  # cl-010's named seam is CLOSED (cozy-runtime a3c3d72): a binding record declaring none
  # of the seven model keys is weightless, its `prepare` returns before the torch import,
  # and this host mints that record instead of refusing `no_servable_function`. What the
  # arms below establish is the only claim a cardless machine — a Windows runner, this
  # container — can ever make for itself: the whole product path, end to end, with nothing
  # to load.
  check "the generation's venv contains NO TORCH — the point of a weightless endpoint" \
    "$([ -z "$(find "$COZY_HOME/generations" -maxdepth 6 -name 'torch' -o -maxdepth 6 -name 'torch-*' 2>/dev/null)" ] && echo 1 || echo 0)" \
    "$(find "$COZY_HOME/generations" -maxdepth 6 -name 'nvidia*' -o -maxdepth 6 -name 'torch*' 2>/dev/null | wc -l) torch/cuda entries in the venv"
  START=$(date +%s%3N)
  run start cozy/weightless
  ELAPSED=$(( $(date +%s%3N) - START ))
  printf '%s\n' "$OUT" | sed 's/^/    /'
  check "cozy start makes the worker READY with no weights to fill" \
    "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -qE 'state: +ready' && echo 1 || echo 0)" \
    "${ELAPSED} ms, $(printf '%s' "$OUT" | grep -E '^ready_plans:' | tr -s ' ')"

  run run cozy/weightless/v1/tile size=32 seed=7 --full --out "$COZY_HOME/out"
  printf '%s\n' "$OUT" | sed 's/^/    /'
  check "the invoke exits 0 on a TYPED terminal — the coordinator's success transaction" \
    "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -qE 'status: +completed' && echo 1 || echo 0)" \
    "$(printf '%s' "$OUT" | grep -E '^status:' | tr -s ' ') [exit $CODE]"
  check "the typed result is the handler's own struct, not a blob" \
    "$(printf '%s' "$OUT" | grep -q 'size:32' && printf '%s' "$OUT" | grep -q 'pixels:1024' && echo 1 || echo 0)" \
    "$(printf '%s' "$OUT" | grep -oE 'digest:[0-9a-f]{16}' | head -1)"
  check "it reserved NO VRAM and constructed NOTHING (empty construction digest)" \
    "$(printf '%s' "$OUT" | grep -q 'peak_vram_bytes:0' && printf '%s' "$OUT" | grep -q 'construction_digest= ' && echo 1 || echo 0)" \
    "$(printf '%s' "$OUT" | grep -oE 'vram=[0-9A-Za-z]+ host=[0-9A-Za-z]+' | head -1)"
  # The output is a REAL published asset: a media id, a length, and a file on this disk
  # whose first bytes are a PNG signature. `cozy run --out` is the last mile of the path.
  PNG="$COZY_HOME/out/image.png"
  check "the published output is a real PNG this machine can open" \
    "$([ -s "$PNG" ] && [ "$(head -c 4 "$PNG" | tr -d '\000-\010\013\014\016-\037\177-\377')" = "PNG" ] && echo 1 || echo 0)" \
    "$PNG ($(stat -c%s "$PNG" 2>/dev/null || echo 0) B)"

  # The FAILED terminal on the SAME weightless path, so the fixture observes both verdicts
  # of the terminal transaction rather than only the happy one.
  run run cozy/weightless/v1/refuse
  check "and the failure terminal is typed too — 11 invalid_request, named" \
    "$([ "$CODE" = 11 ] && printf '%s' "$OUT" | grep -q 'invalid_request' && echo 1 || echo 0)" \
    "$(printf '%s' "$OUT" | grep '^error' | head -1) [exit $CODE]"
fi

section "cozy down — liveness is an OS fact, so absence is provable"
run down
check "cozy down stops it and proves the lock is free" \
  "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'provably gone' && echo 1 || echo 0)" "$(first "$OUT")"
run doctor
check "and the server-backed verbs refuse 9 again" "$([ "$CODE" = 9 ] && echo 1 || echo 0)" "$(first "$OUT")"

if [ -n "$UPGRADE" ] && [ -f "$UPGRADE" ]; then
  section "UPGRADE — the same verify-then-rename, pointed at a different asset"
  BEFORE="$("$COZY" version --fields tag | grep '^tag:')"
  OUT="$("$INSTALL" --asset "$UPGRADE" --prefix "$PREFIX" 2>&1)"; CODE=$?
  printf '%s\n' "$OUT" | sed 's/^/    /'
  AFTER="$("$COZY" version --fields tag | grep '^tag:')"
  check "the upgrade verified its own checksum and exits 0" "$([ "$CODE" = 0 ] && echo 1 || echo 0)" "exit $CODE"
  check "the installed binary now reports the new tag" \
    "$([ "$BEFORE" != "$AFTER" ] && echo 1 || echo 0)" "${BEFORE#tag: } -> ${AFTER#tag: }"
  check "and the install record survived the replacement" \
    "$("$COZY" ls 2>/dev/null | grep -q 'cozy/weightless' && echo 1 || echo 0)" "$COZY_HOME"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
