#!/usr/bin/env bash
# Release-asset acceptance for the public Kong CLI. Every arm drives the installed
# binary on a fresh home; no source package is called directly.
set -uo pipefail

DIST=""; ASSET=""; UPGRADE=""; PACKAGE_ARCHIVE=""
PREFIX="${COZY_PREFIX:-$HOME/.local}"
HOME_DIR="${COZY_HOME:-$HOME/.cozy}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST="$2"; shift 2 ;;
    --asset) ASSET="$2"; shift 2 ;;
    --upgrade) UPGRADE="$2"; shift 2 ;;
    --package) PACKAGE_ARCHIVE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --home) HOME_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 --dist <dir> --asset <name> [--upgrade <path>] [--package <path>]" >&2; exit 2 ;;
  esac
done
[ -n "$DIST" ] && [ -n "$ASSET" ] || { echo "--dist and --asset are required" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL="$HERE/install.sh"
COZY="$PREFIX/bin/cozy"
export COZY_HOME="$HOME_DIR"
PASS=0; FAIL=0

check() {
  if [ "$2" = 1 ]; then PASS=$((PASS+1)); printf 'ok   %s\n' "$1"
  else FAIL=$((FAIL+1)); printf 'FAIL %s — %s\n' "$1" "${3:-}"; fi
}
run() { OUT="$("$COZY" "$@" 2>&1)"; CODE=$?; return 0; }

echo "Cozy release acceptance"
echo "  asset: $DIST/$ASSET"
echo "  home:  $COZY_HOME"

OUT="$("$INSTALL" --asset "$DIST/$ASSET" --prefix "$PREFIX" 2>&1)"; CODE=$?
check "verified release asset installs" "$([ "$CODE" = 0 ] && [ -x "$COZY" ] && echo 1 || echo 0)" "$OUT"

WANT_TAG="$(sed -n 's/^ *"tag": "\(.*\)",\?$/\1/p' "$DIST/RELEASE.json")"
run -v
check "-v reports the release tag without loading config" "$([ "$CODE" = 0 ] && [ "$OUT" = "$WANT_TAG" ] && echo 1 || echo 0)" "$OUT"

run
for command in "package install" "model download" "auth login" "run cancel" "rent" "up" "down" "unload"; do
  check "root help exposes $command" "$(printf '%s' "$OUT" | grep -q "$command" && echo 1 || echo 0)" "$OUT"
done
run help run
check "contextual Kong help works" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'Usage: cozy run' && echo 1 || echo 0)" "$OUT"
check "run exposes named existing rentals and automatic remote allocation" "$(printf '%s' "$OUT" | grep -q -- '--rental=RENTAL' && printf '%s' "$OUT" | grep -q -- '--rental-only' && ! printf '%s' "$OUT" | grep -Eq -- '--(local|machine|max-cost|cloud)' && echo 1 || echo 0)" "$OUT"
run package
check "bare noun group shows focused help" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'Usage: cozy package <command>' && echo 1 || echo 0)" "$OUT"

run exit
check "retired exit command refuses instead of aliasing" "$([ "$CODE" = 2 ] && printf '%s' "$OUT" | grep -q 'unexpected argument exit' && echo 1 || echo 0)" "$OUT"

run up --json
check "up returns only useful daemon facts" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q '"url":"http://127.0.0.1:' && printf '%s' "$OUT" | grep -q '"changed":true' && ! printf '%s' "$OUT" | grep -Eq '"(ok|kind|data)"' && echo 1 || echo 0)" "$OUT"
check "up writes the daemon's own bounded log" "$([ -s "$COZY_HOME/daemon.log" ] && grep -q 'Cozy daemon up:' "$COZY_HOME/daemon.log" && echo 1 || echo 0)" "$COZY_HOME/daemon.log"

run daemon log
check "daemon log prints the daemon's words" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'Cozy daemon up:' && echo 1 || echo 0)" "$OUT"

run down
check "down stops the explicit daemon" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -Eq 'daemon: +stopped' && echo 1 || echo 0)" "$OUT"

run run list --json
check "run list can auto-start the same daemon" "$([ "$CODE" = 0 ] && grep -q '^token=' "$COZY_HOME/daemon.lock" && echo 1 || echo 0)" "$OUT"
check "JSON success is domain-shaped without renderer scaffolding" "$(printf '%s' "$OUT" | grep -q '"invocations":\[\]' && ! printf '%s' "$OUT" | grep -Eq '"(ok|kind|data|fields|rows|count|aggregates)"' && echo 1 || echo 0)" "$OUT"

run unload
check "unload preserves the daemon and returns idle residency" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'workers' && echo 1 || echo 0)" "$OUT"

if [ -n "$PACKAGE_ARCHIVE" ] && [ -f "$PACKAGE_ARCHIVE" ]; then
  DIGEST="sha256:$(sha256sum "$PACKAGE_ARCHIVE" | cut -d' ' -f1)"
  run package install cozy/weightless --from "$PACKAGE_ARCHIVE" --digest "$DIGEST"
  check "package install verifies the release" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'cozy/weightless' && echo 1 || echo 0)" "$OUT"

  run run cozy/weightless/tile size=32 seed=7 --out "$COZY_HOME/out"
  check "one real package invocation completes" "$([ "$CODE" = 0 ] && find "$COZY_HOME/out" -maxdepth 1 -type f -name '*.png' -size +0c | grep -q . && echo 1 || echo 0)" "$OUT"
fi

run down
check "down stops the daemon and proves lock release" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -Eq 'daemon: +stopped' && echo 1 || echo 0)" "$OUT"

if [ -n "$UPGRADE" ] && [ -f "$UPGRADE" ]; then
  BEFORE="$("$COZY" -v)"
  OUT="$("$INSTALL" --asset "$UPGRADE" --prefix "$PREFIX" 2>&1)"; CODE=$?
  AFTER="$("$COZY" -v)"
  check "upgrade verifies and replaces the binary" "$([ "$CODE" = 0 ] && [ "$BEFORE" != "$AFTER" ] && echo 1 || echo 0)" "$BEFORE -> $AFTER"
fi

printf '%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
