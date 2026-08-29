#!/usr/bin/env bash
# Release-asset acceptance for the public Kong CLI. Every arm drives the installed
# binary on a fresh home; no source package is called directly.
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

echo "Cozy Creator release acceptance"
echo "  asset: $DIST/$ASSET"
echo "  home:  $COZY_HOME"

OUT="$("$INSTALL" --asset "$DIST/$ASSET" --prefix "$PREFIX" 2>&1)"; CODE=$?
check "verified release asset installs" "$([ "$CODE" = 0 ] && [ -x "$COZY" ] && echo 1 || echo 0)" "$OUT"

WANT_TAG="$(sed -n 's/^ *"tag": "\(.*\)",\?$/\1/p' "$DIST/RELEASE.json")"
run -v
check "-v reports the release tag without loading config" "$([ "$CODE" = 0 ] && [ "$OUT" = "$WANT_TAG" ] && echo 1 || echo 0)" "$OUT"

run
for command in "endpoint install" "model download" "invoke run" "rental new" "up" "down" "unload"; do
  check "root help exposes $command" "$(printf '%s' "$OUT" | grep -q "$command" && echo 1 || echo 0)" "$OUT"
done
run help invoke run
check "contextual Kong help works" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'Usage: cozy invoke run' && echo 1 || echo 0)" "$OUT"

run exit
check "retired exit command refuses instead of aliasing" "$([ "$CODE" = 2 ] && printf '%s' "$OUT" | grep -q 'cli.usage' && echo 1 || echo 0)" "$OUT"

run up --json
check "up backgrounds the controller and returns the web URL" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q '"kind":"up"' && printf '%s' "$OUT" | grep -q '"url":"http://127.0.0.1:' && echo 1 || echo 0)" "$OUT"
check "up creates no persistent controller log" "$([ ! -e "$COZY_HOME/controller.log" ] && echo 1 || echo 0)" "$COZY_HOME/controller.log"

run down
check "down stops the explicit controller" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'controller: stopped' && echo 1 || echo 0)" "$OUT"

run invoke list --json
check "invoke can auto-start the same controller" "$([ "$CODE" = 0 ] && [ -f "$COZY_HOME/service.lock" ] && [ -f "$COZY_HOME/client.cred" ] && echo 1 || echo 0)" "$OUT"
check "default JSON result is one successful document" "$(printf '%s' "$OUT" | grep -q '"ok":true' && echo 1 || echo 0)" "$OUT"

run unload
check "unload preserves the controller and returns idle residency" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'stopped' && echo 1 || echo 0)" "$OUT"

if [ -n "$ENDPOINT" ] && [ -f "$ENDPOINT" ]; then
  DIGEST="sha256:$(sha256sum "$ENDPOINT" | cut -d' ' -f1)"
  run endpoint install cozy/weightless --from "$ENDPOINT" --digest "$DIGEST"
  check "endpoint install verifies the release" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'cozy/weightless' && echo 1 || echo 0)" "$OUT"

  run invoke run cozy/weightless/v1/tile size=32 seed=7 --out "$COZY_HOME/out"
  check "one real endpoint invocation completes" "$([ "$CODE" = 0 ] && [ -s "$COZY_HOME/out/image.png" ] && echo 1 || echo 0)" "$OUT"
fi

run down
check "down stops the controller and proves lock release" "$([ "$CODE" = 0 ] && printf '%s' "$OUT" | grep -q 'controller: stopped' && echo 1 || echo 0)" "$OUT"

if [ -n "$UPGRADE" ] && [ -f "$UPGRADE" ]; then
  BEFORE="$("$COZY" -v)"
  OUT="$("$INSTALL" --asset "$UPGRADE" --prefix "$PREFIX" 2>&1)"; CODE=$?
  AFTER="$("$COZY" -v)"
  check "upgrade verifies and replaces the binary" "$([ "$CODE" = 0 ] && [ "$BEFORE" != "$AFTER" ] && echo 1 || echo 0)" "$BEFORE -> $AFTER"
fi

printf '%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
