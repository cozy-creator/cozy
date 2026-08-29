#!/usr/bin/env bash
# cl-013's RELEASE ACCEPTANCE FIXTURE — the guard that decides whether a source-tree green
# may be called shipped.
#
#   scripts/accept.sh --dist <release dir> --asset <name> [--upgrade <dir>/<name>]
#                     [--prefix <dir>] [--home <dir>]
#
# It drives THE RELEASE ASSET and nothing else: no Go toolchain, no repository, no build
# tree. Everything it needs is the tarball, its SHA256SUMS, and this script.
#
# It PRINTS what it observed and counts. Every line below is a real process on a real
# machine rather than a simulated release path.
set -uo pipefail

DIST=""; ASSET=""; UPGRADE=""
PREFIX="${COZY_PREFIX:-$HOME/.local}"
HOME_DIR="${COZY_HOME:-$HOME/.cozy}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST="$2"; shift 2 ;;
    --asset) ASSET="$2"; shift 2 ;;
    --upgrade) UPGRADE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --home) HOME_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 --dist <dir> --asset <name> [--upgrade <path>]" >&2; exit 2 ;;
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

section "isolated install target"
check "no prior binary at the target" "$([ ! -e "$COZY" ] && echo 1 || echo 0)" "$COZY"

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

if [ -n "$UPGRADE" ] && [ -f "$UPGRADE" ]; then
  section "UPGRADE — the same verify-then-rename, pointed at a different asset"
  BEFORE="$("$COZY" version --fields tag | grep '^tag:')"
  OUT="$("$INSTALL" --asset "$UPGRADE" --prefix "$PREFIX" 2>&1)"; CODE=$?
  printf '%s\n' "$OUT" | sed 's/^/    /'
  AFTER="$("$COZY" version --fields tag | grep '^tag:')"
  check "the upgrade verified its own checksum and exits 0" "$([ "$CODE" = 0 ] && echo 1 || echo 0)" "exit $CODE"
  check "the installed binary now reports the new tag" \
    "$([ "$BEFORE" != "$AFTER" ] && echo 1 || echo 0)" "${BEFORE#tag: } -> ${AFTER#tag: }"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
