#!/usr/bin/env bash
# Install or upgrade the cozy CLI from a release asset — cl-013's installer.
#
#   scripts/install.sh --asset <cozy-*.tar.gz> [--sums <SHA256SUMS> | --sha256 <hex>]
#                      [--prefix <dir>]
#
# THE CHECKSUM IS VERIFIED BEFORE ANYTHING IS REPLACED. A corrupted or substituted asset
# refuses with the two digests named and leaves the installed binary exactly as it was, so
# a failed upgrade is never a broken installation.
#
# Install and upgrade are ONE act, not two mechanisms: point it at a different asset and
# the same verify-then-rename runs. The replacement is a rename inside the target
# directory, so it is atomic — a reader holding the old inode keeps running it.
#
# There is no download and no channel here on purpose. `latest` resolution and rollback are
# cl-013's first-external-user tier (armed at law 15); a launch-tier installer that invents
# an update server would be a second distribution authority nobody signed.
set -euo pipefail

ASSET=""; SUMS=""; WANT=""; PREFIX="${COZY_PREFIX:-$HOME/.local}"

while [ $# -gt 0 ]; do
  case "$1" in
    --asset) ASSET="$2"; shift 2 ;;
    --sums) SUMS="$2"; shift 2 ;;
    --sha256) WANT="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "usage: $0 --asset <tar.gz> [--sums <file> | --sha256 <hex>] [--prefix <dir>]" >&2; exit 2 ;;
  esac
done

[ -n "$ASSET" ] || { echo "refusing: --asset is required" >&2; exit 2; }
[ -f "$ASSET" ] || { echo "refusing: no such asset: $ASSET" >&2; exit 4; }

if [ -z "$WANT" ]; then
  [ -n "$SUMS" ] || SUMS="$(dirname "$ASSET")/SHA256SUMS"
  [ -f "$SUMS" ] || { echo "refusing: no checksum given and no SHA256SUMS beside the asset" >&2; exit 4; }
  WANT="$(awk -v n="$(basename "$ASSET")" '{ sub(/^\.\//, "", $2); if ($2 == n) print $1 }' "$SUMS")"
  [ -n "$WANT" ] || { echo "refusing: $SUMS names no line for $(basename "$ASSET")" >&2; exit 4; }
fi

GOT="$(sha256sum "$ASSET" | cut -d' ' -f1)"
if [ "$GOT" != "$WANT" ]; then
  echo "refusing: $(basename "$ASSET") does not match its declared checksum" >&2
  echo "  declared sha256:$WANT" >&2
  echo "  actual   sha256:$GOT" >&2
  echo "nothing was replaced" >&2
  exit 13
fi

BIN="$PREFIX/bin"
mkdir -p "$BIN"
WAS="none"
[ -x "$BIN/cozy" ] && WAS="$("$BIN/cozy" version --fields tag 2>/dev/null | grep "^tag:" || echo "tag: unreadable")"

# The staging directory is INSIDE the target directory so the final move is a rename on one
# filesystem. A cross-device install would copy, and a copy can be observed half-written.
STAGE="$(mktemp -d "$BIN/.cozy-install.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
tar -xzf "$ASSET" -C "$STAGE"
[ -f "$STAGE/cozy" ] || { echo "refusing: the asset carries no cozy binary" >&2; exit 13; }
chmod 0755 "$STAGE/cozy"
mv -f "$STAGE/cozy" "$BIN/cozy"

NOW="$("$BIN/cozy" version --fields tag 2>/dev/null | grep "^tag:")"
echo "verified: sha256:$GOT"
echo "prefix:   $PREFIX"
echo "was:      ${WAS#tag: }"
echo "now:      ${NOW#tag: }"
