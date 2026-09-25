#!/usr/bin/env bash
# Install or upgrade cozy and this host's tools (cozy-runtime and TensorFS's tfs).
#
#   scripts/install.sh --asset <cozy-*.tar.gz> [--sums <SHA256SUMS> | --sha256 <hex>]
#                      [--prefix <dir>]
#   scripts/install.sh --binary <built cozy> [--prefix <dir>]
#
# THE CHECKSUM IS VERIFIED BEFORE ANYTHING IS REPLACED. A corrupted or substituted asset
# refuses with the two digests named and leaves the installed binary exactly as it was, so
# a failed upgrade is never a broken installation. The staged binary must run, and the host
# tools must install, before cozy is replaced.
#
# Install and upgrade are ONE act, not two mechanisms: point it at a different asset and
# the same verify-then-rename runs. The replacement is a rename inside the target
# directory, so it is atomic — a reader holding the old inode keeps running it.
#
# The host tools are one uv tool environment: cozy-runtime, with the tfs executable of the
# tensorfs release it resolves. Keep HOST_TOOLS equal to hostruntime.InstallCommand.
#
# There is no download and no channel here on purpose. `latest` resolution and rollback are
# cl-013's first-external-user tier (armed at law 15); a launch-tier installer that invents
# an update server would be a second distribution authority nobody signed.
set -euo pipefail

HOST_TOOLS=(uv tool install --force --python 3.12 --with-executables-from tensorfs 'cozy-runtime[media,model-execution]>=0.18.24')

ASSET=""; BINARY=""; SUMS=""; WANT=""; PREFIX="${COZY_PREFIX:-$HOME/.local}"
usage="usage: $0 (--asset <tar.gz> [--sums <file> | --sha256 <hex>] | --binary <cozy>) [--prefix <dir>]"

while [ $# -gt 0 ]; do
  case "$1" in
    --asset) ASSET="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --sums) SUMS="$2"; shift 2 ;;
    --sha256) WANT="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done

{ [ -n "$ASSET" ] && [ -z "$BINARY" ]; } || { [ -z "$ASSET" ] && [ -n "$BINARY" ]; } ||
  { echo "refusing: give exactly one of --asset or --binary" >&2; echo "$usage" >&2; exit 2; }
command -v uv >/dev/null ||
  { echo "refusing: uv is required to install cozy-runtime and tfs — https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 6; }

if [ -n "$ASSET" ]; then
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
else
  [ -f "$BINARY" ] || { echo "refusing: no such binary: $BINARY" >&2; exit 4; }
fi

BIN="$PREFIX/bin"
mkdir -p "$BIN"
WAS="none"
[ -x "$BIN/cozy" ] && WAS="$("$BIN/cozy" -v 2>/dev/null || echo "unreadable")"

# The staging directory is INSIDE the target directory so the final move is a rename on one
# filesystem. A cross-device install would copy, and a copy can be observed half-written.
STAGE="$(mktemp -d "$BIN/.cozy-install.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
if [ -n "$ASSET" ]; then
  tar -xzf "$ASSET" -C "$STAGE"
  [ -f "$STAGE/cozy" ] || { echo "refusing: the asset carries no cozy binary" >&2; exit 13; }
else
  cp "$BINARY" "$STAGE/cozy"
fi
chmod 0755 "$STAGE/cozy"
NOW="$("$STAGE/cozy" -v)" || { echo "refusing: the new cozy binary does not run; nothing was replaced" >&2; exit 13; }

# A standalone tensorfs tool would still claim tfs: its later upgrade would relink an
# unpinned tfs, and its uninstall would delete the one cozy-runtime installs.
if uv tool list 2>/dev/null | grep -q '^tensorfs '; then uv tool uninstall tensorfs; fi
"${HOST_TOOLS[@]}" ||
  { echo "refusing: host tool installation failed; cozy was not replaced" >&2; exit 13; }
TOOLS="$(uv tool dir --bin)"
RUNTIME="$("$TOOLS/cozy-runtime" version | sed -n 's/^distribution: *//p')"
TFS="$("$TOOLS/tfs" version)"

mv -f "$STAGE/cozy" "$BIN/cozy"

[ -n "$ASSET" ] && echo "verified: sha256:$GOT"
echo "prefix:   $PREFIX"
echo "was:      $WAS"
echo "now:      $NOW"
echo "runtime:  cozy-runtime $RUNTIME"
echo "tfs:      $TFS"
echo "complete: add 'source <(cozy completion bash)' to ~/.bashrc (zsh and fish: cozy completion --help)"
for dir in "$BIN" "$TOOLS"; do
  case ":$PATH:" in *":$dir:"*) ;; *) echo "note:     add $dir to PATH" ;; esac
done | uniq
