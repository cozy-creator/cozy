#!/bin/sh
# Install or upgrade cozy from its public GitHub release, and this host's tools (cozy-runtime
# and TensorFS's tfs) in one uv tool environment:
#
#   curl -fsSL https://github.com/cozy-creator/cozy/releases/latest/download/install.sh | sh
#
# COZY_VERSION=v0.1.0 picks a release (default: the latest), COZY_PREFIX the install prefix
# (default ~/.local), and COZY_RELEASE_URL the directory the assets are read from.
#
# The archive is checked against the release's SHA256SUMS, the new binary must run and the host
# tools must install before anything is replaced. The replacement is a rename, so a running
# daemon keeps its binary and its work; the new one starts after the next `cozy down`.
set -eu

releases=https://github.com/cozy-creator/cozy/releases
if [ -n "${COZY_VERSION:-}" ]; then url="$releases/download/$COZY_VERSION"; else url="$releases/latest/download"; fi
url="${COZY_RELEASE_URL:-$url}"
bin="${COZY_PREFIX:-$HOME/.local}/bin"

fail() { echo "cozy install: $*" >&2; exit 1; }
case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) fail "no release for $(uname -s)" ;; esac
case "$(uname -m)" in x86_64 | amd64) arch=amd64 ;; aarch64 | arm64) arch=arm64 ;; *) fail "no release for $(uname -m)" ;; esac
asset="cozy-$os-$arch.tar.gz"
command -v uv >/dev/null || fail "uv is required for cozy-runtime and tfs: https://docs.astral.sh/uv/getting-started/installation/"

# Staged inside bin/ so the final move is a rename on one filesystem, never a half-written copy.
mkdir -p "$bin"
stage="$(mktemp -d "$bin/.cozy-install.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
curl -fsSL -o "$stage/$asset" "$url/$asset" || fail "cannot download $url/$asset"
curl -fsSL -o "$stage/SHA256SUMS" "$url/SHA256SUMS" || fail "cannot download $url/SHA256SUMS"
want="$(awk -v n="$asset" '$2 == n { print $1 }' "$stage/SHA256SUMS")"
if command -v sha256sum >/dev/null; then got="$(sha256sum "$stage/$asset")"; else got="$(shasum -a 256 "$stage/$asset")"; fi
got="${got%% *}"
[ -n "$want" ] && [ "$got" = "$want" ] || fail "$asset is sha256:$got, SHA256SUMS says ${want:-nothing}; nothing was replaced"
tar -xzf "$stage/$asset" -C "$stage" cozy
now="$("$stage/cozy" -v)" || fail "the new cozy does not run; nothing was replaced"

# A standalone tensorfs tool would still claim tfs: its later upgrade would relink an unpinned
# tfs, and its uninstall would delete the one cozy-runtime installs.
if uv tool list 2>/dev/null | grep -q '^tensorfs '; then uv tool uninstall tensorfs; fi
host_tools() { uv tool install --force --refresh-package cozy-runtime --refresh-package tensorfs --python 3.12 --with-executables-from tensorfs 'cozy-runtime[media,model-execution]>=0.18.67'; }
host_tools || fail "host tool installation failed; cozy was not replaced"
tools="$(uv tool dir --bin)"

was="$("$bin/cozy" -v 2>/dev/null || echo none)"
mv -f "$stage/cozy" "$bin/cozy"
echo "verified: $asset sha256:$got"
echo "cozy:     $was -> $now ($bin/cozy)"
echo "runtime:  $("$tools/cozy-runtime" version | sed -n 's/^distribution: *//p')"
echo "tfs:      $("$tools/tfs" version)"

# Completion goes where each shell autoloads it, written by the binary just installed so an
# upgrade rewrites it. A completion that cannot be written warns; the install already stands.
data="${XDG_DATA_HOME:-$HOME/.local/share}"
conf="${XDG_CONFIG_HOME:-$HOME/.config}"
completion() { # <shell> <file>
  if mkdir -p "$(dirname "$2")" && "$bin/cozy" completion "$1" >"$2.tmp" && mv -f "$2.tmp" "$2"; then
    echo "complete: $1 $2"
  else
    rm -f "$2.tmp"; echo "warning:  $1 completion was not written to $2" >&2
  fi
}
completion bash "$data/bash-completion/completions/cozy"
if command -v fish >/dev/null || [ -d "$conf/fish" ]; then completion fish "$conf/fish/completions/cozy.fish"; fi
if command -v zsh >/dev/null; then
  # zsh has no per-user autoload directory: write one only where this user's zsh already looks.
  if zsh -ic 'print -rl -- $fpath' </dev/null 2>/dev/null | grep -qxF "$data/zsh/site-functions"; then
    completion zsh "$data/zsh/site-functions/_cozy"
  else
    echo "complete: zsh: add 'source <(cozy completion zsh)' to ~/.zshrc after compinit"
  fi
fi

for dir in "$bin" "$tools"; do
  case ":$PATH:" in *":$dir:"*) ;; *) echo "note:     add $dir to PATH" ;; esac
done | uniq
