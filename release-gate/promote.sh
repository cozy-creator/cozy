#!/bin/bash
# promote.sh [--dry-run] <release-gate run id> | --record <URL> <gate dir>...: the one step that hands a passed
# candidate set to users, the same bytes the gate ran (tracker design ci-v2, pre-release and promotion). Testing is
# manual: the evidence is the passing release-gate.yml run the release agent dispatched, or the hand gate's dirs
# with the URL where their result is recorded. verify.sh first; then, for what the set holds, in this order:
#   TensorFS  publish-tensorfs.yaml on tensorfs-v<V> with gate_run: its prerelease's exact files to PyPI;
#   Runtime   cozy-runtime publish.yaml prebuilt with gate_run: its prerelease's exact files to PyPI;
#   cozy      cozy-promote.yaml: the v<V> prerelease becomes latest (`cozy update`, the installer) and Homebrew's;
#   packages  `cozy package yank --undo` of each pinned candidate: the unversioned default again.
# The worker image layer and the Hub pin (/v1/software) stay the release owner's, after this.
set -euo pipefail
DRY=; [ "${1:-}" != --dry-run ] || { DRY=1; shift; }
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = --record ]; then
  gate=${2:-}; shift 2
  [[ $gate == https://* ]] && [ $# -gt 0 ] || { echo "--record <URL> <gate dir>...: the hand gate's dirs and where their result is recorded" >&2; exit 2; }
else
  [ $# -eq 1 ] && [[ $1 =~ ^[0-9]+$ ]] || { sed -n '2,5s/^# \{0,1\}//p' "$0" >&2; exit 2; }
  gate=$1
fi
set_=$("$HERE/verify.sh" "$@")
run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }
field() { jq -r ".$1 // empty" <<< "$set_"; }
ver() { cut -d- -f2 <<< "$1"; }
if [ -n "$(field tensorfs_wheel)" ]; then
  run gh workflow run publish-tensorfs.yaml -R cozy-creator/cozy-creator --ref "tensorfs-v$(ver "$(field tensorfs_wheel)")" -f gate_run="$gate"
fi
if [ -n "$(field runtime_wheel)" ]; then
  v=$(ver "$(field runtime_wheel)")
  sums=$(gh release download "v$v" -R cozy-creator/cozy-runtime -p SHA256SUMS -O - | tr '\n' ' ')
  run gh workflow run publish.yaml -R cozy-creator/cozy-runtime --ref master -f release_tag="v$v" -f mode=prebuilt \
    -f gated_sha256sums="$sums" -f gate_run="$gate"
fi
c=$(field cozy)
if [ -n "$c" ] && [ "$(gh release view "v$c" -R cozy-creator/cozy-creator --json isPrerelease -q .isPrerelease 2>/dev/null)" = true ]; then
  run gh workflow run cozy-promote.yaml -R cozy-creator/cozy-creator -f tag="v$c" -f gate="$gate"
fi
for p in $(jq -r '.pins[]?' <<< "$set_"); do run cozy package yank --undo "$p" -y; done
echo "promoted: $(jq -c . <<< "$set_"). Then the worker image layer and the Hub pin, to this set."
