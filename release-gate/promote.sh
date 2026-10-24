#!/bin/bash
# promote.sh [--dry-run] <gate run id | gate dir>...: the one step that hands a passed candidate set to users, the
# same bytes the gate ran (tracker design ci-v2, pre-release and promotion). verify.sh first; then, for what the set
# holds, in this order:
#   TensorFS  publish-tensorfs.yaml on tensorfs-v<V> with gate_run: its prerelease's exact files to PyPI;
#   Runtime   cozy-runtime publish.yaml prebuilt with gate_run: its prerelease's exact files to PyPI;
#   cozy      release-promote.yml: the v<V> prerelease becomes latest (`cozy update`, the installer) and Homebrew's;
#   packages  `cozy package yank --undo` of each pinned candidate: the unversioned default again.
# The worker image layer and the Hub pin (/v1/software) stay the release owner's, after this.
set -euo pipefail
DRY=; [ "${1:-}" != --dry-run ] || { DRY=1; shift; }
HERE=$(cd "$(dirname "$0")" && pwd)
set_=$("$HERE/verify.sh" "$@")
gate=hand; [ $# -eq 1 ] && [[ $1 =~ ^[0-9]+$ ]] && gate=$1
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
  run gh workflow run release-promote.yml -R cozy-creator/cozy-creator -f tag="v$c" -f gate="$gate"
fi
for p in $(jq -r '.pins[]?' <<< "$set_"); do run cozy package yank --undo "$p" -y; done
echo "promoted: $(jq -c . <<< "$set_"). Then the worker image layer and the Hub pin, to this set."
