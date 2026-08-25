#!/usr/bin/env bash
# Build the SDXL UNet ENDPOINT RELEASE cl-010's live journey installs.
#
# cl-006 served this endpoint from a hand-written `--dev-endpoint` document, because an
# install generation could not yet carry the launch facts a supervisor needs. cl-010
# deletes that door, so the endpoint has to arrive the way a user's endpoint arrives: as a
# release archive with its own source, its own lock over its whole closure, and the two
# wheels that are not on an index yet (cozy-runtime, built from a PINNED read-only
# `git archive`; tensorfs, tfs-007's compiled facade).
#
#   scripts/sdxl-release.sh [--runtime-sha <sha>] [--out <dir>]
#
# Prints the archive path and its sha256 — what `cozy install --digest` checks.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# fabd6fc: cr-009's oneof regression FIXED and A/B re-proved (ready_plans=1). Pinned,
# because a verification whose peer can move underneath it measures nothing.
RUNTIME_SHA="${RUNTIME_SHA:-fabd6fc}"
TENSORFS_WHEEL="${TENSORFS_WHEEL:-/tmp/cozy-wheels/tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl}"
DESCRIBE_PY="${DESCRIBE_PY:-$RUNTIME_REPO/corpus/.venv/bin/python}"
OUT="${OUT:-$HOME/.cache/cozy/cl-010}"
VERSION="${VERSION:-1.0.0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: $0 [--runtime-sha <sha>] [--out <dir>]" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$OUT/runtime"
T="$OUT/tree"
rm -rf "$RT" "$T"
mkdir -p "$RT" "$T/vendor" "$T/corpus"

FULL="$(git -C "$RUNTIME_REPO" rev-parse "$RUNTIME_SHA")"
# READ-ONLY. That repo has a concurrent writer; a checkout of its working tree would make
# this release a photograph of somebody else's edit.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"
cp "$TENSORFS_WHEEL" "$T/vendor/"

# The endpoint's own source: the entrypoints, and the model class they bind.
cp "$RT/corpus/endpoint/sdxl_gpu.py" "$T/sdxl_gpu.py"
cp "$RT/corpus/endpoint/endpoint.toml" "$T/endpoint.toml"
cp "$RT/corpus/sdxl_real.py" "$T/corpus/sdxl_real.py"
: > "$T/corpus/__init__.py"

cat > "$T/pyproject.toml" <<'TOML'
# The SDXL UNet endpoint as a RELEASE. `cozy install` builds its venv with
# `uv sync --locked` against this lock, so the cozy-runtime that serves the endpoint is
# the one the release pinned — never the host's.
[project]
name = "sdxl-unet-endpoint"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "cozy-runtime[corpus]==0.0.1",
    "tensorfs==0.0.1",
]

# uv records path sources RELATIVE to the project root, so an in-tree wheel relocates with
# the archive and an out-of-tree one does not (cl-009's finding).
[tool.uv.sources]
cozy-runtime = { path = "vendor/cozy_runtime-0.0.1-py3-none-any.whl" }
tensorfs = { path = "vendor/tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl" }
TOML

# The committed descriptor, written by the runtime the release itself pins. `cozy install`
# re-derives it in the generation's OWN venv and refuses a stale one (cl-009's exit 13), so
# writing it here with the same pinned tree is what makes the install green.
( cd "$T" && PYTHONPATH="$RT:$RT/src" nice -n 19 "$DESCRIBE_PY" -m cozy_runtime.cli.main \
    --dir "$T" describe --write-descriptor >/dev/null )

( cd "$T" && nice -n 19 uv lock --quiet )

ARCHIVE="$OUT/sdxl-unet-$VERSION.tar.gz"
nice -n 19 python3 "$ROOT/scripts/pack.py" "$T" cozy/sdxl-unet "$VERSION" "$ARCHIVE" | sed 's/^/  /'
echo "  runtime: $FULL (git archive, read-only)"
