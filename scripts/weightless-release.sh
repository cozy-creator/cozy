#!/usr/bin/env bash
# Build the WEIGHTLESS acceptance endpoint release — cl-013's fixture archive.
#
#   scripts/weightless-release.sh [--runtime-sha <sha>] [--out <dir>] [--version <v>]
#
# `fixtures/weightless/` is the endpoint; its only dependency is the BASE cozy-runtime
# wheel (msgspec, protobuf, grpcio — torch-free, tensorfs-free), so the archive is small
# and its install pulls tens of megabytes rather than a CUDA closure. That is what lets the
# clean-machine fixture run an install and a real invoke end to end without a card, without
# the bench store, and without a weights fetch.
#
# The descriptor is written by the runtime the release ITSELF pins, inside the release's
# own venv — never a host venv — so the archive is reproducible from this repo plus one
# cozy-runtime commit.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# f1625f9: the floor, not a preference — this host enters the supervisor through
# `cozy-runtime serve` (db4ab8a), reads `job_descriptor_id` off `describe` (4485f27), stages
# WEIGHTLESS binding records (a3c3d72) and writes the record's NEW key set (f1625f9), so a
# release pinning an older runtime either cannot be served at all or reads a record whose
# every key it refuses as unknown.
RUNTIME_SHA="${RUNTIME_SHA:-f1625f9}"
OUT="${OUT:-$HOME/.cache/cozy/cl-013}"
VERSION="${VERSION:-1.0.0}"
ENDPOINT="${ENDPOINT:-cozy/weightless}"

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    *) echo "usage: $0 [--runtime-sha <sha>] [--out <dir>] [--version <v>]" >&2; exit 2 ;;
  esac
done
SLUG="${ENDPOINT#*/}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$OUT/runtime"
T="$OUT/tree-$SLUG"
rm -rf "$RT" "$T"
mkdir -p "$RT" "$T/vendor"

FULL="$(git -C "$RUNTIME_REPO" rev-parse "$RUNTIME_SHA")"
# READ-ONLY. That repo has a concurrent writer; a checkout of its working tree would make
# this release a photograph of somebody else's edit.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"

cp "$ROOT/fixtures/weightless/weightless.py" "$T/weightless.py"
cp "$ROOT/fixtures/weightless/endpoint.toml" "$T/endpoint.toml"

cat > "$T/pyproject.toml" <<TOML
[project]
name = "cozy-weightless-endpoint"
version = "$VERSION"
requires-python = ">=3.11"
dependencies = ["cozy-runtime==0.0.1"]

[tool.uv.sources]
cozy-runtime = { path = "vendor/cozy_runtime-0.0.1-py3-none-any.whl" }
TOML

( cd "$T" && nice -n 19 uv lock --quiet && nice -n 19 uv sync --locked --quiet )
( cd "$T" && PYTHONPATH="$T" nice -n 19 "$T/.venv/bin/python" -m cozy_runtime.cli.main \
    --dir "$T" describe --write-descriptor >/dev/null )

ARCHIVE="$OUT/$SLUG-$VERSION.tar.gz"
nice -n 19 python3 "$ROOT/scripts/pack.py" "$T" "$ENDPOINT" "$VERSION" "$ARCHIVE" | sed 's/^/  /'
echo "  runtime: $FULL (git archive, read-only)"
