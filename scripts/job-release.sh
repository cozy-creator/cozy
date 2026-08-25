#!/usr/bin/env bash
# Build the JOB ENDPOINT RELEASE cl-004's live verification installs.
#
# The job is cr-009's own `corpus/job/structural_census.py` — the real structural-derivation
# job it shipped, over a real 6.9 GB cozytensors store — taken verbatim from a PINNED
# read-only `git archive` of cozy-runtime. It is not re-implemented here and not edited:
# cl-004 is the coordinator/CLI half, and the thing it must drive is the job that exists.
#
#   scripts/job-release.sh [--runtime-sha <sha>] [--out <dir>] [--endpoint org/name]
#
# Prints the archive path and its sha256 — what `cozy install --digest` checks.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# a3c3d72: the floor, not a preference — this host enters the supervisor through
# `cozy-runtime serve` (db4ab8a), reads `job_descriptor_id` off `describe` (4485f27) and
# stages WEIGHTLESS binding records (a3c3d72), so a release pinning an older runtime either
# cannot be served at all or cannot serve a modelless entrypoint.
RUNTIME_SHA="${RUNTIME_SHA:-a3c3d72}"
TENSORFS_WHEEL="${TENSORFS_WHEEL:-/tmp/cozy-wheels-cl003/tensorfs-0.0.1-cp311-abi3-linux_x86_64.whl}"
DESCRIBE_PY="${DESCRIBE_PY:-$RUNTIME_REPO/corpus/.venv/bin/python}"
OUT="${OUT:-$HOME/.cache/cozy/cl-004}"
VERSION="${VERSION:-1.0.0}"
ENDPOINT="cozy/census"

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    *) echo "usage: $0 [--runtime-sha <sha>] [--out <dir>] [--endpoint org/name]" >&2; exit 2 ;;
  esac
done
SLUG="${ENDPOINT#*/}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$OUT/runtime"
T="$OUT/tree-$SLUG"
rm -rf "$RT" "$T"
mkdir -p "$RT" "$T/vendor"

FULL="$(git -C "$RUNTIME_REPO" rev-parse "$RUNTIME_SHA")"
# READ-ONLY. That repo has a concurrent writer.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"
cp "$TENSORFS_WHEEL" "$T/vendor/"
TFS_WHEEL_NAME="$(basename "$TENSORFS_WHEEL")"

cp "$RT/corpus/job/structural_census.py" "$T/structural_census.py"
cat > "$T/endpoint.toml" <<'TOML'
# cl-004's live job project: cr-009's own structural-derivation job, verbatim.

[application]
object = "structural_census:app"
TOML

cat > "$T/pyproject.toml" <<TOML
[project]
name = "cozy-census-jobs"
version = "$VERSION"
requires-python = ">=3.11"
dependencies = [
    "cozy-runtime==0.0.1",
    "tensorfs==0.0.1",
]

[tool.uv.sources]
cozy-runtime = { path = "vendor/cozy_runtime-0.0.1-py3-none-any.whl" }
tensorfs = { path = "vendor/$TFS_WHEEL_NAME" }
TOML

# The committed descriptor, written by the runtime the release itself pins.
( cd "$T" && PYTHONPATH="$RT:$RT/src:$T" nice -n 19 "$DESCRIBE_PY" -m cozy_runtime.cli.main \
    --dir "$T" describe --write-descriptor >/dev/null )

( cd "$T" && nice -n 19 uv lock --quiet )

ARCHIVE="$OUT/$SLUG-$VERSION.tar.gz"
nice -n 19 python3 "$ROOT/scripts/pack.py" "$T" "$ENDPOINT" "$VERSION" "$ARCHIVE" | sed 's/^/  /'
echo "  runtime: $FULL (git archive, read-only)"
echo "  tensorfs: $TFS_WHEEL_NAME"
