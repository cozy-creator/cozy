#!/usr/bin/env bash
# Build an SDXL ENDPOINT RELEASE — the archive `cozy install --from --digest` verifies.
#
# cl-006 served its endpoint from a hand-written `--dev-endpoint` document, because an
# install generation could not yet carry the launch facts a supervisor needs. cl-010
# deletes that door, so the endpoint has to arrive the way a user's endpoint arrives: as a
# release archive with its own source, its own lock over its whole closure, and the two
# wheels that are not on an index yet (cozy-runtime, built from a PINNED read-only
# `git archive`; tensorfs, tfs-007's compiled facade).
#
#   scripts/sdxl-release.sh [--kind unet|pipeline] [--runtime-sha <sha>] [--out <dir>]
#                           [--endpoint org/name] [--artifact-release <name>]
#
# TWO KINDS, one builder:
#   unet      cr-005's single-component endpoint (`denoise`/`stubborn`/`pair`) — cl-010's
#             journey installs this one.
#   pipeline  cr-008b's FOUR-COMPONENT SDXL text-to-image endpoint (`generate`) — cl-003's
#             M4 proof installs this one. It bundles its own tokenizer vocabularies, which
#             are the endpoint's assets exactly like the model library it imports.
#
# `--artifact-release` rewrites the `[bindings]` table's `release`, which is how the SAME
# endpoint source is released against a different RUNG of the same model repo (cl-003 uses
# it for cr-006's fp8 UNet). Bindings state SELECTION; the code names nothing.
#
# Prints the archive path and its sha256 — what `cozy install --digest` checks.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# No pinned floor: a pinned SHA is a stored artifact in recipe form, and the rev-4 wire
# bump proved it stales (cl-030). internal/live's suite verifies the peer's declared wire
# schema against this tree before invoking this script; standalone use packages HEAD.
RUNTIME_SHA="${RUNTIME_SHA:-HEAD}"
# tfs-007's compiled facade, built from a PINNED read-only `git archive` of tensorfs
# 846532c. The wheel is not incidental: cozytensors' ENCODING REGISTRY ships inside it, so
# the release's pinned tensorfs is what decides which encodings this endpoint can read
# (README §5 — "the pinned registry + reader set IS a runtime's encoding capability").
# cl-003 found this the hard way: an older wheel carried 9 seed digests and no
# `fp8-scaled-scalar/1`, and the fp8 rung refused `unknown_encoding` naming the exact digest
# its own checkpoint cites.
TENSORFS_WHEEL="${TENSORFS_WHEEL:-/tmp/cozy-wheels-cl003/tensorfs-0.0.1-cp311-abi3-linux_x86_64.whl}"
DESCRIBE_PY="${DESCRIBE_PY:-$RUNTIME_REPO/corpus/.venv/bin/python}"
HF="${HF:-$HOME/cozy_v2/tensorfs-bench/hf/sdxl}"
OUT="${OUT:-$HOME/.cache/cozy/cl-010}"
VERSION="${VERSION:-1.0.0}"
KIND="unet"
ENDPOINT=""
ARTIFACT_RELEASE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --artifact-release) ARTIFACT_RELEASE="$2"; shift 2 ;;
    *) echo "usage: $0 [--kind unet|pipeline] [--runtime-sha <sha>] [--out <dir>]" >&2; exit 2 ;;
  esac
done

case "$KIND" in
  unet)     NAME="sdxl-unet";     PROJECT="sdxl-unet-endpoint" ;;
  pipeline) NAME="sdxl-pipeline"; PROJECT="sdxl-pipeline-endpoint" ;;
  *) echo "unknown --kind $KIND (unet|pipeline)" >&2; exit 2 ;;
esac
[ -n "$ENDPOINT" ] || ENDPOINT="cozy/$NAME"
SLUG="${ENDPOINT#*/}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$OUT/runtime"
T="$OUT/tree-$SLUG"
rm -rf "$RT" "$T"
mkdir -p "$RT" "$T/vendor" "$T/corpus"

FULL="$(git -C "$RUNTIME_REPO" rev-parse "$RUNTIME_SHA")"
# READ-ONLY. That repo has a concurrent writer; a checkout of its working tree would make
# this release a photograph of somebody else's edit.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"
RUNTIME_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$RT/pyproject.toml" | head -1)"
RUNTIME_WHEEL="$(find "$T/vendor" -maxdepth 1 -type f -name 'cozy_runtime-*.whl' -printf '%f\n')"
[ -n "$RUNTIME_VERSION" ] && [ -n "$RUNTIME_WHEEL" ] && [ "$(printf '%s\n' "$RUNTIME_WHEEL" | wc -l)" -eq 1 ] || {
  echo "refusing: expected one versioned cozy-runtime wheel from $FULL" >&2
  exit 2
}
cp "$TENSORFS_WHEEL" "$T/vendor/"
TFS_WHEEL_NAME="$(basename "$TENSORFS_WHEEL")"

# The endpoint's own source: the entrypoints, and the model class they bind.
if [ "$KIND" = unet ]; then
  cp "$RT/corpus/endpoint/sdxl_gpu.py" "$T/sdxl_gpu.py"
  cp "$RT/corpus/endpoint/endpoint.toml" "$T/endpoint.toml"
  cp "$RT/corpus/sdxl_real.py" "$T/corpus/sdxl_real.py"
else
  cp "$RT/corpus/pipeline/sdxl_txt2img.py" "$T/sdxl_txt2img.py"
  cp "$RT/corpus/pipeline/endpoint.toml" "$T/endpoint.toml"
  cp "$RT/corpus/sdxl_pipeline.py" "$T/corpus/sdxl_pipeline.py"
  # cl-003's non-cooperative entrypoint, and the application object that carries it: it
  # imports `sdxl_txt2img`, so `generate` and `condition` register beside `stubborn`.
  cp "$ROOT/scripts/sdxl_proof.py" "$T/sdxl_proof.py"
  sed -i 's|^object = .*|object = "sdxl_proof:app"|' "$T/endpoint.toml"
  # The BUNDLED tokenizer vocabularies. `_TOKENIZERS` is the endpoint file's own directory,
  # so they ride the release tree — an asset of the endpoint, never a construction-config
  # path (a path in a construction config refuses, §1.1).
  for sub in tokenizer tokenizer_2; do
    mkdir -p "$T/$sub"
    for f in vocab.json merges.txt tokenizer_config.json special_tokens_map.json; do
      cp "$HF/$sub/$f" "$T/$sub/$f"
    done
  done
fi
: > "$T/corpus/__init__.py"

if [ -n "$ARTIFACT_RELEASE" ]; then
  # The rung the endpoint SELECTS. One line of the closed `[bindings]` grammar; the source
  # above is byte-identical across rungs, which is the point of the two-sided contract.
  sed -i "s|^release = .*|release = \"$ARTIFACT_RELEASE\"|" "$T/endpoint.toml"
fi

cat > "$T/pyproject.toml" <<TOML
# The endpoint as a RELEASE. \`cozy install\` builds its venv with \`uv sync --locked\`
# against this lock, so the cozy-runtime that serves the endpoint is the one the release
# pinned — never the host's.
[project]
name = "$PROJECT"
version = "$VERSION"
requires-python = ">=3.11"
dependencies = [
    "cozy-runtime[corpus]==$RUNTIME_VERSION",
    "tensorfs==0.0.1",
]

# uv records path sources RELATIVE to the project root, so an in-tree wheel relocates with
# the archive and an out-of-tree one does not (cl-009's finding).
[tool.uv.sources]
cozy-runtime = { path = "vendor/$RUNTIME_WHEEL" }
tensorfs = { path = "vendor/$TFS_WHEEL_NAME" }
TOML

# The committed descriptor, written by the runtime the release itself pins. `cozy install`
# re-derives it in the generation's OWN venv and refuses a stale one (cl-009's exit 13), so
# writing it here with the same pinned tree is what makes the install green.
( cd "$T" && PYTHONPATH="$RT:$RT/src" nice -n 19 "$DESCRIBE_PY" -m cozy_runtime.cli.main \
    --dir "$T" describe --write-descriptor >/dev/null )

( cd "$T" && nice -n 19 uv lock --quiet )

ARCHIVE="$OUT/$SLUG-$VERSION.tar.gz"
nice -n 19 python3 "$ROOT/scripts/pack.py" "$T" "$ENDPOINT" "$VERSION" "$ARCHIVE" | sed 's/^/  /'
echo "  runtime: $FULL (git archive, read-only)"
echo "  tensorfs: $TFS_WHEEL_NAME"
