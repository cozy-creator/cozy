#!/usr/bin/env bash
# The per-platform wheel proof: install ONE built wheel into a fresh venv on the platform it
# is tagged for, then run the shipped CLI and the Python facade over the frozen corpus.
# Linux additionally runs scripts/wheel-smoke.sh (a real store end to end) in ci.yaml.
#
#   scripts/wheel-proof.sh <wheel> <workdir>
set -euo pipefail
cd "$(dirname "$0")/.."
wheel=$1
work=$2
py="${PYTHON:-python3}"

rm -rf "$work/venv"
mkdir -p "$work"
"$py" -m venv "$work/venv"
bin="$work/venv/bin"
"$bin/python" -m pip install --quiet "$wheel"
echo "installed $(basename "$wheel") into $("$bin/python" -V)"

"$bin/python" scripts/version_contract.py installed "$wheel"
"$bin/tfs" version
"$bin/tfs" vectors verify vectors
"$bin/tfs" conform
"$bin/python" scripts/facade-conform.py vectors
echo "WHEEL PROOF GREEN: $(basename "$wheel")"
