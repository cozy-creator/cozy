#!/usr/bin/env bash
# The standalone e2e (design §11) in a fresh venv, with no Cozy and no account:
#   scripts/e2e.sh <tensorfs wheel> [workdir]
# Needs uv; downloads torch (CPU), diffusers and transformers, and the tiny repo from Hugging Face.
set -euo pipefail
wheel=$(realpath "$1")
work=${2:-$(mktemp -d)}
here=$(cd "$(dirname "$0")" && pwd)
uv venv --python 3.12 "$work/venv"
UV_TORCH_BACKEND=cpu uv pip install --python "$work/venv/bin/python" "$wheel" \
  torch==2.8.0 diffusers==0.35.1 transformers==4.56.1 numpy==2.3.3
cd "$work"
venv/bin/python "$here/stub_conform.py"
venv/bin/python -I "$here/e2e.py" "$work/run"
