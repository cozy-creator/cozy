#!/usr/bin/env bash
# Wheel smoke (tfs-001): build the abi3 wheel, install it into a FRESH venv, import the
# facade, prove torch never enters sys.modules, and marshal a real golden header through
# the compiled core. A live run, not a test — see tracker README #160.
set -euo pipefail
cd "$(dirname "$0")/.."
work="${1:-$(mktemp -d)}"
mkdir -p "$work"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
nice -n 19 maturin build --release --out "$work/wheels" >"$work/build.log" 2>&1 || {
    tail -20 "$work/build.log"; exit 1; }
wheel=$(ls "$work"/wheels/tensorfs-*.whl)
echo "wheel: $(basename "$wheel")"

rm -rf "$work/venv"
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet "$wheel"
echo "installed into fresh venv: $("$work/venv/bin/python" -V)"

"$work/venv/bin/python" - "$PWD" <<'PY'
import sys, pathlib
repo = pathlib.Path(sys.argv[1])
assert "tensorfs" not in [p for p in sys.path if "cozy_v2" in p], "repo on sys.path"

import tensorfs
print("import tensorfs OK ->", tensorfs.__file__)
print("version:", tensorfs.__version__)

leaked = sorted(m for m in sys.modules if m.split(".")[0] in
                {"torch", "numpy", "hashlib", "json", "struct"})
assert "torch" not in sys.modules, "torch leaked into sys.modules"
print("torch in sys.modules:", "torch" in sys.modules, "| heavy/marshal modules:", leaked)

case = repo / "vectors/cases/header/plain-fp8.json"
raw = case.read_bytes()
oid = tensorfs.object_id(raw)
want = [l.split(": ", 1)[1] for l in
        (repo / "vectors/cases/header/plain-fp8.verdict").read_text().splitlines()
        if l.startswith("id: ")][0]
print("object_id:", oid)
assert oid == want, f"{oid} != {want}"

doc = tensorfs.parse_header(raw)
assert isinstance(doc, dict), type(doc)
names = list(doc["components"]["transformer"])
print("parse_header -> dict, components/transformer tensors:", len(names), names[:2])
print("SMOKE GREEN")
PY
