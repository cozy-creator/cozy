#!/usr/bin/env bash
# Wheel smoke (tfs-001, extended by tfs-007): build the abi3 wheel, install it into a FRESH
# venv, and READ A REAL NAMED TENSOR out of a fresh current-layout store — a lease,
# a plan in the caller's traversal order, a slot ring, and the digest the Rust path banks.
#
# An import smoke proves the wheel loads. This proves the wheel WORKS. It is the repo's
# end-to-end test and CI runs it on every push.
#
#   scripts/wheel-smoke.sh [workdir] [python] [--skip-build] [--wheels <dir>]
set -euo pipefail
cd "$(dirname "$0")/.."
work=""
py=python3
skip_build=""
wheels=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) skip_build=1 ;;
        --wheels) wheels="$2"; shift ;;
        *) if [ -z "$work" ]; then work="$1"; else py="$1"; fi ;;
    esac
    shift
done
work="${work:-$(mktemp -d)}"
mkdir -p "$work"
"$py" scripts/version_contract.py source

wheels="${wheels:-$work/wheels}"
if [ -z "$skip_build" ]; then
    scripts/build-wheel.sh --out "$wheels" >"$work/build.log" 2>&1 || {
        tail -20 "$work/build.log"; exit 1; }
fi
cargo build --quiet --release --bin tfs
fixture=$work/store
write=$(target/release/tfs checkpoint write "$fixture" --blocks 1 --plain)
SNAP=$(sed -n 's/.*manifest_id     sha256:\([0-9a-f]\{64\}\).*/\1/p' <<<"$write")
test -n "$SNAP"
wheel=$(ls "$wheels"/tensorfs-*.whl | head -1)
echo "wheel: $(basename "$wheel")"
"$py" scripts/version_contract.py wheel "$wheel"

rm -rf "$work/venv"
"$py" -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet "$wheel"
echo "installed into fresh venv: $("$work/venv/bin/python" -V)"
"$work/venv/bin/python" scripts/version_contract.py installed "$wheel"
# `py.typed` rides in this wheel, so the stub beside the extension is what every consumer's
# type checker believes. Checked HERE, against the installed pair, because the wheel is what
# a pod runs (tfs-077). A `__getattr__` back in the stub, a binding whose parameters moved,
# a refusal class the facade forgot to re-export: all of them stop here.
"$work/venv/bin/python" scripts/stub_conform.py
"$work/venv/bin/python" scripts/typed-consumer.py
"$work/venv/bin/python" scripts/import-tree.py
"$work/venv/bin/python" scripts/fit-runs.py
"$work/venv/bin/python" scripts/derived-resume.py
"$work/venv/bin/python" scripts/derived-declaration.py
"$work/venv/bin/python" scripts/derived-graft.py
"$work/venv/bin/python" scripts/read-model-parts.py
"$work/venv/bin/python" scripts/derived-writer-concurrency.py
"$work/venv/bin/python" scripts/scoped-derived.py
"$work/venv/bin/python" scripts/scoped-model-reader.py
"$work/venv/bin/python" scripts/read-plan-selection.py
"$work/venv/bin/python" scripts/read-lease-concurrency.py
"$work/venv/bin/python" scripts/plane-host-tier.py
# The frozen corpus, through the FACADE. `cargo test --test vectors` runs the same vectors
# through Rust; this proves the marshal boundary agrees — same ids, same typed refusals.

"$work/venv/bin/python" - "$fixture" "$SNAP" <<'PY'
import hashlib, os, pathlib, sys, tempfile

store_path, snap = pathlib.Path(sys.argv[1]), sys.argv[2]
import tensorfs
print("import tensorfs OK ->", tensorfs.__file__, tensorfs.__version__)

with tempfile.TemporaryDirectory(prefix="tensorfs-default-home-") as default_root:
    os.environ["TENSORFS_HOME"] = default_root
    assert tensorfs.Store.ensure().root == default_root
    assert tensorfs.Store.ensure().root == default_root
    del os.environ["TENSORFS_HOME"]

assert "torch" not in sys.modules, "torch leaked into sys.modules"
print("torch in sys.modules:", "torch" in sys.modules, "| dtypes:", len(tensorfs.DTYPES))

store = tensorfs.Store.open(str(store_path))
info = store.manifest(snap)
assert info["manifest_id"] == "sha256:" + snap
header = tensorfs.parse_header(info["header"])
assert tensorfs.object_id(info["manifest"]) == "sha256:" + snap

# One NAMED tensor, end to end: the traversal is the caller's, the lease is verified, the
# buffer is the caller's, and the bytes are checked against the part digest the header
# declares.
names = [(c, k) for c in header["components"] for k in header["components"][c]]
comp, key = names[0]
tensor = header["components"][comp][key]
print(f"named tensor: {comp}/{key} {tensor['logical']['logical_dtype']} {tensor['logical']['shape']}")
plan = tensorfs.plan(info["header"], names, window=1 << 16)
assert len(plan) > 0 and plan.bytes > 0

with store.acquire_cozytensors(snap) as lease:
    print(f"lease: {len(lease.objects)} objects, {lease.bytes} B verified")
    for asset_name, asset in header["assets"].items():
        asset_bytes = lease.read_asset(info["header"], asset_name)
        assert len(asset_bytes) == asset["logical_length"]
        assert hashlib.sha256(asset_bytes).hexdigest() == asset["logical_sha256"]
    role, part = next(iter(tensor["parts"].items()))
    segments = part.get("segments", [])
    data = bytearray(sum(s["length"] for s in segments) if segments else len(part["inline"]))
    lease.read_part_into(info["header"], comp, key, role, 0, data)
    offset = 0
    for segment in segments:
        chunk = data[offset:offset + segment["length"]]
        assert hashlib.sha256(chunk).hexdigest() == segment["sha256"]
        offset += segment["length"]
    assert segments or data == part["inline"]
print(f"read {comp}/{key} {role}: {len(data)} B, matches the header")

with tempfile.TemporaryDirectory(prefix="tensorfs-transfer-") as transfer_root:
    destination = tensorfs.Store.init(pathlib.Path(transfer_root) / "store")
    operation = destination.begin_operation("wheel-acquire", "tensorhub", "smoke")
    operation.hold_manifest(snap, len(info["manifest"]))
    closure = store.walk_cozytensors(snap)
    copy = pathlib.Path(transfer_root) / "object"
    for reached in closure:
        copy.write_bytes(store.document(reached["id"], reached["length"]))
        destination.put_file(copy, reached["id"], reached["length"])
    admitted = destination.put_manifest(info["manifest"], "sha256:" + snap, len(info["manifest"]))
    assert admitted["id"] == "sha256:" + snap
    operation.commit_release(None, "1.0.0", "cpu", snap, len(info["manifest"]))
    with destination.acquire_manifest(snap) as acquired:
        assert acquired.objects
    print(f"acquisition: {len(closure)} objects -> manifest -> repository -> lease")

try:
    tensorfs.parse_header(info["manifest"])
except tensorfs.errors.MalformedCbor as e:
    print("typed refusal :", e.code, "|", e.detail[:60])

print("SMOKE GREEN")
PY
