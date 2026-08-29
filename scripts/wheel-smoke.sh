#!/usr/bin/env bash
# Wheel smoke (tfs-001, extended by tfs-007): build the abi3 wheel, install it into a FRESH
# venv, and READ A REAL NAMED TENSOR out of the committed store at vectors/store — a lease,
# a plan in the caller's traversal order, a slot ring, and the digest the Rust path banks.
#
# An import smoke proves the wheel loads. This proves the wheel WORKS, which is the only
# thing that stops a matrix entry from rotting quietly. A live run, not a test — see
# tracker README #160.
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

# The committed fixture: `tfs checkpoint write vectors/store --blocks 1 --scale 32`.
SNAP=3efa56a9969b1c4a06d1588da74340b972df9b98ae3b8ecc03cb5796e836f479
DIGEST=a5421bd647401296a540be4f122f2fd03e04e7d1a312023b759e36c44b17ef73

wheels="${wheels:-$work/wheels}"
if [ -z "$skip_build" ]; then
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
    nice -n 19 maturin build --release --out "$wheels" >"$work/build.log" 2>&1 || {
        tail -20 "$work/build.log"; exit 1; }
fi
wheel=$(ls "$wheels"/tensorfs-*.whl | head -1)
echo "wheel: $(basename "$wheel")"
"$py" scripts/version_contract.py wheel "$wheel"

rm -rf "$work/venv"
"$py" -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet "$wheel"
echo "installed into fresh venv: $("$work/venv/bin/python" -V)"
"$work/venv/bin/python" scripts/version_contract.py installed "$wheel"

"$work/venv/bin/python" - "$PWD" "$SNAP" "$DIGEST" <<'PY'
import hashlib, pathlib, sys, tempfile

repo, snap, want = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
import tensorfs
print("import tensorfs OK ->", tensorfs.__file__, tensorfs.__version__)

assert "torch" not in sys.modules, "torch leaked into sys.modules"
print("torch in sys.modules:", "torch" in sys.modules, "| dtypes:", len(tensorfs.DTYPES))

store = tensorfs.Store.open(str(repo / "vectors/store"))
info = store.snapshot(snap)
assert info["checkpoint_id"] == info["snapshot_id"] == "sha256:" + snap
header = tensorfs.parse_header(info["header"])
assert tensorfs.object_id(info["manifest"]) == "sha256:" + snap

# One NAMED tensor, end to end: the traversal is the caller's, the lease is verified, the
# buffer is the caller's, and the bytes are checked against the part digest the header
# declares.
names = [(c, k) for c in header["components"] for k in header["components"][c]]
comp, key = names[0]
tensor = header["components"][comp][key]
print(f"named tensor: {comp}/{key} {tensor['logical']['dtype']} {tensor['logical']['shape']}")

order = names
plan = tensorfs.plan(info["header"], order, window=1 << 16)
slots = [memoryview(bytearray(1 << 18)) for _ in range(3)]
digest = hashlib.sha256()

with store.acquire_snapshot(snap) as lease:
    print(f"lease: {len(lease.objects)} objects, {lease.bytes} B verified")
    first_id, first_length = lease.objects[0]

    def on_batch(b):
        digest.update(slots[b.slot][: b.nbytes])
        b.release()

    st = lease.stream(plan, slots, on_batch, readers=2)

got = digest.hexdigest()
print(f"streamed {st['items']} items, {st['bytes']} B -> sha256 {got[:16]}…")
assert got == want, f"{got} != banked {want}"

with tempfile.TemporaryDirectory(prefix="tensorfs-transfer-") as transfer_root:
    destination = tensorfs.Store.init(transfer_root)
    admitted = tensorfs.transfer_object(store, destination, first_id, first_length)
    skipped = tensorfs.transfer_object(store, destination, first_id, first_length)
    assert admitted == {"id": first_id, "length": first_length, "admitted": True}
    assert skipped == {"id": first_id, "length": first_length, "admitted": False}
    assert destination.verify(first_id)["verdict"] == "verified"
    print(f"transfer:   admitted then skipped {first_id[:23]}… through two Store objects")

try:
    tensorfs.platform_gate("win32")
except tensorfs.errors.PlatformUnsupported as e:
    print("platform gate:", e.code)

try:
    tensorfs.parse_header(info["manifest"])
except tensorfs.errors.MalformedCbor as e:
    print("typed refusal :", e.code, "|", e.detail[:60])

print("SMOKE GREEN")
PY
