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
"$work/venv/bin/python" scripts/source-artifacts.py
"$work/venv/bin/python" scripts/ensure-proof.py "$work/venv/bin/tfs"
"$work/venv/bin/python" scripts/pull-control-proof.py "$work/venv/bin/tfs"
"$work/venv/bin/python" scripts/import-tree.py
"$work/venv/bin/python" scripts/keyed-roots.py
"$work/venv/bin/python" scripts/trust-index.py
"$work/venv/bin/python" scripts/fit-runs.py
"$work/venv/bin/python" scripts/derived-resume.py
"$work/venv/bin/python" scripts/derived-declaration.py
"$work/venv/bin/python" scripts/derived-graft.py
"$work/venv/bin/python" scripts/derived-retention.py
"$work/venv/bin/python" scripts/read-model-parts.py
"$work/venv/bin/python" scripts/derived-writer-concurrency.py
"$work/venv/bin/python" scripts/scoped-derived.py
"$work/venv/bin/python" scripts/scoped-model-reader.py
"$work/venv/bin/python" scripts/read-plan-selection.py
"$work/venv/bin/python" scripts/read-lease-concurrency.py "$fixture" "$SNAP"
"$work/venv/bin/python" scripts/plane-host-tier.py
"$work/venv/bin/python" scripts/stats.py "$fixture" "$SNAP"
# The frozen corpus, through the FACADE. `cargo test --test vectors` runs the same vectors
# through Rust; this proves the marshal boundary agrees — same ids, same typed refusals.
"$work/venv/bin/python" scripts/facade-conform.py

"$work/venv/bin/python" - "$fixture" "$SNAP" <<'PY'
import hashlib, os, pathlib, sys, tempfile

store_path, snap = pathlib.Path(sys.argv[1]), sys.argv[2]
import tensorfs
print("import tensorfs OK ->", tensorfs.__file__, tensorfs.__version__)

with tempfile.TemporaryDirectory(prefix="tensorfs-default-home-") as default_root:
    os.environ["TENSORFS_HOME"] = default_root
    assert tensorfs.home() == default_root
    assert tensorfs.home(pathlib.Path(default_root) / "explicit") == str(
        pathlib.Path(default_root) / "explicit"
    )
    default_store = tensorfs.Store.ensure()
    assert default_store.root == default_root
    default_store.prepare_readers()
    assert default_store.complete_cozytensors_manifests() == []
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

order = names
plan = tensorfs.plan(info["header"], order, window=1 << 16)
slots = [memoryview(bytearray(1 << 18)) for _ in range(3)]
digest = hashlib.sha256()

with store.acquire_cozytensors(snap) as lease:
    print(f"lease: {len(lease.objects)} objects, {lease.bytes} B verified")
    first_id, first_length = lease.objects[0]
    for asset_name, asset in header["assets"].items():
        asset_bytes = lease.read_asset(info["header"], asset_name)
        assert len(asset_bytes) == asset["logical_length"]
        assert hashlib.sha256(asset_bytes).hexdigest() == asset["logical_sha256"]

    def on_batch(b):
        digest.update(slots[b.slot][: b.nbytes])
        b.release()

    st = lease.stream(plan, slots, on_batch, readers=2)

got = digest.hexdigest()
print(f"streamed {st['items']} items, {st['bytes']} B -> sha256 {got[:16]}…")
assert st["items"] > 0 and st["bytes"] > 0

with tempfile.TemporaryDirectory(prefix="tensorfs-transfer-") as transfer_root:
    destination = tensorfs.Store.init(transfer_root)
    operation = destination.begin_operation("wheel-acquire", "tensorhub", "smoke")
    operation.hold_manifest(snap, len(info["manifest"]))
    closure = store.walk_cozytensors(snap)
    for reached in closure:
        operation.hold_blob(reached["id"], reached["length"])
        tensorfs.transfer_object(store, destination, reached["id"], reached["length"])
    admitted_manifest = tensorfs.transfer_manifest(
        store, destination, snap, len(info["manifest"])
    )
    assert admitted_manifest["id"] == "sha256:" + snap
    declared = sorted({(row["id"], row["length"]) for row in closure})
    pull = tensorfs.fetch_plan(
        destination,
        "wheel-runtime-closure",
        snap,
        len(info["manifest"]),
        declared,
    )
    assert not pull["wanted"]
    assert tensorfs.fetch_complete_cozytensors(destination, pull["document"]) == (
        len(info["manifest"]) + sum(length for _, length in declared)
    )
    assert all(isinstance(config, bytes) for config in header["configs"].values())
    repo_bytes = operation.commit_release(None, "1.0.0", "cpu", snap, len(info["manifest"]))
    assert destination.repo_get("tensorhub", "smoke") == repo_bytes
    assert destination.resolve_release("tensorhub", "smoke", "1.0.0", "cpu") == {
        "org": "tensorhub",
        "name": "smoke",
        "version": "1.0.0",
        "revision": 1,
        "lane": "cpu",
        "manifest_digest": "sha256:" + snap,
        "manifest_length": len(info["manifest"]),
    }
    local_a = destination.replace_local(
        None, "dev-model", "sha256:" + "11" * 32,
        snap, len(info["manifest"]),
    )
    assert destination.resolve_local("dev-model") == {
        "name": "dev-model",
        "source_selection": "sha256:" + "11" * 32,
        "manifest_digest": "sha256:" + snap,
        "manifest_length": len(info["manifest"]),
    }
    local_b = destination.replace_local(
        local_a, "dev-model", "sha256:" + "22" * 32,
        snap, len(info["manifest"]),
    )
    assert destination.repo_get("local", "dev-model") == local_b
    assert destination.resolve_local("dev-model")["source_selection"] == "sha256:" + "22" * 32
    try:
        destination.replace_local(
            local_a, "dev-model", "sha256:" + "33" * 32,
            snap, len(info["manifest"]),
        )
        raise AssertionError("stale local alias replacement succeeded")
    except tensorfs.errors.RepositoryConflict:
        pass
    destination.remove_local(local_b, "dev-model")
    try:
        destination.resolve_local("dev-model")
        raise AssertionError("removed local alias resolved")
    except tensorfs.errors.RepositoryAbsent:
        pass
    try:
        destination.resolve_release("tensorhub", "smoke", "1.0.1", "cpu")
        raise AssertionError("missing release resolved")
    except tensorfs.errors.ReleaseAbsent:
        pass
    with destination.acquire_manifest(snap) as acquired:
        assert acquired.objects
    print(f"acquisition: {len(closure)} blobs -> manifest -> repository CAS -> lease")

try:
    tensorfs.platform_gate("win32")
except tensorfs.errors.PlatformUnsupported as e:
    print("platform gate:", e.code)

try:
    tensorfs.parse_header(info["manifest"])
except tensorfs.errors.MalformedCbor as e:
    print("typed refusal :", e.code, "|", e.detail[:60])

try:
    store.prepare_model_source(
        "wheel-source-proof", "sha256:" + "44" * 32, [], [], [],
    )
    raise AssertionError("empty model source preparation succeeded")
except tensorfs.errors.CountCap as e:
    print("source facade :", e.code, "| one compiled planner/executor")

print("SMOKE GREEN")
PY
