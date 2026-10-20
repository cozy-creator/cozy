#!/usr/bin/env bash
# Run only in the hosted Product25 checkout after extracting this exact proof prefix.
set -euo pipefail
product=25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2
proof=${H3_PROOF_SHA:?set the immutable metadata/client proof commit}
[[ "$proof" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$product"
git diff --exit-code "$product" -- crates Cargo.toml Cargo.lock
prefix=agent-tools/h3-a11-metadata-20261020
example=crates/tensorfs/examples/h3_metadata_stage.rs
test ! -e "$example"
mkdir -p "$(dirname "$example")"
cp "$prefix/h3-stage.rs" "$example"
export A11_NATIVE_SOURCE=$product
cargo clippy --locked -p tensorfs --example h3_metadata_stage -- -D warnings
cargo build --release --locked -p tensorfs --example h3_metadata_stage
out=$PWD/dist/testbins/h3-a11
mkdir -p "$out"
python3 "$prefix/stage-only-replay.py" --bundle "$prefix/bundle.json" \
  --migrator "$PWD/target/release/examples/h3_metadata_stage" --destination "$out/staged"
uv run --no-project --with cbor2==6.1.5 python "$prefix/compare-h3.py" \
  "$prefix/bundle.json" "$out/staged" crates/tensorfs/conformance/reader.py
# Native input is entirely the committed metadata fixture; only replay binds loopback.
git diff --exit-code "$product" -- crates Cargo.toml Cargo.lock
export H3_ARTIFACT_OUT=$out H3_PRODUCT_SHA=$product H3_PROOF_SHA=$proof
python3 - <<'PYPROOF'
import hashlib, io, json, os, pathlib, shutil, subprocess, tarfile, gzip
out = pathlib.Path(os.environ['H3_ARTIFACT_OUT'])
prefix = pathlib.Path('agent-tools/h3-a11-metadata-20261020')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
unknown = subprocess.check_output(['git','ls-files','--others','--exclude-standard'], text=True).splitlines()
assert all(p.startswith(str(prefix) + '/') or p == 'crates/tensorfs/examples/h3_metadata_stage.rs' for p in unknown), unknown
binary = pathlib.Path('target/release/examples/h3_metadata_stage')
client = out / 'client'; client.mkdir()
shutil.copy2(binary, client / 'h3_metadata_stage')
shutil.copy2(prefix / 'h3-stage.rs', client / 'h3-stage.rs')
for name in ('stage-receipt.json', 'independent-comparison.json', 'replay-requests.json'):
    shutil.copy2(prefix / name, out / name)
comparison = json.loads((out / 'independent-comparison.json').read_bytes())
assert comparison['unique_retained_chunks'] == 1676 and len(comparison['datasets']) == 1
assert comparison['datasets'][0]['tensor_count'] == 3858
provenance = {
    'purpose': 'Selected H3 int8 A11 metadata-only staging, no live update',
    'product_source': os.environ['H3_PRODUCT_SHA'], 'proof_source': os.environ['H3_PROOF_SHA'],
    'checkout_source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'client_sha256': sha(prefix / 'h3-stage.rs'), 'binary_sha256': sha(binary),
    'bundle_sha256': sha(prefix / 'bundle.json'), 'cargo_lock_sha256': sha(pathlib.Path('Cargo.lock')),
    'migrator_sha256': sha(pathlib.Path('crates/tensorfs/src/migrate/d10.rs')),
    'comparator_sha256': sha(prefix / 'compare-h3.py'), 'replay_sha256': sha(prefix / 'stage-only-replay.py'),
    'rustc': subprocess.check_output(['rustc','-vV'],text=True),
    'source_queries': 0, 'tensor_payload_downloads': 0, 'source_mutations': 0,
    'production_paths_unchanged': True, 'controller_native_execution': False,
}
(out / 'provenance.json').write_text(json.dumps(provenance,indent=2) + '\n')
# Portable FileAsset: selected D10 metadata, original immutable metadata and provenance only.
entries = [(out / name, name) for name in ('provenance.json','stage-receipt.json','independent-comparison.json')]
entries.append((prefix / 'bundle.json', 'source/bundle.json'))
entries.extend((p, 'source/objects/' + p.name) for p in sorted((prefix / 'objects').iterdir()))
entries.extend((p, 'd10/' + str(p.relative_to(out / 'staged')))
               for p in sorted((out / 'staged').rglob('*'))
               if p.is_file() and (p.relative_to(out / 'staged').parts[0] in ('format','blobs','refs')))
manifest = [{'path': name, 'length': p.stat().st_size, 'sha256': sha(p)} for p, name in entries]
raw = io.BytesIO()
with tarfile.open(fileobj=raw, mode='w') as archive:
    for p, name in entries:
        data = p.read_bytes(); info = tarfile.TarInfo(name)
        info.size = len(data); info.mode = 0o644; info.mtime = 0
        archive.addfile(info, io.BytesIO(data))
asset = gzip.compress(raw.getvalue(), mtime=0)
(out / 'h3-int8-d10-metadata.tar.gz').write_bytes(asset)
receipt = {'path':'h3-int8-d10-metadata.tar.gz','length':len(asset),'sha256':hashlib.sha256(asset).hexdigest(),
           'files':manifest,'tensor_payload_files':0,'credential_or_request_logs':False}
(out / 'metadata-bundle-receipt.json').write_text(json.dumps(receipt,indent=2) + '\n')
print(json.dumps({'provenance':provenance, 'metadata_asset':{k:v for k,v in receipt.items() if k != 'files'}}, sort_keys=True))
PYPROOF
