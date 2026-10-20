#!/usr/bin/env bash
set -euo pipefail
client_root=$(pwd)
d10_ref=f7ccd8310e358d39527f591ec144a8bc2ba5c62c
git fetch origin "$d10_ref"
git worktree add --detach "$client_root/.proof-source" "$d10_ref"
mkdir -p .proof-source/crates/cli/examples dist/testbins/a11-proof-publish
cp agent-tools/a11-proof-stage/publish.rs .proof-source/crates/cli/examples/d10_private_proof.rs
D10_NATIVE_SOURCE="$d10_ref" CARGO_BUILD_JOBS=2 cargo build --release --locked --manifest-path .proof-source/Cargo.toml --target-dir "$client_root/target/a11-proof-publish" -p cli --features migrate --example d10_private_proof
cp target/a11-proof-publish/release/examples/d10_private_proof dist/testbins/a11-proof-publish/
cp agent-tools/a11-proof-stage/{publish.rs,prepare.py,build-publish-hosted.sh} dist/testbins/a11-proof-publish/
printf '{"d10_source":"%s"}\n' "$d10_ref" > dist/testbins/a11-proof-publish/provenance.json
(cd dist/testbins/a11-proof-publish && sha256sum d10_private_proof publish.rs prepare.py build-publish-hosted.sh provenance.json > SHA256SUMS)
