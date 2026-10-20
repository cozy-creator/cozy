#!/usr/bin/env bash
set -euo pipefail
client_root=$(pwd)
d10_ref=f7ccd8310e358d39527f591ec144a8bc2ba5c62c
git fetch origin "$d10_ref"
git worktree add --detach "$client_root/.public-proof-source" "$d10_ref"
mkdir -p .public-proof-source/crates/cli/examples dist/testbins/varena-public-proof
cp agent-tools/varena-public-proof/publish.rs .public-proof-source/crates/cli/examples/d10_public_proof.rs
D10_NATIVE_SOURCE="$d10_ref" CARGO_BUILD_JOBS=2 cargo build --release --locked --manifest-path .public-proof-source/Cargo.toml --target-dir "$client_root/target/a11-proof-publish" -p cli --features migrate --example d10_public_proof
cp target/a11-proof-publish/release/examples/d10_public_proof dist/testbins/varena-public-proof/
cp agent-tools/varena-public-proof/{publish.rs,prepare.py,build-hosted.sh} dist/testbins/varena-public-proof/
printf '{"d10_source":"%s"}\n' "$d10_ref" > dist/testbins/varena-public-proof/provenance.json
(cd dist/testbins/varena-public-proof && sha256sum d10_public_proof publish.rs prepare.py build-hosted.sh provenance.json > SHA256SUMS)
