#!/usr/bin/env bash
# One-off clients use the approved source trees directly, with no vendored core.
set -euo pipefail
client_root=$(pwd)
a11_ref=e2c8ded23669b7b18185808c1c1c0934693def3b
d10_ref=4aaa7f425ddefb9be8fe05854979434e5b9b7af1
git fetch origin "$a11_ref" "$d10_ref"
git worktree add --detach "$client_root/.a11-source" "$a11_ref"
git worktree add --detach "$client_root/.d10-source" "$d10_ref"
mkdir -p .a11-source/crates/tensorfs/examples .d10-source/crates/tensorfs/examples dist/testbins/a11-proof-stage
cp agent-tools/a11-proof-stage/stage.rs .a11-source/crates/tensorfs/examples/a11_proof_stage.rs
cp agent-tools/a11-proof-stage/validate.rs .d10-source/crates/tensorfs/examples/d10_validate.rs
A11_NATIVE_SOURCE="$a11_ref" CARGO_BUILD_JOBS=2 cargo build --release --locked --manifest-path .a11-source/Cargo.toml --target-dir "$client_root/target/a11-proof-stage" -p tensorfs --example a11_proof_stage
D10_NATIVE_SOURCE="$d10_ref" CARGO_BUILD_JOBS=2 cargo build --release --locked --manifest-path .d10-source/Cargo.toml --target-dir "$client_root/target/a11-proof-stage" -p tensorfs --no-default-features --example d10_validate
cp target/a11-proof-stage/release/examples/a11_proof_stage dist/testbins/a11-proof-stage/
cp target/a11-proof-stage/release/examples/d10_validate dist/testbins/a11-proof-stage/
cp agent-tools/a11-proof-stage/{stage.rs,validate.rs,build-hosted.sh} dist/testbins/a11-proof-stage/
printf '{"a11_source":"%s","d10_source":"%s"}\n' "$a11_ref" "$d10_ref" > dist/testbins/a11-proof-stage/provenance.json
(cd dist/testbins/a11-proof-stage && sha256sum a11_proof_stage d10_validate stage.rs validate.rs build-hosted.sh provenance.json > SHA256SUMS)
