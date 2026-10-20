#!/usr/bin/env bash
# Run only after extracting this owned prefix into a fresh exact Product25 checkout.
set -euo pipefail
test "$(git rev-parse HEAD)" = 25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2
[[ ${H3_STAGE_SOURCE_COMMIT:?exact client source commit required} =~ ^[0-9a-f]{40}$ ]]
export H3_STAGE_SOURCE_COMMIT
git diff --exit-code HEAD -- crates/tensorfs/src crates/tensorfs/Cargo.toml Cargo.lock
test ! -e crates/tensorfs/examples/h3_chunk_stage.rs
mkdir -p crates/tensorfs/examples
cp agent-tools/h3-chunk-stage/h3_chunk_stage.rs crates/tensorfs/examples/h3_chunk_stage.rs
git rev-parse HEAD
sha256sum agent-tools/h3-chunk-stage/h3_chunk_stage.rs
CARGO_BUILD_JOBS=2 cargo test --locked -p tensorfs --example h3_chunk_stage -- --nocapture
# Compilation of this one-off Rust driver only. No native wheel or AOT build.
CARGO_BUILD_JOBS=2 cargo build --locked -p tensorfs --example h3_chunk_stage
sha256sum target/debug/examples/h3_chunk_stage
# No invocation against the real source is automatic. After separate authorization:
# target/debug/examples/h3_chunk_stage /absolute/path/h3-current-source.json > metadata.json
# target/debug/examples/h3_chunk_stage /absolute/path/h3-current-source.json --download > staging.json
