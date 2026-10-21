#!/usr/bin/env bash
# Hosted source/wire controls only; GPU enforcement remains a real hardware gate.
set -euo pipefail
export CARGO_BUILD_JOBS=2
cargo test --locked -p protocol a_gpu_memory_limit -- --test-threads=2
cargo test --locked -p cli gpu_memory_limit -- --test-threads=2
cargo test --locked -p tensord --lib gpu_memory_limit_alone -- --test-threads=2
cargo test --locked -p tensord --lib a_persistent_cap -- --test-threads=2
cargo build --locked -p tensord -p cozy
export COZY_TEST_TENSORD="$PWD/target/debug/tensord"
cargo run --locked -p protocol --example openapi > /tmp/gpu-cap-machine.json
cmp /tmp/gpu-cap-machine.json crates/protocol/openapi/machine.json
old=b517cd777eb2353dfb2f2ead3a10809b15dfc892
git fetch origin "$old"
git worktree add --detach .old-cap-controller "$old"
cargo build --release --locked --manifest-path .old-cap-controller/Cargo.toml --target-dir "$PWD/target/old-cap-controller" -p cozy
export COZY_TEST_OLD_CALCIFER="$PWD/target/old-cap-controller/release/cozy"
cargo test --locked -p cozy --test machine a_gpu_cap_ -- --test-threads=1 --nocapture
uv pip install --python "$TENSORFS_TEST_PYTHON" "$COZY_TEST_RUNTIME_WHEEL" pytest
PYTHONPATH="$PWD/.runtime/src" "$TENSORFS_TEST_PYTHON" -m pytest -q .runtime/tests/test_device_limit.py .runtime/tests/test_executor_fork.py
