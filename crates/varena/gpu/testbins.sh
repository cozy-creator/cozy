#!/usr/bin/env bash
# Builds varena's native GPU test binary in manylinux_2_28 (it runs on every rental image) into
# dist/testbins/varena-gpu. Run on a hosted runner: ghbuild.sh cozy <ref> none 'bash crates/varena/gpu/testbins.sh'
set -euo pipefail
cargo fetch --locked --target x86_64-unknown-linux-gnu
target="$PWD/target/manylinux"
# A toolchain a failed run left in the cache may be half-installed: the container installs its own.
rm -rf "$target/rustup"
mkdir -p "$target" dist/testbins
docker run --rm --user "$(id -u):$(id -g)" --tmpfs /cargo:rw,exec,mode=1777 \
  -e CARGO_HOME=/cargo -e RUSTUP_HOME=/target/rustup -e CARGO_BUILD_JOBS="$(nproc)" \
  -v "${CARGO_HOME:-$HOME/.cargo}/registry:/cargo/registry" -v "${CARGO_HOME:-$HOME/.cargo}/git:/cargo/git" \
  -v "$PWD:/src:ro" -v "$target:/target" -w /src quay.io/pypa/manylinux_2_28_x86_64:2026.10.03-1 \
  sh -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y -q --no-modify-path --profile minimal --default-toolchain 1.99.1 >&2 &&
         /cargo/bin/cargo test --release --locked --offline --target-dir /target -p varena --test gpu --no-run --message-format=json' \
  > "$target/gpu-build.jsonl" || { jq -r 'select(.reason == "compiler-message") | .message.rendered' "$target/gpu-build.jsonl"; exit 1; }
exe=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "gpu" and .executable != null) | .executable' "$target/gpu-build.jsonl")
install -m 0755 "$target/${exe#/target/}" dist/testbins/varena-gpu
rm -rf "$target/rustup"
strip --strip-debug dist/testbins/varena-gpu
git rev-parse HEAD > dist/testbins/SOURCE
(cd dist/testbins && sha256sum varena-gpu > varena-gpu.sha256)
dist/testbins/varena-gpu --list --ignored
