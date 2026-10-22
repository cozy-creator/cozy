#!/usr/bin/env bash
# Builds the native GPU suite's test binaries in manylinux_2_28 (they run on every rental image):
# varena's `gpu` and TensorFS's `io`, bundled as dist/testbins/bundle.tar (`gpu`, `tensorfs-io`).
# Run on a hosted runner: ghbuild.sh cozy <ref> none 'bash crates/varena/gpu/testbins.sh'
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
         /cargo/bin/cargo test --release --locked --offline --target-dir /target -p varena --test gpu --no-run --message-format=json &&
         /cargo/bin/cargo test --release --locked --offline --target-dir /target -p tensorfs --test io --no-run --message-format=json' \
  > "$target/gpu-build.jsonl" || { jq -r 'select(.reason == "compiler-message") | .message.rendered' "$target/gpu-build.jsonl"; exit 1; }
for pair in gpu:gpu io:tensorfs-io; do
  exe=$(jq -r --arg t "${pair%%:*}" 'select(.reason == "compiler-artifact" and .target.name == $t and .executable != null) | .executable' "$target/gpu-build.jsonl")
  install -m 0755 "$target/${exe#/target/}" "dist/testbins/${pair#*:}"
  strip --strip-debug "dist/testbins/${pair#*:}"
done
rm -rf "$target/rustup"
git rev-parse HEAD > dist/testbins/SOURCE
tar -C dist/testbins -cf dist/testbins/bundle.tar gpu tensorfs-io
(cd dist/testbins && sha256sum bundle.tar > bundle.tar.sha256)
for bin in gpu tensorfs-io; do dist/testbins/$bin --list --ignored; done
