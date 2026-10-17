#!/bin/sh
# Development worker image with this checkout's TensorD: one layer appended to the current
# tensorhub/worker image of KIND (cpu | cuda). `push` pushes it untagged by digest, never onto a
# tag; without it the staged layer is left for a local boot. A third argument names a directory
# of Runtime/TensorFS wheels TensorD's executors use. Rent it with
# `cozy rental new <sku> --image=sha256:<digest>` once a Hub has registered that digest.
set -eu
kind=$1; push=${2:-}; own_wheels=${3:-}; case $kind in cpu) tag=cpu-linux-x86 ;; cuda) tag=torch2.14.0-cu130-linux-x86 ;; *) echo "usage: $0 cpu|cuda" >&2; exit 2 ;; esac
repo=$(cd "$(dirname "$0")/.." && pwd); commit=$(git -C "$repo" rev-parse HEAD); base=$(crane digest "tensorhub/worker:$tag")
target=${CARGO_TARGET_DIR:-$repo/target}/bookworm; stage=$(mktemp -d); mkdir -p "$target" "$stage/opt/cozy/machine" "$stage/usr/local/bin"

# The base is glibc 2.36 (bookworm); a host build needs newer symbols.
nice -n 19 docker run --rm --runtime=runc --user "$(id -u):$(id -g)" -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 -v "$HOME/.cargo:/cargo" -v "$target:/target" -v "$repo:/src" -w /src \
  rust:1.91-bookworm cargo build --release --locked --offline -j 2 --bin tensord
install -m 0755 "$target/release/tensord" "$stage/opt/cozy/machine/tensord"

# Optional: the executor SDK (Runtime/TensorFS wheels) TensorD runs package code with.
if [ -n "$own_wheels" ]; then mkdir -p "$stage/opt/cozy/machine/wheels" && cp "$own_wheels"/*.whl "$stage/opt/cozy/machine/wheels/"; fi
ln -s /opt/cozy/machine/tensord "$stage/usr/local/bin/tensord"
find "$stage" -mindepth 1 -type d -exec chmod 0755 {} +  # a layer's directory entries replace the base's modes
tar --numeric-owner --owner=0 --group=0 --mtime=@0 --sort=name -C "$stage" -czf "$stage.tgz" $(ls -A "$stage")
[ "$push" = push ] || { echo "staged $kind layer: $stage (base tensorhub/worker@$base)"; exit 0; }
pushed=$(crane mutate "tensorhub/worker@$base" --append "$stage.tgz" --entrypoint /usr/local/bin/tensord -l "tensord.commit=$commit" -l "tensord.base=$base")
digest=${pushed#*@}; [ "$digest" != "$pushed" ] || { echo "push failed" >&2; exit 1; }
echo "$kind $digest commit=$commit base=$base"
