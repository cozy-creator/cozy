#!/usr/bin/env bash
# The one wheel build (tfs-051). Builds the `tfs` binary and the extension module from
# the SAME tree at the SAME version and ships both in one wheel: the binary is staged
# into wheel-data/scripts/ (pyproject `data`), so installers put it on PATH beside the
# module. pyproject makes a bare `maturin build` refuse when nothing is staged; this
# script is what stages, and it re-stages unconditionally so a stale binary cannot ride.
#
#   scripts/build-wheel.sh [--out <dir>] [--plane] [maturin build args...]
#
# --plane builds the plane-only wheel Rust-machine executors install: the weight plane,
# headers, read plans and fit, no store, downloader, ingest or GC, and no `tfs` binary.
#
# Portable builds use the existing CI/release manylinux environment for BOTH
# cargo and maturin. A host build can explicitly pass --compatibility linux;
# the final whole-wheel audit refuses a manylinux tag inferred only from the extension.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"

out=target/wheels
plane=false
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift ;;
        --plane) plane=true ;;
        *) args+=("$1") ;;
    esac
    shift
done

"$py" scripts/version_contract.py source

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
if $plane; then
    rm -rf wheel-data && mkdir -p wheel-data
    nice -n 19 maturin build --release --locked --no-default-features --features extension-module \
        --out "$out" ${args[@]+"${args[@]}"}
    exit 0
fi
nice -n 19 cargo build --release --locked -p tensorfs-core --bin tfs

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)
said=$(target/release/tfs version)
case "$said" in
    "tfs $version sha256:"*) ;;
    *) echo "built tfs says \"$said\", not version $version — refusing to stage" >&2; exit 1 ;;
esac

rm -rf wheel-data
mkdir -p wheel-data/scripts
install -m 755 target/release/tfs wheel-data/scripts/tfs

nice -n 19 maturin build --release --locked --out "$out" ${args[@]+"${args[@]}"}

for wheel in "$out"/tensorfs-*.whl; do
    "$py" scripts/version_contract.py wheel "$wheel"
done
