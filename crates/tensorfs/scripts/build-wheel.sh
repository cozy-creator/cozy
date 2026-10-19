#!/usr/bin/env bash
# The one wheel build (tfs-051). Builds the `tfs` binary and the extension module from
# the SAME tree at the SAME version and ships both in one wheel: the binary is staged
# into wheel-data/scripts/ (pyproject `data`), so installers put it on PATH beside the
# module. This script stages it unconditionally; the wheel contract refuses a missing
# or mismatched executable.
#
#   scripts/build-wheel.sh [--out <dir>] [maturin build args...]
#
# Portable builds use the existing CI/release manylinux environment for BOTH
# cargo and maturin. A host build can explicitly pass --compatibility linux;
# the final whole-wheel audit refuses a manylinux tag inferred only from the extension.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"

out=target/wheels
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift ;;
        --plane|--no-default-features)
            echo "tensorfs ships one full Python+CLI wheel; $1 is not a supported wheel option" >&2
            exit 2 ;;
        *) args+=("$1") ;;
    esac
    shift
done

"$py" scripts/version_contract.py source

if [ "$(uname -s)" != Darwin ]; then
    if [ ! -f ../tensorfs-core/kernels/aot/catalog.rs ]; then
        uv run --no-project --python 3.12 --with triton==3.8.0 python kernels/build.py
    fi
    export TENSORFS_REQUIRE_AOT=1
fi

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
nice -n 19 cargo build --release --locked -p tensorfs-core --bin tfs
# The cozy workspace's target directory, not this tree's.
target=$(cargo metadata --format-version 1 --no-deps | "$py" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' ../tensorfs-py/Cargo.toml | head -1)
said=$("$target"/release/tfs version)
case "$said" in
    "tfs $version sha256:"*) ;;
    *) echo "built tfs says \"$said\", not version $version — refusing to stage" >&2; exit 1 ;;
esac

mkdir -p wheel-data/scripts
install -m 755 "$target"/release/tfs wheel-data/scripts/tfs

nice -n 19 maturin build --release --locked --out "$out" ${args[@]+"${args[@]}"}

for wheel in "$out"/tensorfs-*.whl; do
    "$py" scripts/version_contract.py wheel "$wheel"
done
