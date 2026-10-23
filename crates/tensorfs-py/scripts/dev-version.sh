#!/usr/bin/env bash
# Stamps the tree as a development build, <version>+dev.g<commit>: a local version, which PyPI never
# accepts, in every manifest (and the lock that follows them).
set -euo pipefail
cd "$(dirname "$0")/.."
v=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)
dev="$v+dev.g$(git rev-parse --short=8 HEAD)"
perl -pi -e "s/^version = \"\Q$v\E\"\$/version = \"$dev\"/" pyproject.toml Cargo.toml ../{tensorfs,varena}/Cargo.toml
cargo update --workspace --quiet
echo "$dev"
