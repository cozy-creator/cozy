#!/bin/sh
# Build a release as .github/workflows/release.yaml publishes it:
#
#   scripts/release.sh <tag> [<os>/<arch> ...]    ->  dist/<tag>/cozy-<os>-<arch>.tar.gz, SHA256SUMS
#
# cozy-player.tar.gz is web/player, the browser player GitHub Pages serves.
#
# Static (CGO off: SQLite is modernc's pure Go), stamped with the tag minus its `v` (`cozy -v`
# prints 0.1.0) and the commit. The archives are deterministic, so a rebuild reproduces SHA256SUMS.
set -eu
tag="$1"; shift
[ $# -gt 0 ] || set -- linux/amd64 linux/arm64 darwin/arm64 darwin/amd64
cd "$(dirname "$0")/.."
out="dist/$tag"
rm -rf "$out" && mkdir -p "$out/bin"
build=github.com/cozy-creator/cozy/internal/build
for target; do
  os="${target%/*}" arch="${target#*/}"
  GOOS="$os" GOARCH="$arch" CGO_ENABLED=0 go build -trimpath -o "$out/bin/cozy" \
    -ldflags "-s -w -X $build.Version=${tag#v} -X $build.Commit=$(git rev-parse HEAD)" .
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$out/bin" -cf - cozy |
    gzip -n -9 >"$out/cozy-$os-$arch.tar.gz"
done
rm -r "$out/bin"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C web -cf - player | gzip -n -9 >"$out/cozy-player.tar.gz"
cd "$out" && sha256sum cozy-*.tar.gz >SHA256SUMS && cat SHA256SUMS
