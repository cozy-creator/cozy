#!/usr/bin/env bash
# Build the cozy CLI release artifacts — cl-013's distribution half.
#
#   scripts/release.sh [--tag <tag>] [--out <dir>] [--allow-dirty]
#
# A release binary must provably carry its commit, so a DIRTY TREE REFUSES by default: the
# stamp would name a commit whose bytes are not the bytes that were built. `--allow-dirty`
# is the development door and stamps `+dirty` into the tag so the artifact says so itself.
#
# Reproducible: `-trimpath`, no build id path leakage, and every target is built TWICE into
# separate directories and compared. A platform that does not reproduce is reported, not
# quietly shipped.
#
# Output (`dist/<tag>/`):
#   cozy-<tag>-<os>-<arch>.tar.gz   the product artifact
#   SHA256SUMS                       what scripts/install.sh verifies BEFORE replacement
#   RELEASE.json                     tag, commit, toolchain, and the UNBUILT table
#
# UNBUILT is a first-class part of the output. A platform this repository cannot build is
# named with the reason rather than omitted, because an absent artifact and an artifact
# nobody tried to build look identical in a directory listing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/dist"
TAG=""
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    *) echo "usage: $0 [--tag <tag>] [--out <dir>] [--allow-dirty]" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
COMMIT="$(git rev-parse HEAD)"
DIRTY=0
git diff --quiet HEAD -- || DIRTY=1
[ -z "$(git ls-files --others --exclude-standard)" ] || DIRTY=1
if [ "$DIRTY" = 1 ] && [ "$ALLOW_DIRTY" = 0 ]; then
  echo "refusing: the working tree is dirty and a release binary must carry its commit" >&2
  echo "remedy: commit the tree, or pass --allow-dirty to stamp +dirty into the tag" >&2
  exit 3
fi
[ -n "$TAG" ] || TAG="$(git describe --tags --exact-match 2>/dev/null || echo "0.0.0-${COMMIT:0:12}")"
[ "$DIRTY" = 0 ] || TAG="$TAG+dirty"

D="$OUT/$TAG"
rm -rf "$D"
mkdir -p "$D"
STAMP="github.com/cozy-creator/cozy-creator-v2/internal/app"
LDFLAGS="-s -w -X $STAMP.tag=$TAG -X $STAMP.commit=$COMMIT"

# One row per target: GOOS GOARCH CGO CC. CGO is not optional — `internal/records` is the
# ONE local lifecycle authority and its driver is embedded libSQL (cl-009), which is a C
# library. Every wall below is therefore a C wall, not a Go one.
TARGETS="linux/amd64 linux/arm64 darwin/arm64 windows/amd64"
BUILT=""
UNBUILT=""

note_unbuilt() { UNBUILT="$UNBUILT{\"platform\":\"$1\",\"reason\":\"$2\"},"; echo "  UNBUILT $1 — $2"; }

for target in $TARGETS; do
  goos="${target%%/*}"; goarch="${target##*/}"
  # The prebuilt static libSQL archives go-libsql ships. No archive, no platform: this is a
  # DEPENDENCY wall, and it is checked before a compiler is invoked so the reason names the
  # real cause instead of whatever cc says next.
  lib="$(go list -m -f '{{.Dir}}' github.com/tursodatabase/go-libsql)/lib/${goos}_${goarch}/libsql_experimental.a"
  if [ ! -f "$lib" ]; then
    note_unbuilt "$target" "go-libsql ships no static libsql_experimental.a for ${goos}_${goarch}; the embedded libSQL driver (cl-009) has no port here"
    continue
  fi
  if [ "$goos/$goarch" != "$(go env GOOS)/$(go env GOARCH)" ]; then
    note_unbuilt "$target" "cross-compiling CGO needs a ${goos}/${goarch} C toolchain this builder does not have; build it on a ${goos}/${goarch} runner"
    continue
  fi

  ok=1
  for pass in a b; do
    mkdir -p "$D/.build-$pass"
    nice -n 19 env GOOS="$goos" GOARCH="$goarch" CGO_ENABLED=1 \
      go build -trimpath -ldflags "$LDFLAGS" -o "$D/.build-$pass/cozy" ./cmd/cozy || { ok=0; break; }
  done
  [ "$ok" = 1 ] || { note_unbuilt "$target" "the build failed on this host"; continue; }

  a="$(sha256sum "$D/.build-a/cozy" | cut -d' ' -f1)"
  b="$(sha256sum "$D/.build-b/cozy" | cut -d' ' -f1)"
  if [ "$a" != "$b" ]; then
    note_unbuilt "$target" "two identical builds produced different bytes ($a vs $b) — not reproducible"
    continue
  fi

  name="cozy-$TAG-$goos-$goarch.tar.gz"
  # A DETERMINISTIC archive: sorted, epoch mtimes, numeric root ownership, and gzip without
  # its own timestamp. Otherwise the tarball's digest changes on every run and the checksum
  # the installer verifies would be a fact about the clock.
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
      -C "$D/.build-a" -cf - cozy | gzip -n -9 > "$D/$name"
  echo "  built   $target -> $name (binary sha256:$a, reproduced)"
  BUILT="$BUILT{\"platform\":\"$target\",\"artifact\":\"$name\",\"binary_sha256\":\"$a\"},"
  rm -rf "$D/.build-a" "$D/.build-b"
done

( cd "$D" && sha256sum ./*.tar.gz > SHA256SUMS )

python3 - "$D/RELEASE.json" "$TAG" "$COMMIT" "$(go version)" "$BUILT" "$UNBUILT" <<'PY'
import json, sys
out, tag, commit, go, built, unbuilt = sys.argv[1:7]
rows = lambda s: json.loads("[" + s.rstrip(",") + "]") if s.strip(",") else []
json.dump({"tag": tag, "commit": commit, "go": go,
           "artifacts": rows(built), "unbuilt": rows(unbuilt)},
          open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
PY

echo
echo "release $TAG ($COMMIT)"
sed 's/^/  /' "$D/SHA256SUMS"
echo "  -> $D"
