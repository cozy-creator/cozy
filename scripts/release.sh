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
#   RELEASE.json                     tag, commit, toolchain, the artifact table with a
#                                    sha256 per platform, and the UNBUILT table
#
# UNBUILT is a first-class part of the output. A platform this repository cannot build is
# named with the reason rather than omitted, because an absent artifact and an artifact
# nobody tried to build look identical in a directory listing. There is no C wall left to
# name — `cozy` is pure Go (CGO_ENABLED=0) and every target below is a cross-compile from
# this one host — so what remains in the table is Go source that still spells a Unix
# syscall directly, and the reason quotes the compiler saying which.
#
# What a build is NOT: proof the binary RUNS. This host executes linux/amd64 and nothing
# else, so the macOS and Windows rows are static evidence — bytes, size, checksum — until
# a runner of that platform runs `scripts/accept.sh` against them.
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

# One row per target. CGO is OFF for all of them: `internal/records` drives its SQLite
# through modernc.org/sqlite (SQLite transpiled to Go) and `internal/service` holds the
# liveness lock through a per-OS pair, so there is no C in this binary and every target is
# a plain cross-compile from this one host.
TARGETS="linux/amd64 linux/arm64 darwin/arm64 darwin/amd64 windows/amd64"
BUILT=""
UNBUILT=""

note_unbuilt() { UNBUILT="$UNBUILT{\"platform\":\"$1\",\"reason\":\"$2\"},"; echo "  UNBUILT $1 — $2"; }

for target in $TARGETS; do
  goos="${target%%/*}"; goarch="${target##*/}"
  exe="cozy"; [ "$goos" != windows ] || exe="cozy.exe"

  ok=1; why=""
  for pass in a b; do
    mkdir -p "$D/.build-$pass"
    why="$(nice -n 19 env GOOS="$goos" GOARCH="$goarch" CGO_ENABLED=0 \
      go build -trimpath -ldflags "$LDFLAGS" -o "$D/.build-$pass/$exe" ./cmd/cozy 2>&1)" || { ok=0; break; }
  done
  # The compiler's OWN first line, quoted into the reason. "the build failed" names
  # nothing; `undefined: syscall.Kill` names the file that still owes this platform a port.
  [ "$ok" = 1 ] || { note_unbuilt "$target" "$(printf '%s' "$why" | grep -v '^#' | head -1 | tr -d '"' | cut -c1-200)"; continue; }

  a="$(sha256sum "$D/.build-a/$exe" | cut -d' ' -f1)"
  b="$(sha256sum "$D/.build-b/$exe" | cut -d' ' -f1)"
  if [ "$a" != "$b" ]; then
    note_unbuilt "$target" "two identical builds produced different bytes ($a vs $b) — not reproducible"
    continue
  fi

  name="cozy-$TAG-$goos-$goarch.tar.gz"
  # A DETERMINISTIC archive: sorted, epoch mtimes, numeric root ownership, and gzip without
  # its own timestamp. Otherwise the tarball's digest changes on every run and the checksum
  # the installer verifies would be a fact about the clock.
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
      -C "$D/.build-a" -cf - "$exe" | gzip -n -9 > "$D/$name"
  # `file` on the binary, recorded per platform: this host runs linux/amd64 only, so for
  # the other four the format line and the checksum ARE the verification.
  kind="$(file -b "$D/.build-a/$exe" 2>/dev/null | cut -d, -f1-2 || echo unknown)"
  size="$(stat -c%s "$D/.build-a/$exe")"
  ran="no (this builder is $(go env GOOS)/$(go env GOARCH))"
  if [ "$goos/$goarch" = "$(go env GOOS)/$(go env GOARCH)" ]; then
    got="$("$D/.build-a/$exe" version --fields tag 2>/dev/null | sed -n 's/^tag: *//p' || true)"
    ran="ran here, reports tag ${got:-<none>}"
  fi
  echo "  built   $target -> $name (binary sha256:$a, reproduced, $ran)"
  BUILT="$BUILT{\"platform\":\"$target\",\"artifact\":\"$name\",\"binary_sha256\":\"$a\",\"binary_bytes\":$size,\"format\":\"$kind\",\"executed\":\"$ran\"},"
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
