#!/usr/bin/env bash
# tfs-034 live Manifest proof: exact typed admission, deterministic identity, complete walk,
# checkout/materialization, and corruption recovery. Unit tests/vectors own parser mutation
# coverage; this script drives only surviving product commands.
set -euo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-034-manifest-runs}
STORE="$WORK/store"

rm -rf "$WORK"
mkdir -p "$WORK"
cargo build --quiet --release --bin tfs

write=$("$TFS" checkpoint write "$STORE" --blocks 1)
manifest=$(printf '%s\n' "$write" | sed -n 's/.*manifest_id     sha256:\([0-9a-f]\{64\}\).*/\1/p')
test -n "$manifest"
printf '%s\n' "$write" | tail -5

"$TFS" manifest get "$STORE" "$manifest" --out "$WORK/manifest.json"
test "$(sha256sum "$WORK/manifest.json" | cut -d' ' -f1)" = "$manifest"
"$TFS" manifest show "$STORE" "$manifest"
"$TFS" manifest walk "$STORE" "$manifest" --refs "$WORK/refs.jsonl"
"$TFS" manifest verify "$STORE" "$manifest"

"$TFS" checkout "$STORE" "$manifest" "$WORK/checkout"
test ! -e "$WORK/checkout/model.cozytensors"
test -L "$WORK/checkout/unet/config.json"
"$TFS" materialize "$STORE" "$manifest" unet/config.json "$WORK/config.json"
cmp "$WORK/config.json" "$WORK/checkout/unet/config.json"

OTHER="$WORK/other"
"$TFS" store init "$OTHER" >/dev/null
"$TFS" manifest admit "$OTHER" "$WORK/manifest.json" \
  --expect "$manifest" --expect-length "$(stat -c%s "$WORK/manifest.json")"
if "$TFS" manifest verify "$OTHER" "$manifest" >"$WORK/missing.log" 2>&1; then
  echo "manifest without its blob closure unexpectedly verified" >&2
  exit 1
fi
grep -q 'REFUSED OBJECT_ABSENT' "$WORK/missing.log"

again=$("$TFS" checkpoint write "$WORK/deterministic" --blocks 1 | \
  sed -n 's/.*manifest_id     sha256:\([0-9a-f]\{64\}\).*/\1/p')
test "$again" = "$manifest"

printf 'MANIFEST GREEN id=sha256:%s refs=%s deterministic=yes\n' \
  "$manifest" "$(wc -l < "$WORK/refs.jsonl")"
