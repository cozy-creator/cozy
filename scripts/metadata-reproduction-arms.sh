#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tfs="$repo_root/target/release/tfs"
proof_root=$(mktemp -d "${TMPDIR:-/tmp}/tfs029.XXXXXX")
trap 'rm -rf "$proof_root"' EXIT

cargo build --quiet --release --bin tfs --manifest-path "$repo_root/Cargo.toml"
"$tfs" store init "$proof_root/store" >/dev/null
write=$({ "$tfs" checkpoint write "$proof_root/store" --blocks 1; } 2>&1)
header=$(printf '%s\n' "$write" | sed -n 's/.*header_digest   sha256:\([0-9a-f]\{64\}\).*/\1/p')
snapshot=$(printf '%s\n' "$write" | sed -n 's/.*checkpoint_id   sha256:\([0-9a-f]\{64\}\).*/\1/p')
test -n "$header" && test -n "$snapshot"

"$tfs" get "$proof_root/store" "$header" --out "$proof_root/header.cbor" >/dev/null
"$tfs" cbor decode "$proof_root/header.cbor" --out "$proof_root/header.json"
"$tfs" get "$proof_root/store" "$snapshot" --out "$proof_root/snapshot.json" >/dev/null

python3 - "$proof_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
header = json.loads((root / "header.json").read_text())
snapshot = json.loads((root / "snapshot.json").read_text())
entries = []
for row in snapshot["entries"]:
    if row["kind"] == "dir":
        entries.append([row["path"], "dir"])
    elif row["path"] == "model.cozytensors":
        entries.append([row["path"], "header"])
    else:
        entries.append([row["path"], "file", row["object"]])
order = [
    [component[0], tensor[0]]
    for component in header["components"]
    for tensor in component[1]
]
(root / "entries.json").write_text(json.dumps(entries, separators=(",", ":")))
(root / "order.json").write_text(json.dumps(order, separators=(",", ":")))
(root / "header.min.json").write_text(json.dumps(header, separators=(",", ":")))

wrong_order = list(order)
wrong_order[:2] = reversed(wrong_order[:2])
(root / "wrong-order.json").write_text(json.dumps(wrong_order, separators=(",", ":")))

old = dict(header)
old["configs"] = {row[0]: row[1] for row in header["configs"]}
(root / "old-map-header.json").write_text(json.dumps(old, separators=(",", ":")))

unverified = json.loads(json.dumps(entries))
for row in unverified:
    if row[1] == "file" and not row[0].startswith(("assets/", "closure/")):
        row[2]["sha256"] = "0" * 64
        break
(root / "unverified.json").write_text(json.dumps(unverified, separators=(",", ":")))

missing = [
    row for row in entries
    if not (row[0].startswith("closure/") and row[1] == "file")
]
(root / "missing-closure.json").write_text(json.dumps(missing, separators=(",", ":")))
PY

reproduced=$(
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.json" \
    "$proof_root/entries.json" --order "$proof_root/order.json"
)
grep -q "header       sha256:$header" <<<"$reproduced"
grep -q "snapshot     sha256:$snapshot" <<<"$reproduced"
grep -q 'tensor_payload_read 0' <<<"$reproduced"
grep -q 'tensor_payload_hash 0' <<<"$reproduced"

minified=$(
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.min.json" \
    "$proof_root/entries.json" --order "$proof_root/order.json"
)
grep -q "header       sha256:$header" <<<"$minified"
grep -q "snapshot     sha256:$snapshot" <<<"$minified"

refuse() {
  local code=$1
  shift
  if "$@" >"$proof_root/refusal.log" 2>&1; then
    echo "expected $code refusal, command succeeded" >&2
    return 1
  fi
  grep -q "REFUSED $code" "$proof_root/refusal.log"
}

refuse TRAVERSAL_ORDER_MISMATCH \
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.json" \
    "$proof_root/entries.json" --order "$proof_root/wrong-order.json"
refuse WRONG_TYPE \
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/old-map-header.json" \
    "$proof_root/entries.json" --order "$proof_root/order.json"
refuse OBJECT_CORRUPT \
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.json" \
    "$proof_root/unverified.json" --order "$proof_root/order.json"
refuse NOT_CONTAINED \
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.json" \
    "$proof_root/missing-closure.json" --order "$proof_root/order.json"
"$tfs" store bump "$proof_root/store" >/dev/null
refuse OBJECT_CORRUPT \
  "$tfs" checkpoint reproduce "$proof_root/store" "$proof_root/header.json" \
    "$proof_root/entries.json" --order "$proof_root/order.json"

printf 'tfs-029 metadata reproduction: 7/7 arms green\n'
