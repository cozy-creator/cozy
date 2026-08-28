#!/usr/bin/env bash
# tfs-030 focused manual proof: real store plus planted malformed public ReadPlans.
# This is an executable verification script, not a test suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TFS="${TFS:-$ROOT/target/release/tfs}"
WORK="$(mktemp -d /tmp/tfs-read-order.XXXXXX)"
trap 'rm -rf -- "$WORK"' EXIT

output="$($TFS checkpoint write "$WORK/store" --blocks 1 --scale 8)"
snapshot="$(sed -n 's/.*checkpoint_id   sha256:\([0-9a-f]\{64\}\).*/\1/p' <<<"$output")"
test -n "$snapshot"

arms="$($TFS read arms "$WORK/store" "$snapshot")"
for evidence in \
  "construction-order destination gap: RANGE_BOUNDS, destination untouched" \
  "construction-order destination overlap: RANGE_BOUNDS, destination untouched" \
  "construction-order trailing destination byte: RANGE_BOUNDS, destination untouched" \
  "construction-order item past destination: RANGE_BOUNDS, destination untouched" \
  "construction-order item with unaddressable length: RANGE_BOUNDS, destination untouched" \
  "one issue cursor with concurrent disjoint completions covers abcd"
do
  grep -F "$evidence" <<<"$arms" >/dev/null
done

printf '%s\n' \
  "READ ORDER PASS gap overlap trailing past-destination overflow all RANGE_BOUNDS" \
  "destination-untouched=5 issue-cursor=one concurrent-disjoint-completions=green"
