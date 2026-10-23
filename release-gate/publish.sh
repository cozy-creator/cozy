#!/bin/bash
# publish.sh <gate dir> <SKU>: a passed gate's baseline (gate-<SKU>.json and the references it hands the next gate)
# becomes the SKU's baseline: assets of the release-gate-baselines release, which every later gate on the SKU reads.
set -euo pipefail
OUT=$1 SKU=$2 REPO=cozy-creator/cozy-creator TAG=release-gate-baselines
B=$OUT/baseline
grep -qE '^VERDICT (PASS|FAIL .*accepted by the lead)' "$OUT/verdict.txt" && [ -f "$B/gate-$SKU.json" ] \
  || { echo "$OUT holds no passed gate on $SKU"; exit 1; }
gh release view "$TAG" -R "$REPO" >/dev/null 2>&1 \
  || gh release create "$TAG" -R "$REPO" --target master --prerelease --latest=false --title "Release gate baselines" \
       --notes "Each SKU's last passed release gate (release-gate/gate.sh): gate-<SKU>.json and its reference inputs."
for a in $(gh release view "$TAG" -R "$REPO" --json assets -q '.assets[].name' | grep -E "^ref-$SKU\.[a-z]+$" || true); do
  [ -f "$B/$a" ] || gh release delete-asset "$TAG" "$a" -R "$REPO" -y
done
# shellcheck disable=SC2046
gh release upload "$TAG" -R "$REPO" --clobber "$B/gate-$SKU.json" $(ls "$B/ref-$SKU".* "$B/ref-$SKU-clip.mp4" 2>/dev/null)
echo "published the $SKU baseline from $OUT"
