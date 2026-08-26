#!/usr/bin/env bash
# tfs-017 live verification of the publish protocol's document plane. Real `tfs` processes
# over the committed store fixture; the red arms hand-forge documents and watch the plane
# refuse them. Sections: green | red | all.
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-017-runs}
STORE=${STORE:-vectors/store}
SNAP=beb936f23bacbb5679d5cfbc207302503660f53fe543b34a3d752a7e5a38c4ee

hr() { printf '\n===== %s =====\n' "$*"; }

setup() {
  rm -rf "$WORK"; mkdir -p "$WORK/facts"
  "$TFS" cloud closure "$STORE" "$SNAP" qual-1 --out "$WORK/closure.json" | sed 's/^/  /'
}

# Mint the honest fact for every object the closure declares: grant -> the checksum the
# digest requires -> `cloud verified`, which is where the BACKEND's report is adjudicated.
mint_facts() {
  local n=0
  for id in $(python3 -c "
import json;print('\n'.join(o['sha256'] for o in json.load(open('$WORK/closure.json'))['objects']))"); do
    "$TFS" cloud grant "$WORK/closure.json" "$id" --out "$WORK/grant.json" >/dev/null || return 1
    local sum
    sum=$(python3 -c "
import base64;print(base64.b64encode(bytes.fromhex('$id')).decode())")
    "$TFS" cloud verified "$WORK/grant.json" "v-$n" "etag-$n" "$sum" \
      --out "$WORK/facts/$id.json" >/dev/null || return 1
    n=$((n+1))
  done
  echo "  minted $n fact(s), every one through the checksum adjudication"
}

green_section() {
  hr "declare first, mint facts, publish a root over facts only"
  setup
  mint_facts
  "$TFS" cloud publish "$WORK/closure.json" "$WORK/facts" | sed 's/^/  /'
  echo "publish exit=$?"
}

red_section() {
  hr "RED 1 — the backend's own report is adjudicated at mint (pre-existing arm)"
  setup
  local id
  id=$(python3 -c "
import json;print(json.load(open('$WORK/closure.json'))['objects'][0]['sha256'])")
  "$TFS" cloud grant "$WORK/closure.json" "$id" --out "$WORK/grant.json" >/dev/null
  "$TFS" cloud verified "$WORK/grant.json" v1 etag1 "$(python3 -c "
import base64;print(base64.b64encode(b'x'*32).decode())")"
  echo "mint exit=$?"

  # ------------------------------------------------------------------ #553b
  hr "RED 2 — a hand-forged VerifiedObject DOCUMENT authorizes nothing"
  setup
  mint_facts
  echo "--- publish over the honest facts, for contrast ---"
  "$TFS" cloud publish "$WORK/closure.json" "$WORK/facts" | sed 's/^/  /'
  echo "--- now REWRITE one fact's checksum: the byte a forger controls ---"
  local victim; victim=$(ls "$WORK/facts" | head -1)
  python3 - "$WORK/facts/$victim" <<'PY'
import base64, json, sys
p = sys.argv[1]
d = json.load(open(p))
before = d["checksum_sha256"]
d["checksum_sha256"] = base64.b64encode(b"\x00" * 32).decode()
# Canonical form: sorted keys, no spaces — the document plane's own spelling, so the
# forgery is refused on its CONTENT and not on its shape.
open(p, "w").write(json.dumps(d, sort_keys=True, separators=(",", ":")))
print(f"  forged {p.rsplit('/',1)[1][:12]}…: {before} -> {d['checksum_sha256']}")
PY
  echo "--- the forged fact never parses, so the root has nothing to stand on ---"
  "$TFS" cloud publish "$WORK/closure.json" "$WORK/facts"
  echo "publish exit=$?"
  echo "(DESERIALIZATION DOES NOT MINT VERIFICATION: the digest requires one checksum and"
  echo " the document carries another, so the fact refuses before publish ever sees it)"
}

case "${1:-all}" in
  green) green_section ;;
  red)   red_section ;;
  all)   green_section; red_section ;;
  *) echo "usage: $0 green|red|all"; exit 2 ;;
esac
