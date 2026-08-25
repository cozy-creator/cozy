#!/usr/bin/env bash
# tfs-008 live verification: structure fixtures + the executable-encoding registry.
# Nothing here is a test suite — every line runs the real registry, the real geometry
# evaluator, and the real reference decoders over PRODUCER-MINED bytes.
# Sections: fixtures | mine | conform | capability | border | unknown | fences | all
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-008-runs}
V1PY=${V1PY:-$HOME/cozy/python-gen-worker/.venv/bin/python}
SECTION=${1:-all}

hr() { printf '\n===== %s =====\n' "$*"; }
hex() { grep -o '[0-9a-f]\{64\}' | head -1; }

ck() {
  if [ ! -f "$WORK/ck.header" ]; then
    rm -rf "$WORK/ck"; mkdir -p "$WORK/ck"
    nice -n 19 "$TFS" checkpoint write "$WORK/ck" > "$WORK/ck.write"
    grep header_digest "$WORK/ck.write" | hex > "$WORK/ck.header"
  fi
  cat "$WORK/ck.header"
}

fixtures_section() {
  hr "the banked structure fixtures, with provenance"
  "$TFS" structure list

  hr "the REAL H3 fixture is derived from the real shipped header, and says so"
  "$V1PY" - <<'PY'
import json, hashlib, pathlib
p = pathlib.Path.home() / "cozy_v2/tracker-v2/examples/safetensors-headers/minimax_h3_fl2va_fp8_scaled.header.json"
raw = p.read_bytes()
h = json.loads(raw); h.pop("__metadata__", None)
print(f"  source            {p.name}")
print(f"  source sha256     {hashlib.sha256(raw).hexdigest()}")
print(f"  keys in the file  {len(h)}")
print(f"  .weight_scale     {sum(1 for k in h if k.endswith('.weight_scale'))}")
print(f"  .comfy_quant      {sum(1 for k in h if k.endswith('.comfy_quant'))}")
print(f"  logical tensors   {sum(1 for k in h if not k.endswith(('.weight_scale', '.comfy_quant')))}")
PY
  python3 -c "
import json,sys,pathlib
p=pathlib.Path('vectors/structure/h3.fl2va-full-adaln.json')
d=json.loads(p.read_text())
print('  banked keys      ', sum(len(v) for v in d['components'].values()))
print('  provenance       ', d['provenance']['kind'], d['provenance']['source_sha256'][:16]+'...')
print('  note             ', d['provenance']['note'][:110]+'...')
"

  hr "a real checkpoint MATCHES its banked fixture"
  "$TFS" structure match "$WORK/ck" "$(ck)"

  hr "an UNKNOWN structure: identity still derives, nothing matches"
  rm -rf "$WORK/other"; mkdir -p "$WORK/other"
  nice -n 19 "$TFS" checkpoint write "$WORK/other" --blocks 3 > "$WORK/other.write"
  "$TFS" structure match "$WORK/other" "$(grep header_digest "$WORK/other.write" | hex)"
}

mine_section() {
  hr "PRODUCER-MINED vectors: v1's own quantizers, run here, transcribed to raw bytes"
  echo "  producer venv: $V1PY"
  "$V1PY" -c "import torch; print('  torch', torch.__version__)"
  nice -n 19 "$V1PY" scripts/mine-vectors.py all > "$WORK/mined.rows" 2> "$WORK/mined.err"
  cat "$WORK/mined.err" | sed 's/^/  /'
  echo "  rows: $(wc -l < "$WORK/mined.rows"), cases: $(grep -c '^case ' "$WORK/mined.rows")"

  hr "re-mining is byte-stable (the producers are deterministic)"
  for n in fp8-rowwise:fp8-rowwise-16x32 fp8-rowwise-keepdim:fp8-rowwise-keepdim \
           nvfp4-w4a4:nvfp4-w4a4 mxfp8:mxfp8; do
    before=$(sha256sum "vectors/mined/${n%%:*}.json" | cut -d' ' -f1)
    "$TFS" vectors mine "$WORK/mined.rows" "${n%%:*}" --match "${n##*:}" > /dev/null
    after=$(sha256sum "vectors/mined/${n%%:*}.json" | cut -d' ' -f1)
    if [ "$before" = "$after" ]; then echo "  ${n%%:*}: STABLE $after"; else
      echo "  ${n%%:*}: CHANGED $before -> $after"; fi
  done
}

conform_section() {
  hr "the conformance harness: bit-for-bit, against producer bytes"
  "$TFS" conform
}

capability_section() {
  hr "capability is FAIL-CLOSED (the DTYPE_MIN_SM class, under its new owner)"
  echo "-- a qualified (encoding, device) pair:"
  "$TFS" capability x cpu --alias mxfp8/1 2>&1 | sed 's/^/  /'
  echo "-- the SAME encoding on a device nothing was qualified on:"
  "$TFS" capability x cuda.sm89 --alias mxfp8/1 2>&1 | sed 's/^/  /'
  echo "-- a VECTORLESS spec on the device that IS qualified for others:"
  "$TFS" capability x cpu --alias svdq-microscale/1 2>&1 | sed 's/^/  /'
  echo "-- an entirely unknown device string:"
  "$TFS" capability x rocm.gfx1100 --alias plain/1 2>&1 | sed 's/^/  /'
}

border_section() {
  hr "launch border policy: only platform-aliased spec digests admit"
  "$TFS" registry | sed 's/^/  /'
  hr "serving admission against three pins (a foreign digest refuses, naming it)"
  "$TFS" checkpoint admit "$WORK/ck" "$(ck)" 2>&1 | tail -3
  "$TFS" checkpoint admit "$WORK/ck" "$(ck)" --pin plain/1 2>&1 | tail -2
  "$TFS" checkpoint admit "$WORK/ck" "$(ck)" --pin empty 2>&1 | tail -2
}

unknown_section() {
  hr "unknown/hand-edited encoding: storage identity DERIVES, serving REFUSES"
  echo "-- the corpus carries the same bytes through both gates:"
  "$TFS" vectors verify | grep -E 'unaliased-spec-serving|header/plain-fp8 ' | sed 's/^/  /'
  echo
  echo "-- and a vectorless spec stores and inspects but never qualifies:"
  "$TFS" conform svdq-microscale/1 2>&1 | sed 's/^/  /'
}

fences_section() {
  hr "fences green"
  python3 scripts/fence.py

  hr "RED — plant a SECOND shape-relation evaluator and an open rule parser"
  cat > crates/tensorfs-core/src/planted_second_evaluator.rs <<'RS'
pub fn eval(tag: &str, dim: u64, by: u64) -> u64 {
    match tag {
        "ceil_div" => dim.div_ceil(by),
        "elements_div" => dim / by,
        _ => dim,
    }
}
RS
  cat > crates/tensorfs-core/src/planted_rule_parser.rs <<'RS2'
pub struct Arrangement { pub morphism: String }
pub fn parse_quant_rule(s: &str) -> Arrangement {
    Arrangement { morphism: s.to_string() }
}
RS2
  cat > crates/tensorfs-core/src/planted_source_layout.rs <<'RS3'
pub fn source_layout(name: &str) -> Vec<u64> {
    let strides = vec![name.len() as u64];
    strides
}
RS3
  python3 scripts/fence.py
  echo "  (exit $?)"
  rm -f crates/tensorfs-core/src/planted_second_evaluator.rs \
        crates/tensorfs-core/src/planted_rule_parser.rs \
        crates/tensorfs-core/src/planted_source_layout.rs
  hr "plants removed"
  python3 scripts/fence.py
}

mkdir -p "$WORK"
case "$SECTION" in
  fixtures) fixtures_section ;;
  mine) mine_section ;;
  conform) conform_section ;;
  capability) capability_section ;;
  border) border_section ;;
  unknown) unknown_section ;;
  fences) fences_section ;;
  all) fixtures_section; mine_section; conform_section; capability_section; \
       border_section; unknown_section; fences_section ;;
  *) echo "usage: $0 [fixtures|mine|conform|capability|border|unknown|fences|all]"; exit 2 ;;
esac
