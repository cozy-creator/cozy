#!/usr/bin/env bash
# tfs-004 live verification. Not a test harness: this writes a real checkpoint through the
# real CAS, walks its transitive root against the store's own object census, and plants real
# lies in real stored documents to watch the codec refuse them.
# Sections: walk | tensors-only | blob | fuzz | flip | red | fences | db | all
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-004-runs}
BIG="$WORK/big"
RED="$WORK/red"
SECTION=${1:-all}

hr() { printf '\n===== %s =====\n' "$*"; }
hex() { grep -o '[0-9a-f]\{64\}' | head -1; }
objpath() { echo "$1/objects/sha256/${2:0:2}/${2:2:2}/$2"; }

big() {
  if [ ! -f "$WORK/big.snap" ]; then
    rm -rf "$BIG"; mkdir -p "$BIG"
    nice -n 19 "$TFS" checkpoint write "$BIG" | tee "$WORK/big.write" >/dev/null
    grep checkpoint_id "$WORK/big.write" | hex > "$WORK/big.snap"
    grep header_digest "$WORK/big.write" | hex > "$WORK/big.header"
  fi
  cat "$WORK/big.snap"
}

walk_section() {
  local s; s=$(big)
  hr "the real 382 MiB checkpoint, written through the real CAS"
  cat "$WORK/big.write"

  hr "strict decode + decode->re-encode identity"
  "$TFS" snapshot show "$BIG" "$s"

  hr "transitive root: manifest -> header -> closure -> tensor rules -> part runs"
  nice -n 19 "$TFS" snapshot walk "$BIG" "$s"

  hr "full-checkpoint verify walk: every declared byte, through the real store"
  nice -n 19 "$TFS" snapshot verify "$BIG" "$s"

  hr "the same tree in an independent store gives the same snapshot id"
  rm -rf "$WORK/det" "$WORK/det2"; mkdir -p "$WORK/det" "$WORK/det2"
  a=$(nice -n 19 "$TFS" checkpoint write "$WORK/det" --blocks 1 | grep checkpoint_id | hex)
  b=$(nice -n 19 "$TFS" checkpoint write "$WORK/det2" --blocks 1 | grep checkpoint_id | hex)
  echo "  A $a"; echo "  B $b"
  [ "$a" = "$b" ] && echo "  IDENTICAL across two independent stores" || echo "  DIVERGED"
}

tensors_only_section() {
  hr "TENSORS-ONLY: one header plus its closure, zero sibling assets"
  rm -rf "$WORK/solo"; mkdir -p "$WORK/solo"
  nice -n 19 "$TFS" checkpoint write "$WORK/solo" --blocks 1 --no-configs | tee "$WORK/solo.write" | grep -E 'checkpoint_id|configs'
  local s; s=$(grep checkpoint_id "$WORK/solo.write" | hex)
  "$TFS" snapshot show "$WORK/solo" "$s"
  echo "-- the id derivation is the SAME function; nothing branches on 'has a snapshot'"
  "$TFS" snapshot walk "$WORK/solo" "$s"
}

blob_section() {
  hr "a multi-segment blob: admit, verify, repair ONE segment"
  rm -rf "$WORK/blobstore"; mkdir -p "$WORK/blobstore"
  local f="$WORK/big.bin"
  [ -f "$f" ] || nice -n 19 dd if=/dev/urandom of="$f" bs=1M count=200 status=none
  "$TFS" store init "$WORK/blobstore" >/dev/null
  nice -n 19 "$TFS" snapshot blob put "$WORK/blobstore" "$f"
  echo
  nice -n 19 "$TFS" snapshot blob read "$WORK/blobstore" "$f"

  hr "RED — a CHANGED segment under an unchanged whole digest"
  local seg; seg=$(nice -n 19 "$TFS" snapshot blob put "$WORK/blobstore" "$f" | grep '\[1\]' | grep -o '[0-9a-f]\{64\}')
  cp "$(objpath "$WORK/blobstore" "$seg")" "$WORK/seg.bak"
  chmod u+w "$(objpath "$WORK/blobstore" "$seg")"
  printf 'tampered' | dd of="$(objpath "$WORK/blobstore" "$seg")" bs=1 seek=4096 conv=notrunc status=none
  nice -n 19 "$TFS" snapshot blob read "$WORK/blobstore" "$f" 2>&1 | head -3
  echo "  quarantined: $(find "$WORK/blobstore/quarantine" -type f 2>/dev/null | wc -l) object(s)"

  hr "RED — a lie in the DESCRIPTOR: every segment verifies, the reassembly does not"
  nice -n 19 "$TFS" snapshot blob read "$WORK/blobstore" "$f" \
    --whole 0000000000000000000000000000000000000000000000000000000000000000 2>&1 | head -2

  hr "repair exactly that one segment — not the whole file"
  nice -n 19 "$TFS" snapshot blob repair "$WORK/blobstore" "$f" 1
  nice -n 19 "$TFS" snapshot blob read "$WORK/blobstore" "$f"
}

fuzz_section() {
  local s; s=$(big)
  hr "bounded mutation fuzz over real canonical snapshot bytes"
  /usr/bin/time -v nice -n 19 "$TFS" snapshot fuzz "$BIG" "$s" --iters 200000 2>&1 \
    | grep -vE '^\s+(Command|User|System|Average|Major|Minor|Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Exit status)' \
    | sed '/^\t/d'
}

flip_section() {
  local s; s=$(big)
  hr "identity matrix: what moves checkpoint_id, header_digest, topology_digest"
  "$TFS" snapshot flip "$BIG" "$s"
}

red_section() {
  hr "planting real lies in real stored documents"
  rm -rf "$RED"; mkdir -p "$RED"
  nice -n 19 "$TFS" checkpoint write "$RED" --blocks 1 > "$WORK/red.write"
  local s h
  s=$(grep checkpoint_id "$WORK/red.write" | hex)
  h=$(grep header_digest "$WORK/red.write" | hex)
  "$TFS" get "$RED" "$s" --out "$WORK/snap.json" >/dev/null

  hr "RED 1 — a config reference the snapshot does not CONTAIN (path-swapped)"
  local cfg new
  cfg=$("$TFS" snapshot show "$RED" "$s" | grep 'unet/config.json' | grep -o '[0-9a-f]\{64\}')
  # point the manifest entry at a different object while the header still names the old one
  python3 - "$WORK/snap.json" "$cfg" "$WORK/swap.json" <<'PY'
import sys, pathlib
b = pathlib.Path(sys.argv[1]).read_bytes()
old = sys.argv[2].encode()
new = bytes(((c + 1) % 256) if False else c for c in old)
new = (b"0" if old[:1] != b"0" else b"1") + old[1:]
pathlib.Path(sys.argv[3]).write_bytes(b.replace(old, new))
PY
  new=$("$TFS" put "$RED" "$WORK/swap.json" | hex)
  "$TFS" snapshot walk "$RED" "$new" 2>&1 | head -2

  hr "RED 2 — a DANGLING tensor rule: a part segment object removed from the store"
  local seg
  seg=$("$TFS" snapshot walk "$RED" "$s" >/dev/null 2>&1; "$TFS" checkpoint locate "$RED" "$h" unet blocks.0.attn1.to_out.0.weight value 0 8 \
        | grep 'segment 0' | grep -o 'sha256:[0-9a-f]*' | cut -d: -f2)
  seg=$(find "$RED/objects" -name "${seg}*" -printf '%f\n' | head -1)
  cp "$(objpath "$RED" "$seg")" "$WORK/dangle.bak"
  rm -f "$(objpath "$RED" "$seg")"
  "$TFS" snapshot verify "$RED" "$s" 2>&1 | head -2
  cp "$WORK/dangle.bak" "$(objpath "$RED" "$seg")"; chmod 0444 "$(objpath "$RED" "$seg")"

  hr "RED 3 — the header attachment removed from the tree it heads"
  python3 - "$WORK/snap.json" "$WORK/unlisted.json" <<'PY'
import sys, pathlib, re
b = pathlib.Path(sys.argv[1]).read_text()
b = re.sub(r'\{"kind":"file","object":\{"length":\d+,"sha256":"[0-9a-f]{64}"\},"path":"model\.cozytensors"\},', '', b, count=1)
pathlib.Path(sys.argv[2]).write_text(b)
PY
  new=$("$TFS" put "$RED" "$WORK/unlisted.json" | hex)
  "$TFS" snapshot show "$RED" "$new" 2>&1 | head -2

  hr "RED 4 — a MULTIPLY-SELECTED object run is LEGAL and counted once"
  echo "  (the deliberate twin pair: two tensor rules, one byte-identical run)"
  "$TFS" snapshot walk "$BIG" "$(big)" | grep -E 'reached|census'

  hr "RED 5 — the frozen manifest corpus: 19 red arms over exact stored bytes"
  "$TFS" vectors verify | grep -E '^(PASS|FAIL) manifest' | sed 's/^/  /'
}

db_section() {
  hr "canonical bytes are the identity source: there is no DB to mirror them"
  echo "-- any database engine named anywhere in the crates:"
  grep -rniE 'turso|libsql|sqlite|\.db\b|sqlx|rusqlite' crates/ --include='*.rs' --include='*.toml' | sed 's/^/  /' || true
  echo "  (the single hit is store.rs's v1 ERA GATE — the v1 pathname it refuses to open,"
  echo "   not a database. tfs-004 needs none; roots/holds/GC arrive with tfs-006.)"
  echo "-- tensorfs-core dependencies:"
  sed -n '/\[dependencies\]/,$p' crates/tensorfs-core/Cargo.toml | sed 's/^/  /'
  echo "-- every advertised id equals sha256 of the stored object file:"
  local s; s=$(big)
  echo "  advertised sha256:$s"
  echo "  sha256sum   $(sha256sum "$(objpath "$BIG" "$s")" | cut -d' ' -f1)"
}

fences_section() {
  hr "the fences, green on the real tree"
  python3 scripts/fence.py

  hr "RED — plant a second snapshot codec (Go) and a foreign container parser"
  mkdir -p "$WORK/plant"
  cat > crates/tensorfs-core/src/planted_go_codec.go <<'GO'
package snapshot
const Format = "tensorfs.snapshot/1"
func Decode(b []byte) {}
GO
  cat > scripts/planted_py_codec.py <<'PY2'
FORMAT = "tensorfs.snapshot/1"
def decode(b): return b
PY2
  cat > crates/tensorfs-core/src/planted_parser.rs <<'RS'
pub fn read(p: &str) -> Vec<u8> {
    let safetensors = std::fs::read(p).unwrap();
    safetensors
}
RS
  cat > crates/tensorfs-core/src/planted_evaluator.rs <<'RS2'
pub fn eval(tag: &str) -> u64 {
    if tag == "ceil_block" { 1 } else { 0 }
}
RS2
  python3 scripts/fence.py
  echo "  (exit $?)"
  rm -f crates/tensorfs-core/src/planted_go_codec.go scripts/planted_py_codec.py \
        crates/tensorfs-core/src/planted_parser.rs crates/tensorfs-core/src/planted_evaluator.rs

  hr "plants removed — green again"
  python3 scripts/fence.py
}

mkdir -p "$WORK"
case "$SECTION" in
  fences) fences_section ;;
  walk) walk_section ;;
  tensors-only) tensors_only_section ;;
  blob) blob_section ;;
  fuzz) fuzz_section ;;
  flip) flip_section ;;
  red) red_section ;;
  db) db_section ;;
  all) walk_section; tensors_only_section; blob_section; flip_section; fuzz_section; red_section; fences_section; db_section ;;
  *) echo "usage: $0 [walk|tensors-only|blob|fuzz|flip|red|fences|db|all]"; exit 2 ;;
esac
