#!/usr/bin/env bash
# tfs-015 live verification. Not a test harness: this writes a real multi-hundred-MB
# checkpoint through the real CAS, reads every byte back, and plants real lies in real
# stored objects to watch the header codec refuse them.
# Sections: write | read | query | locate | admit | red | size | all
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-015-runs}
BIG="$WORK/big"
RED="$WORK/red"
SECTION=${1:-all}

hr() { printf '\n===== %s =====\n' "$*"; }
hex() { grep -o '[0-9a-f]\{64\}' | head -1; }
objpath() { echo "$1/objects/sha256/${2:0:2}/${2:2:2}/$2"; }

# One shared big checkpoint: written once, read many.
big_header() {
  if [ ! -f "$WORK/big.header" ]; then
    mkdir -p "$BIG"
    "$TFS" checkpoint write "$BIG" | tee "$WORK/big.write" | grep header_digest | hex > "$WORK/big.header"
  fi
  cat "$WORK/big.header"
}

write_section() {
  hr "write a real checkpoint: 3 components, 4 encodings, ~382 MiB through the CAS"
  rm -rf "$BIG" "$WORK/big.header"; mkdir -p "$BIG"
  nice -n 19 "$TFS" checkpoint write "$BIG" | tee "$WORK/big.write"
  grep header_digest "$WORK/big.write" | hex > "$WORK/big.header"
  echo "objects installed: $(find "$BIG/objects" -type f | wc -l)"

  hr "the same tree objectizes deterministically into an independent store"
  rm -rf "$WORK/det"; mkdir -p "$WORK/det"
  nice -n 19 "$TFS" checkpoint write "$WORK/det" --blocks 1 | grep -E 'header_digest|checkpoint_id|topology' > "$WORK/det.a"
  rm -rf "$WORK/det2"; mkdir -p "$WORK/det2"
  nice -n 19 "$TFS" checkpoint write "$WORK/det2" --blocks 1 | grep -E 'header_digest|checkpoint_id|topology' > "$WORK/det.b"
  cat "$WORK/det.a"
  if diff -q "$WORK/det.a" "$WORK/det.b" >/dev/null; then
    echo "  IDENTICAL across two independent stores"
  else
    echo "  DIVERGED"; diff "$WORK/det.a" "$WORK/det.b"
  fi
}

read_section() {
  local h; h=$(big_header)
  hr "read every part back and compare against the regenerated source bytes"
  nice -n 19 "$TFS" checkpoint read "$BIG" "$h"
}

query_section() {
  local h; h=$(big_header)
  hr "component/scope query"
  "$TFS" checkpoint info "$BIG" "$h"
  hr "the same query under strace: which objects were opened?"
  strace -f -e trace=openat -o "$WORK/info.strace" "$TFS" checkpoint info "$BIG" "$h" >/dev/null 2>&1
  echo "object opens: $(grep -c 'objects/sha256' "$WORK/info.strace")"
  grep -o 'objects/sha256/[a-z0-9/]*' "$WORK/info.strace" | sed 's|.*/||' | sort -u | sed 's/^/  opened /'
  echo "  header is $h"
}

locate_section() {
  local h; h=$(big_header)
  hr "offsets derive from the length list alone: a span crossing the 64 MiB grid boundary"
  "$TFS" checkpoint locate "$BIG" "$h" text_encoder embeddings.weight value 67108000 2000 \
    --seed 'text_encoder/embeddings.weight#value'
  hr "and an inline rank-0 scalar"
  "$TFS" checkpoint locate "$BIG" "$h" unet blocks.0.ff.net.0.proj.weight weight_scale_2 0 4 \
    --seed 'unet/blocks.0.ff.net.0.proj.weight#weight_scale_2'
  hr "a range past the declaration refuses without touching a body"
  "$TFS" checkpoint locate "$BIG" "$h" text_encoder embeddings.weight value 134217720 16
}

admit_section() {
  local h; h=$(big_header)
  hr "serving admission against three local pins"
  for pin in "" "plain/1" "empty"; do
    if [ -z "$pin" ]; then
      echo "--- full platform pin"; "$TFS" checkpoint admit "$BIG" "$h"
    else
      echo "--- pin: $pin"; "$TFS" checkpoint admit "$BIG" "$h" --pin "$pin"
    fi
  done
}

# ---------------------------------------------------------------- red arms

red_section() {
  rm -rf "$RED"; mkdir -p "$RED"
  local h; h=$("$TFS" checkpoint write "$RED" --blocks 1 | grep header_digest | hex)
  echo "small checkpoint $h written for planting"

  hr "RED 1 — a segment length lie, refused BEFORE any body is fetched"
  cp "$(objpath "$RED" "$h")" "$WORK/lie.json"
  chmod u+w "$WORK/lie.json"
  sed -i '0,/"length":4194304/s//"length":4194300/' "$WORK/lie.json"
  local lie; lie=$("$TFS" put "$RED" "$WORK/lie.json" | hex)
  echo "doctored header admitted at $lie"
  strace -f -e trace=openat -o "$WORK/lie.strace" "$TFS" checkpoint read "$RED" "$lie" --blocks 1
  echo "object opens during the refusal: $(grep -c 'objects/sha256' "$WORK/lie.strace") \
(header + $(($(grep -c 'objects/sha256' "$WORK/lie.strace") - 1)) spec/config objects; zero tensor segments)"

  hr "RED 2 — the same content, one trailing byte: never readable AS a header"
  cp "$(objpath "$RED" "$h")" "$WORK/twin.json"; chmod u+w "$WORK/twin.json"
  printf '\n' >> "$WORK/twin.json"
  local twin; twin=$("$TFS" put "$RED" "$WORK/twin.json" | hex)
  "$TFS" checkpoint read "$RED" "$twin" --blocks 1

  hr "RED 3 — a spec object missing from the closure: cannot install as self-describing"
  local spec; spec=$("$TFS" checkpoint info "$RED" "$h" | grep '^encoding' | head -1 | hex)
  cp "$(objpath "$RED" "$spec")" "$WORK/spec.bak"
  rm -f "$(objpath "$RED" "$spec")"
  "$TFS" checkpoint read "$RED" "$h" --blocks 1
  cp "$WORK/spec.bak" "$(objpath "$RED" "$spec")"; chmod 444 "$(objpath "$RED" "$spec")"

  hr "RED 4 — a tensor segment corrupted under the store: never reconstructed"
  local seg
  seg=$("$TFS" checkpoint locate "$RED" "$h" unet blocks.0.attn1.to_out.0.weight value 0 8 \
        | grep 'segment 0' | grep -o 'sha256:[0-9a-f]*' | cut -d: -f2)
  seg=$(find "$RED/objects" -name "${seg}*" -printf '%f\n' | head -1)
  cp "$(objpath "$RED" "$seg")" "$WORK/seg.bak"
  chmod u+w "$(objpath "$RED" "$seg")"
  printf 'tampered' | dd of="$(objpath "$RED" "$seg")" bs=1 seek=1000 conv=notrunc status=none
  "$TFS" checkpoint read "$RED" "$h" --blocks 1 2>&1 | head -3
  echo "quarantined: $(find "$RED/quarantine" -type f | wc -l) object(s)"

  hr "RED 5 — whole-document totals, built in memory (too large to freeze as vectors)"
  "$TFS" checkpoint caps
}

size_section() {
  hr "uniform-form header cost at the real selected-H3 cardinality"
  nice -n 19 "$TFS" checkpoint size 3699
  hr "and at SDXL cardinality"
  nice -n 19 "$TFS" checkpoint size 2515
}

mkdir -p "$WORK"
case "$SECTION" in
  write) write_section ;;
  read) read_section ;;
  query) query_section ;;
  locate) locate_section ;;
  admit) admit_section ;;
  red) red_section ;;
  size) size_section ;;
  all) write_section; read_section; query_section; locate_section; admit_section; red_section; size_section ;;
  *) echo "usage: $0 [write|read|query|locate|admit|red|size|all]"; exit 2 ;;
esac
