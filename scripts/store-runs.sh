#!/usr/bin/env bash
# tfs-002 live verification. Not a test harness: this spawns REAL `tfs` processes, kills
# them with SIGKILL at real admission stages, races real writers, plants real corrupt bytes,
# and reports what the store did. Sections: crash | race | red | all.
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
WORK=${WORK:-/tmp/tfs-002-runs}
SECTION=${1:-all}

hr() { printf '\n===== %s =====\n' "$*"; }
fresh() { rm -rf "$1"; mkdir -p "$1"; "$TFS" store init "$1" >/dev/null; }
count_objects() { find "$1/objects" -type f 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- crash injection

crash_section() {
  local S="$WORK/crash" SRC="$WORK/crash.src"
  mkdir -p "$WORK"; fresh "$S"
  head -c 64000000 /dev/urandom > "$SRC"
  local want; want=$(sha256sum "$SRC" | cut -d' ' -f1)
  local len; len=$(stat -c%s "$SRC")
  hr "kill -9 at every admission stage (64 MB object, want sha256:$want)"

  printf '%-9s %8s %8s %6s %8s   %s\n' stage objects partial temps record "resume verdict"
  for stage in open partial written fsync chmod linked record; do
    fresh "$S"   # each kill is independent: one crash, one store, one verdict
    "$TFS" put "$S" "$SRC" --fault "$stage" --ready "$S/ready" >/dev/null 2>&1 &
    local pid=$!
    for _ in $(seq 400); do [ -s "$S/ready" ] && break; sleep 0.05; done
    if ! [ -s "$S/ready" ]; then echo "$stage: never reached"; wait $pid; continue; fi
    kill -9 "$pid" 2>/dev/null
    wait $pid 2>/dev/null
    local objs temps partial
    objs=$(count_objects "$S")
    temps=$(find "$S/tmp" -type f | wc -l)
    # A partial object is any file under objects/ whose name is not its own digest.
    partial=0
    while read -r f; do
      [ -z "$f" ] && continue
      [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$(basename "$f")" ] || partial=$((partial+1))
    done < <(find "$S/objects" -type f)
    local rec; rec=$([ -f "$S/verified/${want:0:2}/${want:2:2}/$want" ] && echo present || echo absent)
    # What a restarted filler concludes about this store, and whether it heals it.
    printf '%s %s %s\n' "$want" "$len" "$SRC" > "$S/plan"
    local resume; resume=$("$TFS" fill "$S" "$S/plan")
    printf '%-9s %8d %8d %6d %8s   %s\n' "$stage" "$objs" "$partial" "$temps" "$rec" \
      "$(printf '%s' "$resume" | head -1)"
    "$TFS" verify "$S" "$want" >/dev/null || echo "  !! post-resume verify FAILED at $stage"
    "$TFS" reap "$S" --max-age 0 >/dev/null
  done
  echo "(partial = a file under objects/ whose bytes do not hash to its own name)"
}

# ---------------------------------------------------------------- concurrent writers

race_section() {
  local S="$WORK/race" SRC="$WORK/race.src" N=${N:-16}
  mkdir -p "$WORK"; fresh "$S"
  head -c 32000000 /dev/urandom > "$SRC"
  local want; want=$(sha256sum "$SRC" | cut -d' ' -f1)
  local len; len=$(stat -c%s "$SRC")
  hr "$N concurrent writer processes, one object (sha256:$want)"
  local pids=()
  for i in $(seq "$N"); do
    "$TFS" put "$S" "$SRC" --expect "$want" --expect-length "$len" > "$S/racer.$i" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "admitted: $(grep -lc '^admitted' "$S"/racer.* 2>/dev/null | wc -l)"
  echo "present:  $(grep -l '^present'  "$S"/racer.* 2>/dev/null | wc -l)"
  echo "objects under objects/: $(count_objects "$S")"
  echo "verification records:   $(find "$S/verified" -type f | wc -l)"
  "$TFS" contains "$S" "$want"
  "$TFS" verify "$S" "$want" --paranoid
  echo "stored bytes match source: $(cmp -s "$SRC" "$S/objects/sha256/${want:0:2}/${want:2:2}/$want" && echo yes || echo NO)"
}

# ---------------------------------------------------------------- red arms

red_section() {
  local S="$WORK/red" SRC="$WORK/red.src"
  mkdir -p "$WORK"; fresh "$S"
  head -c 4000000 /dev/urandom > "$SRC"
  local want; want=$(sha256sum "$SRC" | cut -d' ' -f1)
  local len; len=$(stat -c%s "$SRC")
  local obj="$S/objects/sha256/${want:0:2}/${want:2:2}/$want"

  hr "RED 1 — planted same-length corrupt object: contains true, verified read refuses"
  mkdir -p "$(dirname "$obj")"
  head -c "$len" /dev/urandom > "$obj"; chmod 0444 "$obj"
  echo "planted $(stat -c%s "$obj") bytes at the id of a $len-byte object"
  "$TFS" contains "$S" "$want"
  echo "--- a verified read refuses and quarantines ---"
  "$TFS" get "$S" "$want"; echo "get exit=$?"
  echo "quarantine: $(ls "$S/quarantine")"
  "$TFS" verify "$S" "$want"; echo "verify exit=$?"
  echo "--- plant it again: resume must not skip it either ---"
  rm -rf "${S:?}/quarantine"/*; mkdir -p "$(dirname "$obj")"
  head -c "$len" /dev/urandom > "$obj"; chmod 0444 "$obj"
  printf '%s %s %s\n' "$want" "$len" "$SRC" > "$S/plan"
  "$TFS" fill "$S" "$S/plan"; echo "fill exit=$?"
  echo "quarantine: $(ls "$S/quarantine")"
  echo "healed and verified: $("$TFS" verify "$S" "$want")"
  echo "stored bytes match source: $(cmp -s "$SRC" "$obj" && echo yes || echo NO)"

  hr "RED 2 — mutate a verified inode: the record invalidates"
  fresh "$S"; obj="$S/objects/sha256/${want:0:2}/${want:2:2}/$want"
  "$TFS" put "$S" "$SRC" >/dev/null
  "$TFS" contains "$S" "$want"
  echo "--- (a) same bytes, new inode+mtime (replace the file) ---"
  chmod 0644 "$obj"; cp "$SRC" "$obj.new"; mv "$obj.new" "$obj"; chmod 0444 "$obj"
  "$TFS" contains "$S" "$want"
  "$TFS" verify "$S" "$want"
  echo "--- (b) corrupt the bytes in place, same length ---"
  chmod 0644 "$obj"; printf 'X' | dd of="$obj" bs=1 seek=100 conv=notrunc status=none; chmod 0444 "$obj"
  "$TFS" contains "$S" "$want"
  "$TFS" get "$S" "$want"; echo "get exit=$?"
  echo "quarantine: $(ls "$S/quarantine")"

  hr "RED 3 — store generation bump invalidates every record wholesale"
  fresh "$S"; "$TFS" put "$S" "$SRC" >/dev/null
  "$TFS" contains "$S" "$want"
  "$TFS" store bump "$S" | grep generation
  "$TFS" contains "$S" "$want"
  "$TFS" verify "$S" "$want"
  "$TFS" contains "$S" "$want"

  hr "RED 4 — planted v1-era root refuses to open as v2"
  local V="$WORK/v1"; rm -rf "$V"; mkdir -p "$V/objects" "$V/refs"; : > "$V/metadata.sqlite3"
  "$TFS" store info "$V"; echo "exit=$?"
  echo "--- and a root carrying an explicit v1 era marker ---"
  local V2="$WORK/v1marker"; rm -rf "$V2"; mkdir -p "$V2"
  printf '{"era":"v1","format":"tensorfs.store/1","generation":1,"object_hash":"sha256"}' > "$V2/store.json"
  "$TFS" store info "$V2"; echo "exit=$?"

  hr "RED 4b — a symlink planted at an object path is never followed"
  fresh "$S"; obj="$S/objects/sha256/${want:0:2}/${want:2:2}/$want"
  mkdir -p "$(dirname "$obj")"; ln -s "$SRC" "$obj"
  echo "planted symlink: $(ls -l "$obj" | sed 's/.*-> /-> /')"
  echo "(the target holds exactly the right bytes — only the no-follow open refuses it)"
  "$TFS" get "$S" "$want"; echo "get exit=$?"
  "$TFS" verify "$S" "$want"; echo "verify exit=$?"

  hr "RED 5 — pretty-printed document twin refuses at put"
  fresh "$S"
  local DOC=vectors/cases/header/plain-fp8.json
  local did dlen
  did=$(sha256sum "$DOC" | cut -d' ' -f1); dlen=$(stat -c%s "$DOC")
  python3 -c "import json,sys; d=json.load(open('$DOC')); open('$WORK/pretty.json','w').write(json.dumps(d, indent=2, sort_keys=True))"
  echo "canonical: $dlen bytes  pretty twin: $(stat -c%s "$WORK/pretty.json") bytes"
  "$TFS" put "$S" "$DOC" --expect "$did" --expect-length "$dlen"
  "$TFS" put "$S" "$WORK/pretty.json" --expect "$did" --expect-length "$dlen"; echo "exit=$?"
  echo "--- same LENGTH, different bytes: one flipped byte in the canonical document ---"
  cp "$DOC" "$WORK/flip.json"
  printf 'X' | dd of="$WORK/flip.json" bs=1 seek=40 conv=notrunc status=none
  "$TFS" put "$S" "$WORK/flip.json" --expect "$did" --expect-length "$dlen"; echo "exit=$?"
  echo "--- the pretty twin admits at its OWN id (the store is shape-agnostic) ---"
  "$TFS" put "$S" "$WORK/pretty.json"
  echo "--- but it is not a header: the document plane refuses the twin ---"
  "$TFS" id "$WORK/pretty.json" | head -3

  hr "RED 6 — a killed fill resumes only objects holding valid verification records"
  fresh "$S"
  : > "$S/plan"
  for i in $(seq 12); do
    head -c 3000000 /dev/urandom > "$WORK/fill.$i"
    printf '%s %s %s\n' "$(sha256sum "$WORK/fill.$i" | cut -d' ' -f1)" \
      "$(stat -c%s "$WORK/fill.$i")" "$WORK/fill.$i" >> "$S/plan"
  done
  "$TFS" fill "$S" "$S/plan" --slow-ms 120 > "$S/fill.1" 2>&1 &
  local fpid=$!
  sleep 1.2; kill -9 $fpid; wait $fpid 2>/dev/null
  local done_n; done_n=$(grep -c '^put' "$S/fill.1")
  echo "kill -9 mid-fill after $done_n objects; objects/ holds $(count_objects "$S")"
  echo "--- now plant a same-length corrupt object for a NOT-yet-filled entry ---"
  local last; last=$(tail -1 "$S/plan" | cut -d' ' -f1)
  local llen; llen=$(tail -1 "$S/plan" | cut -d' ' -f2)
  mkdir -p "$S/objects/sha256/${last:0:2}/${last:2:2}"
  head -c "$llen" /dev/urandom > "$S/objects/sha256/${last:0:2}/${last:2:2}/$last"
  chmod 0444 "$S/objects/sha256/${last:0:2}/${last:2:2}/$last"
  echo "--- resume ---"
  "$TFS" fill "$S" "$S/plan"; echo "fill exit=$?"
  echo "quarantine: $(ls "$S/quarantine")"
  hr "RED 6b — scrub finds a planted corruption at rate 1.0 and quarantines it"
  local first; first=$(head -1 "$S/plan" | cut -d' ' -f1)
  local flen; flen=$(head -1 "$S/plan" | cut -d' ' -f2)
  chmod 0644 "$S/objects/sha256/${first:0:2}/${first:2:2}/$first"
  head -c "$flen" /dev/urandom > "$S/objects/sha256/${first:0:2}/${first:2:2}/$first"
  chmod 0444 "$S/objects/sha256/${first:0:2}/${first:2:2}/$first"
  "$TFS" scrub "$S" --rate 1.0; echo "scrub exit=$?"
  echo "quarantine now: $(ls "$S/quarantine" | tr '\n' ' ')"
  echo "--- the same store at a partial rate (one knob) ---"
  "$TFS" scrub "$S" --rate 0.25 --pass 7
}

case "$SECTION" in
  crash) crash_section ;;
  race)  race_section ;;
  red)   red_section ;;
  all)   crash_section; race_section; red_section ;;
  *) echo "usage: $0 crash|race|red|all"; exit 2 ;;
esac
