#!/usr/bin/env bash
# The gguf-v1 planner, driven live (tfs-011). Not a test suite: every green arm runs the
# real binary over REAL banked bytes and every red arm PLANTS a defect in a COPY of them
# and prints the refusal it produced.
#
# The corpus is the six GGUF metadata regions pinned in ~/cozy_v2/h3-evidence — 33 to 75 KB
# probes of 15 to 36 GB files, fetched by range and pinned by digest. No weight byte
# transits this machine and none needs to: a plan is derived from the metadata region.
#
#   scripts/gguf-runs.sh plan     the plan over every banked carrier
#   scripts/gguf-runs.sh twin     logical shapes against the pinned safetensors twins
#   scripts/gguf-runs.sh labels   what a filename Q-label is worth
#   scripts/gguf-runs.sh red      the planted refusals
#   scripts/gguf-runs.sh all
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
BANK=${BANK:-$HOME/cozy_v2/h3-evidence}
W=${W:-/tmp/tfs-011-runs}
SECTION=${1:-all}
PASS=0; MISS=0

hr() { printf '\n===== %s =====\n' "$*"; }

red() {
  local want="$1" what="$2"; shift 2
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if grep -q "REFUSED $want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  RED  %-44s %s\n' "$what" "$want"
    grep -o "REFUSED $want:.*" <<<"$out" | head -1 | cut -c1-160 | sed 's/^/       /'
  else
    MISS=$((MISS+1)); printf '  MISS %-44s expected %s\n' "$what" "$want"
    tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

has() {
  local want="$1" what="$2"; shift 2
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if grep -q -- "$want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %-44s %s\n' "$what" "$want"
  else
    MISS=$((MISS+1)); printf '  MISS %-44s expected %s\n' "$what" "$want"
    tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

# ---------------------------------------------------------------- the banked corpus

stage() {
  mkdir -p "$W"
  [ -f "$W/staged" ] && return
  BANK="$BANK" W="$W" nice -n 19 python3 - <<'PY'
import json, os, pathlib, shutil
B, W = pathlib.Path(os.environ["BANK"]), pathlib.Path(os.environ["W"])
carriers = []
for row in ("gguf-abiray", "gguf-curve"):
    d = json.loads((B / f"rows/{row}.json").read_text())
    for f in d["banked_files"]:
        if f.get("mode") != "gguf":
            continue
        name = f["path"].split("/")[-1]
        shutil.copy(B / "cache" / f["fetch"]["bytes_sha256"], W / name)
        carriers.append((name, f["fetch"]["bytes_len"], f["declared_size"]))
(W / "carriers.txt").write_text("".join(f"{n} {p} {d}\n" for n, p, d in carriers))
print(f"{len(carriers)} banked metadata probes staged, "
      f"{sum(p for _, p, _ in carriers)} B describing {sum(d for _, _, d in carriers)} B")
PY
  # The unquantized twins the shape check runs against: the same pinned Comfy safetensors
  # headers the compatibility plane uses, transcribed by the one transcriber.
  for pair in "fl2va-dit:minimax_h3_fl2va_bf16" "curve-dit:minimax_h3_fl2va_pruned_bf16" \
              "te:qwen3vl_32b_minimax_h3_bf16"; do
    nice -n 19 python3 scripts/h3-pairs.py rows "${pair#*:}" "$W/twin.${pair%%:*}.json" >/dev/null || exit 1
  done
  touch "$W/staged"
}

carriers() { cut -d' ' -f1 "$W/carriers.txt"; }

plan_section() {
  hr "the plan, over every banked carrier — metadata only, zero weight bytes"
  while read -r name probe declared; do
    printf '\n  %s (probe %s B of a %s B carrier)\n' "$name" "$probe" "$declared"
    nice -n 19 "$TFS" gguf plan "$W/$name" --probe | sed 's/^/    /'
  done < "$W/carriers.txt"

  hr "the carrier-blob view against what the publisher's tree declares"
  echo "  a whole-blob view is what a path-consuming engine streams; the plan derives its"
  echo "  length from the tensor table, and the tree API states what the file weighs."
  while read -r name probe declared; do
    derived=$(nice -n 19 "$TFS" gguf plan "$W/$name" --probe | awk '/^carrier blob/ {print $3}')
    printf '  %-46s derived %14s  declared %14s  delta %s\n' \
      "$name" "$derived" "$declared" "$((declared - derived))"
  done < "$W/carriers.txt"
  echo "  A non-zero delta is trailing bytes the tensor table does not describe. The plan"
  echo "  states what it derived and never pads to make a number match: the blob's DIGEST"
  echo "  is an ingest fact over the real file, not something a metadata probe can mint."
}

twin_section() {
  hr "logical shapes against the pinned safetensors twins"
  echo "  GGUF states dimensions fastest-varying first and a producer may reshape a tensor"
  echo "  before quantizing it, so agreement here is a real cross-format result: the same"
  echo "  topology, measured twice through two unrelated packagings."
  has "535 logical shapes agree" "abiray fl2va Q8_0 vs comfy fl2va bf16" \
    "$TFS" gguf plan "$W/MiniMax-H3-FL2VA-Q8_0.gguf" --probe --twin "$W/twin.fl2va-dit.json"
  has "535 logical shapes agree" "abiray ref2va Q8_0 vs the same twin" \
    "$TFS" gguf plan "$W/MiniMax-H3-Ref2VA-Q8_0.gguf" --probe --twin "$W/twin.fl2va-dit.json"
  has "535 logical shapes agree" "abiray fl2va Q4_K_M vs comfy fl2va bf16" \
    "$TFS" gguf plan "$W/MiniMax-H3-FL2VA-Q4_K_M.gguf" --probe --twin "$W/twin.fl2va-dit.json"
  has "532 logical shapes agree" "joeygambino curve Q8_0 vs comfy pruned bf16" \
    "$TFS" gguf plan "$W/MiniMax-H3-fl2va-curve-Q8_0.gguf" --probe --twin "$W/twin.curve-dit.json"
  has "902 logical shapes agree" "the Q4_K_M text encoder vs comfy TE bf16" \
    "$TFS" gguf plan "$W/qwen3vl_32b_minimax_h3-Q4_K_M.gguf" --probe --twin "$W/twin.te.json"
  echo "  and a twin from the OTHER topology refuses rather than half-matching:"
  red MISSING_TENSOR "the 532-key curve carrier against the 535-key twin" \
    "$TFS" gguf plan "$W/MiniMax-H3-fl2va-curve-Q8_0.gguf" --probe --twin "$W/twin.fl2va-dit.json"
}

labels_section() {
  hr "what a producer LABEL is worth — measured, not argued"
  echo "  1. the filename. The same bytes under a different Q-label produce the same plan:"
  cp "$W/MiniMax-H3-FL2VA-Q8_0.gguf" "$W/definitely-Q4_K_M-i-promise.gguf"
  a=$(nice -n 19 "$TFS" gguf plan "$W/MiniMax-H3-FL2VA-Q8_0.gguf" --probe | awk '/^plan digest/ {print $3}')
  b=$(nice -n 19 "$TFS" gguf plan "$W/definitely-Q4_K_M-i-promise.gguf" --probe | awk '/^plan digest/ {print $3}')
  if [ "$a" = "$b" ]; then
    PASS=$((PASS+1)); printf '  ok   %-44s %s\n' "plan digest is filename-independent" "${a:0:26}…"
  else
    MISS=$((MISS+1)); printf '  MISS %-44s %s != %s\n' "plan digest moved with the name" "$a" "$b"
  fi
  rm -f "$W/definitely-Q4_K_M-i-promise.gguf"

  echo
  echo "  2. the label's own CLAIM. Two files ship under the same _K_M label; the mixing"
  echo "     policy that label names is in exactly one of them:"
  for f in MiniMax-H3-FL2VA-Q4_K_M.gguf qwen3vl_32b_minimax_h3-Q4_K_M.gguf; do
    printf '  %-46s ' "$f"
    nice -n 19 "$TFS" gguf plan "$W/$f" --probe |
      awk '/tensors .* B/ {printf "%s:%s ", $1, $2} END {print ""}'
  done
  echo "     Q4_K only on the DiT — no Q5_K/Q6_K anywhere in a file whose name says K_M."
  echo
  echo "  3. general.architecture. The H3 DiT that declares WAN:"
  for f in MiniMax-H3-FL2VA-Q8_0.gguf MiniMax-H3-fl2va-curve-Q8_0.gguf; do
    printf '  %-46s ' "$f"
    nice -n 19 "$TFS" gguf plan "$W/$f" --probe | awk '/^architecture/ {print $2}'
  done
  echo "     Same model family, two publishers, two different declared architectures — and"
  echo "     the twin check above identified both from their tensor tables regardless."
}

red_section() {
  hr "the planted refusals — every plant is a same-length patch of a COPY"
  for p in no-arch mixed-arch no-orig-shape bad-type misalign overlap short-metadata; do
    nice -n 19 python3 scripts/gguf-plant.py "$W/MiniMax-H3-FL2VA-Q8_0.gguf" \
      "$W/planted-$p.gguf" "$p" | sed 's/^/  /'
  done
  red GGUF_ARCH_UNDECLARED "no general.architecture" \
    "$TFS" gguf plan "$W/planted-no-arch.gguf" --probe
  red GGUF_ARCH_MIXED "a KV scoped to another architecture" \
    "$TFS" gguf plan "$W/planted-mixed-arch.gguf" --probe
  red GGUF_TYPE_UNKNOWN "a removed ggml enumerant" \
    "$TFS" gguf plan "$W/planted-bad-type.gguf" --probe
  red GGUF_GEOMETRY "a data offset off the alignment grid" \
    "$TFS" gguf plan "$W/planted-misalign.gguf" --probe
  red CARRIER_OVERLAP "two tensors claiming the same bytes" \
    "$TFS" gguf plan "$W/planted-overlap.gguf" --probe
  red GGUF_METADATA_TRUNCATED "a probe cut mid tensor-info table" \
    "$TFS" gguf plan "$W/planted-short-metadata.gguf" --probe
  echo "  the projection-pairing arm: the reshaped AdaLN projections are 2-D in the carrier"
  echo "  ([256, 1016064]) and 2-D logically ([96768, 2688]) — internally consistent, and a"
  echo "  different tensor. Only the producer's own KV says which:"
  red GGUF_SHAPE_UNRECOVERABLE "orig_shape KVs renamed away" \
    "$TFS" gguf plan "$W/planted-no-orig-shape.gguf" --probe --twin "$W/twin.fl2va-dit.json"
  rm -f "$W"/planted-*.gguf

  hr "the two refusals that need no plant"
  red GGUF_CARRIER_TRUNCATED "a probe planned as if it were the whole carrier" \
    "$TFS" gguf plan "$W/MiniMax-H3-FL2VA-Q8_0.gguf"
  red GGUF_ENCODING_NOT_REVIEWED "admission over a real banked carrier" \
    "$TFS" gguf admit "$W/MiniMax-H3-FL2VA-Q8_0.gguf" --probe

  hr "and the normal door has NOT moved"
  echo "  the planner is not a second ingest door: tfs-003's border still refuses the same"
  echo "  bytes on the normal path, which is what 'consumer-gated' has to mean in code."
  red GGUF_REFUSED "tfs-003's border, same carrier" \
    "$TFS" ingest inspect --source c="$W/MiniMax-H3-FL2VA-Q8_0.gguf" --registry "$W/no-registry.json"
}

stage
case "$SECTION" in
  plan) plan_section ;;
  twin) twin_section ;;
  labels) labels_section ;;
  red) red_section ;;
  all) plan_section; twin_section; labels_section; red_section ;;
  *) echo "unknown section $SECTION"; exit 2 ;;
esac
printf '\n%d arms fired, %d missed\n' "$PASS" "$MISS"
[ "$MISS" -eq 0 ]
