#!/usr/bin/env bash
# The ingest border, driven live (tfs-003). Not a test suite: every arm PLANTS a defect,
# RUNS the real binary against it, and prints the refusal it produced.
#
#   scripts/ingest-runs.sh all           everything below
#   scripts/ingest-runs.sh red           the hostile-carrier and refusal-class arms only
#   scripts/ingest-runs.sh real          the real-SDXL end-to-end + benchmark
#
# The real-artifact arms need a diffusers SDXL tree; point SDXL at one. They skip if absent.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TFS="$ROOT/target/release/tfs"
WORK="${WORK:-/tmp/tfs-003-runs}"
SDXL="${SDXL:-/home/fidika/cozy_v2/tensorfs-bench/hf/sdxl}"
H3="${H3:-$ROOT/../tracker-v2/examples/safetensors-headers/minimax_h3_fl2va_pruned_fp8_scaled.header.json}"
REG="$WORK/registry.json"
MODE="${1:-all}"
PASS=0; MISS=0

banner() { printf '\n===== %s =====\n' "$*"; }

# An arm FIRES when the run refuses with the named code. A green run where a refusal was
# expected is the failure — silence is the defect this whole border exists to prevent.
red() {
  local want="$1" what="$2"; shift 2
  local out; out="$("$@" 2>&1)"
  if grep -q "REFUSED $want" <<<"$out"; then
    PASS=$((PASS+1))
    printf '  RED  %-42s %s\n' "$what" "$want"
    grep -o "REFUSED $want:.*" <<<"$out" | head -1 | cut -c1-150 | sed 's/^/       /'
  else
    MISS=$((MISS+1))
    printf '  MISS %-42s expected %s\n' "$what" "$want"
    tail -2 <<<"$out" | sed 's/^/       /'
  fi
}

green() {
  local what="$1"; shift
  local out; out="$("$@" 2>&1)"
  if [ $? -eq 0 ] && ! grep -q REFUSED <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %s\n' "$what"
  else
    MISS=$((MISS+1)); printf '  MISS %-42s expected success\n' "$what"; tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

rm -rf "$WORK"; mkdir -p "$WORK/carriers"
for v in good pickle pickle-proto0 zip gguf gzip header-cap truncated geometry overlap gap \
         dtype-unknown float-dim no-tensors bnb deep \
         mixed-fp8 mixed-fp8-spoof rowwise-keepdim; do
  python3 "$ROOT/scripts/make-carrier.py" "$v" "$WORK/carriers/$v.safetensors"
done

# One reviewed entry for the well-formed toy carrier, so the arms downstream of
# classification have something authorized to work with.
"$TFS" ingest bank safetensors.diffusers toy v1 diffusers.identity/1 \
  --source toy="$WORK/carriers/good.safetensors" --registry "$REG" \
  --source-name synthetic-toy-carrier >/dev/null
# Two synthetic MIXED carriers with real bytes: the per-tensor split in miniature (one
# module with both scales, one with the weight scale alone, plain tensors beside them) and
# the other scale geometry under the fp8-rowwise alias.
"$TFS" ingest bank safetensors.diffusers toy mixed-fp8 diffusers.identity/1 \
  --source mix="$WORK/carriers/mixed-fp8.safetensors" --registry "$REG" \
  --source-name synthetic-mixed-fp8 >/dev/null
"$TFS" ingest bank safetensors.diffusers toy rowwise-keepdim diffusers.identity/1 \
  --source kd="$WORK/carriers/rowwise-keepdim.safetensors" --registry "$REG" \
  --source-name synthetic-rowwise-keepdim >/dev/null

if [ "$MODE" = all ] || [ "$MODE" = red ]; then

banner "RED 1 — hostile and unsafe carriers, refused on the magic"
red PICKLE_REFUSED  "pickle PROTO opcode (1 MiB file)"     "$TFS" ingest inspect --source c="$WORK/carriers/pickle.safetensors" --registry "$REG"
red PICKLE_REFUSED  "protocol-0 pickle opcode"             "$TFS" ingest inspect --source c="$WORK/carriers/pickle-proto0.safetensors" --registry "$REG"
red ARCHIVE_REFUSED "zip archive (a torch object graph)"   "$TFS" ingest inspect --source c="$WORK/carriers/zip.safetensors" --registry "$REG"
red GGUF_REFUSED    "GGUF (consumer-gated, not this door)" "$TFS" ingest inspect --source c="$WORK/carriers/gguf.safetensors" --registry "$REG"
red ARCHIVE_REFUSED "compressed stream"                    "$TFS" ingest inspect --source c="$WORK/carriers/gzip.safetensors" --registry "$REG"

banner "RED 2 — malformed geometry, refused by arithmetic before any allocation"
red CARRIER_HEADER_CAP "header claims 2^63-1 bytes"        "$TFS" ingest inspect --source c="$WORK/carriers/header-cap.safetensors" --registry "$REG"
red CARRIER_TRUNCATED  "torn source: 32 bytes cut"         "$TFS" ingest inspect --source c="$WORK/carriers/truncated.safetensors" --registry "$REG"
red CARRIER_GEOMETRY   "run length != shape x dtype"       "$TFS" ingest inspect --source c="$WORK/carriers/geometry.safetensors" --registry "$REG"
red CARRIER_OVERLAP    "two keys sharing bytes"            "$TFS" ingest inspect --source c="$WORK/carriers/overlap.safetensors" --registry "$REG"
red CARRIER_GEOMETRY   "unclaimed bytes in the region"     "$TFS" ingest inspect --source c="$WORK/carriers/gap.safetensors" --registry "$REG"
red DTYPE_UNKNOWN      "a dtype outside the closed table"  "$TFS" ingest inspect --source c="$WORK/carriers/dtype-unknown.safetensors" --registry "$REG"
red NON_INTEGER_NUMBER "a fractional dimension"            "$TFS" ingest inspect --source c="$WORK/carriers/float-dim.safetensors" --registry "$REG"
red MISSING_FIELD      "a carrier declaring no tensor"     "$TFS" ingest inspect --source c="$WORK/carriers/no-tensors.safetensors" --registry "$REG"
red DEPTH_CAP          "64-deep nesting in the header"     "$TFS" ingest inspect --source c="$WORK/carriers/deep.safetensors" --registry "$REG"

banner "RED 3 — the KILLED bitsandbytes reader"
red UNREGISTERED_ENCODING "bnb quant state, decoded by nothing" \
    "$TFS" ingest inspect --source c="$WORK/carriers/bnb.safetensors" --registry "$REG"

banner "RED 4 — classification: filenames never authorize"
cp "$WORK/carriers/good.safetensors" "$WORK/carriers/sdxl_base_1.0_official.safetensors"
python3 - "$WORK/carriers/unknown.safetensors" <<'PY'
import json, struct, sys
h = {"totally.different.key": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}
js = json.dumps(h).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(js)) + js + bytes(8))
PY
red UNREGISTERED_FINGERPRINT "an authoritative-looking FILENAME over unknown keys" \
    "$TFS" ingest plan c=plain/1 --source c="$WORK/carriers/unknown.safetensors" --registry "$REG"
green "the same bytes under a lying filename still classify by KEYS" \
    "$TFS" ingest plan toy=plain/1 --source toy="$WORK/carriers/sdxl_base_1.0_official.safetensors" --registry "$REG"

banner "RED 5 — surviving ambiguity refuses loudly"
python3 - "$REG" "$WORK/ambiguous.json" <<'PY'
import json, sys
# A SECOND reviewed entry for the same key set under the same component: a review defect,
# planted here exactly as one would arrive — by two operators banking the same architecture.
r = json.load(open(sys.argv[1]))
e = [x for x in r["entries"] if x["component"] == "toy"][0]
twin = dict(e, task_family="toy-rival", structural_variant="v2", component="toy-rival")
r["entries"] = sorted(r["entries"] + [twin],
                      key=lambda x: (x["keyset_digest"], x["component"]))
json.dump(r, open(sys.argv[2], "w"), separators=(",", ":"), sort_keys=True)
PY
red AMBIGUOUS_CLASSIFICATION "one key set, two reviewed entries" \
    "$TFS" ingest plan toy=plain/1 --source toy="$WORK/carriers/good.safetensors" --registry "$WORK/ambiguous.json"

banner "RED 6 — conversion refusal classes, decided from headers alone"
red REMEDY_FROM_TARGET "plain bf16 offered as fp8-rowwise (the CAST class)" \
    "$TFS" ingest plan toy=fp8-rowwise/1 --spec-variant 1 --source toy="$WORK/carriers/good.safetensors" --registry "$REG"
red SHAPE_MISMATCH "the OTHER scale geometry under the same alias" \
    "$TFS" ingest plan kd=fp8-rowwise/1 --spec-variant 0 --source kd="$WORK/carriers/rowwise-keepdim.safetensors" --registry "$REG"

banner "RED 6a — per-tensor selection: the candidate set, and what it cannot be told"
# A candidate set of ONE is exactly the old single-spec path: the same carrier that refuses
# under the wrong pinned geometry plans green under the right one, and the UNPINNED set of
# two reaches the same topology by reading the carrier's own scale shape.
green "candidate set of one, correct geometry pinned" \
    "$TFS" ingest plan kd=fp8-rowwise/1 --spec-variant 1 --source kd="$WORK/carriers/rowwise-keepdim.safetensors" --registry "$REG"
A=$("$TFS" ingest plan kd=fp8-rowwise/1 --spec-variant 1 --source kd="$WORK/carriers/rowwise-keepdim.safetensors" --registry "$REG" | grep '^topology')
B=$("$TFS" ingest plan kd=fp8-rowwise/1 --source kd="$WORK/carriers/rowwise-keepdim.safetensors" --registry "$REG" | grep '^topology')
if [ "$A" = "$B" ] && [ -n "$A" ]; then
  PASS=$((PASS+1)); printf '  ok   %s\n' "the unpinned set of two reaches the SAME topology as the pinned one"
else
  MISS=$((MISS+1)); printf '  MISS %-42s pinned %s vs unpinned %s\n' "candidate-set-of-two topology" "$A" "$B"
fi
echo "  the mixed synthetic carrier, three encodings from four tensors:"
"$TFS" ingest plan mix=fp8-scaled-scalar/1 --source mix="$WORK/carriers/mixed-fp8.safetensors" --registry "$REG" \
  | grep -E "^encodings|^ +[0-9]+ +|^topology"
# A marker naming an encoding whose companions are ABSENT cannot promote a tensor. It is not
# believed and not consulted: its payload is tensor BYTES, and the plan reads zero of those.
S1=$("$TFS" ingest plan mix=fp8-scaled-scalar/1 --source mix="$WORK/carriers/mixed-fp8.safetensors" --registry "$REG" | grep -E '^topology|^ +[0-9]+ +')
S2=$("$TFS" ingest plan mix=fp8-scaled-scalar/1 --source mix="$WORK/carriers/mixed-fp8-spoof.safetensors" --registry "$REG" | grep -E '^topology|^ +[0-9]+ +')
if [ "$S1" = "$S2" ] && [ -n "$S1" ]; then
  PASS=$((PASS+1)); printf '  RED  %-42s marker dropped, plan identical\n' "a marker claiming fp8 over absent companions"
  echo "       $(grep -c . <<<"$S2") lines identical, topology unchanged by the planted marker"
else
  MISS=$((MISS+1)); printf '  MISS %-42s the planted marker moved the plan\n' "marker spoof"
fi

if [ -f "$H3" ]; then
  "$TFS" ingest bank safetensors.diffusers h3.fl2va pruned-fp8-scaled diffusers.identity/1 \
    --source transformer="$H3" --registry "$REG" --source-name h3-shipped-header >/dev/null
  red DOUBLE_ENCODE "an fp8-with-scales tree offered as plain storage" \
      "$TFS" ingest plan transformer=plain/1 --source transformer="$H3" --registry "$REG"
  red DOUBLE_ENCODE "the real DiT as fp8-rowwise: no role for .input_scale" \
      "$TFS" ingest plan transformer=fp8-rowwise/1 --source transformer="$H3" --registry "$REG"
  red MISSING_COMPANION_ROLE "candidate set pinned to the +input_scale variant" \
      "$TFS" ingest plan transformer=fp8-scaled-scalar/1 --spec-variant 0 --source transformer="$H3" --registry "$REG"
  red DOUBLE_ENCODE "candidate set pinned to the weight-scale-only variant" \
      "$TFS" ingest plan transformer=fp8-scaled-scalar/1 --spec-variant 1 --source transformer="$H3" --registry "$REG"
  # A planted MIXED carrier whose roles match no candidate: one module's .input_scale
  # becomes a .pre_quant_scale — same bytes, same geometry, same logical key set, a role set
  # no fp8-scaled-scalar variant declares.
  python3 - "$H3" "$WORK/h3-unknown-roles.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
victim = next(k for k in h if k.endswith(".input_scale"))
h[victim[: -len(".input_scale")] + ".pre_quant_scale"] = h.pop(victim)
json.dump(h, open(sys.argv[2], "w"))
print("reroled", victim)
PY
  red DOUBLE_ENCODE "one tensor's roles match NO candidate (a .pre_quant_scale)" \
      "$TFS" ingest plan transformer=fp8-scaled-scalar/1 --source transformer="$WORK/h3-unknown-roles.json" --registry "$REG"
  banner "GREEN — the REAL shipped H3 DiT header: per-tensor selection, 0 tensor bytes read"
  "$TFS" ingest plan transformer=fp8-scaled-scalar/1 --source transformer="$H3" --registry "$REG"
  python3 - "$H3" "$WORK/h3-torn.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
victim = next(k for k in h if k.endswith(".weight_scale"))
# REPARENTED, not deleted and not renamed away: deleting leaves unclaimed bytes (the gapless
# check fires first, correctly) and renaming the SUFFIX turns the role into a 944th logical
# key, which moves the key set and classification refuses first (also correctly). Moving the
# role to a module that has no weight keeps the region exact AND the key set intact, and
# orphans exactly one encoded weight — which is what a torn source actually looks like.
h[victim.rsplit(".", 2)[0] + ".NOSUCH.weight_scale"] = h.pop(victim)
json.dump(h, open(sys.argv[2], "w"))
print("removed", victim)
PY
  red MISSING_COMPANION_ROLE "an fp8 weight whose scale twin was removed" \
      "$TFS" ingest plan transformer=fp8-scaled-scalar/1 --source transformer="$WORK/h3-torn.json" --registry "$REG"
else
  echo "  (no H3 header at $H3 — skipping the fp8 arms)"
fi

banner "REAL 0 — the per-tensor split through the REAL transaction, on real bytes"
# Small carrier, real bytes, the whole border: isolated worker -> CAS -> canonical header ->
# Stamp -> receipt -> install -> transitive walk -> delivered-bytes identity. The 20 GB
# artifact's split is proven byte-for-byte here; at scale it stays a pod arm (tfs-018).
MIXSTORE="$WORK/mixstore"
"$TFS" ingest run "$MIXSTORE" mix=fp8-scaled-scalar/1 --registry "$REG" --session mix \
  --source mix="$WORK/carriers/mixed-fp8.safetensors" 2>&1 \
  | grep -E "^encodings|^ +[0-9]+ +|^topology|^stamp|^receipt|parts |tensors "
green "install the mixed candidate" "$TFS" ingest install "$MIXSTORE" mix
MIXSNAP=$("$TFS" ingest roots "$MIXSTORE" | grep installed | head -1 | awk '{print $2}' | cut -d: -f2)
"$TFS" snapshot walk "$MIXSTORE" "$MIXSNAP" | sed -n '2,3p'
green "delivered bytes hash to the digests the header declares" \
    "$TFS" read prove "$MIXSTORE" "$MIXSNAP"
"$TFS" read prove "$MIXSTORE" "$MIXSNAP" | sed 's/^/    /'
MIXHDR=$("$TFS" snapshot show "$MIXSTORE" "$MIXSNAP" | grep attachment | grep -oE '[0-9a-f]{64}')
red CONVERTS_NOTHING "re-ingest of the mixed result changes nothing" \
    "$TFS" ingest reingest "$MIXSTORE" "$MIXHDR" mix=fp8-scaled-scalar/1 --session m0
echo "  a MIXED canonical header re-keyed: three encodings matched by DIGEST, zero bytes hashed"
"$TFS" ingest reingest "$MIXSTORE" "$MIXHDR" mix=fp8-scaled-scalar/1 \
  --rekey mix=qkv.:attn.qkv. --session m1 2>&1 | grep -E "^ops|^objects|bytes_hashed|bytes_inherited" | sed 's/^/    /'

banner "RED 6b — converter geometry: the golden suite and its wrong-order twin"
"$ROOT/target/release/tfs" ingest golden
if "$ROOT/target/release/tfs" ingest golden | grep -q "^  RED"; then
  PASS=$((PASS+1))
else
  MISS=$((MISS+1)); echo "  MISS the wrong-order arm did not fire"
fi

banner "RED 7 — the fourth fence: foreign parsing lives only at the border"
cp "$ROOT/crates/tensorfs-core/src/checkpoint.rs" "$WORK/checkpoint.rs.bak"
cat >> "$ROOT/crates/tensorfs-core/src/checkpoint.rs" <<'RS'
fn planted_rival(safetensors: &[u8]) -> usize { safetensors.len() }
RS
if python3 "$ROOT/scripts/fence.py" 2>&1 | grep -q "foreign container"; then
  PASS=$((PASS+1)); printf '  RED  %-42s fence red\n' "a carrier parser planted in the serving path"
  python3 "$ROOT/scripts/fence.py" 2>&1 | grep "foreign container" | head -1 | sed 's/^/       /'
else
  MISS=$((MISS+1)); printf '  MISS %-42s fence stayed green\n' "planted parser outside the border"
fi
cp "$WORK/checkpoint.rs.bak" "$ROOT/crates/tensorfs-core/src/checkpoint.rs"
python3 "$ROOT/scripts/fence.py" >/dev/null && echo "       plant removed, fence green again"

fi   # red

if [ "$MODE" = all ] || [ "$MODE" = real ]; then

if [ ! -d "$SDXL" ]; then
  echo "  (no SDXL tree at $SDXL — skipping the real-artifact arms)"
else

STORE="$WORK/store"
banner "REAL 1 — hermetic bounded decode of a 6.9 GB diffusers tree"
"$TFS" ingest inspect --registry "$REG" \
  --source unet="$SDXL/unet/diffusion_pytorch_model.fp16.safetensors" \
  --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" \
  --source text_encoder="$SDXL/text_encoder/model.fp16.safetensors" \
  --source text_encoder_2="$SDXL/text_encoder_2/model.fp16.safetensors" 2>&1 | grep -E "carrier|keys |verdict|read "

banner "REAL 2 — bank the reviewed registry, then plan (headers only)"
"$TFS" ingest bank safetensors.diffusers sdxl base-1.0-fp16 diffusers.identity/1 --registry "$REG" \
  --source unet="$SDXL/unet/diffusion_pytorch_model.fp16.safetensors" \
  --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" \
  --source text_encoder="$SDXL/text_encoder/model.fp16.safetensors" \
  --source text_encoder_2="$SDXL/text_encoder_2/model.fp16.safetensors" \
  --source-name stabilityai/stable-diffusion-xl-base-1.0 | tail -1

banner "REAL 3 — the isolated run: carrier bytes -> canonical snapshot under quarantine"
/usr/bin/time -f "  wall %e s   peak RSS %M KiB" \
"$TFS" ingest run "$STORE" unet=plain/1,vae=plain/1,text_encoder=plain/1,text_encoder_2=plain/1 \
  --registry "$REG" --session sdxl \
  --source unet="$SDXL/unet/diffusion_pytorch_model.fp16.safetensors" \
  --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" \
  --source text_encoder="$SDXL/text_encoder/model.fp16.safetensors" \
  --source text_encoder_2="$SDXL/text_encoder_2/model.fp16.safetensors" \
  --config unet="$SDXL/unet/config.json" --config vae="$SDXL/vae/config.json" \
  --config text_encoder="$SDXL/text_encoder/config.json" \
  --config text_encoder_2="$SDXL/text_encoder_2/config.json"

SNAP=$("$TFS" ingest roots "$STORE" 2>/dev/null | true)
banner "REAL 4 — a candidate is NOT a checkpoint"
"$TFS" ingest roots "$STORE"

banner "REAL 5 — install: the coordinator's act, against the receipt"
# An ARM, not a bare command. This step printed `REFUSED SIZE_CAP` for a whole release and
# nothing counted it, because a bare line's refusal is invisible: the subject's candidate
# set is EVERY object the snapshot reaches — 2,601 for this tree — against a 64 KiB cap it
# shared with the fixed-size documents. A step whose failure is not an arm is not verified.
INS=$("$TFS" ingest install "$STORE" sdxl 2>&1)
if grep -q REFUSED <<<"$INS"; then
  MISS=$((MISS+1)); printf '  MISS %-42s %s\n' "install a 2,601-object candidate" "$(head -1 <<<"$INS")"
else
  PASS=$((PASS+1)); printf '  ok   %s\n' "install a 2,601-object candidate (the subject's real candidate set)"
fi
sed 's/^/    /' <<<"$INS"
"$TFS" ingest roots "$STORE"

CK=$("$TFS" ingest roots "$STORE" | grep installed | head -1 | awk '{print $2}' | cut -d: -f2)
banner "REAL 6 — the transitive walk verifies the installed result"
"$TFS" snapshot walk "$STORE" "$CK" | head -6

banner "REAL 7 — inherit by object reference: a re-ingest that hashes NOTHING"
HDR=$("$TFS" snapshot show "$STORE" "$CK" | grep attachment | grep -oE "[0-9a-f]{64}")
red CONVERTS_NOTHING "re-ingest with no change at all" \
    "$TFS" ingest reingest "$STORE" "$HDR" unet=plain/1,vae=plain/1,text_encoder=plain/1,text_encoder_2=plain/1 --session r0
"$TFS" ingest reingest "$STORE" "$HDR" unet=plain/1,vae=plain/1,text_encoder=plain/1,text_encoder_2=plain/1 \
  --rekey unet=down_blocks.:input_blocks. --session rekey

banner "REAL 8 — packaging convergence: the SAME bytes, two packagings, one identity"
python3 "$ROOT/scripts/repack-single-file.py" \
  "$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" "$WORK/vae.single_file.safetensors" \
  "first_stage_model."
"$TFS" ingest bank safetensors.single_file sdxl vae-single-file single_file.identity/1 \
  --registry "$REG" --source vae="$WORK/vae.single_file.safetensors" \
  --source-name repacked-from-the-same-bytes | tail -1
echo "  diffusers packaging, component topology:"
"$TFS" ingest plan vae=plain/1 --registry "$REG" \
  --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" | grep -E "^topology|^ops"
echo "  single-file packaging, component topology:"
"$TFS" ingest run "$WORK/conv" vae=plain/1 --registry "$REG" --session sf \
  --source vae="$WORK/vae.single_file.safetensors" 2>&1 | grep -E "^topology|^ops|bytes_deduped|bytes_hashed|^candidate"
echo "  and the same bytes through the diffusers packaging, into the SAME store:"
"$TFS" ingest run "$WORK/conv" vae=plain/1 --registry "$REG" --session df \
  --source vae="$SDXL/vae/diffusion_pytorch_model.fp16.safetensors" 2>&1 | grep -E "^topology|bytes_deduped|bytes_hashed|^candidate"

banner "REAL 8b — crash mid-ingest: kill -9 leaves no visible checkpoint"
BEFORE=$("$TFS" ingest roots "$STORE" | tail -1)
"$TFS" ingest run "$STORE" unet=plain/1 --registry "$REG" --session crashed \
  --source unet="$SDXL/unet/diffusion_pytorch_model.fp16.safetensors" >"$WORK/crash.log" 2>&1 &
RUNPID=$!
sleep 6
pkill -9 -f "ingest-worker" 2>/dev/null
kill -9 $RUNPID 2>/dev/null
wait $RUNPID 2>/dev/null
echo "  killed mid-stream (-9, no unwinding, no cleanup handler)"
echo "  before: $BEFORE"
echo "  after:  $("$TFS" ingest roots "$STORE" | tail -1)"
red ROOT_ABSENT "installing a session the crash never completed" \
    "$TFS" ingest install "$STORE" crashed
echo "  reaping the expiring roots:"
"$TFS" ingest reap "$STORE" --now | sed 's/^/    /'
red ROOT_ABSENT "installing after the root was reaped" "$TFS" ingest install "$STORE" rekey
echo "  installed checkpoints after the crash and the reap:"
"$TFS" ingest roots "$STORE" | sed 's/^/    /'

fi   # sdxl present
fi   # real

banner "SUMMARY"
echo "  $PASS arms fired as designed, $MISS missed"
[ "$MISS" -eq 0 ]
