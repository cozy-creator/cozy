#!/usr/bin/env bash
# tfs-009 live verification: the canonical Stamp, check_compatibility, hub grading, and the
# hermetic verifier's bounded document interface.
#
# Not a test suite. Every arm runs the real binary; every RED arm plants a real defect and
# prints the refusal it produced. The compatibility arms run against REAL pinned MiniMax-H3
# headers out of ~/cozy_v2/h3-evidence — 33 upstream artifacts, each pinned to a repo
# revision and a sha256 of the exact bytes read. Weights never transit this machine, so the
# logical facts are real and the part object ids are placeholders; `tfs compat import` says
# so on every run and nothing below the topology layer is read by the decision.
#
#   scripts/compat-runs.sh pairs      the four verdict grades, on real banked pairs
#   scripts/compat-runs.sh arms       the red arms
#   scripts/compat-runs.sh verifier   the hermetic seam, including partial output
#   scripts/compat-runs.sh stamp      Stamp through the real border
#   scripts/compat-runs.sh bench      verdict latency, canonical-bytes size, memory
#   scripts/compat-runs.sh all
set -uo pipefail
cd "$(dirname "$0")/.."
TFS=${TFS:-target/release/tfs}
W=${W:-/tmp/tfs-009-runs}
BANK=${BANK:-$HOME/cozy_v2/h3-evidence}
SECTION=${1:-all}
PASS=0; MISS=0

hr() { printf '\n===== %s =====\n' "$*"; }

# A GREEN arm expects one exact grade; a RED arm expects one exact refusal code. Anything
# else is the failure, silence most of all.
grade_is() {
  local want="$1" what="$2"; shift 2
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if grep -q "^grade        $want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %-46s %s\n' "$what" "$want"
    grep -E '^  (registered|target|source|reason) ' <<<"$out" | head -2 | cut -c1-160 | sed 's/^/     /'
  else
    MISS=$((MISS+1)); printf '  MISS %-46s expected %s\n' "$what" "$want"
    tail -4 <<<"$out" | sed 's/^/       /'
  fi
}

red() {
  local want="$1" what="$2"; shift 2
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if grep -q "REFUSED $want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  RED  %-46s %s\n' "$what" "$want"
    grep -o "REFUSED $want:.*" <<<"$out" | head -1 | cut -c1-165 | sed 's/^/       /'
  else
    MISS=$((MISS+1)); printf '  MISS %-46s expected %s\n' "$what" "$want"
    tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

green() {
  local what="$1"; shift
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if [ $? -eq 0 ] && ! grep -q REFUSED <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %s\n' "$what"
  else
    MISS=$((MISS+1)); printf '  MISS %-46s expected success\n' "$what"; tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

# ---------------------------------------------------------------- the real pair corpus

# name : banked artifact : optional unquantized twin supplying logical dtypes
PAIRS="
full-bf16:minimax_h3_fl2va_bf16:
ref2va-bf16:minimax_h3_ref2va_bf16:
curve-bf16:minimax_h3_fl2va_pruned_bf16:
curve-fp8:minimax_h3_fl2va_pruned_fp8_scaled:minimax_h3_fl2va_pruned_bf16
vae-f32:FL2VA/video_vae/source:
vae-f16:vae/minimax_h3_video_vae_fp16:
audio-native:FL2VA/audio_vae:
audio-diffusers:audio_vae/diffusion_pytorch_model:
native-shard1:FL2VA/transformer/model-00001:
diffusers-shard1:transformer/diffusion_pytorch_model-00001:
"

build_pairs() {
  mkdir -p "$W"
  for row in $PAIRS; do
    name="${row%%:*}"; rest="${row#*:}"; art="${rest%%:*}"; twin="${rest#*:}"
    [ -f "$W/$name.header.json" ] && continue
    if [ -n "$twin" ]; then
      nice -n 19 python3 scripts/h3-pairs.py fold "$art" "$W/$name.rows.json" "$twin" >/dev/null || exit 1
    else
      nice -n 19 python3 scripts/h3-pairs.py rows "$art" "$W/$name.rows.json" >/dev/null || exit 1
    fi
    nice -n 19 "$TFS" compat import "$W/$name.rows.json" \
      --out-header "$W/$name.header.json" --out-code "$W/$name.code.json" \
      --out-closure "$W/$name.closure" >"$W/$name.import" || exit 1
  done
}

cl() { ls "$W/$1.closure"/*.json 2>/dev/null | paste -sd, -; }
# The pin: exactly the spec digests the two sides' closures carry. A checkpoint whose spec
# the pin lacks is valid DATA and unsupported EXECUTION, which is its own verdict.
pin_of() { python3 -c "
import sys,pathlib,hashlib
print(','.join('sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
      for p in sorted(pathlib.Path(sys.argv[1]).glob('*.json'))))" "$W/$1.closure"; }

# chk <report-name> <code-of> <header-of> <lane> [extra flags...] — one run, one report.
chk() {
  local name="$1" c="$2" h="$3" lane="$4"; shift 4
  nice -n 19 "$TFS" compat check "$W/$c.code.json" "$W/$h.header.json" \
    --closure "$(cl "$h")$([ "$c" != "$h" ] && printf ,%s "$(cl "$c")")" \
    --pin "$(pin_of "$h"),$(pin_of "$c")" --lane "$lane" --out "$W/compat.$name.json" "$@" 2>&1
}

# verdict_cmd <want> <what> <command...> — for the arms that need their own flags.
verdict_cmd() {
  local want="$1" what="$2"; shift 2
  local out; out="$(nice -n 19 "$@" 2>&1)"
  if grep -q "^verdict      $want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %-46s %s\n' "$what" "$want"
    grep -E '^  (first|required|present) ' <<<"$out" | cut -c1-160 | sed 's/^/     /'
  else
    MISS=$((MISS+1)); printf '  MISS %-46s expected %s\n' "$what" "$want"
    tail -4 <<<"$out" | sed 's/^/       /'
  fi
}

# verdict_of <want> <what> <report-name> <code-of> <header-of> <lane> [flags...]
verdict_of() {
  local want="$1" what="$2"; shift 2
  local out; out="$(chk "$@")"
  if grep -q "^verdict      $want" <<<"$out"; then
    PASS=$((PASS+1)); printf '  ok   %-46s %s\n' "$what" "$want"
    grep -E '^  (first|required|present) ' <<<"$out" | cut -c1-160 | sed 's/^/     /'
  else
    MISS=$((MISS+1)); printf '  MISS %-46s expected %s\n' "$what" "$want"
    grep -E '^(verdict|topology|report)' <<<"$out" | sed 's/^/       /'
    tail -3 <<<"$out" | sed 's/^/       /'
  fi
}

pairs_section() {
  build_pairs
  hr "the real pair corpus, imported from pinned upstream headers"
  for row in $PAIRS; do
    name="${row%%:*}"
    printf '  %-18s %s\n' "$name" "$(grep -m1 '^imported' "$W/$name.import" | cut -c14-120)"
  done

  hr "COMPATIBLE — the fl2va/ref2va task partition (decisions #231), measured"
  verdict_of Satisfied "fl2va contract vs ref2va checkpoint" \
    ref2va full-bf16 ref2va-bf16 canonical
  grade_is COMPATIBLE "graded" "$TFS" grade "$W/compat.ref2va.json"

  hr "COMPATIBLE — the audio VAE is packaging-invariant (native vs Diffusers, 1087/1087)"
  verdict_of Satisfied "native contract vs diffusers checkpoint" \
    audio audio-native audio-diffusers canonical
  grade_is COMPATIBLE "graded" "$TFS" grade "$W/compat.audio.json"

  hr "CONVERTIBLE — the cast-only pair, and the v1 over-constraint recovery on real bytes"
  echo "  the official native video VAE is F32; ComfyUI ships the SAME 560 keys at F16."
  green "register f16 -> f32 over the COMPLETE 16-bit domain" \
    "$TFS" grade register f16-to-f32 widen f16:f32 --out "$W/reg.f16f32.json"
  nice -n 19 "$TFS" grade register f16-to-f32 widen f16:f32 --out "$W/reg.f16f32.json" \
    | grep -E '^round trip' | sed 's/^/     /'
  verdict_of "Mismatch DTYPE_MISMATCH" "F32 contract vs the F16 checkpoint" \
    vae vae-f32 vae-f16 local_unbordered
  grade_is CONVERTIBLE "graded, naming the registered conversion" \
    "$TFS" grade "$W/compat.vae.json" --registry "$W/reg.f16f32.json" \
      --code "$W/vae-f32.code.json" --header "$W/vae-f16.header.json" \
      --closure "$(cl vae-f16)" --pin "$(pin_of vae-f16)"
  echo "  and with NO registration the same pair is INCOMPATIBLE — not over-constraint:"
  echo "  with no registered lossless mapping there is no conversion to name, and inventing"
  echo "  one would be the fabrication the guard exists to prevent."
  grade_is INCOMPATIBLE "same pair, empty registry" "$TFS" grade "$W/compat.vae.json"

  hr "PRODUCIBLE — bf16 contract vs the fp8_scaled checkpoint (a priced quantization job)"
  verdict_of "Mismatch ENCODING_MISMATCH" "plain-bf16 contract vs fp8 checkpoint" \
    fp8 curve-bf16 curve-fp8 canonical
  grade_is PRODUCIBLE "graded" "$TFS" grade "$W/compat.fp8.json"
  echo "  the REVERSE direction is the remedy-from-target arm: an fp8 contract over a bf16"
  echo "  checkpoint must name the fp8 TARGET, never the plain/1 it observed."
  verdict_of "Mismatch ENCODING_MISMATCH" "fp8 contract vs plain-bf16 checkpoint" \
    bf16 curve-fp8 curve-bf16 canonical
  grade_is PRODUCIBLE "graded from the target" "$TFS" grade "$W/compat.bf16.json"

  hr "INCOMPATIBLE — full AdaLN contract vs the AdaLN-curve checkpoint (decisions #233)"
  echo "  measured: 4 keys only in full (time_embedder.*), 1 only in curve (adaln_t_table),"
  echo "  51 adaln_proj shapes changed [96768,2688] -> [96768,8], 102 dtypes bf16 -> f16."
  echo "  in sorted order the FIRST disagreement is the adaln dtype, and it is named:"
  verdict_of "Mismatch DTYPE_MISMATCH" "535-key contract vs the 532-key curve" \
    curve full-bf16 curve-bf16 canonical
  grade_is INCOMPATIBLE "graded" "$TFS" grade "$W/compat.curve.json"
  echo "  and the structural half, isolated to the four keys the curve simply does not have:"
  verdict_of "Mismatch MISSING_TENSOR" "the time_embedder contract vs the curve" \
    timeembed full-timeembed curve-bf16 canonical
  grade_is INCOMPATIBLE "graded — no mapping invents a tensor" "$TFS" grade "$W/compat.timeembed.json"

  hr "the te#185 anchor, reproduced: 66 GB avoided by a header-only verdict"
  echo "  DiffSynth accepts the NATIVE key set; every platform artifact was the DIFFUSERS"
  echo "  one; the two overlap in exactly one name. Found at load, after the download."
  verdict_of "Mismatch MISSING_TENSOR" "native contract vs diffusers checkpoint" \
    te185 native-shard1 diffusers-shard1 local_unbordered
  python3 -c "
import json,pathlib,sys
sys.path.insert(0,'scripts')
a=json.loads(pathlib.Path('$W/native-shard1.rows.json').read_text())
b=json.loads(pathlib.Path('$W/diffusers-shard1.rows.json').read_text())
ka={t['key'] for t in a['tensors']}; kb={t['key'] for t in b['tensors']}
print(f'     native {len(ka)} keys, diffusers {len(kb)} keys, shared {len(ka&kb)}: {sorted(ka&kb)}')"
  grade_is INCOMPATIBLE "graded pre-download" "$TFS" grade "$W/compat.te185.json"

  hr "the containment ladder: topology_digest is an INDEX, never the decision"
  chk ladder1 full-bf16 ref2va-bf16 canonical | grep -E '^topology' | sed 's/^/  /'
  chk ladder2 curve-bf16 curve-fp8 canonical | grep -E '^topology|^verdict' | sed 's/^/  /'
  echo "  the fp8 pair's topology digests are IDENTICAL — same logical dtypes and shapes —"
  echo "  and the verdict is still a mismatch, because encoding is below the index and"
  echo "  above the parts. An index hit is not a match."

  hr "scale rank is a PER-PRODUCER fact (decisions #244), read off the real headers"
  python3 - <<'PY'
import json, pathlib, os
BANK = pathlib.Path(os.path.expanduser("~/cozy_v2/h3-evidence"))
rows = json.loads((BANK / "rows/comfy-org-h3.json").read_text())
for f in rows["banked_files"]:
    if f.get("mode") != "st":
        continue
    name = f["path"].split("/")[-1]
    if not any(x in name for x in ("fp8_scaled", "int8_convrot", "nvfp4")):
        continue
    h = json.loads((BANK / "cache" / f["fetch"]["bytes_sha256"]).read_bytes())
    seen = {}
    for k, v in h.items():
        if k.endswith((".weight_scale", ".input_scale")):
            seen.setdefault((k.rsplit(".", 1)[1], len(v["shape"]), v["dtype"]), 0)
            seen[(k.rsplit(".", 1)[1], len(v["shape"]), v["dtype"])] += 1
    if seen:
        print(f"  {name[:52]:52} " + "  ".join(
            f"{r}=rank{n}/{d}x{c}" for (r, n, d), c in sorted(seen.items())))
PY
  echo "  a reader keyed on \"fp8 implies rank-2\" mis-reads the first row and the third."
  echo "  the registry carries one spec DIGEST per physical variant; the alias groups them."
  "$TFS" registry | grep -E 'fp8|nvfp4' | sed 's/^/  /'
}

# ---------------------------------------------------------------- red arms

arms_section() {
  build_pairs
  hr "direction asymmetry: missing always refuses, surplus depends on the lane"
  echo "  the ComfyUI video VAE carries two keys the official native VAE does not"
  echo "  (latents_mean, latents_std): a REAL surplus, not a planted one."
  verdict_of "Mismatch DTYPE_MISMATCH" "canonical: the dtype disagrees first" \
    surplus vae-f32 vae-f16 canonical
  verdict_of "Mismatch EXTRA_TENSOR" "canonical: the surplus itself refuses, named" \
    surplus2 vae-shared vae-f16 canonical
  echo "  the same surplus on a LOCAL UN-BORDERED snapshot is warn-ignore-confess:"
  chk unbordered vae-shared vae-f16 local_unbordered | grep -E '^unconsumed|^verdict' | sed 's/^/     /'
  red UNCLASSIFIED_EXTRAS "an unclassified-extras snapshot refuses at release cut" \
    "$TFS" compat release-cut "$W/compat.vae.json"
  green "a fully classified candidate cuts" "$TFS" compat release-cut "$W/compat.ref2va.json"

  hr "a lossy gap can NEVER register as a lossless conversion"
  red ROUND_TRIP_LOSSY "f32 -> f16 narrowing (the quantization class)" \
    "$TFS" grade register f32-to-f16 widen f32:f16 --corpus "$W/f32.corpus"
  red ROUND_TRIP_LOSSY "f32 -> bf16 narrowing" \
    "$TFS" grade register f32-to-bf16 widen f32:bf16 --corpus "$W/f32.corpus"
  red UNKNOWN_FIELD "a re-quantization cannot even be PROPOSED" \
    "$TFS" grade register bf16-to-fp8 widen bf16:f8_e4m3fn --corpus "$W/f32.corpus"
  red MISSING_FIELD "an empty corpus proves nothing" \
    "$TFS" grade register empty widen f16:f32 --corpus "$W/empty.corpus"

  hr "the code topology's closed vocabulary"
  red MISSING_FIELD "an empty accepts list" "$TFS" compat check "$W/bad-accepts.code.json" "$W/vae-f16.header.json"
  red UNKNOWN_FIELD "a third satisfaction verdict in a report" "$TFS" grade "$W/bad-verdict.report.json"
  red UNKNOWN_FIELD "a lane outside the closed set" "$TFS" grade "$W/bad-lane.report.json"
  red UNKNOWN_FIELD "a fifth grade in a decision" "$TFS" grade decision "$W/bad-grade.decision.json"

  hr "the pinned registry: an unaliased spec is DATA, not EXECUTION"
  verdict_cmd "Mismatch UNKNOWN_TO_LOCAL_CAPABILITY" "fp8 checkpoint against a plain-only pin" \
    "$TFS" compat check "$W/curve-fp8.code.json" "$W/curve-fp8.header.json" \
      --closure "$(cl curve-fp8)" --lane canonical \
      --pin sha256:88c2f114d68d7e30f0a003a0880a28ded359476ac3cddb7891b6ab7f056d9907

  hr "the registry gap refuses rather than rounding to a spec that fits nothing"
  red UNKNOWN_ENCODING "a measured role geometry no spec describes" \
    "$TFS" compat import "$W/unknown-geometry.rows.json"
}

# ---------------------------------------------------------------- the hermetic seam

verifier_section() {
  build_pairs
  hr "the bounded document interface: one file in, one file out"
  green "build the request" "$TFS" verify-request "$W/vae-f32.code.json" "$W/vae-f16.header.json" \
    --closure "$(cl vae-f16)" --registry "$W/reg.f16f32.json" --lane local_unbordered \
    --pin "$(pin_of vae-f16)" --out "$W/req.json"
  nice -n 19 "$TFS" verify-request "$W/vae-f32.code.json" "$W/vae-f16.header.json" \
    --closure "$(cl vae-f16)" --registry "$W/reg.f16f32.json" --lane local_unbordered \
    --pin "$(pin_of vae-f16)" --out "$W/req.json" | sed 's/^/  /'
  green "run the verifier" "$TFS" verify-compat "$W/req.json" "$W/resp.json"
  nice -n 19 "$TFS" verify-consume "$W/req.json" "$W/resp.json" | sed 's/^/  /'
  echo "  CONVERTIBLE here is PROVED, not asserted: the verifier applies the registration to"
  echo "  the code topology and re-checks; only a Satisfied re-check earns the grade."

  hr "partial output, a crash, and a replayed answer all mint nothing"
  rm -f "$W/resp-partial.json"
  nice -n 19 "$TFS" verify-compat "$W/req.json" "$W/resp-partial.json" --truncate-at 200 >/dev/null
  red VERIFIER_OUTPUT_INCOMPLETE "a truncated response" \
    "$TFS" verify-consume "$W/req.json" "$W/resp-partial.json"
  rm -f "$W/resp-absent.json"
  red VERIFIER_OUTPUT_INCOMPLETE "a crashed verifier leaves no response at all" \
    "$TFS" verify-consume "$W/req.json" "$W/resp-absent.json"
  green "build a DIFFERENT request" "$TFS" verify-request "$W/full-bf16.code.json" "$W/curve-bf16.header.json" \
    --closure "$(cl curve-bf16)" --pin "$(pin_of curve-bf16)" --out "$W/req2.json"
  red VERIFIER_OUTPUT_INCOMPLETE "the answer to another question" \
    "$TFS" verify-consume "$W/req2.json" "$W/resp.json"

  hr "nothing is fetched: the verifier runs with no store, no network, no writable path"
  echo "  linked network stacks in tensorfs-core:"
  nice -n 19 python3 scripts/fence.py | grep 'forbidden dependencies' | sed 's/^/  /'
  local sandbox="$W/sandbox"; rm -rf "$sandbox"; mkdir -p "$sandbox"
  cp "$W/req.json" "$sandbox/req.json"
  green "runs in an empty directory with no store present" \
    env -i PATH=/usr/bin:/bin "$PWD/$TFS" verify-compat "$sandbox/req.json" "$sandbox/resp.json"
  green "and its answer still consumes" "$TFS" verify-consume "$sandbox/req.json" "$sandbox/resp.json"
}

# ---------------------------------------------------------------- Stamp

stamp_section() {
  hr "Stamp through the real border: no receipt without a verified ledger"
  local S="$W/stamp"; rm -rf "$S"; mkdir -p "$S/carriers"
  python3 scripts/make-carrier.py good "$S/carriers/good.safetensors"
  nice -n 19 "$TFS" store init "$S/store" >/dev/null
  nice -n 19 "$TFS" ingest bank safetensors.diffusers toy v1 diffusers.identity/1 \
    --source toy="$S/carriers/good.safetensors" --registry "$S/registry.json" \
    --source-name synthetic-toy-carrier >/dev/null
  green "a clean run stamps and mints" \
    "$TFS" ingest run "$S/store" toy=plain/1 --source toy="$S/carriers/good.safetensors" \
      --registry "$S/registry.json" --session s1 --tenant acme
  nice -n 19 "$TFS" ingest run "$S/store" toy=plain/1 --source toy="$S/carriers/good.safetensors" \
    --registry "$S/registry.json" --session s2 --tenant acme 2>&1 \
    | grep -E '^(stamp|  topology|  candidates|  golden|receipt)' | sed 's/^/  /'
  green "install consumes the stamp AND the receipt" "$TFS" ingest install "$S/store" s2

  hr "restamp: the coordinator's re-verification, byte for byte"
  green "restamping the clean candidate reproduces the exact receipt" \
    "$TFS" ingest restamp "$S/store" s1
  nice -n 19 "$TFS" ingest restamp "$S/store" s1 | sed 's/^/  /'

  hr "Stamp's red arms — one planted fact each, refused by the fact it is protected by"
  # The plants live in the BINARY (beside tfs-002's --fault kill points), never in the
  # library: each corrupts exactly one presented fact and nothing else.
  red TOPOLOGY_MISMATCH "the profile pins a topology the header does not project" \
    "$TFS" ingest restamp "$S/store" s1 --plant topology
  red STAMP_MISMATCH "one candidate dropped from the claimed set" \
    "$TFS" ingest restamp "$S/store" s1 --plant candidate
  red STAMP_MISMATCH "one candidate claimed that nothing reaches" \
    "$TFS" ingest restamp "$S/store" s1 --plant candidate-surplus
  red STAMP_MISMATCH "the profile altered without re-binding the subject" \
    "$TFS" ingest restamp "$S/store" s1 --plant profile-bytes
  red STAMP_MISMATCH "one tensor's logical shape moved (structure)" \
    "$TFS" ingest restamp "$S/store" s1 --plant structure
  red UNREGISTERED_ENCODING "a cited spec outside the profile's allowed set" \
    "$TFS" ingest restamp "$S/store" s1 --plant encoding
  red QUOTA_EXHAUSTED "more tensors than the profile's limit" \
    "$TFS" ingest restamp "$S/store" s1 --plant quota-tensors
  red QUOTA_EXHAUSTED "more bytes moved than the profile's limit" \
    "$TFS" ingest restamp "$S/store" s1 --plant quota-bytes
  red STAMP_MISMATCH "an evidence recipe that is not the profile's converter" \
    "$TFS" ingest restamp "$S/store" s1 --plant converter
  red EVIDENCE_NOT_RESIDENT "a header-invisible source claim, restated as evidence" \
    "$TFS" ingest restamp "$S/store" s1 --plant evidence-metadata
  red GOLDEN_MISMATCH "a golden suite that did not pass" \
    "$TFS" ingest restamp "$S/store" s1 --golden fail
  red GOLDEN_MISMATCH "\"absent\" is not \"pass\"" \
    "$TFS" ingest restamp "$S/store" s1 --plant golden-absent
  red UNKNOWN_FIELD "a golden verdict outside the closed set" \
    "$TFS" ingest restamp "$S/store" s1 --golden probably

  hr "the subject is the identity: generation and tenant ride the receipt binding"
  # Stamp records source generations and tenant; it cannot check them against an outside
  # truth and does not pretend to. What it does is BIND them, so altering either moves the
  # subject digest and the receipt that named the old one stops binding. Both edits below
  # are length-preserving, so the documents stay canonical and only the identity moves.
  local Q="$S/store/ingest/s1"
  cp "$Q/subject.json" "$Q/subject.bak"
  python3 -c "
import pathlib
p=pathlib.Path('$Q/subject.json')
p.write_bytes(p.read_bytes().replace(b'\"generation\":1', b'\"generation\":2'))"
  red CROSS_SUBJECT_REPLAY "a source generation bumped under the same receipt" \
    "$TFS" ingest install "$S/store" s1
  python3 -c "
import pathlib
p=pathlib.Path('$Q/subject.bak')
pathlib.Path('$Q/subject.json').write_bytes(p.read_bytes().replace(b'\"tenant\":\"acme\"', b'\"tenant\":\"acmf\"'))"
  red CROSS_SUBJECT_REPLAY "another tenant on the same candidate (receipt replay)" \
    "$TFS" ingest install "$S/store" s1
  cp "$Q/subject.bak" "$Q/subject.json"; rm -f "$Q/subject.bak"

  hr "a receipt is not a publication, and install binds the stamp"
  cp "$Q/stamp.json" "$Q/stamp.bak"
  python3 -c "
import pathlib
p=pathlib.Path('$Q/stamp.json')
p.write_bytes(p.read_bytes().replace(b'\"isolation_tier\":1', b'\"isolation_tier\":2'))"
  red CROSS_SUBJECT_REPLAY "install with a receipt bound to another stamp" \
    "$TFS" ingest install "$S/store" s1
  cp "$Q/stamp.bak" "$Q/stamp.json"; rm -f "$Q/stamp.bak"
  red ROOT_ABSENT "install with the stamp document removed" \
    bash -c "mv $Q/stamp.json $Q/stamp.hidden; $TFS ingest install $S/store s1; mv $Q/stamp.hidden $Q/stamp.json"
  green "and with everything intact it installs" "$TFS" ingest install "$S/store" s1
}

bench_section() {
  build_pairs
  hr "benchmarks — verdict latency, canonical-bytes size, memory"
  printf '  %-34s %8s %10s %12s %10s\n' pair tensors 'report B' 'verdict' 'peak RSS'
  for p in "ref2va-bf16 full-bf16" "curve-fp8 curve-bf16" "vae-f16 vae-f32" "curve-bf16 full-bf16"; do
    set -- $p
    local out; out="$(chk bench "$2" "$1" canonical 2>&1)"
    local n; n=$(python3 -c "import json,pathlib;print(len(json.loads(pathlib.Path('$W/$1.rows.json').read_text())['tensors']))")
    printf '  %-34s %8s %10s %12s %10s\n' "$2 vs $1" "$n" \
      "$(stat -c%s "$W/compat.bench.json")" \
      "$(grep -oE 'in [0-9]+ us' <<<"$out" | head -1)" \
      "$(grep -oE 'peak RSS [0-9.]+ MiB' <<<"$out" | head -1 | cut -d' ' -f3-)"
  done
  echo
  printf '  %-34s %10s %12s\n' document bytes id
  for d in full-bf16.header full-bf16.code compat.curve; do
    [ -f "$W/$d.json" ] && printf '  %-34s %10s\n' "$d" "$(stat -c%s "$W/$d.json")"
  done
  hr "the registration act, timed"
  nice -n 19 "$TFS" grade register bf16-to-f32 widen bf16:f32 --out "$W/reg.bf16f32.json" | sed 's/^/  /'
  nice -n 19 "$TFS" grade register f16-to-f32 widen f16:f32 --out "$W/reg.f16f32.json" | sed 's/^/  /'
}

fixtures() {
  mkdir -p "$W"
  head -c 4096 /dev/urandom > "$W/f32.corpus"
  : > "$W/empty.corpus"
  build_pairs
  # The 560 keys the two video VAEs actually share, at the F16 the ComfyUI one carries —
  # so the FIRST disagreement is the two-key surplus rather than a dtype.
  if [ ! -f "$W/vae-shared.code.json" ]; then
    python3 -c "
import json,pathlib
d=json.loads(pathlib.Path('$W/vae-f16.rows.json').read_text())
d['tensors']=[t for t in d['tensors'] if t['key'] not in ('latents_mean','latents_std')]
d['provenance']=d['provenance']+' minus the two ComfyUI-only keys'
pathlib.Path('$W/vae-shared.rows.json').write_text(json.dumps(d,sort_keys=True,separators=(',',':')))"
    nice -n 19 "$TFS" compat import "$W/vae-shared.rows.json" \
      --out-header "$W/vae-shared.header.json" --out-code "$W/vae-shared.code.json" \
      --out-closure "$W/vae-shared.closure" >"$W/vae-shared.import"
  fi
  # The four keys the AdaLN-curve topology does not carry at all.
  if [ ! -f "$W/full-timeembed.code.json" ]; then
    python3 -c "
import json,pathlib
d=json.loads(pathlib.Path('$W/full-bf16.rows.json').read_text())
d['tensors']=[t for t in d['tensors'] if t['key'].startswith('time_embedder.')]
d['provenance']=d['provenance']+' time_embedder only'
pathlib.Path('$W/full-timeembed.rows.json').write_text(json.dumps(d,sort_keys=True,separators=(',',':')))"
    nice -n 19 "$TFS" compat import "$W/full-timeembed.rows.json" \
      --out-header "$W/full-timeembed.header.json" --out-code "$W/full-timeembed.code.json" \
      --out-closure "$W/full-timeembed.closure" >"$W/full-timeembed.import"
  fi
  # A code topology whose accepts list is empty: admits nothing, says nothing.
  python3 -c "
import pathlib
p=pathlib.Path('$W/vae-f32.code.json'); b=p.read_bytes()
i=b.index(b'\"accepts\":[')
j=b.index(b']', i)
pathlib.Path('$W/bad-accepts.code.json').write_bytes(b[:i]+b'\"accepts\":[]'+b[j+1:])"
  # A report claiming a third satisfaction verdict, and a decision claiming a fifth grade.
  python3 -c "
import pathlib
p=pathlib.Path('$W/compat.curve.json')
b=p.read_bytes()
pathlib.Path('$W/bad-verdict.report.json').write_bytes(b.replace(b'\"t\":\"mismatch\"', b'\"t\":\"derivable\"'))
pathlib.Path('$W/bad-lane.report.json').write_bytes(b.replace(b'\"lane\":\"canonical\"', b'\"lane\":\"whenever\"'))"
  nice -n 19 "$TFS" grade "$W/compat.curve.json" --out "$W/good.decision.json" >/dev/null
  python3 -c "
import pathlib
b=pathlib.Path('$W/good.decision.json').read_bytes()
pathlib.Path('$W/bad-grade.decision.json').write_bytes(b.replace(b'\"t\":\"INCOMPATIBLE\"', b'\"t\":\"PROBABLY\"'))"
  # A measured role geometry the registry does not describe.
  python3 -c "
import json,pathlib
pathlib.Path('$W/unknown-geometry.rows.json').write_text(json.dumps({
 'component':'transformer','provenance':'planted: a role set no platform spec carries',
 'tensors':[{'key':'blocks.0.attn.qkv_proj.weight','dtype':'bf16','shape':[5376,7168],
             'roles':[{'role':'data','dtype':'f8_e4m3fn','shape':[5376,7168]},
                      {'role':'zero_point','dtype':'i8','shape':[5376,1]}]}]},
 sort_keys=True,separators=(',',':')))"
}

case "$SECTION" in
  pairs)    fixtures; pairs_section ;;
  arms)     fixtures; pairs_section >/dev/null; fixtures; arms_section ;;
  verifier) fixtures; pairs_section >/dev/null; verifier_section ;;
  stamp)    fixtures; stamp_section ;;
  bench)    fixtures; bench_section ;;
  # `fixtures` runs TWICE on purpose: three planted defects are mutations of reports that
  # `pairs_section` produces, so on a cold work directory they have nothing to mutate yet.
  # An arm that only fires on a warm run is not an arm.
  all)      fixtures; pairs_section; fixtures; arms_section; verifier_section; stamp_section; bench_section ;;
  *) echo "sections: pairs | arms | verifier | stamp | bench | all"; exit 2 ;;
esac

printf '\n%d arms fired, %d missed\n' "$PASS" "$MISS"
[ "$MISS" -eq 0 ]
