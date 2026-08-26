#!/usr/bin/env python3
"""Reconcile the pinned H3 evidence bank's own verdicts against what tfs-009's admission
path says about the same bytes (tfs-010, M3 stamp-reconciliation arm).

Two independent routes to one answer:

  BANKED    computed from the bank's RECORDED facts only — the per-file `key_dialect`
            digest, `logical_tensor_count`, `marker_sibling_counts` and `dtype_key_counts`
            that `h3-evidence/scripts/` measured when the row was banked. No header byte is
            re-read here, so a stale or mutated banked fact changes this column and nothing
            else.
  ADMISSION observed by RUNNING the real binary — `tfs compat import` selects a platform
            spec per tensor from the measured role geometry, `tfs compat check` returns
            `Satisfied | Mismatch(details)`, `tfs grade` grades it, and `tfs ingest inspect`
            answers for carriers the normal door never admits.

The third column is the reconciliation: AGREE, or DISAGREE with the code for WHY. Every
relationship is RECORDED, so this is a fence and not a report — if what the bank says and
what admission says stop standing in the recorded relationship, the run goes red. That is
what the red arm exercises: point `--bank` at a COPY with one verdict mutated and the row
that depended on it flips.

  scripts/bank-reconcile.py                 the table
  scripts/bank-reconcile.py --bank DIR      run against another copy of the bank (red arm)
  scripts/bank-reconcile.py --verbose       print each admission command's own output

No weight byte transits this machine: every artifact below is a bounded header, metadata
region or zip directory the bank fetched by range and pinned by digest.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TFS = pathlib.Path(os.environ.get("TFS") or ROOT / "target/release/tfs")
BANK = pathlib.Path(os.environ.get("H3_BANK") or pathlib.Path.home() / "cozy_v2/tracker-v2/h3-evidence")
W = pathlib.Path(os.environ.get("W") or "/tmp/tfs-010-reconcile")
VERBOSE = False


# ---------------------------------------------------------------- the BANKED column

def banked_facts():
    """Every banked artifact's RECORDED facts. Nothing is recomputed from header bytes."""
    out = {}
    for r in sorted((BANK / "rows").glob("*.json")):
        d = json.loads(r.read_text())
        for f in d.get("banked_files", []):
            s = f.get("summary") or {}
            out[f'{d["row_id"]}/{f["path"]}'] = {
                "row": d["row_id"], "path": f["path"],
                "grade": d.get("evidence_grade"),
                "mode": f.get("mode"),
                "sha256": f["fetch"]["bytes_sha256"],
                "dialect": (s.get("key_dialect") or {}).get("dialect_digest"),
                "logical": s.get("logical_tensor_count", s.get("key_count")),
                "siblings": s.get("marker_sibling_counts") or {},
                "dtypes": s.get("dtype_key_counts") or {},
            }
    retired = BANK / "retired.json"
    if retired.exists():
        d = json.loads(retired.read_text())
        for name, rec in sorted(d.get("files", {}).items()):
            out[f"retired/{name}"] = {
                "row": "tracker-committed-header", "path": f"retired/{name}", "grade": "retired-wrong", "mode": "st",
                "sha256": rec["bytes_sha256"], "dialect": rec.get("dialect_digest"),
                "logical": rec.get("logical"), "siblings": {}, "dtypes": {},
            }
    return out


def resolve(facts, needle, modes=("st",)):
    m = [p for p in facts if needle in p and facts[p]["mode"] in modes]
    if len(m) != 1:
        sys.exit(f"{needle!r} matches {len(m)} banked {'/'.join(modes)} artifacts: {m}")
    return m[0]


def banked_relation(a, b):
    """The bank's own verdict about two artifacts, from its recorded facts alone.

    The dialect digest is the bank's identity for a key convention (sha256 of the template
    key set with block indices erased); the sibling table is what it measured hanging off
    those keys; the dtype histogram is what it measured them stored as. Those three facts
    are the whole basis, deliberately — the point of the exercise is to find out where a
    verdict formed from them parts company with a verdict formed by running admission.
    """
    if a["dialect"] != b["dialect"]:
        return "DIFFERENT_TOPOLOGY"
    if b["siblings"] and not a["siblings"]:
        return "ENCODED_TWIN"
    if a["siblings"] or b["siblings"]:
        return "BOTH_ENCODED"
    if a["dtypes"] != b["dtypes"]:
        return "CAST_ONLY"
    return "SAME"


# ---------------------------------------------------------------- the ADMISSION column

def run(*args):
    p = subprocess.run(["nice", "-n", "19", *[str(a) for a in args]],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    if VERBOSE:
        print(f"    $ {' '.join(str(a) for a in args)}\n{out.rstrip()}")
    return p.returncode, out


def refusal(out):
    for line in out.splitlines():
        if line.startswith("REFUSED "):
            return line.split()[1].rstrip(":")
    return None


def build(name, artifact, twin=None):
    """Transcribe one banked artifact and import it. Returns (header, code, closure) or a
    refusal code — an artifact the registry cannot describe never reaches a verdict, and
    that IS its admission answer."""
    rows = W / f"{name}.rows.json"
    env = dict(os.environ, H3_BANK=str(BANK))
    cmd = ["nice", "-n", "19", sys.executable, str(ROOT / "scripts/h3-pairs.py"),
           "fold" if twin else "rows", artifact, str(rows)] + ([twin] if twin else [])
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return f"TRANSCRIBE_FAILED {p.stdout.strip() or p.stderr.strip()}"[:90], None
    code, out = run(TFS, "compat", "import", rows,
                    "--out-header", W / f"{name}.header.json",
                    "--out-code", W / f"{name}.code.json",
                    "--out-closure", W / f"{name}.closure")
    if code != 0:
        r = refusal(out) or "REFUSED"
        key = ""
        for line in out.splitlines():
            if line.startswith("REFUSED "):
                key = line.split(":")[1].strip()
        return f"{r} {key}", None
    return None, name


def closure_of(name):
    return sorted((W / f"{name}.closure").glob("*.json"))


def pin_of(name):
    return ",".join("sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in closure_of(name))


def check(case, contract, checkpoint):
    """One `check_compatibility` run, then one hub grade over its report."""
    cl = closure_of(checkpoint) + (closure_of(contract) if contract != checkpoint else [])
    report = W / f"report.{case}.json"
    pins = pin_of(checkpoint) + ("," + pin_of(contract) if contract != checkpoint else "")
    _, out = run(TFS, "compat", "check", W / f"{contract}.code.json",
                 W / f"{checkpoint}.header.json",
                 "--closure", ",".join(str(p) for p in cl), "--pin", pins,
                 "--lane", CASES_LANE.get(case, "canonical"), "--out", report)
    verdict = next((l.split(None, 1)[1].strip() for l in out.splitlines()
                    if l.startswith("verdict ")), refusal(out) or "NO_VERDICT")
    return verdict, report


def grade(report, registry=None, contract=None, checkpoint=None):
    args = [TFS, "grade", report]
    if registry:
        args += ["--registry", registry, "--code", W / f"{contract}.code.json",
                 "--header", W / f"{checkpoint}.header.json",
                 "--closure", ",".join(str(p) for p in closure_of(checkpoint)),
                 "--pin", pin_of(checkpoint)]
    _, out = run(*args)
    return next((l.split(None, 1)[1].strip() for l in out.splitlines()
                 if l.startswith("grade ")), refusal(out) or "NO_GRADE")


# ---------------------------------------------------------------- the cases

# Every banked source family that has enough facts to reach a verdict, plus the families
# that do not — named, with the reason, rather than quietly dropped.
PAIRS = [
    dict(case="native-partition", banked="SAME", row="minimax-native-*", lane="canonical",
         contract="FL2VA/transformer/model-00001", checkpoint="Ref2VA/transformer/model-00001",
         expect="Satisfied / COMPATIBLE", rel="AGREE",
         why="one topology; the fl2va/ref2va partition is reviewed DATA, not bytes (#231)"),
    dict(case="diffusers-ref", banked="SAME", row="minimax-diffusers-tree", lane="canonical",
         contract="transformer/diffusion_pytorch_model-00001",
         checkpoint="transformer_ref/diffusion_pytorch_model-00001",
         expect="Satisfied / COMPATIBLE", rel="AGREE",
         why="byte-identical configs and indexes; nothing inside either says which partition"),
    dict(case="comfy-partition", banked="SAME", row="comfy-org-h3", lane="canonical",
         contract="minimax_h3_fl2va_bf16", checkpoint="minimax_h3_ref2va_bf16",
         expect="Satisfied / COMPATIBLE", rel="AGREE",
         why="535/535, 0 dtype and 0 shape differences"),
    dict(case="full-vs-curve", banked="DIFFERENT_TOPOLOGY", row="comfy-org-h3", lane="canonical",
         contract="minimax_h3_fl2va_bf16", checkpoint="minimax_h3_fl2va_pruned_bf16",
         expect="Mismatch DTYPE_MISMATCH / INCOMPATIBLE", rel="AGREE",
         why="the AdaLN curve is a different topology and both routes say so"),
    dict(case="curve-fp8", banked="ENCODED_TWIN", row="comfy-org-h3", lane="canonical",
         contract="minimax_h3_fl2va_pruned_bf16",
         checkpoint="minimax_h3_fl2va_pruned_fp8_scaled",
         twin="minimax_h3_fl2va_pruned_bf16",
         expect="Mismatch ENCODING_MISMATCH / PRODUCIBLE", rel="AGREE",
         why="same topology, encoded carrier: a priced job from the named bf16 source"),
    dict(case="dit-int8", banked="ENCODED_TWIN", row="comfy-org-h3", lane="canonical",
         contract="minimax_h3_fl2va_bf16", checkpoint="minimax_h3_fl2va_int8_convrot",
         twin="minimax_h3_fl2va_bf16",
         expect="UNKNOWN_ENCODING blocks.0.adaln_proj.linear.weight", rel="DISAGREE:REGISTRY_GAP",
         why="bank grades it header-verified; admission has no int8 spec at all, so the "
             "artifact is storable evidence and unadmittable execution"),
    dict(case="te-nvfp4", banked="DIFFERENT_TOPOLOGY", row="comfy-org-h3", lane="canonical",
         contract="qwen3vl_32b_minimax_h3_bf16", checkpoint="qwen3vl_32b_minimax_h3_nvfp4_awq",
         twin="qwen3vl_32b_minimax_h3_bf16",
         expect="UNKNOWN_ENCODING model.embed_tokens.weight", rel="DISAGREE:UNDER_INFORMED",
         why="the bank's ninth dialect (1352 keys) counts nvfp4 role siblings as tensors; "
             "folded it is the TE's own 902 keys, 250 of them the registry's mined nvfp4"),
    dict(case="te-int8", banked="ENCODED_TWIN", row="comfy-org-h3", lane="canonical",
         contract="qwen3vl_32b_minimax_h3_bf16",
         checkpoint="qwen3vl_32b_minimax_h3_int8_convrot",
         twin="qwen3vl_32b_minimax_h3_bf16",
         expect="UNKNOWN_ENCODING model.layers.0.mlp.down_proj.weight",
         rel="DISAGREE:REGISTRY_GAP",
         why="same int8 rank-2 [out,1] contract as the DiT's, on the text encoder"),
    dict(case="nf4-dit", banked="ENCODED_TWIN", row="diffsynth-nf4", lane="canonical",
         contract="minimax_h3_fl2va_bf16", checkpoint="minimax-h3-fl2va-nf4",
         twin="minimax_h3_fl2va_bf16",
         expect="UNKNOWN_ENCODING blocks.0.adaln_proj.linear.weight",
         rel="DISAGREE:READER_KILLED",
         why="the bank's own facts read as an encoded twin of the 535-key topology, and "
             "they are; bnb NF4 is not a gap awaiting a spec — the reader is KILLED by "
             "ruling (tensorfs.md q4), remedy is the fp16 base plus our quantize lanes"),
    dict(case="nf4-audio-vae", banked="ENCODED_TWIN", row="diffsynth-nf4", lane="canonical",
         contract="FL2VA/audio_vae", checkpoint="audio_vae_nf4",
         twin="FL2VA/audio_vae",
         expect="UNKNOWN_ENCODING pre_block.attn.proj.weight", rel="DISAGREE:READER_KILLED",
         why="the sharpest form of it: 1081 of 1087 tensors are untouched and SIX carry "
             "bnb state, and six tensors are enough to refuse the whole component"),
    dict(case="video-vae-cast", banked="DIFFERENT_TOPOLOGY", row="minimax-native-fl2va", lane="local_unbordered",
         contract="FL2VA/video_vae/source", checkpoint="minimax_h3_video_vae_fp16",
         registry=("f16-to-f32", "widen", "f16:f32"),
         expect="Mismatch DTYPE_MISMATCH / CONVERTIBLE", rel="DISAGREE:LANE_TIERS_EXTRAS",
         why="the bank reads two surplus keys as a different dialect; admission tiers "
             "surplus BY LANE and the shared 560 are a registered f16->f32 widening"),
    dict(case="audio-vae-packaging", banked="SAME", row="minimax-native-fl2va", lane="canonical",
         contract="FL2VA/audio_vae", checkpoint="audio_vae/diffusion_pytorch_model",
         expect="Satisfied / COMPATIBLE", rel="AGREE",
         why="1087/1087: the audio VAE is packaging-invariant across native and Diffusers"),
    dict(case="audio-vae-comfy", banked="DIFFERENT_TOPOLOGY", row="comfy-org-h3", lane="canonical",
         contract="FL2VA/audio_vae", checkpoint="minimax_h3_audio_vae_fp32",
         expect="Mismatch MISSING_TENSOR / INCOMPATIBLE", rel="AGREE",
         why="Comfy's 917-key audio VAE is a different artifact, not a repackaging"),
    dict(case="te185-anchor", banked="DIFFERENT_TOPOLOGY", row="failure-corpus", lane="local_unbordered",
         contract="FL2VA/transformer/model-00001",
         checkpoint="transformer/diffusion_pytorch_model-00001",
         expect="Mismatch MISSING_TENSOR / INCOMPATIBLE", rel="AGREE",
         why="the corpus anchor: 54 vs 64 keys overlapping in one name, decided from headers"),
    dict(case="retired-header", banked="DIFFERENT_TOPOLOGY", row="tracker-committed-header", lane="canonical",
         contract="minimax_h3_fl2va_pruned_bf16",
         checkpoint="retired/minimax_h3_fl2va_fp8_scaled.header.json",
         expect="Mismatch MISSING_TENSOR / INCOMPATIBLE", rel="AGREE",
         why="the fabricated header decisions #243 retired: admission refuses it against "
             "the artifact it claimed to be"),
]

CASES_LANE = {p["case"]: p["lane"] for p in PAIRS}

CARRIERS = [
    dict(case="gguf-abiray", banked="grade=deferred", row="gguf-abiray",
         artifact="gguf-abiray/unet/MiniMax-H3-FL2VA-Q8_0.gguf",
         expect="GGUF_REFUSED", rel="AGREE",
         why="bank grade is DEFERRED and the normal door refuses GGUF outright (tfs-011)"),
    dict(case="gguf-curve", banked="grade=deferred", row="gguf-curve",
         artifact="gguf-curve/fl2va/MiniMax-H3-fl2va-curve-Q8_0.gguf",
         expect="GGUF_REFUSED", rel="AGREE",
         why="same door, curve topology; the deferred grade is admission's answer too"),
    dict(case="torchao-fp8", banked="grade=format-evidence", row="torchao-pickle",
         artifact="torchao-pickle/pytorch_model.bin", expect="ARCHIVE_REFUSED", rel="AGREE",
         why="the unsafe-pickle row's object state never reaches a decode"),
    dict(case="torchao-int8", banked="grade=format-evidence", row="torchao-pickle-int8",
         artifact="torchao-pickle-int8/pytorch_model.bin",
         expect="ARCHIVE_REFUSED", rel="AGREE", why="same door, int8 publisher"),
    dict(case="torchao-diffusion", banked="grade=format-evidence", row="torchao-pickle-diffusion",
         artifact="diffusion/transformer/diffusion_pytorch_model-00002-of-00002.bin",
         expect="ARCHIVE_REFUSED", rel="AGREE",
         why="a real diffusion torchao pickle; its own bf16 VAE is the control"),
]

# Banked verdicts that CANNOT be reconciled, each with the reason. An unreconcilable row is
# a fact about the bank, not an omission.
INSUFFICIENT = [
    ("h3-adaln-baked", "adaln-topologies",
     "ABSENT — no baked-table artifact exists anywhere, so there is nothing to admit"),
    ("official 535/638/1058 full key sets", "minimax-native-*/diffusers-tree",
     "the bank holds the sharded INDEX (key names only); a verdict needs dtypes and shapes, "
     "so only shard 1 of each reaches admission"),
    ("turbo LoRA (624 keys)", "comfy-org-h3",
     "an adapter is not a checkpoint contract — the A/B target-binding plane is tfs-012"),
    ("84.90 GB coherent candidate", "reconciliation",
     "BLOCKED_ARTIFACT_DOES_NOT_EXIST — it was never produced, so no header exists to admit"),
    ("Stamp's byte-resident facts", "all rows",
     "Stamp verifies candidate objects, quotas and evidence residency against bytes in a "
     "store; the bank holds headers only, so the banked half of admission is the topology "
     "and encoding half — Stamp's own 21 arms run on real border runs in compat-runs.sh"),
]


def main():
    global VERBOSE, BANK
    args = sys.argv[1:]
    if "--verbose" in args:
        VERBOSE = True
        args.remove("--verbose")
    if "--bank" in args:
        i = args.index("--bank")
        BANK = pathlib.Path(args[i + 1])
    if not TFS.exists():
        sys.exit(f"no binary at {TFS} — cargo build --release first")
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    facts = banked_facts()
    print(f"bank {BANK}  ({len(facts)} banked files)\nbinary {TFS}\n")

    rows, bad = [], 0
    for p in PAIRS:
        a, b = resolve(facts, p["contract"]), resolve(facts, p["checkpoint"])
        bankv = banked_relation(facts[a], facts[b])
        cname, kname = p["case"] + ".c", p["case"] + ".k"
        twin = p.get("twin")
        r1, _ = build(cname, p["contract"])
        r2, _ = build(kname, p["checkpoint"], twin)
        if r1 or r2:
            adm = r1 or r2
        else:
            verdict, report = check(p["case"], cname, kname)
            reg = None
            if p.get("registry"):
                reg = W / f"reg.{p['case']}.json"
                run(TFS, "grade", "register", *p["registry"], "--out", reg)
            g = grade(report, reg, cname, kname)
            adm = f"{verdict} / {g}"
        ok = adm == p["expect"] and bankv == p["banked"]
        bad += not ok
        survey = []
        if adm.startswith("UNKNOWN_ENCODING"):
            # The refusal stops at the FIRST gap; the survey is what the whole artifact
            # holds. An artifact mixes encodings per tensor, so a single named key is the
            # start of the answer and never the whole of it.
            _, so = run(TFS, "compat", "import", W / f"{kname}.rows.json", "--survey")
            survey = [l for l in so.splitlines() if l.startswith("  ")]
        rows.append((p["case"], p["row"], bankv, adm, p["rel"] if ok else "STALE", p["why"],
                     ok, f'{p["banked"]} / {p["expect"]}', survey))

    for c in CARRIERS:
        f = facts[resolve(facts, c["artifact"], ("gguf", "pickle"))]
        src = W / f"{c['case']}.carrier"
        shutil.copy(BANK / "cache" / f["sha256"], src)
        _, out = run(TFS, "ingest", "inspect", f"--source", f"c={src}",
                     "--registry", W / "empty-registry.json")
        codes = [l.split()[1].rstrip(":") for l in out.splitlines() if l.startswith("REFUSED ")]
        adm = c["expect"] if c["expect"] in codes else (codes[-1] if codes else "ADMITTED")
        ok = adm == c["expect"] and f"grade={f['grade']}" == c["banked"]
        bad += not ok
        rows.append((c["case"], c["row"], f"grade={f['grade']}", adm,
                     c["rel"] if ok else "STALE", c["why"], ok,
                     f'{c["banked"]} / {c["expect"]}', []))

    w = max(len(r[0]) for r in rows)
    print(f"{'case':<{w}}  {'BANKED':<19} {'ADMISSION':<47} RECONCILED")
    print("-" * (w + 92))
    for case, row, bankv, adm, rel, why, ok, expect, survey in rows:
        print(f"{case:<{w}}  {bankv:<19} {adm:<47} {rel}")
        print(f"{'':<{w}}  {'':<19} {why}")
        for line in survey:
            print(f"{'':<{w}}  {'':<19}{line}")
        if not ok:
            print(f"{'':<{w}}  STALE — recorded {expect!r}, "
                  f"observed {bankv + ' / ' + adm!r}")
    print()
    print("NOT RECONCILABLE — banked verdicts with insufficient facts, named not dropped")
    for what, row, reason in INSUFFICIENT:
        print(f"  {what:<38} [{row}]\n      {reason}")
    agree = sum(1 for r in rows if r[6] and r[4] == "AGREE")
    dis = sum(1 for r in rows if r[6] and r[4].startswith("DISAGREE"))
    print(f"\n{len(rows)} reconciled: {agree} agree, {dis} recorded disagreements, "
          f"{bad} STALE; {len(INSUFFICIENT)} not reconcilable")
    return 1 if bad else 0


sys.exit(main())
