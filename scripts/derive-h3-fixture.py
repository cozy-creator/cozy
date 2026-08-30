#!/usr/bin/env python3
"""Transcribe a REAL shipped H3 DiT header into structure-fixture rows.

This is a TRANSCRIPTION of evidence, not a parser and not a codec: it reads one committed
safetensors header JSON (already extracted from the real artifact — no container is opened
here) and emits `component<TAB>key<TAB>logical_dtype<TAB>d,d,d` rows on stdout. `tfs
structure bank` validates those rows and writes reviewed fixture data.

**The source must be named explicitly, and the old default is GONE.**
`minimax_h3_fl2va_fp8_scaled.header.json` was proven fabricated (2026-08-25): 943 logical
keys including 312 phantom attention/feed-forward biases that appear in zero real H3
artifacts, and a paired index claiming 7 shards where the official repo has shipped 13/14.
The `h3.fl2va/full-adaln` fixture derived from it had its `real` provenance REVOKED — see
the tfs-008 issue note and `tracker-v2/examples/safetensors-headers/PROVENANCE.md`, which
now carries three files pinned to repo + revision + sha256.

Re-deriving is NOT this script's decision to make silently: the replacement fp8 file is a
CURVE artifact (532 logical keys, `adaln_proj.linear`, no `time_embedder`), so transcribing
it under the old `full-adaln` stamp name would put a different topology behind a stamp that
already means something else. Pick the source and the stamp name together, per the tfs-008
table, and pass the source here.

The three derivations, unchanged as a method:

1. `.comfy_quant` marker keys are DROPPED — under an explicit `encoding` digest they have
   nothing left to say.
2. A quantized weight's LOGICAL dtype is bf16, not its F8_E4M3 carrier. The carrier is
   storage; the logical type is the encoding-invariant meaning.
3. `.weight_scale` AND `.input_scale` siblings are DROPPED — ROLES of one cozytensors
   tensor, not tensors. `.input_scale` was added 2026-08-25 when the retired 943-key source
   was replaced by the pinned CURVE fp8 header, which carries 150 of them; the border's own
   vocabulary (`fingerprint::ROLE_SUFFIXES`) has always called it a role, and the rule
   extension is that one word, made explicit here rather than assumed.

Any OTHER role suffix refuses: a role this script has no rule for would otherwise be
transcribed as a logical tensor of its own, which is how a fixture quietly stops describing
the artifact it names.
"""

import hashlib
import json
import pathlib
import sys

# Retired sources, by name and by digest prefix. Naming both means neither a rename nor a
# copy walks past.
RETIRED_NAMES = {"minimax_h3_fl2va_fp8_scaled.header.json"}
RETIRED_DIGESTS = ("abdbc6a1e5293181",)
PROVENANCE = "tracker-v2/examples/safetensors-headers/PROVENANCE.md"

DTYPE = {"BF16": "bf16", "F32": "f32", "F16": "f16", "F8_E4M3": "f8_e4m3fn",
         "F8_E5M2": "f8_e5m2", "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8",
         "U8": "u8", "BOOL": "bool"}
DROPPED_ROLES = (".weight_scale", ".input_scale", ".comfy_quant")


def refuse(why: str) -> None:
    print(f"REFUSED: {why}", file=sys.stderr)
    raise SystemExit(2)


if len(sys.argv) < 3:
    refuse(
        "usage: derive-h3-fixture.py <header.json> <component>\n"
        f"  The source is explicit on purpose. See {PROVENANCE} for the three pinned real\n"
        "  headers and the tfs-008 note for which stamp name each one may carry."
    )

SRC = pathlib.Path(sys.argv[1])
COMPONENT = sys.argv[2]

if SRC.name in RETIRED_NAMES:
    refuse(
        f"{SRC.name} is RETIRED — it was proven fabricated (943 logical keys, 312 phantom "
        f"bias tensors, no upstream revision with that shape). Its fixture's `real` "
        f"provenance is revoked (tfs-008 / tfs-010). Use one of the pinned replacements in "
        f"{PROVENANCE}."
    )

raw = SRC.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
if digest.startswith(RETIRED_DIGESTS):
    refuse(
        f"these bytes are the RETIRED fabricated header (sha256 {digest[:16]}…), whatever "
        f"the file is called. See {PROVENANCE}."
    )

h = json.loads(raw)
h.pop("__metadata__", None)

# The KEY the scale belongs to, not the MODULE. This used to strip ".weight_scale"
# whole, leaving `blocks.0.attn.out_proj`, which matches no key — so rule 2 never fired
# once in the life of this script. It was invisible while the source was the fabricated
# header, whose tensors were declared BF16 anyway; against a real fp8 carrier it banks 200
# F8_E4M3 CARRIER dtypes into fixture data whose whole subject is the logical structure.
quantized = {k[: -len("_scale")] for k in h if k.endswith(".weight_scale")}
unknown = sorted(
    k for k in h
    if "." in k.rsplit(".", 1)[0] + "."
    and k.endswith(tuple(f".{r}" for r in ("smooth_factor", "bias_scale",
                                           "pre_quant_scale", "weight_scale_2")))
)
if unknown:
    refuse(
        f"{len(unknown)} keys carry a role this transcription has no rule for, first "
        f"{unknown[:3]} — a role transcribed as a logical tensor makes the fixture describe "
        f"an artifact that does not exist. Extend the rules deliberately (tfs-008) or pick "
        f"a source whose roles are covered."
    )

rows = []
for key, v in h.items():
    if key.endswith(DROPPED_ROLES):
        continue
    dtype = "bf16" if key in quantized else DTYPE[v["dtype"]]
    rows.append((COMPONENT, key, dtype, ",".join(str(d) for d in v["shape"])))

rows.sort()
for r in rows:
    print("\t".join(r))

print(f"# source {SRC}", file=sys.stderr)
print(f"# source_sha256 {digest}", file=sys.stderr)
print(f"# keys {len(h)} -> logical {len(rows)} (quantized {len(quantized)})", file=sys.stderr)
