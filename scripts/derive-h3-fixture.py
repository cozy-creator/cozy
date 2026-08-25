#!/usr/bin/env python3
"""Transcribe the REAL shipped H3 DiT header into structure-fixture rows.

This is a TRANSCRIPTION of evidence, not a parser and not a codec: it reads one committed
safetensors header JSON (already extracted from the real artifact — no container is opened
here) and emits `component<TAB>key<TAB>logical_dtype<TAB>d,d,d` rows on stdout. `tfs
structure bank` turns those rows into the canonical document, so canonical bytes have
exactly one writer.

Source: tracker-v2/examples/safetensors-headers/minimax_h3_fl2va_fp8_scaled.header.json
        — the header the H3 DiT ships with today (ComfyUI comfy_quant convention).

Two derivations are made here, and both are stated in the fixture's provenance note:

1. The `.comfy_quant` marker keys are DROPPED. They exist only to tell a reader what the
   neighbouring bytes mean; under an explicit `encoding` digest they have nothing to say.
2. A quantized weight's LOGICAL dtype is bf16, not its F8_E4M3 carrier. The carrier is
   storage; the logical type is the encoding-invariant meaning, and every unquantized
   weight in this same artifact is BF16.

The `.weight_scale` siblings are dropped too: they are a ROLE of one cozytensors tensor,
not a tensor of their own.
"""

import hashlib
import json
import pathlib
import sys

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    pathlib.Path(__file__).resolve().parents[2]
    / "tracker-v2/examples/safetensors-headers/minimax_h3_fl2va_fp8_scaled.header.json")

raw = SRC.read_bytes()
h = json.loads(raw)
h.pop("__metadata__", None)

DTYPE = {"BF16": "bf16", "F32": "f32", "F16": "f16", "F8_E4M3": "f8_e4m3fn",
         "F8_E5M2": "f8_e5m2", "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8",
         "U8": "u8", "BOOL": "bool"}

quantized = {k[: -len("_scale")] for k in h if k.endswith(".weight_scale")}
rows = []
for key, v in h.items():
    if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
        continue
    dtype = "bf16" if key in quantized else DTYPE[v["dtype"]]
    rows.append(("transformer", key, dtype, ",".join(str(d) for d in v["shape"])))

rows.sort()
for r in rows:
    print("\t".join(r))

print(f"# source_sha256 {hashlib.sha256(raw).hexdigest()}", file=sys.stderr)
print(f"# keys {len(h)} -> logical {len(rows)} (quantized {len(quantized)})", file=sys.stderr)
