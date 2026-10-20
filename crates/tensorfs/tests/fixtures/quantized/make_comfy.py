#!/usr/bin/env python3
"""Writes the real-comfy-{w4a8,w6a8,w4a4} import fixtures and comfy-provenance.json: 64 rows x 512 columns (two
rotation blocks, whole groups) of a MiniMax-H3 linear in each ComfyUI format, range-read at pinned revisions, with
its .comfy_quant marker, a BF16 norm (the logical dtype), and the same elements of Comfy-Org's bf16 weight.

expect: `source.<key>` is the bf16 reference; `dequant.<key>` is the spec's decode, q8 from comfy-kitchen 0.2.39's
own _dequant_int4_grouped_to_int8 (or its int4 unpack), rotated back exactly.

    uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch==2.14.1+cpu --with comfy-kitchen==0.2.39 \
      --with packaging python -I make_comfy.py
"""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import torch
from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
from comfy_kitchen.backends.eager.w4a8_int8 import _dequant_int4_grouped_to_int8

HERE = Path(__file__).resolve().parent
H3 = ("Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6")
BF16 = H3 + ("diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",)
CASES = {
    "real-comfy-w4a8": (("AX1Y2JP/MiniMax-H3-W4A8-ConvRot", "61a33b9224f9aa119e2332cb24a884051c9cf810",
                         "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"), "blocks.20.attn.qkv_proj", 10752),
    "real-comfy-w6a8": (H3 + ("diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors",), "blocks.20.mlp.fc1", 14336),
    "real-comfy-w4a4": (("Merserk/MiniMax-H3-INT4-ConvRot", "3c167bc916cab6cf3b85b9e3769952757ebb2e5d",
                         "minimax_h3_fl2va_pruned_int4_convrot.safetensors"), "blocks.20.attn.out_proj", 2688),
}
ROWS, COLS, B = 64, 512, 256
NORM = "blocks.20.attn.k_norm.weight"


def get(src, lo, hi) -> bytes:
    url = f"https://huggingface.co/{src[0]}/resolve/{src[1]}/{src[2]}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class File:
    def __init__(self, src):
        self.src = src
        n = struct.unpack("<Q", get(src, 0, 8))[0]
        raw = get(src, 8, 8 + n)
        self.h, self.base, self.sha = json.loads(raw), 8 + n, hashlib.sha256(raw).hexdigest()

    def rows(self, key, r0, r1):
        t = self.h[key]
        width = (t["data_offsets"][1] - t["data_offsets"][0]) // t["shape"][0]
        lo = self.base + t["data_offsets"][0]
        return get(self.src, lo + r0 * width, lo + r1 * width), width

    def whole(self, key):
        t = self.h[key]
        return get(self.src, self.base + t["data_offsets"][0], self.base + t["data_offsets"][1])


def cols(raw, width, lo, hi) -> bytes:
    return b"".join(raw[r * width + lo: r * width + hi] for r in range(len(raw) // width))


def hadamard(b):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.int64)
    h = h4
    while h.shape[0] < b:
        h = torch.kron(h, h4)
    return h


def spec(q8, s) -> bytes:
    """((q8 . H_B) x s[i]) x 2^-4, rounded to BF16."""
    v = (q8.long().reshape(ROWS, COLS // B, B) @ hadamard(B)).float() * s.reshape(ROWS, 1, 1)
    return (v * 2.0 ** -4).reshape(ROWS, COLS).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()


def write_safetensors(path, tensors):
    head, data = {}, bytearray()
    for key, dtype, shape, b in tensors:
        head[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(text).to_bytes(8, "little") + text + bytes(data))


def main():
    ref = File(BF16)
    provenance = {}
    for case, (src, module, r0) in CASES.items():
        f = File(src)
        r1 = r0 + ROWS
        w, width = f.rows(f"{module}.weight", r0, r1)
        k = ref.h[f"{module}.weight"]["shape"][1]
        marker = f.whole(f"{module}.comfy_quant")
        tensors = [(NORM, "BF16", [128], f.whole(NORM)), (f"{module}.comfy_quant", "U8", [len(marker)], marker)]
        if case == "real-comfy-w4a4":
            q = cols(w, width, 0, COLS // 2)
            s = f.rows(f"{module}.weight_scale", r0, r1)[0]
            tq = torch.frombuffer(bytearray(q), dtype=torch.int8).reshape(ROWS, -1)
            dequant = spec(_unpack_int4_row_major(tq), torch.frombuffer(bytearray(s), dtype=torch.float32))
            tensors += [(f"{module}.weight", "I8", [ROWS, COLS // 2], q), (f"{module}.weight_scale", "F32", [ROWS], s)]
        else:
            bits = 4 if case == "real-comfy-w4a8" else 6
            g = k // f.h[f"{module}.weight_s_rel"]["shape"][1]
            q = cols(w, width, 0, COLS // 2)
            if bits == 6:  # the nibble plane's slice, then the high plane's
                q = b"".join(q[r * 256:(r + 1) * 256] + w[r * width + k // 2: r * width + k // 2 + COLS // 4]
                             for r in range(ROWS))
            sr_rows, sr_width = f.rows(f"{module}.weight_s_rel", r0, r1)
            sr = cols(sr_rows, sr_width, 0, COLS // g)
            sc = f.rows(f"{module}.weight_s_channel", r0, r1)[0]
            cb = f.whole(f"{module}.weight_codebook") if bits == 4 else None
            tq = torch.frombuffer(bytearray(q), dtype=torch.int8).reshape(ROWS, -1)
            ts = torch.frombuffer(bytearray(sr), dtype=torch.uint8).reshape(ROWS, -1).view(torch.float8_e4m3fn)
            tc = None if cb is None else torch.frombuffer(bytearray(cb), dtype=torch.float32)
            dequant = spec(_dequant_int4_grouped_to_int8(tq, ts, tc, g),
                           torch.frombuffer(bytearray(sc), dtype=torch.float32))
            tensors += [(f"{module}.weight", "I8", [ROWS, COLS * bits // 8], q)]
            tensors += [(f"{module}.weight_codebook", "F32", [16], cb)] if cb is not None else []
            tensors += [(f"{module}.weight_s_channel", "F32", [ROWS], sc),
                        (f"{module}.weight_s_rel", "F8_E4M3", [ROWS, COLS // g], sr)]
        rw, rwidth = ref.rows(f"{module}.weight", r0, r1)
        source = cols(rw, rwidth, 0, COLS * 2)
        write_safetensors(HERE / case / "model.safetensors", tensors)
        write_safetensors(HERE / f"{case}.expect.safetensors",
                          [(f"dequant.{module}.weight", "BF16", [ROWS, COLS], dequant),
                           (f"source.{module}.weight", "BF16", [ROWS, COLS], source)])
        provenance[case] = {
            "file": {"repo": src[0], "revision": src[1], "path": src[2], "header_sha256": f.sha},
            "reference": {"repo": BF16[0], "revision": BF16[1], "path": BF16[2], "header_sha256": ref.sha},
            "module": module, "rows": [r0, r1], "columns": [0, COLS],
            "sha256": {key: hashlib.sha256(b).hexdigest() for key, _, _, b in tensors},
        }
        print(case, module, f"rows {r0}..{r1}")
    (HERE / "comfy-provenance.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
