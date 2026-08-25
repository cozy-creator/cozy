#!/usr/bin/env python3
"""Synthesize small carriers for the ingest border's red arms.

Not a codec and not a second implementation: it writes ONE tiny well-formed safetensors
file and then damages it in exactly one named way, so the border's refusal can be observed
against a defect whose shape is known. Most variants are a handful of tensors and a few
hundred bytes; the `native-qkv` family is ~64 KB so its permuted members clear the inline cap.
"""
import json
import struct
import sys

VARIANTS = [
    "good", "pickle", "pickle-proto0", "zip", "gguf", "gzip",
    "header-cap", "truncated", "geometry", "overlap", "gap",
    "dtype-unknown", "duplicate-key", "float-dim", "no-tensors", "bnb", "deep",
    "mixed-fp8", "mixed-fp8-spoof", "rowwise-keepdim",
    "native-qkv", "native-qkv-nonorm", "native-qkv-odd", "native-qkv-encoded",
    "native-qkv-vae",
    "marker-malformed",
]

# One module's worth of ComfyUI `scaled_fp8` marker payload, byte-for-byte the shape the
# real artifact ships. The planner never reads these bytes — they are here so a marker CAN
# lie and be seen not to matter.
MARKER = b'{"format": "float8_e4m3fn"}'
MARKER_FP = b'{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}'


def mixed_fp8(spoof):
    """The per-tensor split, in miniature: one module with both scales, one with the weight
    scale alone, and plain tensors beside them — the real fl2va DiT's 150/50/332 shape at
    three tensors instead of five hundred."""
    t = {
        "fc2.bias": ("BF16", [4], 8),
        "fc2.weight": ("F8_E4M3", [4, 8], 32),
        "fc2.weight_scale": ("F32", [], 4),
        "norm.weight": ("BF16", [8], 16),
        "qkv.comfy_quant": ("U8", [len(MARKER)], len(MARKER)),
        "qkv.input_scale": ("F32", [], 4),
        "qkv.weight": ("F8_E4M3", [4, 8], 32),
        "qkv.weight_scale": ("F32", [], 4),
    }
    # The real 150/50 shape: the module with both scales carries the plain marker, the
    # weight-scale-only module carries `full_precision_matrix_mult`. On the real fl2va DiT
    # those 50 are exactly the `mlp.fc2` ones — the fact `inspect --markers` cross-checks.
    t["fc2.comfy_quant"] = ("U8", [len(MARKER_FP)], len(MARKER_FP))
    if spoof:
        # A marker on a PLAIN bf16 tensor that has no companion role anywhere, naming an
        # encoding whose siblings are absent. Selection reads siblings, not markers.
        t["norm.comfy_quant"] = ("U8", [len(MARKER)], len(MARKER))
    return dict(sorted(t.items()))


def native_qkv(kind):
    """H3 native spelling in miniature, with REAL bytes.

    head_dim comes from `q_norm` (4), the fused axis is 24 = 2 groups x 3 x 4, and one row is
    64 BF16 elements = 128 B — so the seam's unit is 512 B and each member is 1,024 B, over
    the 256 B inline threshold, which is what puts the permuted bytes through the real object
    path rather than into the header. The real artifact's numbers are 56 x 128 x 10,752 B;
    the SHAPE of the fact is reproduced here at a size the transaction runs in milliseconds.

    Both attention module families are present, which is the 50-vs-52 defect the v1 seam had.
    """
    def block(p):
        return {
            f"{p}.attn.q_norm.weight": ("BF16", [4], 8),
            f"{p}.attn.k_norm.weight": ("BF16", [4], 8),
            f"{p}.attn.qkv_proj.weight": ("BF16", [24, 64], 3072),
            f"{p}.attn.out_proj.weight": ("BF16", [64, 32], 4096),
            f"{p}.mlp.fc1.weight": ("BF16", [128, 64], 16384),
            f"{p}.mlp.fc2.weight": ("BF16", [64, 64], 8192),
            f"{p}.norm1.weight": ("BF16", [64], 128),
        }
    t = {"rope.inv_freq": ("F32", [2], 8)}
    t.update(block("blocks.0"))
    t.update(block("token_refiner.blocks.0"))
    if kind == "nonorm":
        # The head dim is READ, never assumed: remove the sibling it is read from.
        t.pop("blocks.0.attn.q_norm.weight")
    if kind == "odd":
        # 25 rows do not divide into 3 x 4. Geometry refuses; nothing is guessed.
        t["blocks.0.attn.qkv_proj.weight"] = ("BF16", [25, 64], 3200)
    if kind == "vae":
        # The video VAE's 1x1-conv spelling. A fused attention projection whose head
        # geometry this converter banks no seam for — it must refuse, never carry through.
        t["decoder.transformer_blocks.0.attn.to_qkv.weight"] = ("BF16", [24, 64], 3072)
    if kind == "encoded":
        # A fused key carrying an encoding role: where the head-group cut lands on a scale
        # is banked for no artifact that exists.
        t["blocks.0.attn.qkv_proj.weight"] = ("F8_E4M3", [24, 64], 1536)
        t["blocks.0.attn.qkv_proj.weight_scale"] = ("F32", [], 4)
    return dict(sorted(t.items()))


def body(tensors, payloads=None):
    """{name: (dtype, shape, nbytes)} -> (header dict, payload)"""
    header, blob, off = {}, bytearray(), 0
    for name, (dt, shape, n) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        fixed = (payloads or {}).get(name)
        blob += fixed if fixed is not None else bytes((i * 37 + 11) % 256 for i in range(n))
        off += n
    return header, bytes(blob)


def write(path, header, blob, meta=None, raw_len=None, trunc=0):
    if meta is not None:
        header = {"__metadata__": meta, **header}
    js = json.dumps(header).encode()
    n = raw_len if raw_len is not None else len(js)
    out = struct.pack("<Q", n) + js + blob
    if trunc:
        out = out[:-trunc]
    open(path, "wb").write(out)


def main(variant, path):
    base = {"a.weight": ("F16", [4, 8], 64), "a.bias": ("F16", [4], 8),
            "b.weight": ("F16", [8, 4], 64)}
    h, blob = body(base)

    if variant == "good":
        write(path, h, blob)
    elif variant == "pickle":
        open(path, "wb").write(b"\x80\x04\x95" + b"\x00" * (1 << 20))
    elif variant == "pickle-proto0":
        open(path, "wb").write(b"(dp0\nS'weight'\np1\n" + b"." * 4096)
    elif variant == "zip":
        open(path, "wb").write(b"PK\x03\x04" + b"\x00" * 4096)
    elif variant == "gguf":
        open(path, "wb").write(b"GGUF" + b"\x00" * 4096)
    elif variant == "gzip":
        open(path, "wb").write(b"\x1f\x8b\x08" + b"\x00" * 4096)
    elif variant == "header-cap":
        open(path, "wb").write(struct.pack("<Q", (1 << 63) - 1) + b"{}")
    elif variant == "truncated":
        write(path, h, blob, trunc=32)
    elif variant == "geometry":
        h["a.weight"]["data_offsets"] = [0, 63]     # one byte short of 4*8*2
        write(path, h, blob)
    elif variant == "overlap":
        h["b.weight"]["data_offsets"] = [8, 72]     # walks back into a.bias
        write(path, h, blob)
    elif variant == "gap":
        h["b.weight"]["data_offsets"] = [80, 144]   # 8 unclaimed bytes
        write(path, h, blob + b"\x00" * 8)
    elif variant == "dtype-unknown":
        h["a.weight"]["dtype"] = "F4"
        write(path, h, blob)
    elif variant == "duplicate-key":
        js = json.dumps(h).encode()
        dup = js[:1] + json.dumps({"a.weight": h["a.weight"]}).encode()[1:-1] + b"," + js[1:]
        write(path, {}, blob, raw_len=len(dup))
        open(path, "wb").write(struct.pack("<Q", len(dup)) + dup + blob)
    elif variant == "float-dim":
        js = json.dumps(h).encode().replace(b'"shape": [4, 8]', b'"shape": [4.0, 8]')
        open(path, "wb").write(struct.pack("<Q", len(js)) + js + blob)
    elif variant == "no-tensors":
        write(path, {}, b"")
    elif variant == "bnb":
        b2 = dict(base)
        b2["a.absmax"] = ("F32", [4], 16)
        h2, blob2 = body(b2)
        write(path, h2, blob2)
    elif variant in ("mixed-fp8", "mixed-fp8-spoof"):
        pay = {"qkv.comfy_quant": MARKER, "fc2.comfy_quant": MARKER_FP,
               "norm.comfy_quant": MARKER}
        h2, blob2 = body(mixed_fp8(variant.endswith("spoof")), pay)
        write(path, h2, blob2)
    elif variant == "marker-malformed":
        # A payload that is not the marker vocabulary: `format` absent. The decoder refuses
        # typed rather than defaulting a format nobody wrote.
        bad = b'{"quant_algo": "fp8"}'
        h2, blob2 = body({"a.weight": ("F16", [4, 8], 64),
                          "a.comfy_quant": ("U8", [len(bad)], len(bad))},
                         {"a.comfy_quant": bad})
        write(path, h2, blob2)
    elif variant.startswith("native-qkv"):
        kind = variant[len("native-qkv"):].lstrip("-")
        h2, blob2 = body(native_qkv(kind))
        write(path, h2, blob2)
    elif variant == "rowwise-keepdim":
        h2, blob2 = body(dict(sorted({
            "a.bias": ("BF16", [4], 8),
            "a.weight": ("F8_E4M3", [4, 8], 32),
            "a.weight_scale": ("F32", [4, 1], 16),
        }.items())))
        write(path, h2, blob2)
    elif variant == "deep":
        js = b'{"a.weight":' + b"[" * 64 + b"1" + b"]" * 64 + b"}"
        open(path, "wb").write(struct.pack("<Q", len(js)) + js + blob)
    else:
        raise SystemExit(f"unknown variant {variant}; one of {VARIANTS}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: make-carrier.py <{'|'.join(VARIANTS)}> <path>")
    main(sys.argv[1], sys.argv[2])
