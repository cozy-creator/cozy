#!/usr/bin/env python3
"""Synthesize small carriers for the ingest border's red arms.

Not a codec and not a second implementation: it writes ONE tiny well-formed safetensors
file and then damages it in exactly one named way, so the border's refusal can be observed
against a defect whose shape is known. Every variant is 3 tensors and a few hundred bytes.
"""
import json
import struct
import sys

VARIANTS = [
    "good", "pickle", "pickle-proto0", "zip", "gguf", "gzip",
    "header-cap", "truncated", "geometry", "overlap", "gap",
    "dtype-unknown", "duplicate-key", "float-dim", "no-tensors", "bnb", "deep",
    "mixed-fp8", "mixed-fp8-spoof", "rowwise-keepdim",
]

# One module's worth of ComfyUI `scaled_fp8` marker payload, byte-for-byte the shape the
# real artifact ships. The planner never reads these bytes — they are here so a marker CAN
# lie and be seen not to matter.
MARKER = b'{"format": "float8_e4m3fn"}'


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
    if spoof:
        # A marker on a PLAIN bf16 tensor that has no companion role anywhere, naming an
        # encoding whose siblings are absent. Selection reads siblings, not markers.
        t["norm.comfy_quant"] = ("U8", [len(MARKER)], len(MARKER))
    return dict(sorted(t.items()))


def body(tensors):
    """{name: (dtype, shape, nbytes)} -> (header dict, payload)"""
    header, blob, off = {}, bytearray(), 0
    for name, (dt, shape, n) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        blob += bytes((i * 37 + 11) % 256 for i in range(n))
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
        h2, blob2 = body(mixed_fp8(variant.endswith("spoof")))
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
