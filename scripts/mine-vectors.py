#!/usr/bin/env python3
"""PRODUCER-MINE raw-bit conformance vectors (tfs-008).

Provenance discipline (cozytensors.md §4): vectors are transcribed from REAL PRODUCER
OUTPUT or an independent second implementation — never from the spec author's own reader.
The quantizers below are COPIED VERBATIM from the v1 producers under ~/cozy (owner ruling:
v1 is a quarry), with their file:line citations, so what these vectors pin is what v1
actually shipped — not what v2's Rust reader believes.

    ~/cozy/python-gen-worker/src/gen_worker/models/nvfp4_quant.py   cast_e2m1, pack_e2m1
    ~/cozy/python-gen-worker/src/gen_worker/models/w4a4.py          quantize_nvfp4_tensor
    ~/cozy/python-gen-worker/src/gen_worker/convert/writer.py:546   the fp8 w8a8 quantizer

Run under v1's venv, which has the real torch:

    ~/cozy/python-gen-worker/.venv/bin/python scripts/mine-vectors.py <alias> > out.rows

Output is a deliberately dumb line format, NOT JSON: canonical bytes have exactly one
writer in this repo and it is the Rust one. `tfs vectors mine` turns these rows into the
canonical EncodingVectors document.

    case <name>
    logical <dtype> <d,d>
    expect <base64 of the raw expected logical bytes>
    role <name> <dtype> <d,d> <base64 of the raw role bytes>
"""

import base64
import sys

import torch

E2M1_MAX = 6.0
FP8_MAX = 448.0
SCALE_MIN = 2.0 ** -9
BLOCK = 16

OUT = []


def emit(name, logical_dtype, logical_shape, expect_bytes, roles):
    OUT.append(f"case {name}")
    OUT.append(f"logical {logical_dtype} {','.join(str(d) for d in logical_shape)}")
    OUT.append("expect " + base64.b64encode(expect_bytes).decode())
    for rn, (rdt, rshape, rbytes) in roles.items():
        OUT.append(
            f"role {rn} {rdt} {','.join(str(d) for d in rshape)} "
            + base64.b64encode(rbytes).decode()
        )


def raw(t):
    """The tensor's exact little-endian storage bytes. A rank-0 tensor has no dim to
    reinterpret, so it is flattened first — the bytes are identical either way."""
    return t.contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------- v1, verbatim

def _e2m1_bounds(device):
    # nvfp4_quant.py:33
    return torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        device=device, dtype=torch.float32)


def _e2m1_lut(device):
    # nvfp4_quant.py:24
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device=device, dtype=torch.float32)


def cast_e2m1(t):
    # nvfp4_quant.py:41 — NOT round-half-even: exact midpoints round toward zero
    # EXCEPT {0.75, 1.75, 2.5}, which the `tie` term bumps up.
    v = t.float()
    sign = (v < 0).to(torch.uint8)
    a = v.abs()
    ord_ = torch.searchsorted(_e2m1_bounds(v.device), a.contiguous(),
                              out_int32=True).to(torch.uint8)
    tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(torch.uint8)
    return (sign << 3) + ord_ + tie


def pack_e2m1(codes):
    # nvfp4_quant.py:54 — element 2j in the LOW nibble
    return ((codes[..., 1::2] << 4) | codes[..., 0::2]).contiguous()


def unpack_e2m1(packed):
    # nvfp4_quant.py:59
    lut = _e2m1_lut(packed.device)
    shape = list(packed.shape)
    shape[-1] = shape[-1] * 2
    out = torch.empty(shape, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lut[(packed & 0x0F).long()]
    out[..., 1::2] = lut[(packed >> 4).long()]
    return out


def quantize_nvfp4_tensor(w):
    # w4a4.py:140
    wf = w.float()
    out_f, in_f = wf.shape
    ws2 = (wf.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
    blocks = wf.reshape(out_f, in_f // BLOCK, BLOCK)
    bmax = blocks.abs().amax(dim=-1)
    bs = bmax / (E2M1_MAX * ws2)
    bs[bs == 0] = 1.0
    bs_fp8 = bs.clamp(min=SCALE_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
    q = blocks / (bs_fp8.float().unsqueeze(-1) * ws2)
    codes = cast_e2m1(q.reshape(out_f, in_f))
    return pack_e2m1(codes), bs_fp8, ws2


def dequantize_nvfp4_tensor(packed, weight_scale, weight_scale_2):
    # w4a4.py:157 — the PRODUCER's own inverse, not v2's reader
    vals = unpack_e2m1(packed)
    out_f, in_f = vals.shape
    scales = weight_scale.float().reshape(out_f, in_f // BLOCK, 1)
    return (vals.reshape(out_f, in_f // BLOCK, BLOCK)
            * scales * weight_scale_2.float()).reshape(out_f, in_f)


def quantize_fp8_rowwise(w):
    # convert/writer.py:546 — v1's "w8a8" is fp8-e4m3 PER OUTPUT ROW, not int8.
    wf = w.float()
    scale = (wf.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale  # scale is [out, 1]; v1 stores scale.reshape(-1)


def quantize_mxfp8(w, block=32):
    """INDEPENDENT SECOND IMPLEMENTATION, not a v1 producer: v1 has NO MX producer at all
    (E8M0 appears there only as one decode line, hf_fp8_blockwise.py:275, from which the
    convention `scale = 2^(byte - 127)` is taken). Written from the OCP Microscaling spec:
    the shared block scale is the power of two that puts the block's amax at the top of
    E4M3's range, floor-clamped into E8M0's exponent domain. The E4M3 cast itself is
    torch's real one, so only the exponent choice is ours."""
    wf = w.float()
    out_f, in_f = wf.shape
    nb = (in_f + block - 1) // block
    pad = nb * block - in_f
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    blocks = wf.reshape(out_f, nb, block)
    amax = blocks.abs().amax(dim=-1)
    # exponent such that amax / 2^e <= 448; E4M3's max exponent is 8 (448 = 1.75*2^8)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 8.0,
                    torch.zeros_like(amax))
    e = e.clamp(min=-127.0, max=127.0)
    byte = (e + 127.0).to(torch.uint8)
    scale = torch.exp2(byte.float() - 127.0)
    q = (blocks / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(out_f, nb * block)[:, :in_f].contiguous(), byte, scale


# ---------------------------------------------------------------- the material

def material(rows, cols, seed=0):
    """Real random tensors plus the values that actually break decoders: exact e2m1
    midpoints, signed zero, a whole zero row, and one huge outlier."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    w[0, :8] = torch.tensor([0.0, -0.0, 0.75, -0.75, 1.75, 2.5, 3.5, 5.0])
    w[1, :] = 0.0
    w[2, 0] = 60.0
    return w


def mine_fp8_rowwise():
    w = material(16, 32, 1)
    q, scale2d = quantize_fp8_rowwise(w)
    deq = q.float() * scale2d
    for lg in ("f32", "bf16"):
        t = deq if lg == "f32" else deq.to(torch.bfloat16)
        # rank-1 scale: v1's own `scale.reshape(-1)` (convert/writer.py:554)
        emit(f"fp8-rowwise-16x32-{lg}", lg, [16, 32], raw(t), {
            "data": ("f8_e4m3fn", [16, 32], raw(q)),
            "scale": ("f32", [16], raw(scale2d.reshape(-1))),
        })
        # rank-2 scale: the SAME producer bytes with the keepdim axis kept — the shipped
        # H3 comfy_quant shape. One producer run, two physical variants.
        emit(f"fp8-rowwise-keepdim-16x32-{lg}", lg, [16, 32], raw(t), {
            "data": ("f8_e4m3fn", [16, 32], raw(q)),
            "scale": ("f32", [16, 1], raw(scale2d)),
        })


def mine_nvfp4():
    w = material(16, 64, 2)
    packed, bs, ws2 = quantize_nvfp4_tensor(w)
    deq = dequantize_nvfp4_tensor(packed, bs, ws2)
    # nvfp4's spec admits bf16/f16 logically and NOT f32 — it is a 4-bit weight format for
    # 16-bit activations. Mining an f32 case would be mining a rung the spec refuses.
    for lg in ("bf16", "f16"):
        t = deq.to(torch.bfloat16 if lg == "bf16" else torch.float16)
        emit(f"nvfp4-w4a4-16x64-{lg}", lg, [16, 64], raw(t), {
            "weight": ("u8", [16, 32], raw(packed)),
            "weight_scale": ("f8_e4m3fn", [16, 4], raw(bs)),
            "weight_scale_2": ("f32", [], raw(ws2.reshape(()))),
        })


def mine_mxfp8():
    w = material(4, 64, 3)
    q, byte, scale = quantize_mxfp8(w)
    deq = q.float() * scale.repeat_interleave(32, dim=1)[:, :64]
    for lg in ("f32", "bf16"):
        t = deq if lg == "f32" else deq.to(torch.bfloat16)
        emit(f"mxfp8-4x64-{lg}", lg, [4, 64], raw(t), {
            "data": ("f8_e4m3fn", [4, 64], raw(q)),
            "scale": ("u8", [4, 2], raw(byte)),
        })


MINERS = {
    "fp8-rowwise": mine_fp8_rowwise,
    "nvfp4-w4a4": mine_nvfp4,
    "mxfp8": mine_mxfp8,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in MINERS.items():
        if which in ("all", name):
            fn()
    print("\n".join(OUT))
    print(f"# torch {torch.__version__}", file=sys.stderr)
    print(f"# {len([l for l in OUT if l.startswith('case ')])} cases", file=sys.stderr)
