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

`int8-convrot256` is mined from v2's own producer, cozy-runtime's `encode_int8_convrot256`,
copied verbatim below with its file citation.

`fp8-scaled-scalar` is mined a step further: nothing is copied at all. The producer class
is IMPORTED and CALLED — ComfyUI's own `TensorCoreFP8E4M3Layout.quantize(..., "recalculate")`
and comfy_kitchen's own `quantize_fp8` / `dequantize_fp8` kernels, which run on CPU. It
needs ComfyUI's venv rather than v1's:

    ~/cozy_v2/comfy-bench/ComfyUI/.venv/bin/python scripts/mine-vectors.py <alias> > out.rows

Everything else runs under either. Weights never enter this: the material is a 16x32 random
tensor with the values that break decoders planted in it.

Output is a deliberately dumb line format, NOT JSON: canonical bytes have exactly one
writer in this repo and it is the Rust one. `cargo run -p tensorfs-core --example mine-vectors
-- <rows> <name>` turns these rows into the canonical EncodingVectors document.

    case <name>
    logical <dtype> <d,d>
    expect <base64 of the raw expected logical bytes>
    role <name> <dtype> <d,d> <base64 of the raw role bytes>
"""

import base64
import os
import pathlib
import sys

import numpy as np
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


# ------------------------------------------------- cozy-runtime's int8 producer, verbatim

# cozy-runtime src/cozy_runtime/derive/microscale.py: hadamard256 and
# encode_int8_convrot256, the producer of every `int8-convrot256/1` tensor.
def hadamard256(values):
    rows, cols = values.shape
    work = values.reshape(-1, 4, 4, 4, 4)
    for axis in range(1, 5):
        a, b, c, d = (np.take(work, i, axis=axis) for i in range(4))
        work = np.stack((a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a), axis=axis)
    return work.reshape(rows, cols)


def quantize_int8_convrot256(w):
    rotated = hadamard256(w.astype(np.float64)) / 16.0
    amax = np.abs(rotated).max(axis=-1)
    scale = np.maximum(amax / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.rint(rotated / scale.astype(np.float64)[:, None]), -127.0, 127.0)
    return q.astype(np.int8), scale


def mine_int8_convrot256():
    """The value law: `scale[i] * (data[i] . H)[j] / 16`, H the regular Hadamard
    H4(x)H4(x)H4(x)H4 per 256 input channels. The Hadamard sum of int8 is an exact integer;
    it is multiplied by the f32 scale IN F32 (one rounding), by 1/16 (exact), then cast to the
    logical dtype. An outlier, a zero row and a negative extreme are planted."""
    w = material(4, 512, 6).numpy()
    w[3, 300] = -9.25
    q, scale = quantize_int8_convrot256(w)
    sums = hadamard256(q.astype(np.int32)).astype(np.float32)
    deq = torch.from_numpy((sums * scale[:, None]) * np.float32(0.0625))
    for lg, dt in (("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)):
        emit(f"int8-convrot256-4x512-{lg}", lg, [4, 512], raw(deq.to(dt)), {
            "data": ("i8", [4, 512], q.tobytes()),
            "scale": ("f32", [4], scale.tobytes()),
        })


# ---------------------------------------------------------------- ComfyUI, imported

# ComfyUI v0.33.0, 2f35f4a08176d993cded35dac3332be4f7287f41; comfy_kitchen 0.2.31.
COMFY = os.environ.get("COMFY_ROOT") or str(pathlib.Path.home() / "cozy_v2/comfy-bench/ComfyUI")


def _comfy_fp8_layout():
    """ComfyUI's own per-tensor fp8 layout class, imported rather than reimplemented.

    `quantize(t, scale="recalculate")` is comfy/quant_ops.py:118 — scale =
    amax(|t|)/finfo(e4m3).max as an f32 SCALAR, then comfy_kitchen's compiled
    `quantize_fp8`. That scalar is what ComfyUI stores as `<module>.weight_scale`, and it is
    the whole reason `fp8-scaled-scalar/1` is a different spec from the rank-2 variants.
    """
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    import comfy.quant_ops as qo

    return qo.TensorCoreFP8E4M3Layout


def mine_fp8_scaled_scalar():
    """The two role sets ComfyUI's `scaled_fp8` emits in ONE file, mined separately.

    Measured on the pinned pruned fp8 H3 DiT: 150 modules carry `.weight_scale` AND
    `.input_scale` (out_proj, qkv_proj, mlp.fc1) and 50 carry the weight scale alone
    (mlp.fc2, the `full_precision_matrix_mult` ones — no activation is quantized there, so
    there is no activation scale to store). Two role sets, two spec digests, two vector
    documents.

    The activation scale is produced by the SAME producer call on a synthetic activation
    batch, because that is what it is: `QuantizedTensor.from_float(input, layout, scale=...)`
    (comfy/ops.py:1408). It takes no part in reconstructing the weight, which is precisely
    the fact these vectors pin — a decoder that folds it in is one of the wrong arms.
    """
    import comfy_kitchen as ck

    layout = _comfy_fp8_layout()
    w = material(16, 32, 4)
    # one large-but-not-448 outlier so the recalculated scale is nowhere near 1.0: a scale
    # of exactly 1 would let a scale-ignoring decoder pass and prove nothing
    w[3, 7] = -137.5
    act = material(8, 32, 5) * 3.25
    for lg, dt in (("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)):
        q, params = layout.quantize(w.to(dt), scale="recalculate")
        _, aparams = layout.quantize(act.to(dt), scale="recalculate")
        deq = ck.dequantize_per_tensor_fp8(q, params.scale, dt)
        roles = {
            "data": ("f8_e4m3fn", [16, 32], raw(q)),
            "weight_scale": ("f32", [], raw(params.scale.reshape(()))),
        }
        emit(f"fp8-scaled-scalar-wonly-16x32-{lg}", lg, [16, 32], raw(deq), dict(roles))
        roles["input_scale"] = ("f32", [], raw(aparams.scale.reshape(())))
        emit(f"fp8-scaled-scalar-16x32-{lg}", lg, [16, 32], raw(deq), roles)


MINERS = {
    "fp8-rowwise": mine_fp8_rowwise,
    "nvfp4-w4a4": mine_nvfp4,
    "fp8-scaled-scalar": mine_fp8_scaled_scalar,
    "int8-convrot256": mine_int8_convrot256,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in MINERS.items():
        if which in ("all", name):
            fn()
    print("\n".join(OUT))
    print(f"# torch {torch.__version__}", file=sys.stderr)
    print(f"# {len([l for l in OUT if l.startswith('case ')])} cases", file=sys.stderr)
