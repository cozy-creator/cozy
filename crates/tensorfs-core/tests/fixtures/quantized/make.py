"""Quantized checkpoints as their producers write them, for `tests/quantized_import.rs`.

Each case is a folder of the producer's files (`<case>/`) and `<case>.expect.safetensors`: for every
quantized weight `<key>`, `source.<key>` (the BF16 weight before quantization) and `dequant.<key>` (the
producer's own dequantization to BF16, ported line by line, in the producer's arithmetic).

The quantizers and byte layouts are ported from:
- ModelOpt: nvidia-modelopt 0.44.0, modelopt/torch/quantization/qtensor/nvfp4_tensor.py
  (`get_weights_scaling_factor{,_2}`, `_cast_fp4`, packing `(q[..., 1::2] << 4) | q[..., 0::2]`: the even
  element in the LOW nibble; row-major E4M3 block scales; dequantize `q * (weight_scale * weight_scale_2)`)
  and mxfp8_tensor.py (E8M0 exponent `ceil(log2(amax / 448))` stored as uint8, row-major); the export
  declares itself in `hf_quant_config.json`.
- compressed-tensors 0.19.0: compressors/nvfp4 (`weight_packed`, `pack_fp4_to_uint8`: `low | high << 4`),
  quantization/lifecycle/forward.py `_dequantize` (`scale / global_scale`, then `x_q * scale`),
  compressors/mxfp8 (uint8 E8M0 `weight_scale`); declared by `config.json` `quantization_config`.
- ComfyUI (comfy/float.py: FP4 `(fp4[0::2] << 4) | fp4[1::2]`, the even element in the HIGH nibble, :114;
  `to_blocked`, cuBLAS 128x4 tiles padded to 128 rows and 4 block columns, :118; comfy/ops.py
  `.comfy_quant` markers, comfy/utils.py `_quantization_metadata` and the legacy `scaled_fp8`) and
  comfy-kitchen 0.2.37 (backends/eager/quantization.py `quantize_nvfp4`/`dequantize_nvfp4`,
  `quantize_mxfp8`/`dequantize_mxfp8`, `quantize_per_tensor_fp8`/`dequantize_per_tensor_fp8`,
  `quantize_int8_{tensorwise,rowwise}`, `quantize_int8_convrot_weight`, `dequantize_int8_*`).

Run from anywhere: uv run --python 3.12 --with numpy python -I make.py <this directory>
"""

import json
import sys
from pathlib import Path

import numpy as np

F32 = np.float32


# ------------------------------------------------------------------ element formats


def bf16(x):
    """f32 values rounded to BF16 (nearest even), as f32."""
    u = np.asarray(x, F32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(F32)


def _e4m3_values():
    out = []
    for c in range(127):  # 0x7F is NaN
        e, m = c >> 3, c & 7
        out.append(m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(out, np.float64)


E4M3 = _e4m3_values()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)


def encode(table, sign_bit, x):
    """Codes of `x` in a sign-magnitude format whose non-negative values are `table`: round to
    nearest, ties to the even code, saturating (torch's casts and kitchen's `_f32_to_floatx_unpacked`)."""
    a = np.abs(np.asarray(x, np.float64))
    hi = np.clip(np.searchsorted(table, a), 1, len(table) - 1)
    lo = hi - 1
    dlo, dhi = a - table[lo], table[hi] - a
    code = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    code = np.where(a >= table[-1], len(table) - 1, code)
    return (code | np.where(np.signbit(x), sign_bit, 0)).astype(np.uint8)


def e4m3(x):
    return encode(E4M3, 0x80, x)


def e4m3_value(c):
    c = np.asarray(c, np.uint8)
    v = E4M3[np.minimum(c & 0x7F, 126)].astype(F32)
    return np.where(c & 0x80, -v, v).astype(F32)


def e2m1(x):
    return encode(E2M1, 0x8, x)


def e2m1_value(c):
    v = E2M1[c & 7].astype(F32)
    return np.where(c & 8, -v, v).astype(F32)


def to_blocked(m):
    """comfy/float.py `to_blocked(.., flatten=False)`: cuBLAS 128x4 tiles, zero padded."""
    rows, cols = m.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    padded = np.zeros((nrb * 128, ncb * 4), m.dtype)
    padded[:rows, :cols] = m
    blocks = padded.reshape(nrb, 128, ncb, 4).transpose(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)


# ------------------------------------------------------------------ producers


def modelopt_nvfp4(w):
    g = F32(np.abs(w).max()) / F32(6.0 * 448.0)
    blocks = w.reshape(*w.shape[:-1], -1, 16)
    s = np.abs(blocks).max(-1) / (F32(6.0) * g)
    s[s == 0] = 1.0
    s = e4m3(s)
    q = e2m1((blocks / (e4m3_value(s) * g)[..., None]).reshape(w.shape))
    packed = (q[..., 1::2] << 4) | q[..., 0::2]
    # dequantize: q * (weight_scale * weight_scale_2), in f32, then .to(bf16)
    total = e4m3_value(s) * g
    deq = (e2m1_value(q).reshape(blocks.shape) * total[..., None]).reshape(w.shape)
    return {"weight": packed, "weight_scale": s, "weight_scale_2": np.array(g, F32)}, bf16(deq)


def ct_nvfp4(w):
    # compressed-tensors' global scale is the reciprocal: weight_global_scale = 448 * 6 / amax.
    gs = F32(448.0 * 6.0) / F32(np.abs(w).max())
    blocks = w.reshape(*w.shape[:-1], -1, 16)
    s = e4m3((np.abs(blocks).max(-1) / F32(6.0)) * gs)
    eff = e4m3_value(s) / gs  # _dequantize: scale = scale / global_scale
    q = e2m1((blocks / eff[..., None]).reshape(w.shape))
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    deq = (e2m1_value(q).reshape(blocks.shape) * eff[..., None]).reshape(w.shape)
    return {"weight_packed": packed, "weight_scale": s, "weight_global_scale": np.array([gs], F32)}, bf16(deq)


def comfy_nvfp4(w, g):
    """kitchen `quantize_nvfp4(hi_first=True)` of a matrix, scales `to_blocked`; `dequantize_nvfp4` in
    the output dtype: `total = g.to(bf16) * block_scale.to(bf16)`, `value * total`."""
    blocks = w.reshape(w.shape[0], -1, 16)
    s = e4m3(np.minimum(np.abs(blocks).max(-1) / F32(6.0) / g, F32(448.0)))
    q = e2m1((blocks / (g * e4m3_value(s))[..., None]).reshape(w.shape))
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    total = bf16(bf16(g) * bf16(e4m3_value(s)))
    deq = bf16(e2m1_value(q).reshape(blocks.shape) * total[..., None]).reshape(w.shape)
    return packed, to_blocked(s), deq


def comfy_global(w):
    """`TensorCoreNVFP4Layout.quantize`: `amax(|w|) / (448 * 6)` in the weight's dtype, then f32."""
    return bf16(F32(np.abs(w).max()) / F32(448.0 * 6.0))


def mxfp8(w):
    """ModelOpt `MXFP8QTensor` / kitchen `quantize_mxfp8`: E8M0 = ceil(log2(amax / 448)) + 127, both
    dequantize `q.float() * 2^(e - 127)` (exact)."""
    blocks = w.reshape(*w.shape[:-1], -1, 32)
    amax = np.abs(blocks).max(-1).astype(F32)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(amax / F32(448.0))), -127)
    e = np.clip(e + 127, 0, 254).astype(np.uint8)
    scale = (2.0 ** (e.astype(np.float64) - 127)).astype(F32)
    q = e4m3(np.clip(blocks / scale[..., None], -448, 448)).reshape(w.shape)
    deq = (e4m3_value(q).reshape(blocks.shape) * scale[..., None]).reshape(w.shape)
    return q, e, bf16(deq)


def comfy_fp8(w, rowwise):
    """`TensorCoreFP8Layout.quantize(scale="recalculate")` (or that with a per-row scale): kitchen
    `quantize_per_tensor_fp8`, `x * (1 / scale).to(bf16)`; `dequantize_per_tensor_fp8`,
    `x.to(bf16) * scale.to(bf16)`."""
    amax = np.abs(w).max(-1, keepdims=True) if rowwise else np.abs(w).max()
    scale = (amax.astype(F32) / F32(448.0)).astype(F32)
    q = e4m3(np.clip(bf16(w * bf16(F32(1.0) / scale)), -448, 448))
    return q, scale, bf16(e4m3_value(q) * bf16(scale))


def comfy_int8(w, rowwise):
    """kitchen `quantize_int8_{tensorwise,rowwise}`: `round(x / scale.to(bf16))` in bf16;
    `dequantize_int8_simple`: `q.float() * scale`, then `.to(bf16)`."""
    amax = np.abs(w).max(-1, keepdims=True) if rowwise else np.abs(w).max()
    scale = np.maximum(amax.astype(F32) / F32(127.0), F32(1e-30)).astype(F32)
    q = np.clip(np.round(bf16(w / bf16(scale))), -128, 127).astype(np.int8)
    return q, scale, bf16(q.astype(F32) * scale)


def hadamard(n):
    """kitchen `_build_hadamard`: H4 ⊗ ... ⊗ H4, normalized by 1/sqrt(n)."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float64)
    h = h4
    while h.shape[0] < n:
        h = np.kron(h, h4)
    return h / np.sqrt(n)


def comfy_convrot(w, group):
    """kitchen `quantize_int8_convrot_weight`: `W @ H^T` per group (in bf16), then per-row int8;
    `dequantize_int8_convrot_weight`: `(q.float() * scale) @ H^T` per group, then `.to(bf16)`. The
    group sums are taken exactly and rounded once to f32: a GEMM's summation order is its backend's."""
    rows, cols = w.shape
    h = hadamard(group)
    rot = bf16((w.astype(np.float64).reshape(rows, -1, group) @ h.T).reshape(rows, cols).astype(F32))
    q, scale, _ = comfy_int8(rot, True)
    x = (q.astype(F32) * scale).astype(np.float64)
    back = (x.reshape(rows, -1, group) @ h.T).reshape(rows, cols)
    return q, scale, bf16(back.astype(F32))


# ------------------------------------------------------------------ files


def safetensors(path, tensors, metadata=None):
    """`tensors`: (key, safetensors dtype, shape, numpy array) in file order."""
    head, data = {}, bytearray()
    if metadata:
        head["__metadata__"] = metadata
    for key, dtype, shape, array in tensors:
        raw = np.ascontiguousarray(array).tobytes()
        head[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(text).to_bytes(8, "little") + text + bytes(data))


def bf16_bytes(x):
    return (np.asarray(x, F32).view(np.uint32) >> 16).astype("<u2")


def marker(module, conf):
    """comfy/ops.py: a module's `comfy_quant` marker, its JSON as uint8."""
    m = np.frombuffer(json.dumps(conf).encode(), np.uint8)
    return (f"{module}.comfy_quant", "U8", [len(m)], m)


class Case:
    def __init__(self, root, name, seed):
        self.dir, self.name = root / name, name
        self.rng = np.random.default_rng(seed)
        self.expect = []

    def weight(self, shape, spread=0.05):
        w = self.rng.standard_normal(shape).astype(F32) * F32(spread)
        w.reshape(-1, shape[-1])[0, :8] *= 9  # outliers: blocks and rows of very different scale
        return bf16(w)

    def expected(self, key, source, deq):
        self.expect += [(f"dequant.{key}", "BF16", source.shape, bf16_bytes(deq)),
                        (f"source.{key}", "BF16", source.shape, bf16_bytes(source))]

    def write(self, tensors, metadata=None, files=None, name="model.safetensors"):
        safetensors(self.dir / name, tensors, metadata)
        for path, doc in (files or {}).items():
            (self.dir / path).write_text(json.dumps(doc, indent=2) + "\n")
        safetensors(self.dir.parent / f"{self.name}.expect.safetensors", self.expect)


def bias(case, key, n):
    return (key, "BF16", [n], bf16_bytes(case.weight([n])))


def main(root: Path):
    # ModelOpt NVFP4: a matrix and a 3-D expert stack (block scales on the last axis).
    c = Case(root, "modelopt-nvfp4", 1)
    tensors = []
    for key, shape in [("proj", [32, 64]), ("experts", [2, 16, 64])]:
        w = c.weight(shape)
        parts, deq = modelopt_nvfp4(w)
        c.expected(f"{key}.weight", w, deq)
        packed = list(shape[:-1]) + [shape[-1] // 2]
        tensors += [(f"{key}.weight", "U8", packed, parts["weight"]),
                    (f"{key}.weight_scale", "F8_E4M3", list(shape[:-1]) + [shape[-1] // 16], parts["weight_scale"]),
                    (f"{key}.weight_scale_2", "F32", [], parts["weight_scale_2"])]
        if key == "proj":
            tensors += [("proj.input_scale", "F32", [], np.array(0.0123, F32)), bias(c, "proj.bias", 32)]
    tensors.append(("norm.weight", "BF16", [64], bf16_bytes(c.weight([64]))))
    c.write(tensors, files={"hf_quant_config.json": {
        "producer": {"name": "modelopt", "version": "0.44.0"},
        "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                         "exclude_modules": ["norm"]}}})

    # ModelOpt MXFP8.
    c = Case(root, "modelopt-mxfp8", 2)
    w = c.weight([32, 64])
    q, e, deq = mxfp8(w)
    c.expected("proj.weight", w, deq)
    c.write([("proj.weight", "F8_E4M3", [32, 64], q), ("proj.weight_scale", "U8", [32, 2], e),
             bias(c, "proj.bias", 32)],
            files={"hf_quant_config.json": {
                "producer": {"name": "modelopt", "version": "0.44.0"},
                "quantization": {"quant_algo": "MXFP8", "group_size": 32, "exclude_modules": []}}})

    # compressed-tensors NVFP4 (divided global scale) and MXFP8.
    def ct_config(fmt, weights):
        return {"architectures": ["Toy"], "torch_dtype": "bfloat16", "quantization_config": {
            "quant_method": "compressed-tensors", "format": fmt, "quantization_status": "compressed",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights, "format": fmt}},
            "ignore": []}}

    c = Case(root, "ct-nvfp4", 3)
    w = c.weight([32, 64])
    parts, deq = ct_nvfp4(w)
    c.expected("proj.weight", w, deq)
    c.write([("proj.weight_packed", "U8", [32, 32], parts["weight_packed"]),
             ("proj.weight_scale", "F8_E4M3", [32, 4], parts["weight_scale"]),
             ("proj.weight_global_scale", "F32", [1], parts["weight_global_scale"]),
             ("proj.input_global_scale", "F32", [1], np.array([81.25], F32)),
             bias(c, "proj.bias", 32)],
            files={"config.json": ct_config("nvfp4-pack-quantized", {
                "num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
                "symmetric": True, "dynamic": False})})

    c = Case(root, "ct-mxfp8", 4)
    w = c.weight([32, 64])
    q, e, deq = mxfp8(w)
    c.expected("proj.weight", w, deq)
    c.write([("proj.weight", "F8_E4M3", [32, 64], q), ("proj.weight_scale", "U8", [32, 2], e)],
            files={"config.json": ct_config("mxfp8-quantized", {
                "num_bits": 8, "type": "float", "strategy": "group", "group_size": 32,
                "scale_dtype": "torch.uint8", "symmetric": True, "dynamic": False})})

    # ComfyUI NVFP4: 144 rows (two row tiles), 80 columns (5 blocks: two block-column tiles), and an
    # expert stack whose experts are tiled one by one.
    c = Case(root, "comfy-nvfp4", 5)
    w = c.weight([144, 80])
    g = comfy_global(w)
    packed, tiles, deq = comfy_nvfp4(w, g)
    c.expected("blocks.0.attn.qkv.weight", w, deq)
    tensors = [("blocks.0.attn.qkv.weight", "U8", [144, 40], packed),
               ("blocks.0.attn.qkv.weight_scale", "F8_E4M3", [256, 8], tiles),
               ("blocks.0.attn.qkv.weight_scale_2", "F32", [], np.array(g, F32)),
               ("blocks.0.attn.qkv.input_scale", "F32", [], np.array(0.0071, F32)),
               marker("blocks.0.attn.qkv", {"format": "nvfp4"}),
               bias(c, "blocks.0.attn.qkv.bias", 144)]
    w = c.weight([2, 128, 16])
    g = comfy_global(w)
    experts = [comfy_nvfp4(w[e], g) for e in range(2)]  # MoEExperts: tiles per expert, one scale_2
    c.expected("blocks.0.experts.weight", w, np.stack([d for _, _, d in experts]))
    tensors += [("blocks.0.experts.weight", "U8", [2, 128, 8], np.stack([p for p, _, _ in experts])),
                ("blocks.0.experts.weight_scale", "F8_E4M3", [2, 128, 4], np.stack([t for _, t, _ in experts])),
                ("blocks.0.experts.weight_scale_2", "F32", [], np.array(g, F32)),
                marker("blocks.0.experts", {"format": "nvfp4", "num_experts": 2}),
                ("final_norm.weight", "BF16", [80], bf16_bytes(c.weight([80])))]
    c.write(tensors)

    # ComfyUI MXFP8, declared by the file's `_quantization_metadata`: 5 blocks of 32 per row.
    c = Case(root, "comfy-mxfp8", 6)
    w = c.weight([32, 160])
    q, e, deq = mxfp8(w)
    c.expected("proj.weight", w, deq)
    c.write([("proj.weight", "F8_E4M3", [32, 160], q),
             ("proj.weight_scale", "F8_E8M0", [128, 8], to_blocked(e)),
             bias(c, "proj.bias", 32)],
            metadata={"_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": {"proj": {"format": "mxfp8"}}})})

    # ComfyUI fp8: a tensorwise scale with an input scale, and a per-row scale.
    c = Case(root, "comfy-fp8", 7)
    tensors = []
    for key, rowwise in [("attn.to_q", False), ("attn.to_k", True)]:
        w = c.weight([32, 64])
        q, scale, deq = comfy_fp8(w, rowwise)
        c.expected(f"{key}.weight", w, deq)
        tensors += [(f"{key}.weight", "F8_E4M3", [32, 64], q),
                    (f"{key}.weight_scale", "F32", [32, 1] if rowwise else [], scale)]
        if not rowwise:
            tensors.append((f"{key}.input_scale", "F32", [], np.array(0.25, F32)))
        tensors += [marker(key, {"format": "float8_e4m3fn"}), bias(c, f"{key}.bias", 32)]
    c.write(tensors)

    # ComfyUI's legacy `scaled_fp8` checkpoints: `scale_weight` and `scale_input` siblings.
    c = Case(root, "comfy-scaled-fp8", 8)
    w = c.weight([32, 64])
    q, scale, deq = comfy_fp8(w, False)
    c.expected("attn.to_v.weight", w, deq)
    c.write([("scaled_fp8", "F8_E4M3", [0], np.zeros(0, np.uint8)),
             ("attn.to_v.weight", "F8_E4M3", [32, 64], q),
             ("attn.to_v.scale_weight", "F32", [1], np.array([scale], F32)),
             ("attn.to_v.scale_input", "F32", [1], np.array([1.0], F32)),
             bias(c, "attn.to_v.bias", 32)])

    # ComfyUI int8_tensorwise: a scalar scale, a per-row scale, and ConvRot.
    c = Case(root, "comfy-int8", 9)
    tensors = []
    for key, kind, shape in [("ff.up", "tensorwise", [32, 64]), ("ff.gate", "rowwise", [32, 64]),
                             ("ff.down", "convrot", [16, 128])]:
        w = c.weight(shape)
        if kind == "convrot":
            q, scale, deq = comfy_convrot(w, 64)
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
        else:
            q, scale, deq = comfy_int8(w, kind == "rowwise")
            conf = {"format": "int8_tensorwise"}
        c.expected(f"{key}.weight", w, deq)
        tensors += [(f"{key}.weight", "I8", shape, q),
                    (f"{key}.weight_scale", "F32", [shape[0], 1] if kind != "tensorwise" else [], scale),
                    marker(key, conf), bias(c, f"{key}.bias", shape[0])]
    c.write(tensors)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
