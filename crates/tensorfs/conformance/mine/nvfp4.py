#!/usr/bin/env python3
"""Mines nvfp4's pre_quant_scale and per-matrix scale_2 cases into vectors/cases/nvfp4.json (its own
keys; the other cases are kept) and the real-producer import fixtures into
tensorfs-core/tests/fixtures/quantized/{modelopt-nvfp4-awq,comfy-nvfp4-experts}.

Producers, all on CPU:
- ModelOpt 0.47.0 over a tiny Llama (BF16, F16, F32): `mtq.quantize(NVFP4_DEFAULT_CFG)` and
  `export_hf_checkpoint` for NVFP4. For NVFP4_AWQ, max calibration, then awq_lite's own finalisation
  per linear (`_amax_for_smoothing`, `apply_pre_quant_scale_and_smooth(module, 1 / best_scale)`,
  `get_scale` at alpha 0.5), then the same export. Only awq_lite's alpha search is skipped: it
  fake-quantizes weights, which ModelOpt runs on CUDA only (and the search is a package job).
  ModelOpt keeps a pre_quant_scale on o_proj only; it folds the others into the norm and up_proj.
- ComfyUI: Comfy-Org/Lens gpt_oss_20b_nvfp4, a fused-expert file (`weight_scale_2 [E]`, tiles per
  expert), range-read: layers.1 down_proj, experts 11, 14, 0, rows 2816..2879 (the last, half-padded
  tile band), columns 0..63. comfy_kitchen 0.2.39 lays out the ComfyUI-layout twin and dequantizes.
- OpenAI gpt-oss-20b, the MXFP4 original of the same elements, range-read: the Lens slice decodes to
  it bit for bit.

Expected bytes are the spec's f32 decode. Each is compared with the producer's own dequantize
(ModelOpt NVFP4QTensor.dequantize x pre_quant_scale; comfy_kitchen per expert, as ComfyUI's
MoEExperts slices it) and the differences are printed.

    uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch==2.14.1+cpu --with nvidia-modelopt==0.47.0 \
      --with transformers==5.14.1 --with accelerate --with requests --with datasets \
      --with comfy-kitchen==0.2.39 python -I conformance/mine/nvfp4.py
"""

import base64
import json
import shutil
import struct
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import modelopt.torch.quantization as mtq
import modelopt.torch.export.unified_export_hf as unified
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.model_calib import apply_pre_quant_scale_and_smooth
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4, to_blocked
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

HERE = Path(__file__).resolve().parent
CASES = HERE.parent / "vectors" / "cases" / "nvfp4.json"
FIXTURES = HERE.parents[2] / "tensorfs-core" / "tests" / "fixtures" / "quantized"
LENS = ("Comfy-Org/Lens", "393618fd7db3e370509f84390b9162b8fd839a25", "text_encoders/gpt_oss_20b_nvfp4.safetensors")
GPT_OSS = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "model-00000-of-00002.safetensors")
EXPERTS, ROWS, N = [11, 14, 0], (2816, 2880), 64
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
NAME = {v: k for k, v in TORCH.items()}
torch.set_num_threads(1)


def b64(b) -> str:
    return base64.b64encode(bytes(b)).decode()


def raw(t) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def get(repo, rev, path, lo, hi) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def header(src):
    n = struct.unpack("<Q", get(*src, 0, 8))[0]
    return json.loads(get(*src, 8, 8 + n)), 8 + n


# ------------------------------------------------------------------ ModelOpt


def llama(dtype, seed):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=False, architectures=["LlamaForCausalLM"])
    model = LlamaForCausalLM(config).to(dtype).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2 and "embed" not in name:  # heavy-tailed, a few outlier input channels
                w = torch.randn(p.shape) * 0.05
                w[:, torch.randperm(p.shape[1])[:3]] *= 12
                p.copy_(w.to(dtype))
    g = torch.Generator().manual_seed(seed + 1)
    batches = [torch.randint(0, 96, (2, 16), generator=g) for _ in range(4)]

    def loop(m):
        with torch.no_grad():
            for b in batches:
                m(b)
    return model, loop


def awq_finalise(model, stats, dtype):
    """model_calib.py awq_lite.postprocess, with get_weight_scale and get_scale verbatim."""
    for name, module in model.named_modules():
        if name not in stats:
            continue
        mean, amax, n = stats[name]
        w = module.weight.contiguous().view(-1, 16).abs()
        w_scale = (w / (w.amax(dim=1, keepdim=True) + torch.finfo(w.dtype).tiny)).view(module.weight.shape)
        w_scale = w_scale.mean(0).to(torch.float32)
        scales = ((mean / n).pow(0.5) / (w_scale.pow(0.5) + torch.finfo(torch.float32).tiny)).clamp(1e-4, 1e4)
        best_scale = scales / (scales.max() * scales.min()).sqrt()
        iq = module.input_quantizer
        iq._amax_for_smoothing = amax.to(dtype)
        iq.reset_amax()
        iq.axis = None
        iq.amax = amax.to(dtype).amax()
        iq.enable()
        apply_pre_quant_scale_and_smooth(module, 1.0 / best_scale)


def modelopt(dtype, awq, out: Path):
    """Exports; returns {weight key: the weight ModelOpt quantized, after its export-time smoothing}."""
    model, loop = llama(dtype, 2 if awq else 1)
    stats = {}

    def hook(name):
        def f(_, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1]).abs()
            s = stats.setdefault(name, [0, torch.zeros(x.shape[-1]), 0])
            s[0], s[1], s[2] = s[0] + x.mean(0), torch.maximum(s[1], x.amax(0)), s[2] + 1
        return f
    handles = [m.register_forward_pre_hook(hook(n)) for n, m in model.named_modules()
               if awq and isinstance(m, nn.Linear) and "layers." in n]
    loop(model)
    for h in handles:
        h.remove()
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, loop)
    with torch.no_grad():
        awq_finalise(model, stats, dtype)
    names, quantized = {id(m): n for n, m in model.named_modules()}, {}
    export = unified._export_quantized_weight

    def recording(module, dt, weight_name="weight"):
        quantized[f"{names[id(module)]}.{weight_name}"] = getattr(module, weight_name).detach().clone()
        return export(module, dt, weight_name)
    unified._export_quantized_weight = recording
    try:
        export_hf_checkpoint(model, export_dir=str(out))
    finally:
        unified._export_quantized_weight = export
    return quantized


def modelopt_dequant(f, m):
    w, s, g = f[m + ".weight"], f[m + ".weight_scale"], f[m + ".weight_scale_2"]
    shape = torch.Size([w.shape[0], w.shape[1] * 2])
    d = NVFP4QTensor(shape, torch.float32, w).dequantize(torch.float32, scale=s, double_scale=g, block_sizes={-1: 16})
    return d * f[m + ".pre_quant_scale"].float() if m + ".pre_quant_scale" in f else d


# ------------------------------------------------------------------ the spec's decode


def spec(shape, w, s, g, p=None, divide=False, high_first=False):
    """f32, left to right: g per matrix of the stack, then x pre_quant_scale[j]."""
    w = np.frombuffer(w, np.uint8)
    if high_first:
        w = ((w << 4) | (w >> 4)).astype(np.uint8)
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    q = E2M1[np.stack([w & 15, w >> 4], -1).reshape(rows, cols // 16, 16)]
    s = s.reshape(rows, cols // 16, 1)
    g = np.asarray(g, np.float32).reshape(-1)
    g = g[np.arange(rows) // shape[-2] if g.size > 1 else np.zeros(rows, int)].reshape(rows, 1, 1)
    out = (q * (s / g) if divide else (q * s) * g).reshape(rows, cols)
    return out if p is None else out * np.asarray(p, np.float32)


def e4m3(b) -> np.ndarray:
    return torch.frombuffer(bytearray(b), dtype=torch.float8_e4m3fn).float().numpy()


def to_l(dtype, x) -> bytes:
    return raw(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(TORCH[dtype]))


def differ(dtype, a, b) -> str:
    """Elements of different value (+0 equals -0), the most steps of L apart, signed-zero flips."""
    bits = 16 if dtype != "F32" else 32
    x, y = (np.frombuffer(v, f"<u{bits // 8}").astype(np.int64) for v in (a, b))
    mag = (1 << (bits - 1)) - 1
    signed = lambda v: np.where(v >> (bits - 1), -(v & mag), v & mag)  # noqa: E731
    zero = ((x & mag) == 0) & ((y & mag) == 0)
    d = (signed(x) != signed(y)) & ~zero
    steps = int(np.abs(signed(x) - signed(y))[d].max(initial=0))
    return f"{int(d.sum())}/{x.size} differ (at most {steps} step), {int(((x != y) & zero).sum())} signed zeros"


# ------------------------------------------------------------------ cases

PARAMS = {"block_size": 16, "global_scale": "multiply", "nibble_order": "low_first", "scale_layout": "row_major"}
COMFY = dict(PARAMS, nibble_order="high_first", scale_layout="cublas_128x4")


def part(role, dtype, shape, bits):
    return {"bits": b64(bits), "dtype": dtype, "role": role, "shape": list(shape)}


def case(key, dtype, shape, params, parts, expect=None):
    c = {"component": "nvfp4" if expect is not None else "refuse", "dtype": dtype,
         "encoding": {"params": params, "scheme": "nvfp4"}, "key": key, "parts": parts, "shape": list(shape)}
    c.update({"expect": b64(expect)} if expect is not None else {"refuse": "decode"})
    return c


def modelopt_cases(tmp: Path, greens: list, reds: list):
    for dtype, awq in [(torch.bfloat16, False), (torch.bfloat16, True), (torch.float16, True), (torch.float32, True)]:
        L = NAME[dtype]
        out = tmp / f"{'awq' if awq else 'nvfp4'}-{L}"
        quantized = modelopt(dtype, awq, out)
        f = load_file(out / "model.safetensors")
        m = "model.layers.0.self_attn.o_proj"
        w, s, g, i = (f[m + x] for x in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"))
        shape = [w.shape[0], w.shape[1] * 2]
        p = f.get(m + ".pre_quant_scale")
        expect = to_l(L, spec(shape, raw(w), e4m3(raw(s)), g.numpy(), None if p is None else p.float().numpy()))
        key = f"modelopt-{'awq' if awq else 'nvfp4'}-{L.lower()}"
        print(f"{key}: against ModelOpt's dequantize: {differ(L, expect, raw(modelopt_dequant(f, m).to(dtype)))}")
        params = dict(PARAMS, **({"pre_quant_scale_dtype": NAME[p.dtype]} if awq else {}))
        parts = [part("weight", "F4", shape, raw(w)), part("scale", "F8_E4M3", s.shape, raw(s)),
                 part("scale_2", "F32", g.shape, raw(g)), part("input_scale", "F32", i.shape, raw(i))]
        if awq:
            parts.append(part("pre_quant_scale", NAME[p.dtype], p.shape, raw(p)))
        greens.append(case(key, L, shape, params, parts, expect))
        if L == "BF16" and awq:
            # ComfyUI's layout (the H3 text encoder's): kitchen's tiles, the even element high.
            tiles = to_blocked(s.view(torch.uint8).reshape(shape[0], -1), flatten=False)
            hw = ((w << 4) | (w >> 4)).to(torch.uint8)
            cparts = [part("weight", "F4", shape, raw(hw)), part("scale", "F8_E4M3", tiles.shape, raw(tiles))]
            greens.append(case("comfy-awq-bf16", L, shape, dict(COMFY, pre_quant_scale_dtype="BF16"),
                               cparts + parts[2:], expect))
            k = (dequantize_nvfp4(hw, g, tiles.view(torch.float8_e4m3fn), torch.bfloat16) * p).to(torch.bfloat16)
            print(f"comfy-awq-bf16: against comfy_kitchen x pre_quant_scale: {differ(L, expect, raw(k))}")
            red = lambda k, pr, ps: reds.append(case(k, L, shape, pr, ps))  # noqa: E731
            no_pqs = [x for x in parts if x["role"] != "pre_quant_scale"]
            red("awq-param-missing", PARAMS, parts)
            red("awq-part-missing", params, no_pqs)
            red("awq-dtype-mismatch", dict(params, pre_quant_scale_dtype="F32"), parts)
            red("awq-dtype-fp8", dict(params, pre_quant_scale_dtype="F8_E4M3"), parts)
            red("awq-pqs-count", params, no_pqs + [part("pre_quant_scale", "BF16", [32], raw(p[:32]))])
            two = [part("scale_2", "F32", [2], raw(g.reshape(1).repeat(2))) if x["role"] == "scale_2" else x
                   for x in parts]
            red("matrix-scale2-two", params, two)
            awq_bf16 = (out, quantized)
    return awq_bf16


def lens_slice():
    """The Lens slice's parts and bias, and the MXFP4 original's BF16 bytes for the same elements."""
    h, base = header(LENS)
    t = {k[len("layers.1.mlp.experts.down_proj."):]: v for k, v in h.items()
         if k.startswith("layers.1.mlp.experts.down_proj.")}
    assert t["weight"]["shape"] == [32, 2880, 1440] and t["weight_scale"]["shape"] == [32, 2944, 180]
    at = lambda name, off, n: get(*LENS, base + t[name]["data_offsets"][0] + off,  # noqa: E731
                                  base + t[name]["data_offsets"][0] + off + n)
    r0, r1 = ROWS
    w = b"".join(at("weight", (e * 2880 + r0) * 1440, (r1 - r0) * 1440)[r * 1440: r * 1440 + N // 2]
                 for e in EXPERTS for r in range(r1 - r0))
    s = b"".join(at("weight_scale", e * 2944 * 180 + (r0 // 128 * 45) * 512, 512) for e in EXPERTS)
    g = at("weight_scale_2", 0, 128)
    g = b"".join(g[4 * e: 4 * e + 4] for e in EXPERTS)
    bias = b"".join(at("bias", (e * 2880 + r0) * 2, (r1 - r0) * 2) for e in EXPERTS)
    h, base = header(GPT_OSS)
    B, S = h["model.layers.1.mlp.experts.down_proj_blocks"], h["model.layers.1.mlp.experts.down_proj_scales"]
    vals = []
    for e in EXPERTS:
        a = base + B["data_offsets"][0] + (e * 2880 + r0) * 1440
        blk = np.frombuffer(get(*GPT_OSS, a, a + (r1 - r0) * 1440), np.uint8).reshape(-1, 90, 16)[:, :N // 32]
        a = base + S["data_offsets"][0] + (e * 2880 + r0) * 90
        sc = np.frombuffer(get(*GPT_OSS, a, a + (r1 - r0) * 90), np.uint8).reshape(-1, 90)[:, :N // 32]
        q = E2M1[np.stack([blk & 15, blk >> 4], -1).reshape(-1, N // 32, 32)]
        vals.append(q * np.ldexp(np.float32(1), sc.astype(np.int32) - 127)[..., None])
    source = to_l("BF16", np.stack(vals).reshape(-1, N))
    return w, s, g, bias, source


def expert_cases(lens, greens, reds):
    w, s, g, _, source = lens
    E = len(EXPERTS)
    shape = [E, ROWS[1] - ROWS[0], N]
    i, b = np.arange(shape[1])[:, None], np.arange(N // 16)[None, :]
    rowmajor = np.stack([np.frombuffer(s, np.uint8)[e * 512:][16 * (i % 32) + 4 * (i % 128 // 32) + b]
                         for e in range(E)])
    expect = to_l("BF16", spec(shape, w, e4m3(rowmajor.tobytes()), np.frombuffer(g, "<f4"), high_first=True))
    assert expect == source, "the Lens slice decodes to OpenAI's MXFP4 original"
    q = torch.frombuffer(bytearray(w), dtype=torch.uint8).reshape(E, shape[1], N // 2)
    t = torch.frombuffer(bytearray(s), dtype=torch.uint8).reshape(E, 128, 4).view(torch.float8_e4m3fn)
    gt = torch.frombuffer(bytearray(g), dtype=torch.float32)
    k = torch.stack([dequantize_nvfp4(q[e], gt[e], t[e], torch.bfloat16) for e in range(E)])
    print(f"comfy-experts: against comfy_kitchen per expert: {differ('BF16', expect, raw(k))}; "
          "equal to the MXFP4 original")
    parts = [part("weight", "F4", shape, w), part("scale", "F8_E4M3", [E, 128, 4], s), part("scale_2", "F32", [E], g)]
    greens.append(case("comfy-experts-3x64x64-bf16", "BF16", shape, COMFY, parts, expect))
    # ModelOpt's layout, global scales divided (their reciprocals are exact: powers of two), input_scale per expert.
    inv = (np.float32(1) / np.frombuffer(g, "<f4")).astype("<f4")
    low = ((np.frombuffer(w, np.uint8) << 4) | (np.frombuffer(w, np.uint8) >> 4)).astype(np.uint8)
    greens.append(case("experts-divide-3x64x64-bf16", "BF16", shape, dict(PARAMS, global_scale="divide"), [
        part("weight", "F4", shape, low.tobytes()), part("scale", "F8_E4M3", [E, shape[1], 4], rowmajor.tobytes()),
        part("scale_2", "F32", [E], inv.tobytes()), part("input_scale", "F32", [E], np.full(E, .5, "<f4").tobytes())],
        expect))
    swap = lambda role, p: [p if x["role"] == role else x for x in parts]  # noqa: E731
    reds.append(case("experts-scale2-count", "BF16", shape, COMFY, swap("scale_2", part("scale_2", "F32", [2], g[:8]))))
    reds.append(case("experts-cublas-flat", "BF16", shape, COMFY,
                     swap("scale", part("scale", "F8_E4M3", [256, 4], s[:1024]))))


# ------------------------------------------------------------------ fixtures


def write_safetensors(path, tensors, metadata=None):
    head, data = ({"__metadata__": metadata} if metadata else {}), bytearray()
    for key, dtype, shape, b in tensors:
        head[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(text).to_bytes(8, "little") + text + bytes(data))


def fixtures(awq_bf16, lens):
    out, quantized = awq_bf16
    case_dir = FIXTURES / "modelopt-nvfp4-awq"
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "hf_quant_config.json"):
        shutil.copy(out / name, case_dir / name)
    f = load_file(out / "model.safetensors")
    expect = []
    for key, w in quantized.items():
        # source: the weight the logical tensor stands for; quantized: what ModelOpt's RTN was given.
        m = key.removesuffix(".weight")
        p = f.get(m + ".pre_quant_scale")
        src = w.float() * (1 if p is None else p.float())
        expect += [(f"dequant.{key}", "BF16", w.shape, raw(modelopt_dequant(f, m).to(torch.bfloat16))),
                   (f"source.{key}", "BF16", w.shape, raw(src.to(torch.bfloat16)))]
        if p is not None:
            expect.append((f"quantized.{key}", "BF16", w.shape, raw(w)))
    write_safetensors(FIXTURES / "modelopt-nvfp4-awq.expect.safetensors", expect)

    w, s, g, bias, source = lens
    E, n = len(EXPERTS), ROWS[1] - ROWS[0]
    m = "layers.1.mlp.experts.down_proj"
    conf = json.dumps({"format": "nvfp4", "num_experts": E, "full_precision_matrix_mult": True}).encode()
    write_safetensors(FIXTURES / "comfy-nvfp4-experts" / "model.safetensors", [
        (f"{m}.bias", "BF16", [E, n], bias), (f"{m}.comfy_quant", "U8", [len(conf)], conf),
        (f"{m}.weight", "U8", [E, n, N // 2], w), (f"{m}.weight_scale", "F8_E4M3", [E, 128, 4], s),
        (f"{m}.weight_scale_2", "F32", [E], g)])
    q = torch.frombuffer(bytearray(w), dtype=torch.uint8).reshape(E, n, N // 2)
    t = torch.frombuffer(bytearray(s), dtype=torch.uint8).reshape(E, 128, 4).view(torch.float8_e4m3fn)
    gt = torch.frombuffer(bytearray(g), dtype=torch.float32)
    k = torch.stack([dequantize_nvfp4(q[e], gt[e], t[e], torch.bfloat16) for e in range(E)])
    write_safetensors(FIXTURES / "comfy-nvfp4-experts.expect.safetensors", [
        (f"dequant.{m}.weight", "BF16", [E, n, N], raw(k)), (f"source.{m}.weight", "BF16", [E, n, N], source)])


def main():
    greens, reds = [], []
    with tempfile.TemporaryDirectory() as tmp:
        awq_bf16 = modelopt_cases(Path(tmp), greens, reds)
        lens = lens_slice()
        expert_cases(lens, greens, reds)
        fixtures(awq_bf16, lens)
    mine = {c["key"] for c in greens + reds}
    kept = [c for c in json.loads(CASES.read_text())["cases"] if c["key"] not in mine]
    at = max(i for i, c in enumerate(kept) if c["component"] == "nvfp4") + 1
    cases = kept[:at] + greens + kept[at:] + reds  # make_vectors.py's one form
    CASES.write_text('{"cases":[\n' + ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":"))
                                                   for c in cases) + "\n]}\n")
    print(f"{len(greens)} cases and {len(reds)} refusals into {CASES.name}; fixtures in {FIXTURES.name}/")


if __name__ == "__main__":
    main()
