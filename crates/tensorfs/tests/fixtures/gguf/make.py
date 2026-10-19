"""GGUF files as gguf-py writes them, for `tests/gguf.rs`.

- `llama-moe.gguf`: a llama-arch Mixtral (2 experts, 4 heads, 2 KV heads) as llama.cpp's
  `convert_hf_to_gguf.py` lays it out: Q/K rows through `LlamaModel.permute`, experts stacked in
  order, ggml's NVFP4 stack with its per-expert `.scale` and `.input_scale`, `rope_freqs`, every KV
  value type and a 64-byte alignment. `llama-moe.expect.safetensors` holds each upstream tensor's
  stored bytes (`<key>` for a plain tensor, `<key>:<role>` for a part), computed here from the
  upstream rows the converter started from.
- `flux.gguf`: city96 ComfyUI-GGUF keys kept as they are; every tier-1 block type, an i-quant, the
  plain types, and a tensor reshaped to `[n/256, 256]` with `comfy.gguf.orig_shape`.

Block bytes come from gguf-py's quantizers where it has them, else random: these files test the
container, names and layouts, not decoding.

Producer: gguf-py from llama.cpp ff5a23376725e5a889fa0bfb987b7541dc304da0 (0.19.0 plus Q2_0):
  uv run --python 3.12 --with numpy \
    --with 'gguf @ git+https://github.com/ggml-org/llama.cpp@ff5a23376725e5a889fa0bfb987b7541dc304da0#subdirectory=gguf-py' \
    python -I make.py <this directory>
"""

import json
import sys
from pathlib import Path

import gguf
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as V

RNG = np.random.default_rng(49)


def permute(w, n_head, n_head_kv):
    """llama.cpp conversion/llama.py `LlamaModel.permute`, verbatim in numpy."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def blocks(qtype, rows, cols):
    """Rows of `cols` elements in `qtype`, as uint8 rows of block structs."""
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    x = RNG.standard_normal((rows, cols)).astype(np.float32)
    try:
        return gguf.quants.quantize(x, qtype)
    except (NotImplementedError, gguf.QuantError):
        return RNG.integers(0, 256, (rows, cols // block * size), dtype=np.uint8)


def safetensors(path, tensors):
    head, data = {}, b""
    for key, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        head[key] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)


def llama_moe(out: Path):
    w = gguf.GGUFWriter(str(out / "llama-moe.gguf"), "llama")
    w.add_custom_alignment(64)
    w.add_block_count(1)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_expert_count(2)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_key_value("test.u8", 200, V.UINT8)
    w.add_key_value("test.i8", -100, V.INT8)
    w.add_key_value("test.u16", 60000, V.UINT16)
    w.add_key_value("test.i16", -30000, V.INT16)
    w.add_key_value("test.i32", -2_000_000_000, V.INT32)
    w.add_key_value("test.f32", 0.1, V.FLOAT32)
    w.add_key_value("test.nan", float("nan"), V.FLOAT32)
    w.add_key_value("test.bool", True, V.BOOL)
    w.add_key_value("test.u64", 2**63 + 5, V.UINT64)
    w.add_key_value("test.i64", -(2**62), V.INT64)
    w.add_key_value("test.f64", 1 / 3, V.FLOAT64)
    w.add_key_value("test.nested", [[1, 2], [3]], V.ARRAY)
    w.add_token_list(["<unk>", "<s>", "é", "日本"])
    w.add_token_scores([0.0, -1.5, 2.25, 1e-30])
    expect = {}

    def add(name, arr, qtype=None):
        if qtype is None:
            w.add_tensor(name, arr)
        else:
            w.add_tensor(name, arr, raw_dtype=qtype)

    add("token_embd.weight", blocks(Q.Q8_0, 32, 64), Q.Q8_0)
    add("output_norm.weight", RNG.standard_normal(64).astype(np.float32))
    add("output.weight", blocks(Q.Q4_0, 32, 64), Q.Q4_0)
    add("rope_freqs.weight", RNG.standard_normal(8).astype(np.float32))
    add("blk.0.attn_norm.weight", RNG.standard_normal(64).astype(np.float32))
    # Q and K: upstream rows, then the converter's permutation; quantized after, as it does.
    q_up = RNG.standard_normal((64, 64)).astype(np.float32)
    k_up = RNG.standard_normal((32, 64)).astype(np.float32)
    add("blk.0.attn_q.weight", gguf.quants.quantize(permute(q_up, 4, 4), Q.Q8_0), Q.Q8_0)
    add("blk.0.attn_k.weight", gguf.quants.quantize(permute(k_up, 4, 2), Q.Q4_1), Q.Q4_1)
    expect["model.layers.0.self_attn.q_proj.weight:blocks"] = gguf.quants.quantize(q_up, Q.Q8_0)
    expect["model.layers.0.self_attn.k_proj.weight:blocks"] = gguf.quants.quantize(k_up, Q.Q4_1)
    v = RNG.standard_normal((32, 64)).astype(np.float16)
    add("blk.0.attn_v.weight", v)
    expect["model.layers.0.self_attn.v_proj.weight"] = v
    add("blk.0.attn_output.weight", gguf.quants.quantize(RNG.standard_normal((64, 64)).astype(np.float32), Q.BF16), Q.BF16)
    add("blk.0.ffn_norm.weight", RNG.standard_normal(64).astype(np.float32))
    add("blk.0.ffn_gate_inp.weight", RNG.standard_normal((2, 64)).astype(np.float32))
    for name, w_id, qtype, (n, k) in [
        ("ffn_gate_exps", "w1", Q.Q5_0, (32, 64)),
        ("ffn_down_exps", "w2", Q.Q5_1, (64, 32)),
        ("ffn_up_exps", "w3", Q.NVFP4, (32, 64)),
    ]:
        experts = [blocks(qtype, n, k) for _ in range(2)]
        add(f"blk.0.{name}.weight", np.stack(experts), qtype)
        for e, x in enumerate(experts):
            expect[f"model.layers.0.block_sparse_moe.experts.{e}.{w_id}.weight:blocks"] = x
    scales = np.array([0.5, 3.0], np.float32)
    inputs = np.array([1.25, 7.0], np.float32)
    add("blk.0.ffn_up_exps.scale", scales)
    add("blk.0.ffn_up_exps.input_scale", inputs)
    for e in range(2):
        expect[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight:scale"] = scales[e : e + 1]
        expect[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight:input_scale"] = inputs[e : e + 1]
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    safetensors(out / "llama-moe.expect.safetensors", expect)


def flux(out: Path):
    w = gguf.GGUFWriter(str(out / "flux.gguf"), "flux")
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_K_S)
    cols = {Q.Q4_0: 32, Q.Q4_1: 32, Q.Q5_0: 32, Q.Q5_1: 32, Q.Q8_0: 64, Q.MXFP4: 64, Q.NVFP4: 64, Q.Q1_0: 128, Q.Q2_0: 64, Q.IQ4_NL: 32}
    for qtype in [Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K, Q.MXFP4, Q.NVFP4, Q.Q1_0, Q.Q2_0, Q.IQ4_NL]:
        w.add_tensor(f"double_blocks.0.{qtype.name.lower()}.weight", blocks(qtype, 3, cols.get(qtype, 256)), raw_dtype=qtype)
    w.add_tensor("double_blocks.0.img_attn.qkv.bias", RNG.standard_normal(6).astype(np.float32))
    w.add_tensor("img_in.weight", RNG.standard_normal((4, 8)).astype(np.float16))
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.float64]:
        w.add_tensor(f"plain.{np.dtype(dtype).name}", RNG.integers(-100, 100, (2, 3)).astype(dtype))
    # ComfyUI-GGUF's shape fix: a conv weight whose last extent is not whole blocks.
    conv = RNG.standard_normal((8, 4, 4, 4)).astype(np.float32)
    w.add_tensor("input_blocks.0.0.weight", gguf.quants.quantize(conv.reshape(2, 256), Q.Q8_0), raw_dtype=Q.Q8_0)
    w.add_array("comfy.gguf.orig_shape.input_blocks.0.0.weight", [8, 4, 4, 4])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    llama_moe(out)
    flux(out)
