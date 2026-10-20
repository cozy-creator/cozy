# cozytensors/1 conformance

These executable fixtures accompany the current checked-in reader. The TensorFS 0.13.0
[target specification](../spec/cozytensors.md), [schemas](../spec/schema/) and
[readable examples](../spec/examples/) are ahead of that implementation while it is integrated.

For anyone implementing [cozytensors/1](../spec/cozytensors.md).

| Path | What |
|---|---|
| `schema/header.cddl` | The header's CBOR structure (RFC 8610). |
| `schema/{manifest,repo}.schema.json` | JSON Schema (2020-12) of the manifest and repo. Byte lengths, bounds, duplicate keys, integer-only numbers and version uniqueness are spec rules the schema cannot express. |
| `vectors/cases/<family>.json` | Decode cases of one scheme family (`core`: plain tensors, spans, an unknown scheme), in header order: a tensor's parts (base64 bits, in order) and its expected logical bytes, or the refusal code (`decode`, `unknown_scheme`). |
| `vectors/store/` | A static origin holding those cases: one repo per family, `Conformance/<family>` at `refs/conformance/<family>.json`, lane `all` (every case, laid out by the chunk rule). `core` adds lane `spans` (a range across three chunks) and a yanked release 2 that swaps its lanes. |
| `vectors/cores.json` | Cases of the decode building blocks (spec §6): affine and table over the producers' packings, and gguf-py's own q4_0/q8_0 block structs. |
| `vectors/green/header-*.cbor` | Headers a reader must accept, each with its projection `header-*.json`: the CBOR tree as JSON, byte strings as `{"hex": ...}`. |
| `vectors/red/header-*.cbor` | Headers a reader must refuse: twins of `header-basic` in bytes other than its one encoding (non-preferred heads, indefinite lengths, unordered keys, encodings not in first-use order), forbidden items, bad tags and structure. `header-pre-cut*` (a real pre-cut header, and the provisional JSON) must be refused as pre-cut. |
| `vectors/{green,red}/{manifest,repo}-*.json` | Manifests and repos to accept or refuse. |
| `reader.py` | An independent reader and header encoder (Python, numpy, cbor2); its decoders are `schemes/`, one module per family, registered in `schemes/__init__.py`. |
| `make_vectors.py` | Writes the header vectors from the projections and defects it lists, and checks and normalizes every `cases/<family>.json`. |
| `mine/` | Producer miners: `mine/cores.py` writes `cores.json`; a family's miner writes its cases (`mine/gguf.py`: ggml's own C, fetched at llama.cpp ff5a233 and built here). |

A conforming implementation:
- reads the store, verifies every object, and decodes every case to its expected bytes or refuses it with its code;
- decodes every `cores.json` case to its expected values;
- reads every green header as its projection, and as a writer encodes each projection to the green bytes (the ones
  with a minor tag or later fields are read-only); refuses every red document;
- as a writer, reproduces the store byte for byte from the case files, keeping their order: components by first
  appearance, tensors and parts as listed (construction order is data, spec §4).

Run both implementations:

```bash
uv run --no-project --python 3.12 --with numpy --with cbor2==6.1.4 python -I conformance/reader.py conformance/vectors
cargo test -p tensorfs --no-default-features --test format_conformance --test format_docs --test format_cores
```

After a deliberate change: `make_vectors.py` rewrites the header vectors and normalizes the case files, and
`cargo test ... --test format_conformance -- --ignored regenerate` rewrites `vectors/store/`. A new family adds only
its own files: `cases/<family>.json`, its repo and its objects in the store.

**Provenance.** The `fp8.rowwise`, `fp8.tensorwise`, `int8.convrot`, `nvfp4` and `mxfp8` (4×64) cases are
producer-mined (ComfyUI's fp8_scaled quantizer, cozy-runtime's int8 ConvRot, NVFP4 and MXFP8 producers);
cozy-runtime's decoders reproduce them bit for bit, except `fp8.tensorwise` at `BF16`/`F16`: ComfyUI rounds the scale
to the logical dtype first, the spec never rounds a stored scale, so those expected bytes are the spec's decode of the
mined parts (within one step of ComfyUI's). The ComfyUI-layout `nvfp4` and `mxfp8` cases and the expert stacks
hold the same values as their mined twins in the other nibble order, scale tiling or shape. The other cases
(per-column scales, `F8_E5M2` weights, `BF16`/`F16` tensorwise, rowwise and convrot scales, `int8.tensorwise`, `int8.rowwise`, a ragged `mxfp8` block, a divided global
scale, rounding of 3,072 f32 patterns to BF16/F16, plain and unknown dtypes, refusals) were computed exactly from the
spec's formulas.

`mine/nvfp4.py` mines the `nvfp4` `modelopt-*`, `comfy-*` and `experts-*` cases: parts from ModelOpt 0.47.0's own
NVFP4 and NVFP4_AWQ export (`pre_quant_scale` in BF16, F16 and F32), and a range-read slice of ComfyUI's fused-expert
file `Comfy-Org/Lens` (`weight_scale_2 [E]`, tiles per expert). Expected bytes are the spec's decode: ModelOpt's own
dequantize gives the same values at BF16 and F16 (its table has no −0) and differs by at most 2 steps at F32
(it multiplies `s × g` first); comfy-kitchen's BF16 arithmetic differs by one step. The Lens slice decodes bit for bit
to OpenAI's MXFP4 original of the same weights. `mine/mxfp8.py` mines the stacked `mxfp8` cases (`[3, 80, 96]`, so
`cublas_128x4` pads each expert's 80 rows to 128) with comfy-kitchen's own `quantize_mxfp8` per expert, as ComfyUI's
`MoEExperts` stores them; kitchen's dequantize gives the same bytes. No published ComfyUI mxfp8 file holds an expert
stack.

`fp8_blockwise` and `mxfp4` are mined by `mine/<family>.py` from transformers' FineGrainedFP8, compressed-tensors,
gpt-oss's triton_kernels and torchao on CPU, each producer's own dequantize giving the expected bytes, plus rows of
real checkpoints read by HTTP range at pinned revisions (DeepSeek-V3.1 and V4, Qwen3, RedHatAI, gpt-oss, AMD Quark,
MLX, compressed-tensors). A case with `source` holds the `F32` values its producer quantized; TensorFS's encoder
reproduces its parts bit for bit (`tests/format_encode.rs`).

The `gguf` cases are ggml's own C (`quantize_row_<type>_ref`, `dequantize_row_<type>` from llama.cpp ff5a233's
`ggml-quants.c`, gcc for baseline x86-64 with `-ffp-contract=off`); gguf-py's numpy dequantizers (same commit) agree
on every F32 case.

`gptq`, `awq` and `ct` are mined by `mine/<family>.py` on CPU from GPTQModel 7.6.0, AutoGPTQ 0.7.1 and AutoAWQ 0.2.9
(their own files from the PyPI sdists), and compressed-tensors 0.16.0 and 0.19.0 (both sides of its packing change),
each producer's own dequantizer giving the expected bytes, plus 8-row bands of real Qwen GPTQ and AWQ and RedHatAI
w4a16 checkpoints read by HTTP range. A case with `pack` holds the codes and zeros its producer packed: TensorFS's
`pack` reproduces its parts bit for bit (`tests/format_pack.rs`).

`bnb` is mined by `mine/bnb.py` from bitsandbytes 0.50.2's CPU backend: Linear4bit and Linear8bitLt state dicts, and
bitsandbytes' own dequantize as the expected bytes, plus rows of unsloth's and lllyasviel's nf4 checkpoints read by HTTP
range at pinned revisions. bitsandbytes' fallback for an odd last extent reads FP4 code 8 as +0 where its kernels (and
the spec) read −0; the odd-extent FP4 cases hold no code 8.

**comfy.** `mine/comfy.py` runs comfy-kitchen 0.2.39's eager quantizers (PyTorch 2.14.1 CPU) and range-reads rows of
real ComfyUI files (AX1Y2JP W4A8, Comfy-Org W6A8, Merserk W4A4 of MiniMax-H3). Expected bytes are the spec's decode;
kitchen's own dequantize is within 2.5·10^−3 relative at `BF16` (it rotates back in `L`) and f32 rounding at `F32`.
Cases kitchen quantized with its defaults carry `source`; those with other options carry `kitchen` (source and
options), which `tests/encode_comfy.rs` reproduces.
