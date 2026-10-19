# cozytensors/1

Status: normative, version 1.1. Start with the [introduction and usage guide](../README.md).
Conformance vectors and an independent reader: [`../conformance/`](../conformance/).

This is the authoritative specification of the stored format and its decode semantics. A reader can implement it
without TensorFS, Cozy or Tensorhub. Library APIs, cache policy and GPU execution are outside the format contract.

cozytensors/1 stores model weights as immutable, content-addressed objects. A **checkpoint** is a manifest (a file
tree) whose `model.cozytensors` file is a **header**; the header names **chunks** (objects of tensor bytes) and
describes every tensor in them, plain or encoded, by component. A **repo** document maps a model's releases and lanes
to checkpoints. Everything a reader needs to read and decode a tensor is in these documents and this spec.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. §1–§8 define meaning; §9 defines the byte encoding
of documents; §10 is informative writer guidance. Appendix A holds the schemes' fixed tables.

## 1. Objects

An object is an immutable byte string named by an **object ref** `{sha256, length}`: `sha256` is the SHA-256 of the
bytes as 64 lowercase hex digits, `length` their count. In a store or static origin the object lives at

```
blobs/<sha256[0:2]>/<sha256[2:4]>/<sha256>
```

A reader MUST check the length and digest of every object before using its bytes.

## 2. Documents

Manifests, headers and repos are documents. Each carries a `format` tag `<kind>/<major>` or `<kind>/<major>.<minor>`,
where major and minor are decimal digits:

| Document | `format` kind | Encoding (§9) | Usual name |
|---|---|---|---|
| manifest | `tensorfs.manifest` | JSON object | an object, cited by a repo |
| header | `cozytensors` | CBOR array | the manifest file `model.cozytensors` |
| repo | `tensorfs.repo` | JSON object | `refs/<org>/<name>.json` |

- A reader MUST refuse a document whose kind differs or whose major is not `1`. It MUST accept any minor.
- A reader MUST ignore what a later minor adds: unknown fields of a JSON document, at every level, and trailing
  elements of the header's records (§4). Adding them never needs a new major.
- Values are booleans, integers in [−2^63, 2^64), strings, lists and maps with string keys. There are no null values
  and no floating-point numbers anywhere in a document. A map MUST NOT repeat a key.
- `metadata`, where allowed, is a free-form map of such values.

**Names.** A *segment* is a non-empty string that is not `.` or `..` and contains no `/`, `\` or NUL. A *path* is one
or more segments joined by `/`.

## 3. Manifest

```
{format: "tensorfs.manifest/1", files: {path: object-ref}, metadata?}
```

- Every key of `files` MUST be a path. The tree is an ordinary model folder: `model.cozytensors`,
  `<component>/config.json`, tokenizer files, a license.
- `model.cozytensors`, when present, is the checkpoint's header. A manifest without it holds no tensors.

## 4. Header

A header is an array of records, each an array whose fields are known by position (CDDL:
[`../conformance/schema/header.cddl`](../conformance/schema/header.cddl)):

```
header    = [format, chunks: [* chunk], encodings: [* encoding], components: [* component], metadata]
chunk     = [sha256: bytes(32), length]
encoding  = [scheme, params: {* name => value}]
component = [name, tensors: [* tensor]]
tensor    = [key, dtype, shape, 0, location]                 ; plain
          / [key, dtype, shape, encoding_index, [+ part]]    ; encoded, index ≥ 1
part      = [role, dtype, shape, location]
location  = [chunk_index, offset, length]
```

- **Construction order is data.** `components`, each component's `tensors` and each encoded tensor's parts are
  arrays in construction order: the order in which the model builds and fills them (its `state_dict()` order). The
  order is part of the header's bytes, so reordering changes its digest. A writer MUST take the order from its
  producer (a converter's rows, a base checkpoint, a source's constructor order) and MUST NOT sort names. Readers
  iterate in this order and look names up through an index.
- **Later fields.** A record MAY have more elements than listed; a reader MUST skip them (they are a later minor's).
  A reader MUST refuse a record with fewer.
- `chunks` lists the chunk objects (§1) by their 32-byte SHA-256 and length.
- `encodings` lists each distinct encoding (§6) once, in order of first citation by the tensors; every entry MUST be
  cited. A tensor's `encoding_index` `i ≥ 1` cites `encodings[i − 1]`; `0` marks a plain tensor.
- Component names MUST be segments (they name folders on export) and unique. Tensor keys MUST be non-empty and unique
  within their component; they are the state-dict names within the component. Roles MUST be non-empty and unique
  within their tensor.
- A **tensor** has a key, a logical dtype (§5) and shape (non-negative integers; `[]` is a scalar). A plain tensor's
  bytes are at its location. An encoded tensor's parts, each with its own dtype, shape and location, become the
  tensor through its encoding.
- A **location** `[chunk_index, offset, length]` names bytes in the chunks. The range starts at `offset` in
  `chunks[chunk_index]` and, when `length` exceeds the rest of the chunk, continues at offset 0 of the following chunks
  in list order. For every location with `length > 0`, a reader MUST refuse the header unless `chunk_index` is in
  range, `offset` is a multiple of 4096, `offset < chunks[chunk_index].length`, and `offset + length` is at most the
  summed lengths of `chunks[chunk_index..]`. A zero-length location reads no bytes (writers emit `[0, 0, 0]`).
- When the dtype is known, `length` MUST equal the byte size of the dtype and shape (§5).
- Locations MAY overlap: parts of different tensors may cite the same bytes (a shared codebook, a shared factor).

Tensor bytes are row-major (C order) and little-endian. Writers fill gaps between ranges with zeros; readers never
read them.

### Canonical names

- Component folders, layer names and tensor keys are the upstream on-disk names: the component's Diffusers or
  Transformers folder (`unet`, `text_encoder`, `transformer`, `vae`) and the keys of its upstream `*.safetensors`
  state dict, unchanged.
- LoRAs are stored under the upstream module names they adapt, with PEFT's `lora_A` and `lora_B` keys.
- A writer MUST NOT deviate from these names unless the deviation is listed below with its written, measured reason.

| Model | Upstream name | Stored name | Measured reason |
|---|---|---|---|
| — | — | — | No deviations. |

Per-head QKV layouts and the H3 gate/value order are the upstream layout (measured bit-identical, #508c, tfs-094),
so they are not deviations.

## 5. Dtypes

The known dtypes are the 22 safetensors 0.8 names:

| Bits | Dtypes |
|---|---|
| 4 | `F4` (E2M1) |
| 6 | `F6_E2M3`, `F6_E3M2` |
| 8 | `BOOL`, `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ` |
| 16 | `I16`, `U16`, `F16`, `BF16` |
| 32 | `I32`, `U32`, `F32` |
| 64 | `C64`, `F64`, `I64`, `U64` |

- The byte size of a known dtype and shape is `bits × Π shape / 8`. A reader MUST refuse a tensor or part whose
  element bits are not a multiple of 8, or whose product overflows 64 bits at any step (`bits`, then each extent in
  order).
- Sub-byte elements pack from the low bits: `F4` element 2k is bits 0–3 of byte k and element 2k+1 bits 4–7; `F6`
  elements fill bits continuously from bit 0 of byte 0.
- `BOOL` is one byte, 0 or 1. `C64` is two `F32` (real, imaginary).
- Dtypes are these names everywhere in a header. A dtype is any non-empty name; a name outside this table is a newer
  dtype: a reader still reads the document and checks bounds, takes the `length` as written, and refuses only a
  typed view of the bytes or a decode that needs it.

Element formats used by the schemes:

| Format | Meaning of byte (or nibble) `b` |
|---|---|
| `F8_E4M3` | sign bit 7, exponent bits 3–6 (bias 7), mantissa bits 0–2. `e = 0`: `m × 2^−9`; otherwise `(1 + m/8) × 2^(e−7)`. `S.1111.111` is NaN; no infinities. |
| `F8_E5M2` | sign bit 7, exponent bits 2–6 (bias 15), mantissa bits 0–1. `e = 0`: `m × 2^−16`; `e = 31`: infinity if `m = 0`, else NaN; otherwise `(1 + m/4) × 2^(e−15)`. |
| `F4` (E2M1) | sign bit 3; bits 0–2 index `0, 0.5, 1, 1.5, 2, 3, 4, 6`. |
| `F8_E8M0` | `2^(b−127)` for `b < 255`; `255` is NaN. |
| `I8` | two's complement. |
| UE4M3 (ggml, inside structs) | bits 3–6 exponent (bias 7), bits 0–2 mantissa, bit 7 ignored; `0x7F` is zero and there is no NaN. `e = 0`: `m × 2^−9`; otherwise `(1 + m/8) × 2^(e−7)`. |

## 6. Encodings

An encoded tensor cites an **encoding** `[scheme, params]`. With this section they fully define how the parts become
the logical tensor; no decode depends on anything outside the header, its chunks and this spec.

**Names** are `family.variant` in lowercase, in the words of the ecosystem that made the format: generic scaled
formats are `element.granularity` (`fp8.tensorwise`, `int8.rowwise`, `fp8.blockwise`), a producer's own format
is `producer.type` with the producer's type name verbatim (`gguf.q4_k`, `gptq.int4`, `awq.int4`, `ct.int4`; later
`bnb.nf4`), and a vendor-neutral standard keeps its bare name (`mxfp4`, `mxfp8`, `nvfp4`).

**Params** are snake_case keys in the producer's vocabulary (`axis`, `block_size`, `rotation`, `rotation_size`,
`global_scale`, `nibble_order`, `scale_layout`, `scale_dtype`). Every param the decode reads is required and none has a
default, so the encoding shows the whole recipe. A scale whose dtype the scheme leaves free carries `scale_dtype`:
`F32`, `BF16` or `F16` (and `F8_E8M0` for `fp8.blockwise`), as the producer stored it (the fixed scales of `mxfp4`,
`mxfp8` and `nvfp4` do not). Where a param
repeats a part's dtype (`scale_dtype`), a reader MUST refuse a mismatch.

**Four rules** bring any format in without new grammar:
1. *Keep the producer's bytes.* Parts are the producer's packed bytes (int words, nibbles, block structs, swizzled
   scales); the scheme fixes the bit order and params name what varies, so import and export are copies.
2. *Fixed codebooks are spec tables.* A constant of a scheme (a lookup table, a lattice grid) is given in an appendix
   of this spec with its SHA-256; a learned codebook is a part.
3. *Extra terms are parts.* Low-rank factors, per-channel factors (`pre_quant_scale`), permutations (`g_idx`) and
   float constants are named parts that the scheme's formula uses.
4. *Nesting only for sparsity.* A sparsity scheme MAY carry an inner `[scheme, params]` in its params; nothing else
   nests.

**Reading and decoding.**
- A tensor with an unknown scheme still reads: its parts are located and verified like any other. Only decoding
  refuses, and the refusal MUST name the scheme.
- A decoder MUST refuse a known scheme when a required part is missing, a part not listed for the scheme is present,
  a param is missing, unknown or has a value not listed, a part's dtype or shape breaks the scheme's rule, or the
  logical dtype is not `F32`, `BF16` or `F16`. Params drive the decode, so an unknown one is a refusal, never ignored.

**Shapes.** The tensorwise schemes take any logical shape, as one row of every element (`R = 1`). The others take
rank 2 or more and view the tensor as `[R, C]`: `C` is the last extent and `R` the product of the others, so an expert
stack `[E, N, K]` is `R = E·N` rows of `K`. `i` indexes rows and `j` columns; block schemes act along `j`. The
ecosystem's `[N, K]` weight is `R = N` output rows of `K` inputs.

**Arithmetic.** Each formula is evaluated element by element in IEEE-754 binary32 (`f32`): every `+`, `−`, `×` and `÷`
is one f32 operation rounded to nearest even, applied left to right as written; element formats (§5), codes and stored
scales convert to f32 exactly, and no scale is rounded before use. The result is then rounded once to the logical
dtype `L`, written `round_L`:
- `F32`: unchanged; `BF16`, `F16`: round to nearest, ties to even, keeping subnormals; `F16` overflows to ±infinity.
- Any NaN result is written as the canonical quiet NaN: `0x7FC00000` (`F32`), `0x7FC0` (`BF16`), `0x7E00` (`F16`).

This arithmetic defines the decode, not any producer's kernel. A producer's own dequantization may round elsewhere
(comfy-kitchen multiplies in `BF16`, for example) and differ from it by a step of `L`; that is within the determinism
the format promises, and readers follow this spec.

`input_scale` (optional where listed, `F32`, one element; `nvfp4` also allows one per matrix) is the activation scale
of a quantized matmul; it never enters a decode. It applies as `global_scale` says for `nvfp4`, and multiplies
elsewhere.

Each scheme below lists its params (with their allowed values), its parts (role, dtypes, stored shape) and its decode,
built from the building blocks that follow.

### Building blocks

**Scale grids.** A part on a grid (a scale, a zero or a bias) gives element `(i, j)` the value `s(i, j)`:

| Grid | Stored | `s(i, j)` |
|---|---|---|
| tensor | one element, any shape | `s[0]` |
| row | `R` elements, any shape | `s[i]` |
| column | `C` elements, any shape | `s[j]` |
| row blocks of `B` | `W = ⌈C/B⌉` per row, in a layout below | `s(i, b)`: `b = ⌊j/B⌋`, or `b = g_idx[j]` where the scheme has a `g_idx` |
| tiles of `Bᵣ × B_c` | `[lead…, ⌈N/Bᵣ⌉, ⌈C/B_c⌉]`: the logical shape with its last two extents, `N` and `C`, counted in tiles | `s[m, ⌊r/Bᵣ⌋, ⌊j/B_c⌋]`, where row `i = m·N + r` is row `r` of matrix `m` |

The last block of a row, and the last tile of a matrix's rows or columns, may be short. A `g_idx` holds `C` values, each below `W`. Row blocks are stored:
- `row_major`: the logical shape with its last extent replaced by `W` (`[R, W]` for a matrix); `s(i, b)` is element
  `i·W + b`.
- `transposed`: `[W, R]`; `s(i, b)` is element `b·R + i` (GPTQ's and AWQ's `[K/g, N]`).
- `cublas_128x4`: cuBLAS's block-scaling layout, as ComfyUI stores it. Each matrix `[N, C]` of the logical shape
  `[…stack, N, C]` (the whole tensor at rank 2) is tiled on its own: the part is
  `[…stack, 128·⌈N/128⌉, 4·⌈W/4⌉]`, zero-padded, in tiles of 128 rows × 4 blocks. Row `i` is row `r = i mod N` of
  matrix `m = ⌊i/N⌋`, and `s(i, b)` is element
  `m·128·⌈N/128⌉·4·⌈W/4⌉ + 512·(⌊r/128⌋·⌈W/4⌉ + ⌊b/4⌋) + 16·(r mod 32) + 4·⌊(r mod 128)/32⌋ + (b mod 4)`.

**Bit packing.** A packed part holds the codes `q[i, j]` of the `[R, C]` view, stored as `[R, C]` or, where the
scheme says *transposed*, as `[C, R]`. Along one axis of that stored array, the *pack axis*, each lane of `n` codes
is held in words of `w` bits: `w` is 8, 16 or 32 (the part's dtype `U8`/`I8`, `U16`/`I16` or `U32`/`I32`),
little-endian, its bits read unsigned. The part's shape is the stored array's with the pack axis counted in words
(a view stored as is and packed along its last axis keeps the logical leading extents). With `b` bits per code
(1 ≤ `b` ≤ 8) and `p = ⌊w/b⌋` codes to a word, the *fill* places code `t` of a lane:
- `word`: in word `⌊t/p⌋`, slot `σ(t mod p)`; the lane takes `⌈n/p⌉` words.
- `planar`: in word `t mod m`, slot `σ(⌊t/m⌋)`, where `m = n/p` words (`n` MUST be a multiple of `p`).
- `dense`: the lane's words form one bit stream, word `k` holding stream bits `k·w … k·w + w − 1` (bit 0 first); code
  `t` is stream bits `t·b … t·b + b − 1`, crossing words; the lane takes `⌈n·b/w⌉` words.

Slot `s` is bits `s·b … s·b + b − 1` of its word. Bits in no used slot are ignored by readers and zero from writers.
The *order* `σ` takes a code's index within its word to its slot: `low_first` `σ(e) = e`; `high_first`
`σ(e) = p − 1 − e`; or a permutation `π` listed by the scheme, slot `s` holding code `π[s]` (AWQ:
`[0, 2, 4, 6, 1, 3, 5, 7]`). A dense stream is always low first. A slot's bits `r` are the code: `q = r − o` for an
unsigned code with offset `o`, or `r` as a `b`-bit two's complement integer for a signed one.

**Affine.** `value(i, j) = (q − z(i, j)) × s(i, j)`, `q × s(i, j) + c(i, j)` or `q × s(i, j)`, where `q` is the packed
code and the scale `s`, zero `z` and bias `c` are parts on one grid. Codes and integer zeros convert to f32 exactly.

**Table.** `value(i, j) = T[q] × s(i, j)`: `T` is a fixed table (an element format's codes, or one in Appendix A) or a
part, indexed by the code; a code outside `T` refuses the decode.

**Block structs.** A `U8 [lead…, C/n, S]` part of fixed-size structs of `S` bytes, each holding `n` consecutive
elements of a row in order (`C` a multiple of `n`; the leading extents are the logical shape's). The scheme defines
each struct's little-endian fields and the formula of its `n` elements, using the blocks above within the struct.

**Transposed storage.** A scheme with the param `transposed` (`false` or `true`) whose value is `true` stores the
logical tensor `[lead…, A, B]` as `S = [lead…, B, A]`: its view `[R, C]`, its parts' shapes and its formula all
apply to `S`, and `value[lead…, a, b] = S[lead…, b, a]`. Its blocks therefore run along the logical axis −2. A
producer that keeps a weight as `[out, in]` where its framework's tensor is `[in, out]` (gpt-oss) is read under
the framework's key and shape. `mxfp4` has the param; ModelOpt's fused gpt-oss NVFP4 export (`[E, K/2, N]`) is
the next user, through `nvfp4`.

### `fp8.tensorwise`

- Params: `scale_dtype`.
- Parts: `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `scale_dtype`, one element. `input_scale`
  (optional).
- Decode, any shape: `round_L(weight × scale)`.

### `fp8.rowwise`

- Params: `axis`: `0` or `1`; `scale_dtype`.
- Parts: `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `scale_dtype`, on the row grid (`axis: 0`) or
  the column grid (`axis: 1`).
- Decode: `round_L(weight[i, j] × s(i, j))`.

### `int8.tensorwise`

- Params, parts and decode as `fp8.tensorwise`, with `weight` `I8`.

### `int8.rowwise`

- Params, parts and decode as `fp8.rowwise`, with `weight` `I8`.

### `int8.convrot`

- Params: `rotation: "hadamard"`; `rotation_size: B`, `B = 4^k`, `1 ≤ k ≤ 8` (256 in practice); `scale_dtype`.
- Parts: `weight`: `I8`, the logical shape; `C` MUST be a multiple of `B`. `scale`: `scale_dtype`, on the row grid.
- Decode: `value[i, gB + j] = round_L((y[i, g, j] × scale[i]) × 2^−k)`, with
  `y[i, g, ·] = weight[i, gB : gB + B] · H_B`.

`H_B = H₄ ⊗ … ⊗ H₄` (`k` factors) with entries ±1, `H₄` as Appendix A.1. `H_B` is symmetric and `H_B · H_B = B · I`, so
`2^−k · H_B` is the orthonormal rotation ComfyUI's ConvRot applies before quantizing each row. `y` is an exact integer
(|y| ≤ 128·B ≤ 2^23), so its conversion to f32 is exact. One way to compute it: view each group of `B` as `k` base-4
digits (most significant first) and apply `H₄` along every digit.

### `mxfp8`

The OCP Microscaling MXFP8 format: a power-of-two scale per 32 consecutive elements of a row (the last block of a row
may be short).
- Params: `block_size: 32`; `scale_layout`: `row_major` or `cublas_128x4` (ComfyUI's).
- Parts: `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `F8_E8M0` on row blocks of 32, stored
  `scale_layout`.
- Decode: `round_L(weight[i, j] × s(i, j))`.

### `nvfp4`

NVIDIA's NVFP4: an E2M1 element per weight, an E4M3 scale per 16 consecutive elements of a row, a global scale per
matrix, and optionally AWQ's per-column factor.
- Params: `block_size: 16`; `global_scale`: `multiply` or `divide`; `nibble_order`: `low_first` or `high_first`;
  `scale_layout`: `row_major` or `cublas_128x4`; `pre_quant_scale_dtype`: `F32`, `BF16` or `F16`, present exactly
  when the `pre_quant_scale` part is.
- Parts: `weight`: `F4`, the logical shape, `C` a multiple of 16; its bytes are 4-bit codes packed `word`-fill in `U8`
  words along the last axis, `nibble_order` being the order (`low_first` is §5's `F4` packing). `scale`: `F8_E4M3` on
  row blocks of 16, stored `scale_layout`. `scale_2`: `F32`, one element, or `M` elements, one per matrix of the
  logical shape `[…stack, N, C]` (`M = R/N`; any shape). `pre_quant_scale` (optional): `pre_quant_scale_dtype` on the
  column grid. `input_scale` (optional): one element or `M`.
- Decode, with `T` the 16 E2M1 values (§5), `g` the element of `scale_2` for row `i` (its only one, or element
  `⌊i/N⌋`) and `p` the factor `pre_quant_scale[j]`: `multiply`: `round_L(((T[q] × s(i, j)) × g) × p)`; `divide`:
  `round_L((T[q] × (s(i, j) ÷ g)) × p)`. Without a `pre_quant_scale`, `× p` is left out.

`global_scale` says how both global scales, `scale_2` and `input_scale`, apply: multiplied (ModelOpt's
`weight_scale_2` and `input_scale`, ComfyUI) or divided (compressed-tensors' `weight_global_scale` and
`input_global_scale`, each the reciprocal). ModelOpt's `weight` (`uint8 [R, C/2]`), `weight_scale`,
`weight_scale_2` and `pre_quant_scale` are `weight`, `scale`, `scale_2` and `pre_quant_scale` here; ModelOpt and
compressed-tensors write `low_first`, `row_major`; ComfyUI writes `high_first`, `cublas_128x4`.

`pre_quant_scale` is AWQ's per-input-channel factor (ModelOpt's `NVFP4_AWQ`, and ComfyUI files made from it). The
producer quantizes the smoothed weight `W · diag(1/p)` and applies `p` to the activations at run time (ComfyUI:
`input * pre_quant_scale`), so the logical weight is the dequantized one times `p` along its columns. ModelOpt stores
`p` in the model's dtype, and folds the factors it can into the layer before (a norm, `v_proj`, `up_proj`): those
tensors carry no part and decode to the weight the model computes with.

A stack's `scale_2` is per matrix where the producer quantized each matrix on its own: ComfyUI's fused experts
(`num_experts`) write `weight_scale_2 [E]` beside `[E, N, C]` weights, and tile each expert's scales. ModelOpt and
compressed-tensors write experts as separate matrices, each with one `scale_2`. ModelOpt's fused Llama-4 and gpt-oss
export (`[E, C/2, N]`, blocks along the second-to-last axis) is not this scheme.

### `fp8.blockwise`

FP8 elements and one scale per tile of each matrix (DeepSeek-V3 and later, Qwen3-FP8, Kimi-K2, MiniMax-M2, GLM-5-FP8;
compressed-tensors' block strategy).
- Params: `block_size: [Bᵣ, B_c]`, two positive integers (`[128, 128]` in practice); `scale_dtype`: `F32`, `BF16`,
  `F16` or `F8_E8M0`, as the producer stored the scale.
- Parts: `weight`: `F8_E4M3`, the logical shape. `scale`: `scale_dtype`, on tiles of `Bᵣ × B_c`.
- Decode: `round_L(weight[i, j] × s(i, j))`.

The scale multiplies. DeepSeek's and transformers' `weight_scale_inv` (the inverse of the factor the weight was
multiplied by when quantized; DeepSeek-V4 names it `scale`) and compressed-tensors' `weight_scale` are this `scale`.
Under `scale_fmt: ue8m0` every scale is a power of two, stored `F32` (DeepSeek-V3.1) or `F8_E8M0` (DeepSeek-V4).

### `mxfp4`

The OCP Microscaling MXFP4 format: an E2M1 element per weight and a power-of-two scale per 32 consecutive elements of
a row (the last block of a row may be short).
- Params: `block_size: 32`; `transposed`: `false` or `true` (transposed storage, above).
- Parts: `weight`: `F4`, the logical shape, or `S` when `transposed` (§5's packing). `scale`: `F8_E8M0` on row
  blocks of 32, `row_major`.
- Decode, with `T` the 16 E2M1 values (§5): `round_L(T[q] × s(i, j))`.

Every producer seen writes these bytes, the even element in the low nibble and the scales row-major: gpt-oss's
`<module>_blocks` (`U8 [E, N, K/32, 16]`, a row's blocks in order) and `<module>_scales`, read as transformers'
`<module>` `[E, K, N]` with `transposed: true`; compressed-tensors' `weight_packed` and `weight_scale`, AMD Quark's
`weight` and `weight_scale` (`pack_method: reorder`), MLX's `weight` (`U32` words, little-endian) and `scales`, and
DeepSeek-V4's FP4 experts (`weight` `I8`, `scale`).

### `gguf.*`

ggml's block types as llama.cpp writes them in a GGUF (`ggml-common.h`, and `dequantize_row_*` in `ggml-quants.c`,
llama.cpp `ff5a233`): one scheme per type, named `gguf.<type>` with ggml's type name in lowercase. ggml's plain types
(`f32`, `f16`, `bf16`, `i8`…`i64`, `f64`) are plain tensors.
- Params: `block_size`: the type's `n` below.
- Parts: `blocks`: `U8`, the type's structs of `S` bytes, as block structs `[lead…, C/n, S]` when `n` divides the last
  extent `C`, else `[N/n, S]` for a tensor of `N` elements (ComfyUI-GGUF stores a tensor whose last extent is not
  whole blocks reshaped). Either way struct `k` holds elements `k·n … k·n + n − 1` of the tensor in row-major order.
  `gguf.nvfp4` adds `scale` and `input_scale` (optional, `F32`, one element each): ggml's `<name>.scale` and
  `<name>.input_scale` tensors.
- Decode, any shape whose element count is a multiple of `n`: element `t` of a struct is `round_L(v)`, with `v` below
  evaluated in f32 in ggml's own order, as written. `d`, `m` and `dmin` are the struct's `F16` fields, `lo(x) = x & 15`,
  `hi(x) = x >> 4`, `bit(x, k) = (x >> k) & 1`, and `K` is Appendix A.2.

| Type | `n` | `S` | Fields (byte offset) | `v` for element `t` |
|---|---|---|---|---|
| `q4_0` | 32 | 18 | `d`@0, `qs[16]`@2 | `(q − 8) × d`, `q = lo(qs[t])` for `t < 16`, else `hi(qs[t − 16])` |
| `q4_1` | 32 | 20 | `d`@0, `m`@2, `qs[16]`@4 | `q × d + m`, `q` as `q4_0` |
| `q5_0` | 32 | 22 | `d`@0, `qh` U32@2, `qs[16]`@6 | `(q − 16) × d`, `q` = `q4_0`'s nibble `+ 16·bit(qh, t)` |
| `q5_1` | 32 | 24 | `d`@0, `m`@2, `qh` U32@4, `qs[16]`@8 | `q × d + m`, `q` as `q5_0` |
| `q8_0` | 32 | 34 | `d`@0, `qs` I8[32]@2 | `qs[t] × d` |
| `q2_k` | 256 | 84 | `scales[16]`@0, `qs[64]`@16, `d`@80, `dmin`@82 | `(d × lo(c)) × q − dmin × hi(c)`, `c = scales[b]`, `q = (qs[32h + 16·(b mod 2) + l] >> 2j) & 3` |
| `q3_k` | 256 | 110 | `hmask[32]`@0, `qs[64]`@32, `scales[12]`@96, `d`@108 | `(d × (c − 32)) × (q − 4 + 4·bit(hmask[16·(b mod 2) + l], ⌊t/32⌋))`, `q` as `q2_k`, `c = lo4 + 16·((scales[8 + b mod 4] >> 2·⌊b/4⌋) & 3)` with `lo4 = lo(scales[b])` for `b < 8`, else `hi(scales[b − 8])` |
| `q4_k` | 256 | 144 | `d`@0, `dmin`@2, `scales[12]`@4, `qs[128]`@16 | `(d × c) × q − dmin × e`, `q = lo` (`t mod 64 < 32`) or `hi` of `qs[32·⌊t/64⌋ + t mod 32]`, `(c, e)` the 6-bit pair `a` below |
| `q5_k` | 256 | 176 | `d`@0, `dmin`@2, `scales[12]`@4, `qh[32]`@16, `qs[128]`@48 | as `q4_k` with `q + 16·bit(qh[t mod 32], a)` |
| `q6_k` | 256 | 210 | `ql[128]`@0, `qh[64]`@128, `scales` I8[16]@192, `d`@208 | `(d × scales[8h + ⌊l/16⌋ + 2j]) × (q − 32)`, `q = lo` (`j` even) or `hi` of `ql[64h + 32·(j mod 2) + l]`, `+ 16·((qh[32h + l] >> 2j) & 3)` |
| `mxfp4` | 32 | 17 | `e` U8@0, `qs[16]`@1 | `K[q] × 2^(e − 128)`, `q` as `q4_0`; every `e` is a power of two (no NaN) |
| `nvfp4` | 64 | 36 | `d[4]` UE4M3@0, `qs[32]`@4 | `K[q] × (UE4M3(d[s]) × 0.5)`, `s = ⌊t/16⌋`, `q = lo` (`t mod 16 < 8`) or `hi` of `qs[8s + t mod 8]`; then `× scale` when present |
| `q1_0` | 128 | 18 | `d`@0, `qs[16]`@2 | `d` if `bit(qs[⌊t/8⌋], t mod 8)`, else `−d` |
| `q2_0` | 64 | 18 | `d`@0, `qs[16]`@2 | `(q − 1) × d`, `q = (qs[⌊t/4⌋] >> 2·(t mod 4)) & 3` |

In the k-types, `b = ⌊t/16⌋` (`q2_k`, `q3_k`) is the sub-block, `h = ⌊t/128⌋`, `j = ⌊(t mod 128)/32⌋`, and
`l = t mod 16` (`q2_k`, `q3_k`) or `t mod 32` (`q6_k`). `q4_k` and `q5_k` take sub-block
`a = ⌊t/32⌋`'s scale `c` and min `e` from `scales` (ggml's `get_scale_min_k4`): for `a < 4`, `c = scales[a] & 63`,
`e = scales[a + 4] & 63`; else `c = lo(scales[a + 4]) + 16·(scales[a − 4] >> 6)`, `e = hi(scales[a + 4]) +
16·(scales[a] >> 6)`.

Every product of a code with a scale is exact in f32 (`q6_k`'s second product aside), so only the add or subtract of
`q4_1`, `q5_1`, `q2_k`, `q4_k` and `q5_k`, `q6_k`'s second product and `nvfp4`'s `scale` round; a build of ggml that
fuses multiply-adds agrees. ggml's own encoder for each type is its `quantize_row_<type>_ref`; TensorFS's is a port,
bit-exact with it.

### `gptq.int2`, `gptq.int3`, `gptq.int4`, `gptq.int8`

GPTQ's checkpoint layout (AutoGPTQ, GPTQModel, and the GPTQ exports of AutoRound and Optimum): `b`-bit codes (`b` is
the variant's 2, 3, 4 or 8), a zero and a scale per group of `g` inputs. The logical weight is `[N, K]` (`R = N`,
`C = K`) and `W = ⌈K/g⌉`.
- Params: `group_size`: `g`, a positive integer, or `-1` for one group per row (`g = K`); `symmetric`: boolean;
  `act_order`: boolean; `zero_offset`: `0` or `1`.
- Parts:
  - `qweight`: `I32 [⌈K·b/32⌉, N]`: the codes, stored transposed and packed `dense` along K (for `b` = 2, 4 and 8
    the same bytes as `word`), unsigned.
  - `qzeros`: `I32 [W, ⌈N·b/32⌉]`: each group's stored zero `r` as `[W, N]`, packed `dense` along N, unsigned.
  - `scales`: `F16` or `BF16` on row blocks of `g`, `transposed` (`[W, N]`).
  - `g_idx`: `I32 [K]`: each input's block of the zeros' and scales' grid.
- Decode: `z = (r + zero_offset) mod 2^b` per slot; `round_L((q − z(i, j)) × s(i, j))`.
- `act_order: false` requires `g_idx[j] = ⌊j/g⌋`, and `true` takes the groups from `g_idx` (GPTQ's `desc_act`).
  `symmetric: true` requires every `z` to be `2^(b−1)`. A reader MUST refuse either mismatch.

`zero_offset` is the zero convention. GPTQ's original checkpoints (`checkpoint_format: gptq`: AutoGPTQ, and
GPTQModel's default export) store `z − 1`, so `zero_offset: 1`; GPTQModel's `gptq_v2` stores `z`, so `0`. A v1 zero
of 0 is stored `2^b − 1` and decodes to 0, as AutoGPTQ's and GPTQModel's offline dequantizers read it. GPTQModel's v1
export subtracts 1 from every slot in one 32-bit subtraction, so the slot after a zero of 0 borrows and decodes one
low (vector `v1-zero-borrow`).

### `awq.int4`

AutoAWQ's GEMM layout (`version: gemm`; GPTQModel's AWQ export writes the same bytes): 4-bit codes, a zero and a scale
per group of `g` inputs. The logical weight is `[N, K]` and `W = ⌈K/g⌉`.
- Params: `group_size`: `g`, a positive integer, or `-1` (`g = K`).
- Parts:
  - `qweight`: `I32 [K, ⌈N/8⌉]`: the codes, stored transposed and packed `word` along N with the order
    `[0, 2, 4, 6, 1, 3, 5, 7]` (slot `s` of word `t` holds column `8t + order[s]`), unsigned.
  - `qzeros`: `I32 [W, ⌈N/8⌉]`: each group's zero as `[W, N]`, packed the same way.
  - `scales`: `F16` or `BF16` on row blocks of `g`, `transposed` (`[W, N]`).
- Decode: `round_L((q − z(i, j)) × s(i, j))`.

AWQ's per-input activation scale is folded into the previous operation by its producers, so it is no part here.
AutoAWQ's GEMV and GEMV_FAST layouts are other schemes.

### `ct.int1` … `ct.int8`

compressed-tensors' `pack-quantized` format (llm-compressor's W4A16 and W8A16, Kimi-K2-Thinking): signed `b`-bit
codes (`b` is the variant's 1 to 8), and per group of `g` inputs a scale and, when asymmetric, a zero point. `N` is
the logical shape's second-last extent, a matrix's rows, and `W = ⌈C/g⌉`.
- Params: `group_size`: `g`, a positive integer; `strategy`: `group`, or `channel` with `g = C`; `symmetric`: boolean;
  `packing`: `dense` (compressed-tensors 0.17 and later) or `word` (before: `⌊32/b⌋` codes to a word). The two are the
  same bytes when `b` divides 32.
- Parts:
  - `weight_packed`: `I32`, the logical shape with C counted in words: the codes packed `packing` along C, each
    stored as `q + 2^(b−1)`.
  - `weight_scale`: `F32`, `BF16` or `F16` on row blocks of `g`, `row_major`.
  - `weight_zero_point`: `I32`, present exactly when `symmetric` is false: each matrix's zero points `[N, W]`, offset
    the same way and packed `packing` along N; its shape is the logical one before N, then N counted in words, then
    `W`.
  - `weight_shape`: `I32` or `I64 [rank]`: the logical shape; a reader MUST refuse another.
  - `weight_g_idx` (optional): `I32 [C]`: each column's block of the grid, from a group act order (`actorder:
    group`, which compressed-tensors wrote until 0.18).
- Decode: `round_L((q − zp(i, j)) × s(i, j))`, or `round_L(q × s(i, j))` when symmetric.

compressed-tensors groups the columns sorted by `weight_g_idx` in runs of `g`, which is the stated group only when
each group holds its `g` columns (the last, `C − (W − 1)·g`); a reader MUST refuse another. A `−1` anywhere in it is
compressed-tensors' unset index: the column order.

<!-- bnb.* -->

<!-- comfy.* -->

### Plain tensors

A plain tensor's bytes are its value; reading it is a copy, whatever its dtype.

### Adding a scheme

A new scheme is a minor version (§8). It ships:
1. a `### <scheme>` subsection here (params and their values, parts with the producer's role names, dtypes and shapes,
   the decode in the building blocks' terms), and any fixed table in Appendix A with its SHA-256;
2. a reference decoder in `tensorfs-core` (`format/decode/<family>.rs`, one row in `SCHEMES`) and in the
   independent reader (`conformance/schemes/<family>.py`, one row in `FAMILIES`);
3. conformance vectors mined from the real producer, `conformance/vectors/cases/<family>.json`, with refusals;
4. encoders (`format/encode/<family>.rs`, one row in `ENCODERS`) and an importer row for its native container.

## 7. Repo, store and origin

```
{format: "tensorfs.repo/1", name: "<org>/<name>",
 releases: [{version, lanes: {lane: object-ref}, yanked?, revision?}], checkpoints?: [object-ref]}
```

- `name` is two segments. Versions MUST be non-empty and unique; lane names MUST be non-empty. Each lane cites a
  manifest. Releases are listed in publication order, newest last.
- `yanked` (boolean, default `false`) withdraws a release from unversioned lookups. `revision` (non-negative integer,
  default 0) is the release's expected-revision counter: every change to the release increments it, and a writer
  that changes a release conditions the change on the revision it read.
- `checkpoints` (default empty) lists manifests uploaded but not yet in any release.
- **Resolution.** An unversioned lookup takes the newest release that is not yanked; an explicit version takes that
  release, yanked or not (a reader MAY warn). The lane is then looked up in that release.
- **Roots.** Every lane of every release, yanked included, and every checkpoint are roots: a store keeps their
  closures.
- Repo names are case-insensitive: `Acme/SDXL` and `acme/sdxl` are one repo. Keys and paths use the name with ASCII
  `A`–`Z` lowered (`refs/acme/sdxl.json`); the document's `name` keeps the writer's spelling.
- A **store** or **static origin** is a directory, bucket or HTTP prefix holding `refs/<org>/<name>.json` and
  `blobs/…` (§1). A reader resolves name → release → lane → manifest → header → chunks, verifying every object.

**Closure.** A checkpoint's closure is its manifest, every object its `files` cite, and, for `model.cozytensors`,
every object in the header's `chunks`. A copy of a checkpoint is complete when its whole closure is present. A
reader that needs only some components MAY fetch only the chunks their locations touch.

## 8. Versioning

- Minor versions add fields, trailing record elements, schemes and dtypes; their absence keeps the old meaning, and
  readers of the same major read them.
- A new scheme or dtype is a minor change: every location carries its `length`, so an old reader reads the document,
  locates and verifies every byte, and refuses only a decode or typed view of the tensors that use it. A new param of
  an existing scheme is too: an old decoder refuses the tensors whose params it does not know (§6).
- A change that alters the meaning of an existing field, scheme or param is a new major.

Minors: 1.1 adds `nvfp4`'s `pre_quant_scale` and a `scale_2` per matrix, and tiles each matrix of a stack on its own
in `cublas_128x4` (before any stacked file was written); the schemes `fp8.blockwise`, `mxfp4`, `gguf.*`, `gptq.int2`,
`gptq.int3`, `gptq.int4`, `gptq.int8`, `awq.int4` and `ct.int1` … `ct.int8`; and the `tiles` grid and transposed
storage.

## 9. Encoding

**Header: deterministic CBOR.** A header is CBOR (RFC 8949) in core deterministic encoding (§4.2.1), so equal headers
have equal bytes and equal object refs. A reader MUST refuse:
- anything but unsigned and negative integers, byte and text strings, arrays, maps with text keys, `true` and
  `false`: tags, floats, `null`, `undefined` and other simple values are refused;
- a head that is not in its shortest form, an indefinite length, map keys out of the bytewise order of their
  encodings or repeated, invalid UTF-8, and bytes after the root;
- a byte string anywhere but a chunk's SHA-256 (and later fields);
- an integer outside [−2^63, 2^64).

Decoding and re-encoding a header of this version reproduces its bytes. A header that fails to read and is a
**pre-cut** layout, the provisional JSON (first byte `{`) or a well-formed root of exactly five elements whose fifth is
an array, MUST be refused as such ("pre-cut cozytensors header (old layout); re-import or migrate"): the tag is the
same, so only the structure tells them apart.

**Manifest and repo: JSON.** UTF-8 JSON (RFC 8259) whose top level is an object; §2's values map to JSON in the obvious
way. A reader MUST refuse invalid UTF-8 or JSON, a top level that is not an object, a repeated key in any object, any
number with a fraction or exponent, `NaN` or `Infinity`, any integer outside [−2^63, 2^64), `null`, and a field of the
wrong JSON type. Whitespace between tokens is allowed. A writer MUST produce the canonical form:
- object keys sorted by their UTF-8 bytes; no whitespace;
- integers in shortest decimal (no `+`, no leading zeros, `-` only for negatives);
- strings with `"` and `\` escaped as `\"` and `\\`, the controls U+0008, U+0009, U+000A, U+000C and U+000D as `\b`,
  `\t`, `\n`, `\f`, `\r`, other controls below U+0020 as `\u00XX` (lowercase hex), and every other character as its
  UTF-8 bytes (Python: `json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`);
- optional fields at their default omitted: empty `metadata` and `checkpoints`, `yanked: false`, `revision: 0`.

## 10. Chunks (informative)

Readers need only locations. Writers SHOULD lay tensors out by this rule, which keeps objects few (16–64 MiB),
shares unchanged large tensors between lanes, and aligns every part for direct I/O:

1. **Order.** Parts in header order (components, their tensors, each tensor's parts); the layout never reorders.
2. **Standalone.** A part of 16 MiB (2^24 bytes) or more is its own chunk(s), split every 64 MiB from its first byte.
   Its objects depend only on its bytes, so an unchanged large tensor dedupes across lanes and versions.
3. **Packs.** Smaller parts are packed in order, each at a 4 KiB-aligned offset (zero padding). The open pack closes
   before a part when the part belongs to another component; when the pack holds 16 MiB or more and the part's block
   differs from the previous part's; or when the part would end past 64 MiB. A part's **block** is its key up to and
   including the last dot-separated segment made only of digits (`transformer_blocks.12.attn.to_q.weight` →
   `transformer_blocks.12`; none → empty). A pack's chunk index is assigned when it opens. Standalone parts do not
   close the open pack.
4. Chunks never span components. Zero-length parts take no space.

The chunk concatenation is not a safetensors file (padding); export rebuilds one.

**Construction order of an import.** A writer that converts with a table takes the table's row order, which its
generator took from the target model's `state_dict()`. A source already in the canonical layout (a Diffusers or
Transformers folder) carries no order, since safetensors files sort their keys; a writer orders it by the reference
of the class its `config.json` names (`_class_name`, or the first of `architectures`): that class's `state_dict()`
order, generated on the meta device and checked in as data (`crates/tensorfs-core/orders/<class>.json`,
`task -d crates/tensorfs orders`). It applies the reference structurally:

- a key is a path of dot-separated segments; a segment made only of digits is an index;
- children of one parent come in the order the reference gives children of the same parent pattern (indices written
  `#`), merged over every instance of that parent; indices ascend;
- a child the reference does not hold, or every key of a class without a reference, keeps the source's data order
  after the known children, and is reported.

## Appendix A. Fixed tables

A scheme's constants (rule 2). Each table is listed in row-major order with its dtype and shape; its SHA-256 is that
of its little-endian bytes in that dtype, and a reader MAY check its copy against it.

### A.1 `H₄` (`int8.convrot`)

`I8 [4, 4]`, SHA-256 `8f2b2e629d1df5e0be97981cbf4b4d117c320dfec29a7cd72b2f497dcb680cea`:

```
[[ 1,  1,  1, -1],
 [ 1,  1, -1,  1],
 [ 1, -1,  1,  1],
 [-1,  1,  1,  1]]
```

### A.2 `K` (`gguf.mxfp4`, `gguf.nvfp4`)

ggml's `kvalues_fp4`: E2M1's values doubled, by code.

`I8 [16]`, SHA-256 `adf76b672d47eb7fc0afd4db132d548d59d3190aa0ad118fdbf569d5fd478f97`:

```
[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
```


<!-- bnb tables -->

<!-- comfy tables -->
