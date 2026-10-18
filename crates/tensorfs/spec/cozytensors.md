# cozytensors/1

Status: normative, version 1.0. Conformance vectors and an independent reader: [`../conformance/`](../conformance/).

cozytensors/1 stores model weights as immutable, content-addressed objects. A **checkpoint** is a manifest (a file
tree) whose `model.cozytensors` file is a **header**; the header names **chunks** (objects of tensor bytes) and
describes every tensor in them, plain or encoded, by component. A **repo** document maps a model's releases and lanes
to checkpoints. Everything a reader needs to read and decode a tensor is in these documents and this spec.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. §1–§8 define meaning; §9 defines the byte encoding
of documents; §10 is informative writer guidance.

## 1. Objects

An object is an immutable byte string named by an **object ref** `{sha256, length}`: `sha256` is the SHA-256 of the
bytes as 64 lowercase hex digits, `length` their count. In a store or static origin the object lives at

```
blobs/<sha256[0:2]>/<sha256[2:4]>/<sha256>
```

A reader MUST check the length and digest of every object before using its bytes.

## 2. Documents

Manifests, headers and repos are documents. Each is a map with a `format` tag `<kind>/<major>` or
`<kind>/<major>.<minor>`, where major and minor are decimal digits:

| Document | `format` kind | Usual name |
|---|---|---|
| manifest | `tensorfs.manifest` | an object, cited by a repo |
| header | `cozytensors` | the manifest file `model.cozytensors` |
| repo | `tensorfs.repo` | `refs/<org>/<name>.json` |

- A reader MUST refuse a document whose kind differs or whose major is not `1`. It MUST accept any minor.
- A reader MUST ignore fields it does not know, at every level. A writer MAY add fields; adding a field never needs a
  new major.
- Values are null, booleans, integers in [−2^63, 2^64), strings, lists and maps with string keys. There are no
  floating-point numbers anywhere in a document. A map MUST NOT repeat a key.
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

```
{format: "cozytensors/1", chunks: [object-ref], components: [{name, tensors: [tensor]}], metadata?}
```

- **Construction order is data.** `components`, each component's `tensors` and each encoded tensor's `parts` are
  arrays in construction order: the order in which the model builds and fills them (its `state_dict()` order). The
  order is part of the header's bytes, so reordering changes its digest. A writer MUST take the order from its
  producer (a converter's rows, a base checkpoint, a source's constructor order) and MUST NOT sort names. Readers
  iterate in this order and look names up through an index.
- Component `name`s MUST be segments (they name folders on export) and unique. Tensor `key`s MUST be non-empty and
  unique within their component; they are the state-dict names within the component.
- A **tensor** has a `key`, a logical `dtype` (§5) and `shape` (a list of non-negative integers; `[]` is a scalar),
  and exactly one body:
  - **plain**: `data`, a location of its bytes; or
  - **encoded**: `encoding: {scheme, params?}` with a non-empty scheme name, and a non-empty array `parts` (§6).
- A **part** is `{role, dtype, shape, chunk, offset, length}`: a non-empty role unique within the tensor, the part's
  own dtype and shape, and its location.
- A **location** `{chunk, offset, length}` names bytes in the chunks. `chunk` indexes `chunks`. The range starts at
  `offset` in that chunk and, when `length` exceeds the rest of the chunk, continues at offset 0 of the following
  chunks in list order.
- For every location with `length > 0`, a reader MUST refuse the header unless `chunk` is in range,
  `offset < chunks[chunk].length`, and `offset + length` is at most the summed lengths of `chunks[chunk..]`. A
  zero-length location reads no bytes (writers emit `{0, 0, 0}`).
- When the dtype is known, `length` MUST equal the byte size of the dtype and shape (§5), for plain data and for
  every part.

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
- A dtype is any non-empty name. A name outside this table is a newer dtype: a reader still reads the document and
  checks bounds, takes the `length` as written, and refuses only a typed view of the bytes or a decode that needs it.

Element formats used by the schemes:

| Format | Meaning of byte (or nibble) `b` |
|---|---|
| `F8_E4M3` | sign bit 7, exponent bits 3–6 (bias 7), mantissa bits 0–2. `e = 0`: `m × 2^−9`; otherwise `(1 + m/8) × 2^(e−7)`. `S.1111.111` is NaN; no infinities. |
| `F8_E5M2` | sign bit 7, exponent bits 2–6 (bias 15), mantissa bits 0–1. `e = 0`: `m × 2^−16`; `e = 31`: infinity if `m = 0`, else NaN; otherwise `(1 + m/4) × 2^(e−15)`. |
| `F4` (E2M1) | sign bit 3; bits 0–2 index `0, 0.5, 1, 1.5, 2, 3, 4, 6`. |
| `F8_E8M0` | `2^(b−127)` for `b < 255`; `255` is NaN. |
| `I8` | two's complement. |

## 6. Encodings

An encoded tensor cites an **encoding**: a scheme name and its params. With this section they fully define how the
parts become the logical tensor; no decode depends on anything outside the header, its chunks and this spec.

**Names** are `family.variant` in lowercase: the element family (`fp8`, `int8`; later `int4`, …) and the method or scale
granularity (`rowwise`, `tensorwise`, `convrot`; later `gptq`, `awq`, `block`). A format with an established name keeps
it: `mxfp8` (OCP Microscaling), `nvfp4` (NVIDIA ModelOpt).

**Params** are complete: every param a scheme defines is written, even where the name implies its value, so the params
alone tell a reader the decode. Keys and text values are snake_case; dtypes are §5's names, the one dtype vocabulary of
a header, in parts and params alike.

| Param | Meaning |
|---|---|
| `weight_dtype` | The `weight` part's dtype. |
| `scale` | The `scale` part's granularity: `per_tensor` (one), `per_row` (one per row `i`), `per_block` (one per row and block of `block` columns). |
| `scale_dtype` | The `scale` part's dtype. |
| `block` | Columns per block: of a `per_block` scale, or of a rotation group. |
| `rotation` | The transform applied to each block before quantizing: `hadamard`. |
| `global_scale`, `global_scale_dtype` | A second, per-tensor scale (the `global_scale` part) and its dtype. |

No scheme here has a zero point.

**Reading and decoding.**
- A tensor with an unknown scheme still reads: its parts are located and verified like any other. Only decoding
  refuses, and the refusal MUST name the scheme.
- A decoder MUST refuse a known scheme when a required part is missing, a part not listed for the scheme is present,
  a param the scheme defines is missing or has another value, a part's dtype or shape breaks the scheme's rule, or the
  logical dtype is not `F32`, `BF16` or `F16`. It MUST ignore params the scheme does not define.

**Arithmetic.** Each formula is evaluated element by element in IEEE-754 binary32 (`f32`): every `×` is one f32
multiplication rounded to nearest even, applied left to right as written; element formats (§5) convert to f32 exactly.
The result is then rounded once to the logical dtype `L`, written `round_L`:
- `F32`: unchanged; `BF16`, `F16`: round to nearest, ties to even, keeping subnormals; `F16` overflows to ±infinity.
- Any NaN result is written as the canonical quiet NaN: `0x7FC00000` (`F32`), `0x7FC0` (`BF16`), `0x7E00` (`F16`).

`R × C` below is the logical shape `[R, C]`; `i` indexes rows and `j` columns. `input_scale` (optional where listed,
`F32`, one element) is the activation scale of a quantized matmul; it never enters a decode.

### `fp8.rowwise`

| | |
|---|---|
| Params | `scale: per_row`, `scale_dtype: F32`, `weight_dtype: F8_E4M3` or `F8_E5M2`. |
| Parts | `weight`: `weight_dtype`, `[R, C]`. `scale`: `F32`, `R` elements (`[R]` or `[R, 1]`). |
| Decode | `value[i, j] = round_L(weight[i, j] × scale[i])` |

### `fp8.tensorwise`

| | |
|---|---|
| Params | `scale: per_tensor`, `scale_dtype: F32`, `weight_dtype: F8_E4M3` or `F8_E5M2`. |
| Parts | `weight`: `weight_dtype`, the logical shape (any rank). `scale`: `F32`, one element. `input_scale` (optional). |
| Decode | `value = round_L(weight × s)` where `s` is `scale` rounded to `L` and read back as f32 (`s = scale` when `L` is `F32`). |

### `int8.rowwise`

| | |
|---|---|
| Params | `scale: per_row`, `scale_dtype: F32`, `weight_dtype: I8`. |
| Parts | `weight`: `I8`, `[R, C]`. `scale`: `F32`, `R` elements. |
| Decode | `value[i, j] = round_L(weight[i, j] × scale[i])` |

### `int8.convrot`

| | |
|---|---|
| Params | `block: B = 4^k`, `1 ≤ k ≤ 8` (256 in practice), with `C` a multiple of `B`. `rotation: hadamard`, `scale: per_row`, `scale_dtype: F32`, `weight_dtype: I8`. |
| Parts | `weight`: `I8`, `[R, C]`. `scale`: `F32`, `R` elements. |
| Decode | `value[i, gB + j] = round_L((y[i, g, j] × scale[i]) × 2^−k)`, with `y[i, g, ·] = weight[i, gB : gB + B] · H_B` |

`H_B = H₄ ⊗ … ⊗ H₄` (`k` factors) with entries ±1, where

```
H₄ = [[ 1,  1,  1, -1],
      [ 1,  1, -1,  1],
      [ 1, -1,  1,  1],
      [-1,  1,  1,  1]]
```

`H_B` is symmetric and `H_B · H_B = B · I`, so `2^−k · H_B` is the orthonormal rotation ComfyUI's ConvRot applies before
quantizing each row. `y` is an exact integer (|y| ≤ 128·B ≤ 2^23), so its conversion to f32 is exact. One way to
compute it: view each group of `B` as `k` base-4 digits (most significant first) and apply `H₄` along every digit.

### `mxfp8`

| | |
|---|---|
| Params | `block: 32`, `scale: per_block`, `scale_dtype: F8_E8M0`, `weight_dtype: F8_E4M3` or `F8_E5M2`. |
| Parts | `weight`: `weight_dtype`, `[R, C]`. `scale`: `F8_E8M0`, `[R, ⌈C/32⌉]` (the last block may be short). |
| Decode | `value[i, j] = round_L(weight[i, j] × scale[i, ⌊j/32⌋])` |

This is the OCP Microscaling MXFP8 format: a power-of-two scale per 32 consecutive elements of a row.

### `nvfp4`

| | |
|---|---|
| Params | `block: 16`, `global_scale: per_tensor`, `global_scale_dtype: F32`, `scale: per_block`, `scale_dtype: F8_E4M3`, `weight_dtype: F4`. `C` MUST be a multiple of 16. |
| Parts | `weight`: `F4`, `[R, C]` (two elements per byte, §5). `scale`: `F8_E4M3`, `[R, C/16]`. `global_scale`: `F32`, one element. `input_scale` (optional). |
| Decode | `value[i, j] = round_L((weight[i, j] × scale[i, ⌊j/16⌋]) × global_scale)` |

This is NVIDIA ModelOpt's NVFP4 layout: ModelOpt's `weight` (`uint8 [R, C/2]`, element 2k in the low nibble),
`weight_scale` and `weight_scale_2` are `weight`, `scale` and `global_scale` here.

### Plain tensors

A plain tensor's bytes are its value; reading it is a copy, whatever its dtype.

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

- Minor versions add fields or values whose absence keeps the old meaning; readers of the same major read them.
- A new scheme or dtype is a minor change: every location carries its `length`, so an old reader reads the document,
  locates and verifies every byte, and refuses only a decode or typed view of the tensors that use it.
- A change that alters the meaning of an existing field, scheme or param is a new major.

## 9. Encoding

Documents are UTF-8 JSON (RFC 8259) whose top level is an object; §2's values map to JSON in the obvious way.

**Reading.** A reader MUST refuse:
- invalid UTF-8 or JSON, or a top level that is not an object;
- a repeated key in any object;
- any number with a fraction or exponent, `NaN` or `Infinity`, and any integer outside [−2^63, 2^64);
- a field of the wrong JSON type (for example `null` for `metadata` or `params`; a JSON `null` for `data`,
  `encoding` or `parts` counts as absent).

Whitespace between tokens is allowed.

**Writing.** A writer MUST produce the canonical form, so equal documents have equal bytes and equal object refs:
- object keys sorted by their UTF-8 bytes; no whitespace;
- integers in shortest decimal (no `+`, no leading zeros, `-` only for negatives);
- strings with `"` and `\` escaped as `\"` and `\\`, the controls U+0008, U+0009, U+000A, U+000C and U+000D as `\b`,
  `\t`, `\n`, `\f`, `\r`, other controls below U+0020 as `\u00XX` (lowercase hex), and every other character as its
  UTF-8 bytes (Python: `json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`);
- optional fields at their default omitted: empty `metadata`, `params` and `checkpoints`, `yanked: false`,
  `revision: 0`.

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
