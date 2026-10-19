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

## 6. Encodings

An encoded tensor cites an **encoding** `[scheme, params]`. With this section they fully define how the parts become
the logical tensor; no decode depends on anything outside the header, its chunks and this spec.

**Names** are `family.variant` in lowercase, in the words of the ecosystem that made the format: generic scaled
formats are `element.granularity` (`fp8.tensorwise`, `int8.rowwise`; later `fp8.blockwise`), a producer's own format
is `producer.type` with the producer's type name verbatim (later `gguf.q4_k`, `gptq.int4`, `bnb.nf4`), and a
vendor-neutral standard keeps its bare name (`mxfp8`, `nvfp4`; later `mxfp4`).

**Params** are snake_case keys in the producer's vocabulary (`axis`, `block_size`, `rotation`, `rotation_size`,
`global_scale`, `nibble_order`, `scale_layout`, `scale_dtype`). Every param the decode reads is required and none has a
default, so the encoding shows the whole recipe. Where a param repeats a part's dtype (`scale_dtype`), a reader MUST
refuse a mismatch.

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

**Shapes.** The tensorwise schemes take any logical shape. The others take rank 2 or more and view the tensor as
`[R, C]`: `C` is the last extent and `R` the product of the others, so an expert stack `[E, N, K]` is `R = E·N` rows of
`K`. `i` indexes rows and `j` columns; block schemes act along `j`.

**Arithmetic.** Each formula is evaluated element by element in IEEE-754 binary32 (`f32`): every `×` and `÷` is one f32
operation rounded to nearest even, applied left to right as written; element formats (§5) and stored scales convert to
f32 exactly, and no scale is rounded before use. The result is then rounded once to the logical dtype `L`, written
`round_L`:
- `F32`: unchanged; `BF16`, `F16`: round to nearest, ties to even, keeping subnormals; `F16` overflows to ±infinity.
- Any NaN result is written as the canonical quiet NaN: `0x7FC00000` (`F32`), `0x7FC0` (`BF16`), `0x7E00` (`F16`).

This arithmetic defines the decode, not any producer's kernel. A producer's own dequantization may round elsewhere
(comfy-kitchen multiplies in `BF16`, for example) and differ from it by a step of `L`; that is within the determinism
the format promises, and readers follow this spec.

`input_scale` (optional where listed, `F32`, one element) is the activation scale of a quantized matmul; it never
enters a decode. It applies as `global_scale` says for `nvfp4`, and multiplies elsewhere.

| Scheme | Params | Parts | Decode |
|---|---|---|---|
| `fp8.tensorwise` | `scale_dtype`: `F32`, `BF16` or `F16`, as the producer stored it | `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `scale_dtype`, one element. `input_scale`. | `round_L(weight × scale)` |
| `int8.tensorwise` | `scale_dtype`, as above | `weight`: `I8`, the logical shape. `scale`: `scale_dtype`, one element. `input_scale`. | `round_L(weight × scale)` |
| `fp8.rowwise` | `axis`: `0` or `1` | `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `F32`, `R` elements (`axis: 0`) or `C` (`axis: 1`), any shape. | `round_L(weight[i, j] × scale[axis = 0 ? i : j])` |
| `int8.rowwise` | `axis`: `0` or `1` | `weight`: `I8`, the logical shape. `scale` as `fp8.rowwise`. | as `fp8.rowwise` |
| `int8.convrot` | `rotation: "hadamard"`, `rotation_size: B` | `weight`: `I8`, the logical shape. `scale`: `F32`, `R` elements. | below |
| `mxfp8` | `block_size: 32`, `scale_layout` | `weight`: `F8_E4M3` or `F8_E5M2`, the logical shape. `scale`: `F8_E8M0`, `⌈C/32⌉` per row (below). | `round_L(weight[i, j] × scale(i, ⌊j/32⌋))` |
| `nvfp4` | `block_size: 16`, `global_scale`, `nibble_order`, `scale_layout` | `weight`: `F4`, the logical shape, `C` a multiple of 16. `scale`: `F8_E4M3`, `C/16` per row. `scale_2`: `F32`, one element. `input_scale`. | `multiply`: `round_L((weight[i, j] × scale(i, ⌊j/16⌋)) × scale_2)`; `divide`: `round_L(weight[i, j] × (scale(i, ⌊j/16⌋) ÷ scale_2))` |

### `int8.convrot`

`B = 4^k`, `1 ≤ k ≤ 8` (256 in practice), and `C` MUST be a multiple of `B`:

`value[i, gB + j] = round_L((y[i, g, j] × scale[i]) × 2^−k)`, with `y[i, g, ·] = weight[i, gB : gB + B] · H_B`

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

### Block scales: `scale_layout`

`scale(i, b)` is row `i`'s scale for block `b`, with `W` blocks per row (`⌈C/32⌉` for `mxfp8`, `C/16` for `nvfp4`):
- `row_major`: the `scale` part has the logical shape with its last extent replaced by `W` (`[R, W]` for a matrix), and
  `scale(i, b)` is element `i·W + b`.
- `cublas_128x4`: cuBLAS's block-scaling layout, as ComfyUI stores it. The `scale` part is
  `[128·⌈R/128⌉, 4·⌈W/4⌉]`, zero-padded, in tiles of 128 rows × 4 blocks; `scale(i, b)` is element
  `512·(⌊i/128⌋·⌈W/4⌉ + ⌊b/4⌋) + 16·(i mod 32) + 4·⌊(i mod 128)/32⌋ + (b mod 4)`.

### `nvfp4`

NVIDIA's NVFP4: an E2M1 element per weight, an E4M3 scale per 16 consecutive elements of a row, and a per-tensor
`scale_2`.
- `nibble_order`: which nibble of a byte holds the even element of each pair. `low_first` is §5's `F4` packing
  (ModelOpt, compressed-tensors); `high_first` swaps the nibbles of every byte (ComfyUI).
- `global_scale`: how both per-tensor scales, `scale_2` and `input_scale`, apply: `multiply` (ModelOpt's
  `weight_scale_2` and `input_scale`, ComfyUI) or `divide` (compressed-tensors' `weight_global_scale` and
  `input_global_scale`, each the reciprocal).
- `scale_2` is one element for the whole tensor; a pre-quantization factor (`pre_quant_scale`) is not a part of this
  scheme (a later minor may add one, rule 3).
- ModelOpt's `weight` (`uint8 [R, C/2]`), `weight_scale` and `weight_scale_2` are `weight`, `scale` and `scale_2`
  here; ModelOpt and compressed-tensors write `low_first`, `row_major`; ComfyUI writes `high_first`, `cublas_128x4`.

### `mxfp8`

The OCP Microscaling MXFP8 format: a power-of-two `F8_E8M0` scale per 32 consecutive elements of a row (the last
block of a row may be short). ComfyUI stores its scales `cublas_128x4`, others `row_major`.

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

- Minor versions add fields, trailing record elements, schemes and dtypes; their absence keeps the old meaning, and
  readers of the same major read them.
- A new scheme or dtype is a minor change: every location carries its `length`, so an old reader reads the document,
  locates and verifies every byte, and refuses only a decode or typed view of the tensors that use it. A new param of
  an existing scheme is too: an old decoder refuses the tensors whose params it does not know (§6).
- A change that alters the meaning of an existing field, scheme or param is a new major.

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
