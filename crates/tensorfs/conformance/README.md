# cozytensors/1 conformance

For anyone implementing [cozytensors/1](../spec/cozytensors.md).

| Path | What |
|---|---|
| `schema/header.cddl` | The header's CBOR structure (RFC 8610). |
| `schema/{manifest,repo}.schema.json` | JSON Schema (2020-12) of the manifest and repo. Byte lengths, bounds, duplicate keys, integer-only numbers and version uniqueness are spec rules the schema cannot express. |
| `vectors/cases.json` | Decode cases, in header order: a tensor's parts (base64 bits, in order) and its expected logical bytes, or the refusal code (`decode`, `unknown_scheme`). |
| `vectors/store/` | A static origin holding those cases: repo `Conformance/Schemes` at `refs/conformance/schemes.json`, lane `all` (every case, laid out by the chunk rule) and lane `spans` (a range across three chunks). |
| `vectors/green/header-*.cbor` | Headers a reader must accept, each with its projection `header-*.json`: the CBOR tree as JSON, byte strings as `{"hex": ...}`. |
| `vectors/red/header-*.cbor` | Headers a reader must refuse: twins of `header-basic` in bytes other than its one encoding (non-preferred heads, indefinite lengths, unordered keys, encodings not in first-use order), forbidden items, bad tags and structure. `header-pre-cut*` (a real pre-cut header, and the provisional JSON) must be refused as pre-cut. |
| `vectors/{green,red}/{manifest,repo}-*.json` | Manifests and repos to accept or refuse. |
| `reader.py` | An independent reader and header encoder (Python, numpy, cbor2) with reference decoders. |
| `make_vectors.py` | Writes the header vectors from the projections and defects it lists. |

A conforming implementation:
- reads the store, verifies every object, and decodes every case to its expected bytes or refuses it with its code;
- reads every green header as its projection, and as a writer encodes each projection to the green bytes (the ones
  with a minor tag or later fields are read-only); refuses every red document;
- as a writer, reproduces the store byte for byte from `cases.json`, keeping its order: components by first
  appearance, tensors and parts as listed (construction order is data, spec §4).

Run both implementations:

```bash
uv run --python 3.12 --with numpy --with cbor2==6.1.4 python -I conformance/reader.py conformance/vectors
cargo test -p tensorfs-core --no-default-features --test format_conformance --test format_docs
```

After a deliberate change: `make_vectors.py` rewrites the header vectors, and
`cargo test ... --test format_conformance -- --ignored regenerate` rewrites `vectors/store/`.

**Provenance.** The `fp8.rowwise`, `fp8.tensorwise`, `int8.convrot`, `nvfp4` and `mxfp8` (4×64) cases are
producer-mined (ComfyUI's fp8_scaled quantizer, cozy-runtime's int8 ConvRot, NVFP4 and MXFP8 producers);
cozy-runtime's decoders reproduce them bit for bit, except `fp8.tensorwise` at `BF16`/`F16`: ComfyUI rounds the scale
to the logical dtype first, the spec never rounds a stored scale, so those expected bytes are the spec's decode of the
mined parts (within one step of ComfyUI's). The ComfyUI-layout `nvfp4` and `mxfp8` cases and the expert stacks
hold the same values as their mined twins in the other nibble order, scale tiling or shape. The other cases
(per-column scales, `F8_E5M2` weights, `BF16`/`F16` tensorwise, rowwise and convrot scales, `int8.tensorwise`, `int8.rowwise`, a ragged `mxfp8` block, a divided global
scale, rounding of 3,072 f32 patterns to BF16/F16, plain and unknown dtypes, refusals) were computed exactly from the
spec's formulas.
