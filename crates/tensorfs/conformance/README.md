# cozytensors/2 conformance

For anyone implementing [cozytensors/2](../spec/cozytensors-2.md).

| Path | What |
|---|---|
| `schema/{header,manifest,repo}.schema.json` | JSON Schema (2020-12) of each document's structure. Byte lengths, bounds, duplicate keys, integer-only numbers and version uniqueness are spec rules the schema cannot express. |
| `vectors/cases.json` | Decode cases: a tensor's parts (base64 bits) and its expected logical bytes, or the refusal code (`decode`, `unknown_scheme`). |
| `vectors/store/` | A static origin holding those cases: `refs/conformance/schemes.json`, lane `all` (every case, laid out by the chunk rule) and lane `spans` (a range across three chunks). |
| `vectors/green/`, `vectors/red/` | Single documents, named `<kind>-<case>.json`, that a reader must accept or refuse. |
| `reader.py` | An independent reader (Python stdlib and numpy) with reference decoders. |

A conforming implementation:
- reads the store, verifies every object, and decodes every case to its expected bytes or refuses it with its code;
- accepts every green document and refuses every red one;
- as a writer, reproduces the store byte for byte from `cases.json` (components by name, keys in natural order).

Run both implementations:

```bash
uv run --with numpy python -I conformance/reader.py conformance/vectors
cargo test -p tensorfs-core --no-default-features --test format_conformance --test format_docs
```

`cargo test ... --test format_conformance -- --ignored regenerate` rewrites `vectors/store/` after a deliberate change.

**Provenance.** The `fp8-rowwise`, `fp8-tensorwise`, `int8-convrot`, `nvfp4` and `mxfp8` cases are the producer-mined
vectors of cozytensors/1 (ComfyUI's fp8_scaled quantizer, cozy-runtime's int8 ConvRot and NVFP4 producers), renamed to
v2 parts; cozy-runtime's decoders reproduce them bit for bit. The other cases (per-column scales, `F8_E5M2` weights,
`int8-rowwise`, a ragged `mxfp8` block, rounding of 3,072 f32 patterns to BF16/F16, plain dtypes, refusals) were
computed exactly from the spec's formulas.
