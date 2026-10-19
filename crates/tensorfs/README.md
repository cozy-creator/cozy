# TensorFS

Safetensors provides a fast, memory-mappable tensor file. Its [standard header](https://github.com/safetensors/safetensors#format)
records names, dtypes, shapes and byte offsets. Distributing and serving quantized models still
leaves applications to supply several things:

| What safetensors leaves to the application | How TensorFS addresses it |
|---|---|
| Quantization meaning: how scales, packed weights and producer-specific transforms reconstruct a tensor. These are conventions beyond the standard tensor declarations. | **cozytensors** declares named encodings, complete parameters and stored parts, with specified decoding math. |
| Reusing unchanged bytes across checkpoints: the file format defines no content-addressed chunks shared between files. | Immutable chunks identified by SHA-256 and length are shared across models and precision lanes; pulls fetch missing objects. |
| Verifying partial downloads: tensor offsets permit selective reads, but the format includes no per-chunk content hashes. | Each downloaded chunk is verified independently before use, so a selected component can be fetched and checked without downloading the whole checkpoint. |

TensorFS combines that format with a local store, fetcher and Rust/Python readers. It packs small
parts into aligned chunks and preserves construction order for efficient reads. Safetensors remains
an import/export format, so existing Diffusers and Transformers workflows can use the same weights.

Use TensorFS through its Rust API, Python library or `tfs` command-line tool. It works with local
files, static origins and Tensorhub, without requiring Cozy, a Tensorhub account or PyTorch.
The [cozytensors specification](spec/cozytensors.md) defines the format and exact decoding rules.

## Timing versus safetensors

Measured disk → host → GPU transfer of the same real INT8 lane bytes, represented as `U8` parts
in both formats. Times are **milliseconds, median of five interleaved repetitions**, on NVIDIA L4.
Each load includes open, registration, allocation/mapping and transfer through GPU completion;
persistent reader initialization and fixture preparation are excluded.

| Workload (logical payload) | TensorFS cold / warm | Safetensors cold / warm | Run |
|---|---:|---:|---|
| SDXL, full checkpoint, 4.08 GB | **306 / 344** | 392 / 405 | 11566 |
| Qwen-Image, full checkpoint, 19.2 GB | **1,384 / 1,684** | 1,659 / 2,019 | 11566 |
| SDXL, selected block, 187 MB logical | **24 / 23** | 79 / 45 | 11624 |
| Qwen-Image, selected block, 218 MB logical | **33 / 23** | 61 / 37 | 11624 |

The baseline is the fastest tested safetensors path per cell: `fastsafetensors 0.5.0` for full
loads and cold blocks; `safetensors 0.8.0`'s `safe_open` CPU→CUDA path for warm blocks. Cold requests
payload-cache eviction with `POSIX_FADV_DONTNEED`; warm is the immediate second read. Chunk boundaries
make the selected-block reads 211 MB and 227 MB respectively.

These are separate runs and reader revisions: the full-load run predates the final CBOR header
and lazy window registration; the block run includes them. They measure serialized-byte loading,
excluding dequantization, model construction and inference. Hardware and cache conditions affect
results. [Measurements and provenance](gates/speed/measurements.json) include the earlier failed
window cell; [the recorded harness](https://github.com/cozy-creator/cozy-creator/blob/0a6e71ee2c919db3e6e9b5844c8234ca9749cedb/crates/tensorfs/gates/speed/speedgate.py)
defines the workloads. Download projections are separate from these measured timings.

## Get started

Python 3.12 or later is required. The wheel includes both the Python module and `tfs`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install tensorfs

tfs import hf://hf-internal-testing/tiny-sdxl-pipe --name local/example
tfs inspect local/example
tfs export local/example ./example --safetensors
```

The export is a Diffusers/Transformers folder, including configs and tokenizer files. Plain tensors
retain their bytes. Encoded tensors retain their parts and encoding metadata; add
`--dequantize bf16` to export ordinary BF16 values for a consumer without that encoding.

Read the imported checkpoint from Python:

```python
import tensorfs

store = tensorfs.Store("~/.tensorfs")
checkpoint = store.checkpoint("local/example")
info = checkpoint.tensors["unet"]["conv_in.weight"]
value = checkpoint.read("unet", "conv_in.weight")
print(info.dtype, info.shape, value.nbytes)
```

`read` returns logical CPU values, decoding when needed. The returned array supports DLPack for
frameworks and dtypes that support it. `checkpoint.file(path)` reads a companion file. Parsing
header bytes with `tensorfs.Checkpoint.parse(data)` permits metadata inspection without a Store;
reading tensors also requires their chunks.

## How checkpoints fit together

A **checkpoint** is a model folder described by an immutable **manifest**. It contains configuration,
tokenizer and other files, plus a `model.cozytensors` **header** when it has tensors. The header lists
components such as `unet` or `text_encoder`, their tensors and the chunks holding their bytes.
A tokenizer-only checkpoint can have no header.

```text
repo → release → lane → manifest
                         ├─ model.cozytensors → tensor chunks
                         └─ configs, tokenizer files, license, …
```

A **lane** selects one representation of a release, such as BF16, FP8 or INT8. Each has its own
manifest. Every immutable object is named by its SHA-256 and length. A repo document holds the mutable
release and lane pointers; changing a pointer leaves existing checkpoint bytes intact.

## Import and share

`tfs import` accepts files, directories, `hf://`, `civitai://` and HTTPS sources. Imports pin the
source, download into temporary storage and normalize recognized layouts through converter tables.
Canonical component and tensor names follow upstream Diffusers/Transformers names. An unrecognized
layout can be stored unchanged; this alone does not make it runnable by a particular model loader.
Use `--converter` when the source needs an explicit layout declaration.

Source layout matters even when names and shapes match. H3's native-base converter handles Q/K/V
rows interleaved per head; its trainer-LoRA converter expects contiguous Q, K and V thirds.
Selecting a converter declares the source layout; inspect an unfamiliar producer rather than infer
that layout from shapes. LoRAs retain upstream module names and PEFT `lora_A`/`lora_B` identities.

Pull a released lane from Tensorhub, or use the same format on an ordinary HTTP server or directory:

```bash
tfs pull acme/model@1.0.0 --lane int8
tfs export local/example ./origin --origin
tfs --store ./second-store pull ./origin local/example
```

Tensorhub is the default origin; public lanes need no account. Private pulls use
`TENSORFS_HUB_TOKEN`, and `--hub` selects another Hub. For a static HTTP origin, replace `./origin`
with its base URL. The origin holds repo documents and immutable objects in the same layout as the
Store. Fetching verifies each object's hash and length and can select only the chunks needed by
the requested components. Authentication and permission policy belong to the origin.

## Write and retain checkpoints

`store.writer(base=manifest, key=optional_resume_key)` creates a writer. `put` adds a plain tensor;
`put_encoded` supplies a scheme, explicit params and named parts; `copy` reuses a tensor from another
checkpoint; `file` attaches an ordinary file; `commit` returns the new manifest. Supply tensors in
construction order. A base checkpoint supplies inherited tensors and files, with explicit writes
replacing them. A keyed writer journals completed work for same-store restart; `has` reports it and
`abandon` releases it. Closing a handle is not a commit or an abandonment.

The [Python signatures](python/tensorfs/_ext.pyi) and Rust API documentation describe arguments,
return types and error families. Python errors expose `.code` and `.detail`. An unknown encoding
remains inspectable and transferable; decoding it requires an implementation of that scheme.

## Storage and ownership

```text
<root>/
  format                     Store format marker
  blobs/aa/bb/<sha256>        Immutable objects, including manifests and headers
  refs/<org>/<name>.json      Release/lane pointers and unreleased checkpoints
  pins/<id>.json             Retained work and checkpoint roots
  tmp/                       In-progress admission, import and writer data
```

The CLI uses `--store`, then `TENSORFS_HOME`, then `~/.tensorfs`. Python takes an explicit root.
Admission streams and hashes bytes into a temporary file before installing the immutable object.
Durable stores flush objects before publication; ephemeral stores batch the flush before publishing
records that name the objects. Verified local objects are trusted on ordinary reads. `tfs verify` rehashes them,
removes corrupt objects and reports missing ones so they can be fetched again.

Repo refs and pins retain their reachable objects. GC marks their manifests, files and tensor
chunks, then collects unreferenced objects older than its grace period. `tfs gc` performs that sweep;
`--older-than` also expires idle pins, while repo refs remain roots. Shared objects are counted once.
Managed-cache eviction is the embedding application's policy; a reference or pin must be released
before its objects become collectible.
Use the library's locking and retention APIs rather than edit the Store's files by hand.

TensorFS is a library, with no separate daemon. In Cozy, TensorD owns the managed Store, credentials,
fetch priorities and cache policy. Runtime constructs and executes models using granted sources;
package code uses Runtime's channels. GPU allocation, residency and paging belong to `varena` and
its caller. A standalone Store gives its caller storage access; installing the Python wheel does
not grant access to a managed execution.

## Format and development

The header is deterministic CBOR; manifests and repo documents are JSON. Component, tensor and
part order is construction order and contributes to identity. Small parts are packed at 4 KiB
alignment; large parts keep independent chunks, normally up to 64 MiB. The specification separates
required reader behavior from this writer policy. Export reconstructs safetensors files from chunks.

The [source-header fixtures](vectors/) pin import provenance and geometry without model payloads.
The Rust `tensorfs-core` crate contains the format and reference decoders; its `store` feature adds
storage and I/O. `tensorfs-py` supplies Python bindings, and `varena` supplies memory mechanisms.
The [conformance corpus](conformance/) checks the format independently of those APIs.
The standalone [end-to-end check](scripts/e2e.sh) imports a tiny public model, exports it for
Diffusers, compares its result and pulls the same checkpoint through a static origin.
