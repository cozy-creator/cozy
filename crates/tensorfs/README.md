# TensorFS

Model weights are large, and applications need to download, verify, store and read them. Different
source layouts and quantized encodings complicate this: packed numbers need the right scales and
decoding rules to become usable tensors. Keeping a separate copy of every checkpoint also wastes
space when models or precision variants share unchanged weights.

**TensorFS handles that path in one library.** It imports recognized sources into **cozytensors**,
a format whose header describes each tensor's shape, dtype and encoding. It stores the bytes in
immutable chunks named by their content, so checkpoints can share unchanged chunks. Pulls reuse
objects already in the store, verify newly downloaded bytes and can fetch just the chunks needed
by selected components. Readers use the declared encoding to reconstruct tensor values; exports
reassemble ordinary safetensors files for other tools.

Use TensorFS through its Rust API, Python library or `tfs` command-line tool. It works with local
files, static origins and Tensorhub, without requiring Cozy, a Tensorhub account or PyTorch.
The [cozytensors specification](spec/cozytensors.md) defines the format and exact decoding rules.

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
