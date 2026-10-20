# TensorFS Python API

Open a model and read its tensors. TensorFS fetches missing data into its shared store
automatically; normal loading does not create a snapshot folder. This guide covers the
Python API, with optional filesystem and repository tools kept separate. The
[Rust API](../tensorfs/API.md) and [file-format specification](../tensorfs/spec/cozytensors.md)
are separate documents.

- [Import and open a model](#python-import-and-open-a-model)
- [Use the returned objects](#python-use-the-returned-objects)
- [Write, convert and publish](#python-write-convert-and-publish)
- [Configuration and advanced operations](#python-configuration-and-advanced-operations)
- [Errors and value types](#errors-and-value-types)

The [temporary type-declaration dump](API-TYPES.md) is available for signature and naming review.

## Python: import and open a model

Install the Python package, then import it:

```bash
uv pip install tensorfs
```

```python
import tensorfs

checkpoint = tensorfs.open("me/model", revision="1.0", variant="bf16")
weight = checkpoint.components["unet"].tensors.read("conv_in.weight")
```

`open` returns a `Checkpoint`. By default (`lazy=True`), it loads metadata; reading
tensors then downloads the objects those tensors need. `revision` selects a release or immutable reference;
`variant` selects a representation such as `bf16` or `int8`. Without a variant, a
single-variant tree is unambiguous; a multi-variant tree requires a choice. No separate
download call is required. The same `open` accepts a local `.cozytensors` header or an
explicitly materialized folder; those are optional input forms.

To download the complete checkpoint into the shared store before continuing:

```python
checkpoint = tensorfs.open("me/model", lazy=False)
```

This waits for all declared chunks and component assets. It does not create a folder,
unpack quantized weights or place them on a GPU. To preload only selected components,
open lazily and call `checkpoint.download(components=["unet"])`.

For a private model, pass `token=...`. Other options are `endpoint`, `cache_dir` and
`local_files_only`. `local_files_only=True` forbids network access and fails if the required
data is unavailable locally. A private CA is set once, with `tensorfs.configure(ca_der=...)`.

## Python: use the returned objects

### Inspect the pipeline and components

```python
print(checkpoint.pipeline.cls if checkpoint.pipeline else "no pipeline")
print(list(checkpoint.components))

unet = checkpoint.components["unet"]
print(unet.cls)
print(unet.config)
```

A `Pipeline` has `cls` and `config`. A `Component` exposes:

| Attribute or call | What you get |
|---|---|
| `name` | Component name, such as `unet`. |
| `cls` | Model class identifier. |
| `config` | Parsed JSON configuration as a dictionary. |
| `config_file` | The file name `config` exports as: `config.json` unless the writer named another. |
| `tensors` | A `Tensors` mapping, described below. |
| `assets` | Supporting filenames mapped to `(sha256_hex, byte_length)` references; this does not load file contents. |
| `asset(name)` | Read a supporting file, fetching it if needed; returns `bytes`. |

An asset is a supporting file needed by the component, such as a tokenizer's
`tokenizer.json`, `vocab.json` or `merges.txt`. All of these files belong to the checkpoint;
they are stored by content hash alongside the tensor data. Native consumers read `config`
and file bytes directly through TensorFS; no directory export is part of loading a
cozytensors checkpoint:

```python
tokenizer = checkpoint.components["tokenizer"]
print(list(tokenizer.assets))
tokenizer_bytes = tokenizer.asset("tokenizer.json")
```

For import/export compatibility, `config_file` is the filename to use when recreating a
component's configuration file. For example, a scheduler can require
`scheduler_config.json` rather than `config.json`. Loading directly from `config` does not
require a filename; conversion and a `Writer(base=...)` rewrite preserve it.

### Inspect tensor metadata without reading weights

```python
print(list(unet.tensors))
info = unet.tensors["conv_in.weight"]
print(info.dtype, info.shape, info.encoding)
```

`Tensors` behaves like a read-only mapping: `len`, iteration, membership, indexing,
`keys()`, `values()`, `items()` and `get(key, default)` work normally. A missing indexed
key raises `KeyError`; `get` returns its default.

The returned `TensorMetadata` describes the whole tensor: its logical `dtype`, `shape`
and optional `encoding`. It contains metadata, not tensor values. Inspect its shape/dtype,
then call `read`. Ordinary reads hide physical chunk locations and byte offsets;
advanced storage integrations can inspect them through `tensorfs.raw` below.

### Read a tensor, or a component's weights

```python
weight = unet.tensors.read("conv_in.weight")
weights = unet.tensors.read()
selected = unet.tensors.read(["conv_in.weight", "conv_in.bias"])
```

`read(keys=None, device="cpu", decode=False)` accepts a tensor name, a sequence of
names, or no selection. A single name returns one value; a sequence returns a name-to-value
dictionary; no selection (or `None`) reads every tensor in this component. An empty list
returns an empty dictionary without fetching data. Explicit selections keep the requested
order, with duplicate names included once; an all-tensor read uses construction order.
An unknown name raises `KeyError`.

All forms accept `device="cpu"` (the default), `device="cuda"` or `device="cuda:N"`,
and the same `decode` option.

`decode=False` is the default: a plain tensor returns an `Array`, and a quantized tensor
returns `Encoded` with its stored arrays and encoding parameters. Reading does not
automatically expand quantized weights. An `Array` has `dtype`, `shape`, `nbytes` and
`tobytes()`.

With explicit `decode=True`, quantized weights are unpacked into the tensor's declared
logical dtype (`TensorMetadata.dtype`), such as BF16, F16 or F32; BF16 is not forced.
Plain tensors keep their dtype with either setting. The default destination is CPU;
`device="cuda:0"` requests GPU arrays. Supported GPU decoders unpack there; schemes with
a host-only decoder use the CPU fallback.

Decode when the model's compute code needs ordinary floating-point weights. For example,
a standard PyTorch layer can consume a decoded BF16 array, but cannot directly consume
int8 weights together with scales and rotation parameters. A matching quantized compute
kernel can use those encoded arrays directly. Decoding provides compatibility at the
cost of an unpacking step and a larger weight allocation.

`Array` is TensorFS's own array object. It implements DLPack, a standard protocol for
sharing tensor memory between libraries, rather than a separate Python array library.
Frameworks such as PyTorch can wrap a compatible array's existing memory without copying.
For example:

```python
import torch

weight = unet.tensors.read("conv_in.weight", decode=True)
torch_weight = torch.from_dlpack(weight)
gpu_weights = unet.tensors.read(device="cuda:0", decode=True)
state_dict = {name: torch.from_dlpack(value) for name, value in gpu_weights.items()}
```

The returned arrays and their framework consumers retain the storage they need.

### Keep quantized weights encoded

```python
value = unet.tensors.read("some_quantized.weight")  # decode=False is the default
print(value.scheme, value.params)
print(value.parts.keys())

# GPU placement does not require decoding:
gpu_value = unet.tensors.read("some_quantized.weight", device="cuda:0")
```

`Encoding` describes how the stored arrays reconstruct the logical tensor: a scheme
name such as `int8.rowwise`, plus its parameters. For example, a logical weight matrix
with shape `(4096, 4096)` can be stored as an int8 array named `weight` with that shape
and a separate scale array named `scale` with shape `(4096,)`. Each row is reconstructed
as `weight[row, column] * scale[row]`.

These are ordinary arrays with their own dtype and shape. Consumers receive their values;
TensorFS handles the underlying chunk locations and byte offsets.

For an encoded tensor, `decode=False` returns `Encoded`: the logical `dtype` and `shape`,
the loaded `scheme` and `params`, and a role-to-`Array` mapping named `parts`. For a plain
tensor it still returns `Array`.
A consumer of encoded weights must understand that scheme and its parameters.
`device` chooses where the arrays live; `decode` chooses their representation. Our
Runtime keeps supported quantized weights encoded for its matching compute kernels.

### Read into an existing allocation

```python
destination = bytearray(weight.nbytes)
unet.tensors.read_into("conv_in.weight", destination, offset=0)
copied_bytes = bytes(destination)

# For stored encoded bytes, size the destination for that part and select its role.
encoded = unet.tensors.read("some_quantized.weight", decode=False)
packed = bytearray(encoded.parts["weight"].nbytes)
unet.tensors.read_into("some_quantized.weight", packed, role="weight", offset=0)

# Or its decoded logical bytes, from byte 64:
window = bytearray(256)
unet.tensors.read_into("some_quantized.weight", window, decode=True, offset=64)
```

`read_into(key, out, *, role=None, decode=False, offset=0)` follows `read`. The destination
determines the number of bytes copied, and `offset` is a byte offset. A plain tensor gives
its data. `role` selects one stored part of an encoded tensor, and `decode=True` gives its
decoded logical bytes. An encoded tensor with neither raises `InvalidError` naming its
roles, and a role with `decode=True` is refused too. This call does not convert to the
destination's dtype; provide the appropriate size and representation.

### Download, verify and manage a checkpoint

```python
def on_progress(stage, completed, total):
    print(f"{stage}: {completed}/{total}")

checkpoint.download(components=["unet"], priority=0, progress=on_progress)
```

`download` returns when all requested chunks and supporting files are verified and locally
available, or raises an error. The open checkpoint retains them. Closing its last handle
allows ordinary cache policy to apply again; a saved header file roots its data while it
holds that header.
Progress callbacks receive `(stage, completed, total)`; download counts are locally available
bytes out of requested bytes, while conversion counts are tensors.

For optional availability inspection or an explicit integrity audit:

```python
missing = checkpoint.missing(components=["unet"])
report = checkpoint.verify(components=["unet"])
print(report.ok, report.missing, report.corrupt)
```

`missing` checks local availability without downloading or rehashing. `verify` re-reads
and hashes local objects, returning a `VerificationReport` with separate `missing` and
`corrupt` object-reference lists. `ok` is true when both are empty. Verification never
downloads, repairs or deletes data; other I/O failures raise the usual typed error.
There is no need to call it after every download: incoming data is already verified when
admitted to the store.

All three methods accept the same component selection. `None` covers the checkpoint's
declared chunks and component assets; a component list covers those components' complete
chunks and assets. They do not materialize a repository folder or fetch README/LICENSE
files. Verification always includes the header, even with `components=[]`. A chunk
shared by selected and unselected tensors is checked as one complete stored object.

Additional checkpoint properties and operations:

| Attribute or call | Meaning |
|---|---|
| `id`, `header_ref` | Header identity: `sha256:…` string and `(hex, length)` pair. |
| `tree`, `tree_ref` | Repository-tree identity in the same two forms, when known. |
| `repo_id` | Repository name, when known. |
| `revision` | Release label, when known. |
| `variant` | Selected repository variant, when known. A standalone header does not store this label. |
| `pipeline`, `components` | Pipeline description and ordered component mapping. |
| `metadata` | Loading/provenance metadata. |
| `chunks` | Chunk object references in header order, as `(sha256_hex, byte_length)` pairs. |
| `close()` | Release this checkpoint handle. Existing child handles remain valid. |

Use `with tensorfs.open(...) as checkpoint:` for automatic close. `Array` implements
`__dlpack__` and `__dlpack_device__`, including stream/device/copy negotiation. Objects
provide normal debug representations (`repr`); an `Encoding` also has a string form.
These representations are not serialization formats.

### Use TensorFS inside an async application

The Python API is synchronous: `open`, `download` and `read` wait for their results.
The native engine can still overlap work internally. In an asyncio application, move
the blocking call to a worker thread so the event loop remains responsive:

```python
import asyncio

async def load_model():
    return await asyncio.to_thread(tensorfs.open, "me/model", lazy=False)
```

The same pattern works for `checkpoint.download` and tensor reads. Cancelling the
await does not automatically stop an operation already running in the worker thread.

### Advanced: inspect stored ranges

Storage and memory integrations can inspect physical layout without fetching data:

```python
from tensorfs import raw

layout = raw.parts(checkpoint, "unet", "conv_in.weight")
for role, part in layout.items():
    chunk_ref = checkpoint.chunks[part.chunk]
    print(role, part.dtype, part.shape, chunk_ref, part.offset, part.length)
```

`raw.parts` returns `dict[str, raw.Part]`. A plain tensor has one entry with the empty
role `""`; an encoded tensor has its named stored arrays. Each `raw.Part` reports the
stored array's `dtype` and `shape`, a `chunk` index, and byte `offset` and `length` within
that chunk. This call only inspects metadata. Normal model code uses `TensorMetadata`
and `read` instead of managing these ranges.

### Advanced: decode resident encoded parts

`tensorfs.raw.decode_into` is for integrations that already hold encoded parts in memory and own the
destination. It does not open a checkpoint or fetch data. Ordinary model code continues to
use `component.tensors.read(..., decode=True)`. Its signature in `tensorfs.raw` is:

```python
def decode_into(
    scheme: str,
    parts: Mapping[str, tuple[str, Sequence[int], Buffer | DLPack]],
    out: Buffer | DLPack,
    *,
    dtype: str,
    shape: Sequence[int],
    params: Mapping[str, object] | None = None,
) -> None: ...
```

Each input is `role: (format_dtype, format_shape, data)`, using the same role names and
encoded representation as `ComponentWriter.put_encoded`. The tuple describes the stored
format; `data` supplies exactly that many contiguous bytes. Its carrier dtype/shape may
differ: packed F4 data can use a U8 carrier, and F8 bytes can use a U8 view. Marshalling
reinterprets those bytes; it does not numerically convert the encoded weights.

`dtype` and `shape` describe the logical result and its final format rounding. `out` must
already have exactly that actual dtype, shape and byte length, with writable contiguous
storage. It is not cast or resized. Primitive buffer outputs support F32 and F16; BF16
outputs use a supported DLPack carrier. For example, this planned CPU call decodes stored
rowwise INT8 bytes into an existing F32 array:

```python
import numpy as np
from tensorfs import raw

weight = np.array([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=np.int8)
scale = np.array([0.5, 2.0], dtype=np.float32)
out = np.empty((2, 4), dtype=np.float32)

raw.decode_into(
    "int8.rowwise",
    {
        "weight": ("I8", weight.shape, memoryview(weight)),
        "scale": ("F32", scale.shape, scale.tobytes()),
    },
    memoryview(out),
    dtype="F32",
    shape=[2, 4],
    params={"axis": 0, "scale_dtype": "F32"},
)
```

Host inputs and a host destination use the existing CPU decoder. The initial resident
CUDA path supports `comfy.convrot_w4a4`, `comfy.asym_w4a8_int8` and `comfy.w6a8_int8`,
with F32, BF16 or F16 output. All parts and the destination must be on the same CUDA
device. Mixed host/device placement and other resident CUDA schemes raise
`UnsupportedError`; there is no CPU fallback or implicit weight upload. Device metadata
conversion and decode scratch may still require temporary allocations.

The call is synchronous: success means the output is ready. Do not mutate inputs or
access the destination concurrently with the call. Nonempty input/output spans must not overlap;
duplicate read-only input spans are allowed. The binding holds buffer leases and DLPack
owners through completion and orders CUDA work against their producer streams.

Buffer/descriptor and overlap refusals occur before output writes. Unknown schemes raise
`UnknownSchemeError`; invalid encoding/value cases can raise `DecodeError`. After a
value or runtime error, output contents are unspecified. If CUDA completion cannot be
proved, the binding retains the actual input/output owners rather than freeing storage
still in use; the output is not a valid result.

## Python: write, convert and publish

### Save a checkpoint file

`Writer` writes the cozytensors format. To create Safetensors files, first commit the
checkpoint and then use `tensorfs.materialize(..., format="safetensors")`. This writes
weight files plus the separate configuration and supporting files required by that layout.

```python
from array import array
import tensorfs

with tensorfs.Writer("./model.cozytensors") as writer:
    writer.metadata["source"] = "my-training-run"
    writer.pipeline("my_package.MyPipeline", {"version": 1})
    model = writer.component("model", cls="my_package.MyModel", config={"width": 2})
    model.put("weight", array("f", [1, 2]), dtype="F32", shape=[2])
    model.asset("vocab.txt", b"one\ntwo\n")

saved = writer.commit()
print(saved.id)       # immutable header identity
print(saved.tree)     # repository-tree identity, when present
```

The context commits on normal exit. `commit()` returns a `Checkpoint` and returns the same
result on later successful calls. A raw Python buffer needs explicit `dtype` and `shape`;
a DLPack input supplies them, and contradictory explicit values are refused.

A `.cozytensors` file is a small handle into the shared local store, not a self-contained
file. The destination gets only the header; tensor and supporting-file bytes are admitted
once into the store:

```text
./model.cozytensors          # binary metadata and content-hash references
$TENSORFS_HOME/blobs/...     # shared tensor and supporting-file bytes
```

The header contains pipeline/component configuration, tensor names, dtypes, shapes,
encodings and references. Store paths, credentials and endpoint URLs stay in configuration,
not in the file. Opening it uses the configured store and, when needed and allowed, the
configured source.

The store keeps the saved data while that file still holds this header. Once the file is
deleted, moved or overwritten with another header, the data ages out like any other cache
entry; open checkpoints keep their bytes meanwhile. To copy a checkpoint to another machine,
write the portable form with `tensorfs.materialize(saved, local_dir=...)` (described below).

The file identifies one checkpoint, not an enclosing repository tree. It contains no release
label or repository README/LICENSE files; those remain available on a checkpoint opened
through its repository.

`Writer(dest=None, ...)` keeps the checkpoint in the shared store as a cache entry, without a
header file. `dest` is a literal file path, not a directory; TensorFS does not infer a
variant label from the filename. `variant` names `model.<variant>.cozytensors` in a
`repo_id` upload and raises `InvalidError` without `repo_id`. To build a multi-variant folder
locally, write each header to its own `model.<variant>.cozytensors` path in that folder.

Its options and members are:

| Option or member | Use |
|---|---|
| `variant` | The uploaded file's label, `model.<variant>.cozytensors`; only with `repo_id`. |
| `base` | Start from an existing checkpoint and reuse unchanged data. |
| `resume` | Resume keyed local write state. |
| `repo_id`, `endpoint`, `token` | Stream the new checkpoint to a repository using supplied access. |
| `cache_dir` | Choose the local store/cache. |
| `metadata` | Mutable loading/provenance metadata dictionary. |
| `local_checkpoint` | A committed local result retained after an upload failure, if one exists. |
| `pipeline(cls, config)` | Set the pipeline description. |
| `component(name, cls=..., config=..., config_file=None)` | Create or reopen a component and return its `ComponentWriter`. `config_file` names the file its config exports as (`scheduler_config.json`); the default keeps a reopened or base component's, else `config.json`. |
| `has(component, key)` | Check a current/resumed write, excluding untouched tensors inherited from the base. |
| `commit()` | Finish and return the checkpoint; a first commit error is cached and re-raised. |
| `abandon()` | End without producing a new checkpoint. |

An exceptional context exit abandons the writer unless a resume key requests retained state.
An upload failure remains an error; it can also carry `error.checkpoint` and populate
`writer.local_checkpoint`. Retry publication of the retained result without rerunning training.

A `ComponentWriter` supports:

| Call | Use |
|---|---|
| `put(key, data, dtype=None, shape=None)` | Write plain values from a buffer or an array implementing DLPack, such as a Torch tensor. |
| `put_encoded(key, dtype=..., shape=..., scheme=..., params=..., parts=...)` | Preserve encoded data; each part is `role: (dtype, shape, buffer)`. |
| `copy(key, source_component, as_key=None)` | Reuse source key `key`, optionally renaming its destination. |
| `asset(name, data)` | Write a supporting file. |
| `order(keys)` | Set tensor construction order. |
| `has(key)` | Check a current/resumed write. |

### Import an existing model

```python
checkpoint = tensorfs.convert_to_cozytensors("hf://owner/model")
print(list(checkpoint.components))
```

`convert_to_cozytensors` imports an existing model into the cozytensors format and shared
store, then returns a `Checkpoint` you can read normally. It accepts one source or a
sequence: supported local files/folders, Hugging Face, Civitai and HTTPS. Options are
`converter`, `variant`, `cache_dir`, `token` and `progress`.
For a Hugging Face source, `hf://owner/model@revision` selects an upstream tag, branch or
commit. Without `@revision`, the importer resolves Hugging Face's `main` revision. It pins
and records the exact upstream commit automatically before importing; no separate
resolution call is needed. These are upstream source revisions, not Tensorhub branches.

## Python: configuration and advanced operations

### Choose defaults and credentials

```python
tensorfs.configure(endpoint="https://tensorhub.com", cache_dir="/path/to/cache", token=my_token,
                   ca_der=private_ca_der)
print(tensorfs.__version__)
```

`configure` optionally sets process-wide defaults for future calls. It returns `None`,
does not write a configuration file and does not change existing checkpoint handles.
You can instead pass settings directly to each operation:

```python
checkpoint = tensorfs.open("me/model", endpoint="https://tensorhub.com", token=my_token)
```

The default endpoint is `https://tensorhub.com`; environment settings include
`TENSORFS_ENDPOINT`, `TENSORFS_HOME` and `TENSORFS_TOKEN`. Public reads need no account.

`configure` updates only non-`None` arguments. Its token applies to open, materialize,
`download_file`, `upload_folder` and `Writer`; pass provider-specific tokens explicitly to
`convert_to_cozytensors`. `ca_der` (DER bytes) adds a private CA as a trusted root, on top of
the system roots, for every later call in the process; no other call takes it. An explicitly
constructed `Endpoint` has its own `token` and `ca_der`.

For refreshable access, supply a callable or object whose `headers(method, url, headers,
payload)` returns request headers. `payload` is the known body SHA-256, not body contents.
An optional `retry(method, url, headers, payload, status, response_headers)` returns whether
to try again after a response. These are duck-typed protocols, not classes you must inherit.
Progress and repository-change callback exceptions can propagate normally. An ordinary
exception from a token provider is wrapped as `DeniedError`; a TensorFS exception retains
its family. Credentials do not follow cross-origin redirects.

### Optional filesystem escape hatches

These operations are for tools that require paths or portable folders. They are not
steps in the normal `open`/`read` workflow.

#### Materialize a checkpoint or repository

`materialize` creates a regular folder and returns its path. Pass an open checkpoint to
write that checkpoint, or a repository name to reproduce repository content:

```python
folder = tensorfs.materialize(saved, local_dir="./portable")
st_folder = tensorfs.materialize(
    saved, local_dir="./safetensors-model", format="safetensors", dequantize="BF16"
)
repo_folder = tensorfs.materialize("me/model", revision="1.0", local_dir="./repo")
```

The default `format="cozytensors"` writes a native portable folder: `model.cozytensors`
(or its named variant) plus `chunks/`, retaining the same header bytes and content hashes.
`format="safetensors"` writes Safetensors weights with ordinary configuration and supporting
files. It can retain supported encoded representations, or use `dequantize="BF16"`, for
example, to write ordinary floating-point weights. Native cozytensors materialization
preserves encodings and rejects `dequantize`.

Materialization fetches required missing objects or fails; it does not silently create an
incomplete output. A checkpoint produces its own data and any known repository files.
A bare header cannot supply an unknown README, LICENSE or other variants. With a repository
name in native mode, omitting `variant` includes all variants; `components` can select
which components' payloads are materialized.

Supply `local_dir` for a checkpoint or for Safetensors output. A checkpoint already has
its source and credentials, so repository-selection/configuration keywords do not apply.
Safetensors output requires one selected checkpoint and does not accept `components`;
choose `variant` when a named repository has multiple variants. Native repository mode
also accepts `revision`, `variant`, `components`, `local_files_only`, `cache_dir`, `endpoint`
and `token`; without `local_dir`, it returns a managed snapshot folder.

#### Export for file-based libraries

Direct TensorFS/cozytensors loading is the normal integration path, including in our
Runtime. Libraries should consume component configuration, supporting-file bytes and
tensor arrays directly. Export is a convenience for libraries that still require
filenames; it is not a required preparation step.

```python
# Only for a consumer that requires a directory:
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained(checkpoint.components["tokenizer"].export_files())
folder = checkpoint.components["tokenizer"].export_files("./tokenizer-files")
```

`Component.export_files(directory=None)` writes configuration (as `config_file`) and
supporting files, fetching any missing asset bytes, and returns the folder's path. It never
exports tensor weights. Without a directory it writes to a folder the cache manages and
evicts like its other entries; with one it writes there and leaves it to you. The
full-checkpoint `tensorfs.materialize` operation also writes weight files when that is
needed.

#### Download files or publish a folder

```python
license_path = tensorfs.download_file("me/model", "LICENSE")
tree_id = tensorfs.upload_folder(repo_id="me/model", folder_path="./portable", token=my_token)
```

`my_token` in the examples is a credential you obtained for your endpoint.

`download_file` returns one file path and accepts revision, location and authentication
options, without component or variant selection.

`upload_folder` returns the exact published tree id. Its options are `revision`, `token`,
`endpoint` and `cache_dir`. Without a revision it appends an unreleased checkpoint;
it does not create a public release. Repository files such as README and LICENSE are
ordinary files in the uploaded folder.

### Work directly with repository references

```python
endpoint = tensorfs.Endpoint("https://tensorhub.com", token=my_token)
current = endpoint.repo("me/model")
if current is not None:
    document, etag = current
```

`Endpoint(url, token=None, ca_der=None, cache_dir=None)` provides `repo(repo_id)` and
`update_repo(repo_id, change)`. The latter returns the resulting document; `change` receives
the current document or `None` and returns its replacement or `None` to empty/remove it.
A conflict can cause the callback to run again, so it must tolerate retries.

### All Python entry points at a glance

| Entry point | Returns |
|---|---|
| `configure` | `None` |
| `open` | `Checkpoint`; `lazy=True` loads metadata, `lazy=False` waits for all checkpoint data locally |
| `convert_to_cozytensors` | `Checkpoint` |
| `materialize` | Local folder path; native cozytensors or Safetensors output |
| `download_file` | Local file path |
| `upload_folder` | Published tree id |

Normal loading uses `open` followed by component tensor reads. Use
`checkpoint.download(components=...)` to preload data into the shared store;
`checkpoint.missing(components=...)` inspects availability, and
`checkpoint.verify(components=...)` returns a `VerificationReport` for an explicit local
integrity audit. `materialize` and `download_file` are optional filesystem conveniences.
All of these Python calls are synchronous.

The returned data types are `Checkpoint`, `Pipeline`, `Component`, `Tensors`,
`TensorMetadata`, `Encoding`, `Array`, `Encoded` and `VerificationReport`. Writing uses `Writer` and `ComponentWriter`;
repository references use `Endpoint`. They are covered above. The companion `tensorfs.varena` placement API is separate and is unnecessary for
ordinary TensorFS or Diffusers/ComfyUI use.

## Errors and value types

Catch `tensorfs.Error` to handle TensorFS failures; inspect `.code` and `.detail`.
More specific Python exception classes are:

| Family | Python exception |
|---|---|
| Malformed request, document or value | `InvalidError` |
| Wrong document kind or major version | `FormatError` |
| Missing object/reference | `NotFoundError` |
| Hash/length mismatch | `IntegrityError` |
| Unknown encoding | `UnknownSchemeError` |
| Decode failure | `DecodeError` |
| Storage I/O | `StorageError` |
| Insufficient storage | `DiskFullError` |
| Network | `NetworkError` |
| Permission | `DeniedError` |
| Unsupported operation | `UnsupportedError` |
| Concurrent update | `ConflictError` |

Ordinary argument and mapping errors can still be `TypeError` or `KeyError`;
progress and repository-change callbacks can propagate their own exceptions.

Loading metadata and encoding parameters accept booleans, integers in the signed-64 through
unsigned-64 range, strings, lists and string-keyed maps. They do not accept floats, `None`
or bytes. Pipeline/component configuration is ordinary JSON, including finite floats and
null values.
