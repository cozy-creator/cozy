# TensorFS Python API

Import TensorFS, perform an operation, then work with the checkpoint, component or array
it returns. This guide covers the complete planned Python API by task.

This is the **0.13 target**, currently being integrated. The implementation on `master`
can still expose the preceding interface. The [Rust API](../tensorfs/API.md) and
[technical file-format specification](../tensorfs/spec/cozytensors.md) are separate documents.

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
# The same call accepts a local model folder:
local = tensorfs.open("./model")
print(list(local.components))
```

`open` returns a `Checkpoint`. It fetches metadata first; reading tensors fetches the
objects those tensors need. `revision` selects a release or immutable reference;
`variant` selects a representation such as `bf16` or `int8`. Without a variant, a
single-variant tree is unambiguous; a multi-variant tree requires a choice.

For a private model, pass `token=...`. Other options are `endpoint`, `cache_dir`,
`local_files_only` and `ca_der` (private CA bytes). `local_files_only=True` forbids network
access and fails if the required data is unavailable locally.

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
| `config_file` | Configuration filename, usually `config.json`. |
| `tensors` | A `Tensors` mapping, described below. |
| `assets` | Asset names mapped to `(sha256_hex, byte_length)` references. |
| `asset(name)` | The named asset's bytes. |
| `path` | A folder containing this component's configuration and assets; obtaining it does not fetch tensor weights. |

For example, a tokenizer that needs files can use
`checkpoint.components["tokenizer"].path`.

### Inspect tensor metadata without reading weights

```python
print(list(unet.tensors))
info = unet.tensors["conv_in.weight"]
print(info.dtype, info.shape, info.encoding)
```

`Tensors` behaves like a read-only mapping: `len`, iteration, membership, indexing,
`keys()`, `values()`, `items()` and `get(key, default)` work normally. A missing indexed
key raises `KeyError`; `get` returns its default.

The returned `Info` describes the whole tensor: its logical `dtype`, `shape`, optional
`encoding`, and stored `parts`. It contains metadata, not the tensor values. For ordinary
use, inspect its shape/dtype and then call `read` or `read_all`. The storage details below
are useful when consuming an encoding directly.

### Read a tensor, or a component's weights

```python
weight = unet.tensors.read("conv_in.weight")
weights = unet.tensors.read_all()
selected = unet.tensors.read_all(["conv_in.weight", "conv_in.bias"])
```

`read` returns CPU data. `read_all(keys=None, device="cpu", decode=True)` returns a
name-to-value dictionary and also accepts `device="cuda"` or `device="cuda:N"`.

A decoded value is an `Array`, with `dtype`, `shape`, `nbytes` and `tobytes()`. To hand an
array to a framework, use DLPack:

```python
import torch

torch_weight = torch.from_dlpack(weight)
gpu_weights = unet.tensors.read_all(device="cuda:0")
state_dict = {name: torch.from_dlpack(value) for name, value in gpu_weights.items()}
```

The returned arrays and their framework consumers retain the storage they need.

### Keep quantized weights encoded

```python
value = unet.tensors.read("some_quantized.weight", decode=False)
print(value.scheme, value.params)
print(value.parts.keys())
```

`Encoding` describes how the stored arrays reconstruct the logical tensor: a scheme
name such as `int8.rowwise`, plus its parameters. For example, a logical weight matrix
with shape `(4096, 4096)` can be stored as an int8 array named `weight` with that shape
and a separate scale array named `scale` with shape `(4096,)`. Each row is reconstructed
as `weight[row, column] * scale[row]`.

`Part` is the current name for the metadata of **one of those stored arrays**: its
`dtype`, `shape`, `chunk`, byte `offset` and byte `length`. It does not mean a slice of
the logical tensor or a storage chunk. `Info` describes the whole logical tensor;
its `parts` mapping describes the arrays used to represent it.

For an encoded tensor, `decode=False` returns `Encoded`: the loaded `scheme`, `params`,
and a role-to-`Array` mapping named `parts`. For a plain tensor it still returns `Array`.
A consumer of encoded weights must understand that scheme and its parameters.

### Read into an existing allocation

```python
destination = bytearray(weight.nbytes)
unet.tensors.read_into("conv_in.weight", destination, offset=0)
copied_bytes = bytes(destination)

# For stored encoded bytes, size the destination for that part and select its role.
encoded = unet.tensors.read("some_quantized.weight", decode=False)
packed = bytearray(encoded.parts["weight"].nbytes)
unet.tensors.read_into("some_quantized.weight", packed, role="weight", offset=0)
```

The destination determines the number of bytes copied. `offset` is a byte offset, and
`role` selects an encoded part. Without a role, encoded weights are decoded. This call
does not convert to the destination's dtype; provide the appropriate size and representation.

### Prefetch, verify and manage a checkpoint

```python
def on_progress(stage, completed, total):
    print(f"{stage}: {completed}/{total}")

missing = checkpoint.missing(components=["unet"])
checkpoint.fetch(components=["unet"], priority=0, progress=on_progress)
missing_or_corrupt = checkpoint.verify()
```

`missing(components=None)` reports absent object references. `fetch(components=None,
priority=0, progress=None)` fetches selected or all components. Progress callbacks receive
`(stage, completed, total)`; download counts are bytes, conversion counts are tensors.
`verify()` hashes available objects and reports missing or corrupt references.

Additional checkpoint properties and operations:

| Attribute or call | Meaning |
|---|---|
| `id`, `header_ref` | Header identity: `sha256:…` string and `(hex, length)` pair. May be absent for a parsed-only header. |
| `tree`, `tree_ref` | Repository-tree identity in the same two forms, when known. |
| `name`, `repo_id` | Repository name; `repo_id` is an alias. |
| `release`, `revision` | Release label, when known; `revision` is an alias. |
| `variant` | Selected variant. |
| `pipeline`, `components` | Pipeline description and ordered component mapping. |
| `metadata` | Loading/provenance metadata. |
| `chunks`, `chunk_paths` | Chunk references and their local paths. |
| `path` | Fetch and materialize the whole checkpoint as a folder. |
| `Checkpoint.parse(header_bytes)` | Inspect a binary header without a store; it does not grant access to its objects. |
| `open_chunk(index)` | Return a new read-only OS file descriptor; the caller must close it. |
| `close()` | Release this checkpoint handle. Existing child handles remain valid. |

Use `with tensorfs.open(...) as checkpoint:` for automatic close. `Array` implements
`__dlpack__` and `__dlpack_device__`, including stream/device/copy negotiation. Objects
provide normal debug representations (`repr`); an `Encoding` also has a string form.
These representations are not serialization formats.

## Python: write, convert and publish

### Write a model folder

```python
from array import array
import tensorfs

with tensorfs.Writer("./output", variant="f32") as writer:
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

`Writer(dest=None, ...)` writes to the shared store when no folder is supplied. Its options are:

| Option or member | Use |
|---|---|
| `variant` | Name the representation being written. |
| `base` | Start from an existing checkpoint and reuse unchanged data. |
| `resume` | Resume keyed local write state. |
| `repo_id`, `endpoint`, `token`, `ca_der` | Stream the new checkpoint to a repository using supplied access. |
| `cache_dir` | Choose the local store/cache. |
| `metadata` | Mutable loading/provenance metadata dictionary. |
| `local_checkpoint` | A committed local result retained after an upload failure, if one exists. |
| `pipeline(cls, config)` | Set the pipeline description. |
| `component(name, cls=..., config=..., config_file="config.json")` | Create or reopen a component and return its `ComponentWriter`. |
| `has(component, key)` | Check a current/resumed write, excluding untouched tensors inherited from the base. |
| `commit()` | Finish and return the checkpoint; a first commit error is cached and re-raised. |
| `abandon()` | End without producing a new checkpoint. |

An exceptional context exit abandons the writer unless a resume key requests retained state.
An upload failure remains an error; it can also carry `error.checkpoint` and populate
`writer.local_checkpoint`. Retry publication of the retained result without rerunning training.

A `ComponentWriter` supports:

| Call | Use |
|---|---|
| `put(key, data, dtype=None, shape=None)` | Write plain values from a buffer or DLPack array. |
| `put_encoded(key, dtype=..., shape=..., scheme=..., params=..., parts=...)` | Preserve encoded data; each part is `role: (dtype, shape, buffer)`. |
| `copy(key, source_component, as_key=None)` | Reuse source key `key`, optionally renaming its destination. |
| `asset(name, data)` | Write a supporting file. |
| `order(keys)` | Set tensor construction order. |
| `has(key)` | Check a current/resumed write. |

### Convert an existing model and export ordinary files

```python
checkpoint = tensorfs.convert("hf://owner/model", variant="bf16")
files = tensorfs.export(checkpoint, "./exported", dequantize="BF16")
```

`convert` accepts one source or a sequence. Sources include supported local files/folders,
Hugging Face, Civitai and HTTPS. Options are `converter`, `variant`, `cache_dir`, `token`
and `progress`. It returns a `Checkpoint`. `resolve_source(source, token=None)` returns the
pinned source reference without performing the conversion.

`export(checkpoint, directory, format="safetensors", dequantize=None)` returns written
paths. Omit `dequantize` to retain supported encoded representations in the export.

### Download files or publish a folder

```python
folder = tensorfs.snapshot_download("me/model", variant="bf16")
license_path = tensorfs.hub_download("me/model", "LICENSE")
tree_id = tensorfs.upload_folder(repo_id="me/model", folder_path="./output", token=my_token)
```

`my_token` in the examples is a credential you obtained for your endpoint.

`snapshot_download` returns a folder; options are `revision`, `variant`, `components`,
`local_dir`, `local_files_only`, `cache_dir`, `endpoint`, `token` and `ca_der`. With no
variant it downloads all variants. `hub_download` returns one file path and accepts the
same location/authentication options, without component or variant selection.

`upload_folder` returns the exact published tree id. Its options are `revision`, `token`,
`endpoint`, `cache_dir` and `ca_der`. Without a revision it appends an unreleased checkpoint;
it does not create a public release. Repository files such as README and LICENSE are
ordinary files in the uploaded folder.

## Python: configuration and advanced operations

### Choose defaults and credentials

```python
tensorfs.configure(endpoint="https://tensorhub.com", cache_dir="/path/to/cache", token=my_token)
print(tensorfs.__version__)
```

The default endpoint is `https://tensorhub.com`; environment settings include
`TENSORFS_ENDPOINT`, `TENSORFS_HOME` and `TENSORFS_TOKEN`. Public reads need no account.
Private CA input is DER bytes through `ca_der`.

In the current frozen wrapper, `configure` updates only non-`None` arguments. Its configured
token applies to open/download/upload/Writer. Pass provider-specific tokens explicitly to
`convert` and `resolve_source`; an explicitly constructed `Endpoint` has its own token.
Configuration scoping is being reviewed as the common Rust client is integrated.

For refreshable access, supply a callable or object whose `headers(method, url, headers,
payload)` returns request headers. `payload` is the known body SHA-256, not body contents.
An optional `retry(method, url, headers, payload, status, response_headers)` returns whether
to try again after a response. These are duck-typed protocols, not classes you must inherit.
Progress and repository-change callback exceptions can propagate normally. An ordinary
exception from a token provider is wrapped as `DeniedError`; a TensorFS exception retains
its family. Credentials do not follow cross-origin redirects.

The optional `DPoP` helper supports proof-bearing access where the endpoint requires it.
Its `key` argument is a private P-256 JWK encoded as a JSON string:

| Call or property | Meaning |
|---|---|
| `DPoP.token(key, access_token, resource=None, endpoint=None)` | Use an existing token and signing key. |
| `DPoP.grant(key, token_endpoint=..., client_id=..., resource=..., claims=..., endpoint=None, ca_der=None)` | Create a provider for the endpoint's supported grant flow. |
| `issued_token()` | A token obtained by the provider, when present. |

Given that key and an access token from your authorization setup:

```python
provider = tensorfs.DPoP.token(private_jwk_json, access_token)
checkpoint = tensorfs.open("me/private-model", token=provider)
```

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
| `open` | `Checkpoint` |
| `convert` | `Checkpoint` |
| `resolve_source` | Pinned source string |
| `export` | List of written paths |
| `snapshot_download` | Local folder path |
| `hub_download` | Local file path |
| `upload_folder` | Published tree id |

The returned data types are `Checkpoint`, `Pipeline`, `Component`, `Tensors`, `Info`,
`Part`, `Encoding`, `Array` and `Encoded`. Writing uses `Writer` and `ComponentWriter`;
advanced authorization/reference operations use `DPoP` and `Endpoint`. They are covered
above. The companion `tensorfs.varena` placement API is separate and is unnecessary for
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
