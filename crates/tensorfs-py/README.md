# TensorFS

TensorFS (Tensor Filesystem) is a library written in Rust, with python bindings, for downloading, storing, and loading tensors into memory efficiently. It uses an alternative storage-format we call `cozytensors`, designed to address safetensors' shortcomings.

### Problem 1: Safetensors is not self-describing

Safetensors describes arrays: their names, dtypes, shapes and byte offsets. It does not
standardize how those arrays form a quantized tensor. An INT8 weight and a scale array do not
tell a reader which axis to scale, how values were packed, or whether a rotation must be
undone. Its optional metadata can carry conventions, but the reader must already understand
them.

Each ecosystem therefore supplies its own rules, sometimes reserving tensor names for entries
that are actually metadata rather than model weights:

- ComfyUI's [MXFP8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/mxfp8.md)
  stores `<layer>.comfy_quant` as a `uint8` tensor containing JSON such as
  `{"format": "mxfp8"}`. Its loader recognizes that special name, parses the bytes as JSON,
  and interprets `<layer>.weight_scale` using ComfyUI's particular block layout.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
  packs non-tensor quantization properties into a tensor under a key such as
  `<layer>.weight.quant_state.bitsandbytes__nf4`. Its loader looks for that naming convention
  and unpacks the metadata to recover properties such as the logical dtype and block size.
  Transformers also publishes [quantization configuration alongside the weights](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes).

To safetensors, those metadata entries are just arrays with arbitrary names. It defines no
standard relationship between them and the weights they describe. Saving a valid file therefore
does not guarantee that another library can reconstruct the intended tensors: the reader must
implement the same conventions as the writer. A matching loader can read them reliably; a
different loader may reject the file or misinterpret its contents. Compatibility depends on
finding a loader that knows those conventions, rather than on the `.safetensors` extension alone.

cozytensors makes that meaning part of the format: each logical tensor declares its encoding,
parameters and stored arrays, with specified decoding rules. TensorFS can preserve the encoded
representation for a compatible compute kernel or decode a supported encoding into ordinary
tensors. Components, configuration and supporting assets are described in the same checkpoint,
so consumers do not have to reverse-engineer filenames and producer-specific tensor names.

### Problem 2: Safetensors has no built-in de-duplication

A safetensors file contains its own payload. Two checkpoints containing the same text encoder
or VAE still store those bytes in each file unless another storage layer deduplicates them.
The format itself has no reference that lets both files share one stored copy.

Hugging Face addresses this with [Xet](https://huggingface.co/docs/xet/en/download-protocol),
a general file-storage layer: it groups content-addressed chunks into containers called
*xorbs*, then reconstructs files from ranges within them. Xet works on consumer machines too,
and Hugging Face's [local file cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#shared-blobs-across-repos)
can share identical files across repositories. Its normal download path reconstructs an
ordinary file for the model loader; deduplication sits underneath that file interface.

TensorFS uses the shared chunk store as the model's native storage. A `.cozytensors` header
references immutable chunks; checkpoints sharing a chunk keep one copy on disk, and readers
load directly from that store without reconstructing a separate full model file. A fine-tune
can reuse its base checkpoint's unchanged chunks and add only new ones. This saves local disk
space as well as transfer bytes on an ordinary workstation, without requiring a deduplicating
filesystem or a hosted storage service.



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
The [cozytensors specification](../tensorfs/spec/cozytensors.md) defines the format and exact decoding rules.

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

These receipts predate the 0.13 API. They are separate runs and reader revisions: the full-load run predates the final CBOR header
and lazy window registration; the block run includes them. They measure serialized-byte loading,
excluding dequantization, model construction and inference. Hardware and cache conditions affect
results. [Measurements and provenance](gates/speed/measurements.json) include the earlier failed
window cell; [the recorded harness](https://github.com/cozy-creator/cozy-creator/blob/0a6e71ee2c919db3e6e9b5844c8234ca9749cedb/crates/tensorfs/gates/speed/speedgate.py)
defines the workloads. Download projections are separate from these measured timings.

## Get started

The target TensorFS 0.13 interface uses the component API and hard-cut header/tree format. Python 3.12 or later is
required; the current master implementation is still being integrated. The wheel includes the Python module and `tfs`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install tensorfs

tfs import hf://hf-internal-testing/tiny-sdxl-pipe --name local/example
tfs inspect local/example
tfs export local/example ./example --safetensors
```

```python
import tensorfs

checkpoint = tensorfs.open("local/example", local_files_only=True)
unet = checkpoint.components["unet"]
info = unet.tensors["conv_in.weight"]
value = unet.tensors.read("conv_in.weight", decode=True)
print(unet.cls, info.dtype, info.shape, value.nbytes)
```

`read` accepts one name, a list of names, or no selection for the whole component.
It preserves the stored representation by default: plain tensors return TensorFS `Array` objects,
and quantized tensors return their encoded arrays and parameters. Use `decode=True`, as
above, when the consumer needs ordinary floating-point weights. Frameworks wrap arrays
through their DLPack entry point. DLPack is the memory-sharing protocol, not a separate
array library to install. `component.config` is structured JSON, `component.asset(name)` reads an asset,
and explicit `component.export_files(directory)` is only a compatibility convenience for legacy path-only loaders.

## Checkpoints and repositories

A cozytensors variant is one header plus the chunks and assets it names. Its identity is the header's object ref.
The header contains a pipeline class/config and named components with class/config, tensors and assets. Run data
is self-contained; README, LICENSE, images and code remain ordinary repository files.

```text
repo → release → tree
                  ├─ model.int8.cozytensors → header → chunks and assets
                  ├─ model.bf16.cozytensors → header → chunks and assets
                  └─ README.md, LICENSE, …
```

A release names a `tensorfs.tree/1` tree. `variant=` selects a cozytensors file; `open` requires an explicit variant
when there is more than one. `revision="sha256:…"` can select an unpublished tree or header. Opening downloads
metadata by default; each tensor read fetches only the objects it needs. Set `lazy=False` to wait for the whole
checkpoint in the shared store, or call `checkpoint.download(components=["unet"])` to preload a selection.
`materialize` returns a plain folder, including all variants by default, while `download_file` returns one
repository file. These folder/file operations are optional compatibility tools.

## Convert, write and share

`convert_to_cozytensors` accepts files, directories, `hf://`, `civitai://` and HTTPS sources. It pins upstream revisions and records
source metadata. Recognized source layouts use converter tables; `converter=` declares a layout that cannot be
inferred safely. `materialize(checkpoint, local_dir=directory, format="safetensors")` writes ordinary Hugging Face filenames, configs and assets.
Encoded tensors retain their parts unless `dequantize=` requests ordinary values.

```python
from array import array
import tensorfs

with tensorfs.Writer("./model.cozytensors") as writer:
    component = writer.component("model", cls="", config={"width": 2})
    component.put("weight", array("f", [1, 2]), dtype="F32", shape=[2])
    component.asset("tokenizer.json", b"{}")
checkpoint = writer.commit()
tensorfs.materialize(checkpoint, local_dir="./portable", format="cozytensors")

tensorfs.upload_folder(repo_id="local/example", revision="draft", folder_path="./portable")
tensorfs.upload_folder(repo_id="acme/example", folder_path="./portable", endpoint="file:///srv/origin")
```

A normal save writes one .cozytensors header file, with its tensors and supporting-file bytes retained in
the shared store. The file contains content hashes, not physical store paths. `Writer()` without a destination
keeps the checkpoint in the store without writing a user-visible header file. An explicit native portable export
writes `model.cozytensors` (or a named variant) plus complete `chunks/<sha256>` objects. `base=` reuses unchanged
tensors and `resume=` journals local work. `Writer(repo_id=…, endpoint=…, token=…)` streams new objects
and appends an unreleased tree. Its successful checkpoint carries the exact published tree; an upload failure keeps
`writer.local_checkpoint` and remains an error.

The default endpoint is `https://tensorhub.com`; `configure(endpoint=…)` or `TENSORFS_ENDPOINT` changes it.
`token=` accepts a string, a native DPoP provider, a callable, or an object with `headers(method, url, headers,
payload)` and optional `retry(method, url, headers, payload, status, response_headers)`. The payload is its SHA-256
hex digest when known. Credentials are presented afresh per request and removed on redirects to another origin.
Public data needs no account. [Distribution](../tensorfs/spec/distribution.md) defines endpoint routes and publication.

## Storage and ownership

The shared local store is internal. Its default root is `~/.cache/tensorfs`; `cache_dir=` and `TENSORFS_HOME` override it. Processes
share immutable objects and coordinated records. Admission hashes bytes before installing them by rename; ordinary
opens check lengths. Saved local roots are retained separately from evictable download-cache entries; deleting a header file only removes that handle. Explicit verification rehashes on demand. TensorFS is a library and has no daemon.

Runtime and TensorD use this same public API. Callers read and write the shared store and upload with a supplied
credential. TensorFS owns I/O and decoding; varena owns memory placement. Installing the wheel does
not grant access to private data or managed executions.

The [usage guide](../tensorfs/README.md) summarizes the target public calls. The
[native type declarations](python/tensorfs/_ext.pyi) describe the checked-in implementation while the 0.13 bindings are integrated. Every format, storage and transport failure derives from `tensorfs.Error`; `.code` identifies one of
twelve families, including `denied` and `conflict`. Unknown encodings remain inspectable and transferable; decoding
requires that scheme's implementation.

## Format and development

The [cozytensors specification](../tensorfs/spec/cozytensors.md) is the format contract. The header is deterministic CBOR;
trees and repository documents are canonical JSON. Construction order contributes to identity. The
[conformance corpus](../tensorfs/conformance/) checks independent readers, and the [source fixtures](../tensorfs/vectors/) retain source
provenance without model payloads. The [standalone end-to-end check](scripts/e2e.sh) converts a tiny public model,
exports it for Diffusers, compares its render and downloads the same bytes through an ordinary endpoint.
