# TensorFS

Start with the [Python API and examples](API.md) for calls and returned objects.
The [Rust API](../tensorfs/API.md) is documented separately.

This guide describes the TensorFS 0.13.0 target API under integration. The current master
implementation and published wheels may still expose the preceding interface.

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
It preserves the stored representation by default: plain tensors return DLPack arrays,
and quantized tensors return their encoded arrays and parameters. Use `decode=True`, as
above, when the consumer needs ordinary floating-point weights. Frameworks wrap arrays
through their DLPack entry point. `component.config` is structured JSON, `component.asset(name)` reads an asset,
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
metadata; each tensor read fetches only the objects it needs. `snapshot_download` returns a plain folder, including
all variants by default, while `hub_download` returns one repository file.

## Convert, write and share

`convert` accepts files, directories, `hf://`, `civitai://` and HTTPS sources. It pins upstream revisions and records
source metadata. Recognized source layouts use converter tables; `converter=` declares a layout that cannot be
inferred safely. `export(checkpoint, directory)` renders ordinary Hugging Face filenames, configs and assets.
Encoded tensors retain their parts unless `dequantize=` requests ordinary values.

```python
from array import array
import tensorfs

with tensorfs.Writer("./model.cozytensors") as writer:
    component = writer.component("model", cls="", config={"width": 2})
    component.put("weight", array("f", [1, 2]), dtype="F32", shape=[2])
    component.asset("tokenizer.json", b"{}")
checkpoint = writer.commit()
tensorfs.export(checkpoint, "./portable", format="cozytensors")

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
