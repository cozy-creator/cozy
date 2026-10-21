# TensorFS

TensorFS (Tensor Filesystem) is a library written in Rust, with python bindings, for downloading, storing, and loading tensors into memory efficiently. It uses an alternative storage-format we call `cozytensors`, designed to address safetensors' shortcomings.

### Problem 1: Safetensors is Not Self-Describing

Safetensors' header gives a list of keys, their names, dtypes, shapes, and where to find them in the file using byte offsets, but does not describe how those bytes form a quantized tensor.

For example, all the bytes can reach GPU memory intact, with valid dtypes and shapes, and still be interpreted incorrectly:

- **Scaling the wrong axis.** Suppose an INT8 matrix is `[[10, 20], [30, 40]]` and its
  scales are `[0.1, 0.2]`. Scaling each row gives weights `[[1, 2], [6, 8]]`; scaling each
  column gives `[[1, 4], [3, 8]]`. Both are valid 2×2 matrices. With input `[1, 1]`, they
  produce `[3, 14]` and `[5, 11]` respectively. Nothing about reading the file detects
  that the second result is wrong.
- **Unpacking in the wrong order.** For simple 4-bit integer weights, the byte `0x21`
  can represent `[1, 2]` when the low four bits come first, or `[2, 1]` when the high four
  bits come first. A reader that assumes the wrong order still loads valid numbers, but
  assigns them to the wrong input channels. With input `[1, 10]`, the dot product becomes
  `12` instead of `21`.

The file is readable in both cases. Across a model, these mistakes can turn otherwise
successful inference into nonsense. What is missing is the agreed meaning of its arrays:
the scale axis, packing order and other encoding rules.

It does not
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

Much of the model-loading ecosystem is therefore held together by agreements between producers
and loaders: expected filenames, tensor names, metadata keys, configuration schemas and code
that knows how to interpret them. These agreements can be documented and reliable within one
ecosystem, but they are not part of the safetensors format. To a generic safetensors reader,
the metadata entries above are just arrays with arbitrary names, with no standard relationship
to the weights they describe.

Two tools can both support safetensors and still disagree about how to reconstruct the model.
A valid file is not enough: compatibility depends on finding a loader that implements the
writer's conventions. A matching loader can read it reliably; a different loader may reject
it or misinterpret its contents. The `.safetensors` extension alone does not establish that
shared contract.

cozytensors makes that meaning part of the format: each logical tensor declares its encoding,
parameters and stored arrays, with specified decoding rules. TensorFS can preserve the encoded
representation for a compatible compute kernel or decode a supported encoding into ordinary
tensors. Components, configuration and supporting assets are described in the same checkpoint,
so consumers do not have to reverse-engineer filenames and producer-specific tensor names.

#### Why the exact encoding matters

These are concrete cases from our format and importer work, captured by the
[encoding definitions](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs/spec/cozytensors.md):

| Case | Why dtype, shape or names were not enough | What cozytensors records |
|---|---|---|
| Ordinary INT8 versus ConvRot INT8 | Both can contain an INT8 matrix and row scales, but ConvRot quantized rotated weights. Multiplying by the scales alone does not reconstruct the original weight. | Distinct schemes, the Hadamard rotation and its size, and the decoding formula. |
| NVFP4 from different producers | ComfyUI and ModelOpt can order the two 4-bit values in a byte differently and lay out block scales differently. Global scale conventions can multiply or divide. | Nibble order, scale layout and scale convention, rather than guessing from an “NVFP4” label. |
| H3's NVFP4 AWQ text encoder | Our audit had to establish that `pre_quant_scale` multiplies each input column. Omitting it or dividing by it produces different weights. | The optional per-column factor and its exact role in decoding. |
| NF4 versus FP4 | Both use 4-bit codes, but the codes index different value tables. Some files also quantize the scales themselves. | The codebook's identity, block size and whether nested scale decoding is required. |
| GPTQ and other packed 4-bit formats | GPTQ versions differ in how zero points are stored; AWQ uses a different packed element order. The same integer dtype does not identify either rule. | The specific scheme, group size, zero-point convention and any required index/permutation arrays. |

Our earlier ComfyUI W4A4 survey also confused a kernel's tile size with quantization grouping;
the format actually used per-row scales. That was a mistake in interpreting the producer's
format, not something a tensor's dtype or shape could correct for us.

We also tested guessing NVFP4 layouts from tensor shapes or from how plausible decoded values
looked. Neither reliably distinguished the formats. An importer must resolve the producer's
convention or refuse an ambiguous source. Once imported, those decisions are explicit in the
cozytensors header, so every reader can apply the same rules.

Successful loading does not prove correct interpretation. In a
[reported GPTQ compatibility failure](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ/discussions/12),
the model ran but produced incoherent replies; the maintainer identified a mismatch between
the writer's activation-order setting and the reader's `desc_act` configuration, and the user
confirmed that correcting it resolved the problem. cozytensors makes that setting and the
group-index mapping part of the GPTQ encoding. A format cannot eliminate buggy kernels or bad
weights, but it can specify what the bytes mean instead of leaving that agreement implicit.

ComfyUI has also had a [WAN 2.2 regression](https://github.com/Comfy-Org/ComfyUI/issues/13238)
where inference ran but produced noise, with `scaled_fp8`, `scale_input` and `scale_weight`
reported as unexpected keys. The [fix](https://github.com/Comfy-Org/ComfyUI/commit/e2ddf28d78f190b27d136668a7dc15c7f0ec75dc)
re-ran legacy quantization conversion after stripping the checkpoint's tensor-name prefix;
the reporter confirmed that it resolved the problem. The arrays were readable, but the
conventions carrying their quantization meaning had not been applied correctly.

#### Choose the representation your runtime needs

In these examples `me/model` stands for a repository you can read, here with `int8` and `bf16`
variants. For a quantized matrix, inspect and retain its stored encoding when your runtime has a
matching compute kernel:

```python
import tensorfs

checkpoint = tensorfs.open("me/model", variant="int8")
tensors = checkpoint.components["unet"].tensors
stored = tensors.read("time_embedding.linear_1.weight", decode=False)
print(stored.scheme, stored.params)
# stored.parts holds the encoded arrays; device="cuda:0" puts them on the GPU for that kernel.
```

A runtime that expects ordinary floating-point weights can instead request decoding:

```python
import torch

dense = tensors.read("time_embedding.linear_1.weight", decode=True)
weight = torch.from_dlpack(dense)  # ordinary values; no input-format-specific unpacker
weight = weight.to(torch.bfloat16)  # only if this runtime requires BF16
```

`decode=True` reconstructs the header's declared logical dtype; it does not always mean BF16.
The same call can decode supported ConvRot, GPTQ, NF4 or other stored schemes. `device="cuda:0"`
chooses the destination, not a different quantization.

#### Convert to another quantization at load time

Changing from one quantization to another means decoding the stored values, then applying
the target encoder. For example, a runtime that expects symmetric, per-row INT8 can prepare
weights decoded from ConvRot or another supported encoding as follows:

```python
values = torch.from_dlpack(tensors.read("time_embedding.linear_1.weight", decode=True)).float()
scales = values.abs().amax(dim=1, keepdim=True) / 127
scales = torch.where(scales == 0, torch.ones_like(scales), scales)
weights_i8 = (values / scales).round().clamp(-127, 127).to(torch.int8)
row_scales = scales.squeeze(1)
# weights_i8 and row_scales are ready for a matching rowwise-INT8 kernel.
```

This target representation is `int8.rowwise` with an `F32` `scale` part per row.
TensorFS handles the source encoding; the target quantizer above is ordinary PyTorch code.
TensorFS has no re-encoding call: `ComponentWriter.put_encoded(...)` stores already-encoded arrays.
Prepare this once when loading, rather than on every forward pass. Re-quantizing can add error;
methods requiring calibration need their own calibration step, not merely a dtype cast.

### Problem 2: The weights file is not a complete model package

A usable model needs more than its weight arrays: its architecture and configuration,
the components in its pipeline, and supporting data such as tokenizer vocabularies.
Safetensors does not standardize how to package those together. The diffusion ecosystem has
two common [layouts](https://huggingface.co/docs/diffusers/main/en/using-diffusers/other-formats):

- **Single-file checkpoints**, common in ComfyUI and Automatic1111, collect model weights in
  one `model.safetensors` file. Tensor-name prefixes and loader conventions identify its
  components. One file is convenient to share, but it does not necessarily include the
  architecture configuration, tokenizer files or other data needed to run the pipeline.
- **Diffusers directories** put components in separate subfolders and describe them with
  `model_index.json` and configuration files. A typical directory includes:

```text
model_index.json
unet/config.json
unet/diffusion_pytorch_model.safetensors
scheduler/scheduler_config.json
tokenizer/tokenizer.json
...
```

The loader knows which filenames to seek, how subfolders identify components, and which
library classes the JSON describes. Passing only the weights file to another tool loses
that surrounding contract. A different ecosystem may use another layout or infer the model
from its tensor names. The array container can be valid while the model package is incomplete
or unintelligible to that reader.

These are different organizations of model data, not different tensor file types: both can
use safetensors. Even Diffusers' `from_single_file()` may infer the architecture from tensor
keys and obtain configuration from a known base-model repository, or require an explicit
`config` argument. A locally available weights file alone does not guarantee a fully local,
self-contained load.

A [confirmed Transformers regression](https://github.com/huggingface/transformers/pull/47988)
illustrates the consequence: OLMo loaded and ran, but produced repetitive nonsense because
`tokenizer.json` and `tokenizer_config.json` disagreed about end-of-sequence handling and a
loader change stopped applying the override. The weights were not the problem; interpretation
of the accompanying metadata was.

cozytensors puts the component descriptions, class names, parsed configurations and asset
references in its header. `checkpoint.components["unet"].config` reads the configuration
directly; a tokenizer's `asset("tokenizer.json")` reads its supporting data. Consumers do not
need to discover that structure by walking a particular folder layout. The header supplies
the manifest while payloads remain in the shared store, and the consuming library supplies
the model code. Exporting ordinary folders is a compatibility convenience.

### Problem 3: Safetensors has no built-in de-duplication

A safetensors file contains its own payload. Two checkpoints containing the same text encoder
or VAE still store those bytes in each file unless another storage layer deduplicates them.
The format itself has no reference that lets both files share one stored copy.

Hugging Face also layers [sharding](https://huggingface.co/docs/transformers/main/en/big_models#sharded-checkpoints)
on top of safetensors to handle large models: split the checkpoint into files such as
`model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`, then add a
`model.safetensors.index.json` whose `weight_map` maps each tensor name to its shard.
This lets loaders process one shard at a time and bound transient memory use. It is another
file-layout convention that the application must understand, not a feature defined by the
safetensors container itself.

Sharding alone does not share unchanged regions between different shard files. An identical
whole shard can be reused by a file cache, but partially changed shards are still distinct
files. Splitting files and deduplicating their contents solve different problems.

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

TensorFS also divides large payloads, but the tensor-to-chunk references are part of the
checkpoint format. Readers follow those references directly; they do not need a separate
framework-specific shard index or numbered shard filenames to locate each tensor.



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
The [cozytensors specification](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs/spec/cozytensors.md) defines the format and exact decoding rules.

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
results. [Measurements and provenance](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs-py/gates/speed/measurements.json) include the earlier failed
window cell; [the recorded harness](https://github.com/cozy-creator/cozy-creator/blob/0a6e71ee2c919db3e6e9b5844c8234ca9749cedb/crates/tensorfs/gates/speed/speedgate.py)
defines the workloads. Download projections are separate from these measured timings.

## Get started

It needs CPython 3.10 or later. Wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows
(x86_64, arm64) include the Python module and the `tfs` command:

```bash
uv venv
source .venv/bin/activate
uv pip install tensorfs

tfs convert hf://hf-internal-testing/tiny-sdxl-pipe --name local/example@1
tfs inspect local/example@1
tfs export local/example@1 ./example --safetensors
```

```python
import tensorfs

checkpoint = tensorfs.open("local/example", revision="1", local_files_only=True)
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
and `component.export_files()` (a cache-managed folder, or the directory you name) is only a convenience for
path-only loaders.

## Checkpoints and repositories

A cozytensors variant is one header plus the chunks and assets it names. Its identity is the header's object ref.
The header contains a pipeline class/config and named components with class/config, tensors and assets. Run data
is self-contained; README, LICENSE, images and code remain ordinary repository files.

```text
repo → tag → commit (tree)
               ├─ model.int8.cozytensors → header → chunks and assets
               ├─ model.bf16.cozytensors → header → chunks and assets
               └─ README.md, LICENSE, …
```

A repo is commits and tags, as in git. A commit is an immutable `cozytensors.tree/1` tree; a tag is a movable name
for one. Public tags are releases, named by versions; private tags are the owner's. `revision=` names a tag, a commit
(`sha256:<tree>`) or a variant header (`sha256:<header>`); without one, `open` takes the newest public final release.
`variant=` selects a cozytensors file; `open` requires an explicit variant when there is more than one. Opening downloads
metadata by default; each tensor read fetches only the objects it needs. Set `lazy=False` to wait for the whole
checkpoint in the shared store, or call `checkpoint.download(components=["unet"])` to preload a selection.
`materialize` returns a plain folder, including all variants by default, while `download_file` returns one
repository file. These folder/file operations are optional compatibility tools.

## Convert, write and share

`convert_to_cozytensors` accepts files, directories, `hf://`, `civitai://` and HTTPS sources. It pins upstream revisions and records
source metadata. Recognized source layouts use converter tables; `converter=` declares a layout that cannot be
inferred safely. `materialize(checkpoint, local_dir=directory, format="safetensors")` writes ordinary Hugging Face filenames, configs and assets.
Encoded tensors retain their parts unless `decode=True` writes them in their logical dtypes.

```python
from array import array
from pathlib import Path
import tensorfs

with tensorfs.Writer("./model.cozytensors") as writer:
    component = writer.component("model", config={"width": 2})
    component.put("weight", array("f", [1, 2]), dtype="F32", shape=[2])
    component.asset("tokenizer.json", b"{}")
checkpoint = writer.commit()
tensorfs.materialize(checkpoint, local_dir="./portable", format="cozytensors")

tensorfs.upload_folder(repo_id="local/example", revision="draft", folder_path="./portable")  # private tag "draft"
tensorfs.upload_folder(repo_id="acme/example", folder_path="./portable", endpoint=Path("origin").absolute().as_uri())
```

A `.cozytensors` file is a small handle into the shared local store, not a self-contained file: a save writes the
header, and its tensors and supporting-file bytes stay in the store while that file holds this header. Delete or
overwrite the file and the data expires like any other local copy. The copy for another machine is
`materialize(checkpoint, local_dir=…)`: `model.cozytensors` (or a named variant) plus complete `chunks/<sha256>`
objects. `Writer()` without a destination keeps the checkpoint in the store as a cache entry. `base=` reuses
unchanged tensors and `resume=` journals local work. `Writer(repo_id=…, revision=…, endpoint=…, token=…)` streams
new objects, makes a commit and moves the private tag `revision` (default `main`) to it; `variant=` names the file
`model.<variant>.cozytensors` and needs `repo_id`. The commit goes on top of the tag: its other variants and files
stay, so a bf16 and then an int8 writer to `main` leave both. `upload_folder` does the same for a folder (removing
only what `delete_patterns` names) and returns the commit.
Publishing (making a tag public, a release) is the CLI's `cozy model publish`. A successful checkpoint carries the exact
commit; an upload failure keeps `writer.local_checkpoint`, which can be uploaded again for 7 days, and remains an error.

The default endpoint is `https://tensorhub.com`; `configure(endpoint=…)` or `TENSORFS_ENDPOINT` changes it.
`token=` accepts a string, a callable, or an object with `headers(method, url, headers,
payload)` and optional `retry(method, url, headers, payload, status, response_headers)`. The payload is its SHA-256
hex digest when known. Credentials are presented afresh per request and removed on redirects to another origin.
Public data needs no account. [Distribution](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs/spec/distribution.md) defines endpoint routes and publication.

## Storage and ownership

The shared local store is internal. Its default root is `~/.cache/tensorfs`; `cache_dir=` and `TENSORFS_HOME` override it. Processes
share immutable objects and coordinated records. Admission hashes bytes before installing them by rename; ordinary
opens check lengths. Local copies expire 7 days after their last use, least recently used first when the disk runs low;
header files (written by `Writer(dest)` or opened by path) and `local/` references keep their data while they exist.
Explicit verification rehashes on demand. TensorFS is a library and has no daemon.

Runtime and TensorD use this same public API. Callers read and write the shared store and upload with a supplied
credential. TensorFS owns I/O and decoding; varena owns memory placement. Installing the wheel does
not grant access to private data or managed executions.

The [Python API](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs-py/API.md) and [Rust API](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs/API.md) guides cover every call; the
wheel ships type declarations. Every format, storage and transport failure derives from `tensorfs.Error`; `.code` identifies one of
twelve families, including `denied` and `conflict`. Unknown encodings remain inspectable and transferable; decoding
requires that scheme's implementation.

## Format and development

The [cozytensors specification](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs/spec/cozytensors.md) is the format contract. The header is deterministic CBOR;
trees and repository documents are canonical JSON. Construction order contributes to identity. The
[conformance corpus](https://github.com/cozy-creator/cozy-creator/tree/master/crates/tensorfs/conformance) checks independent readers, and the [source fixtures](https://github.com/cozy-creator/cozy-creator/tree/master/crates/tensorfs/vectors) retain source
provenance without model payloads. The [end to end check](https://github.com/cozy-creator/cozy-creator/blob/master/crates/tensorfs-py/scripts/e2e.py) converts a tiny
public model, publishes and reopens it, and runs every example in these documents against it in CI.
