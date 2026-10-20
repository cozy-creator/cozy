# TensorFS Rust API

Create a client, perform an operation, then use the checkpoint, component or byte data
it returns. This guide covers the application-facing Rust API by task. The
[Python API](../tensorfs-py/API.md) and [file-format specification](spec/cozytensors.md)
are separate documents.

- [Create a client and read a model](#rust-create-a-client-and-read-a-model)
- [Write, convert and publish](#rust-write-convert-and-publish)
- [Options and returned objects](#rust-options-and-returned-objects)
- [Decode existing host or GPU buffers](#advanced-decode-existing-host-or-gpu-buffers)
- [Errors and value types](#errors-and-value-types)

[API-TYPES.md](API-TYPES.md) lists every declaration.

## Rust: create a client and read a model

Inside an async Rust function, create one client and reuse it:

```rust
use tensorfs::{Client, Config, Download, Open, Selection};

let client = Client::new(Config::from_env())?;
let checkpoint = client.open("me/model", &Open {
    revision: Some("1.0".into()),
    variant: Some("bf16".into()),
    ..Default::default()
}).await?;

checkpoint.download(&Download {
    selection: Selection::Components(vec!["unet".into()]),
    ..Default::default()
}).await?;

let unet = checkpoint.component("unet")?;
let bytes = unet.read_decoded("conv_in.weight")?;
println!("{} bytes", bytes.len());
```

`Client::open` also accepts a local .cozytensors file, a portable folder, or an immutable
reference. The default `Open { lazy: true, .. }` loads metadata and retains source access.
Set `lazy: false` to wait until the checkpoint's declared chunks and component assets are
available locally before returning. This does not unpack weights, load a GPU or create a
snapshot folder.

Rust network operations use your runtime. With a lazy checkpoint, call `download` for
the required selection before synchronous `read`, `asset` or `export_files` operations.
Only the placement `source` fetches on its own, on the process runtime.

A borrowed component follows Rust's ordinary borrow rules. Clone the `Component` if it
must outlive the parent checkpoint; owned child handles retain their source and data holds.

## Rust: write, convert and publish

```rust
use std::path::Path;
use tensorfs::{format::Dtype, WriterOptions};

let mut writer = client.writer(WriterOptions {
    dest: Some(Path::new("./model.cozytensors")),
    ..Default::default()
})?;
let bytes: Vec<u8> = [1.0_f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
let mut input = bytes.as_slice();
{
    let mut model = writer.component("model", "", Default::default(), None)?;
    model.put("weight", Dtype::F32, &[2], &mut input)?;
}
let saved = writer.commit().await?;
println!("saved {}", saved.id().id());
```

A `.cozytensors` file is a small handle into the shared local store, not a self-contained
file: `dest` gets only the header, and the tensor and asset bytes stay in the store. The
store keeps them while that file still holds this header. Once the file is deleted, moved or
overwritten with another header, the data ages out like any other cache entry. Open
checkpoints keep their bytes in the meantime. To copy a checkpoint to another machine,
write the portable form with `saved.materialize(dir, Format::Cozytensors, None)`.

`dest` is the exact filename; TensorFS does not append a variant or create a directory.
`dest` and `repo_id` exclude each other. `variant` names `model.<variant>.cozytensors` in a
`repo_id` upload and is `Invalid` without `repo_id`. With no destination, the writer keeps
the checkpoint in the shared store as a cache entry.

A standalone header identifies one checkpoint. It does not contain its enclosing tree,
release label, README or LICENSE; repository context is available only when separately known.

The component borrows its writer while being filled. `commit(&mut self)` is async for both
local and remote writes. After an upload failure, `writer.committed()` can return the strong
local checkpoint; the operation still returns an error. Successful publication records the
exact remote tree.

| Client operation | Inputs and result |
|---|---|
| `Client::new(config)` | Validate configuration and create a reusable client; returns `Result<Client>`. |
| `open(name_or_path, &Open)` | Open metadata, optionally waiting for all checkpoint data with `lazy: false`; returns `Result<Checkpoint>`. |
| `materialize(&Ref, &Download, local_dir, local_files_only)` | Make selected repository content available as a normal folder; returns `Result<PathBuf>`. |
| `download_file(&Ref, filename, local_dir, local_files_only)` | Fetch one repository file; returns `Result<PathBuf>`. |
| `upload_folder(&Ref, folder, progress)` | Publish the folder; returns `Result<ObjectRef>`. |
| `convert_to_cozytensors(&[String], &Convert)` | Convert pinned sources; returns `Result<Checkpoint>`. |
| `writer(WriterOptions)` | Begin a write; returns `Result<Writer>`. |

For example, parse a repository reference and use the returned folder or tree:

```rust
use tensorfs::{Format, Ref};

let model: Ref = "me/model@1.0/bf16".parse()?;
let folder = client.materialize(&model, &Download::default(), None, false).await?;
println!("downloaded to {}", folder.display());

saved.materialize(Path::new("./portable"), Format::Cozytensors, None).await?;
let destination: Ref = "me/output".parse()?;
let tree = client.upload_folder(&destination, Path::new("./portable"), None).await?;
println!("published {}", tree.id());
```

Checkpoint materialization includes that checkpoint and any known repository files.
Use `Client::materialize` for an entire native repository, including all variants when no
variant is selected. A standalone header cannot supply unknown repository files or release
metadata.

All operations involving network/source resolution above are async except construction and
`writer`. `checkpoint.materialize(directory, format, dequantize).await` returns
`Result<PathBuf>`; choose `Format::Safetensors` or `Format::Cozytensors`. It first makes
the required checkpoint data and known repository files available, then writes the folder.
Missing or denied source objects fail the operation.
`Format::Cozytensors` requires `None` for `dequantize`; `Format::Safetensors` accepts
`Some(Dtype::F32)`, `Some(Dtype::BF16)` or `Some(Dtype::F16)`, or `None`.
Other combinations return `Invalid` before source preparation.

A `Writer` also has `pipeline(class, config)`, `component(name, class, config, config_file)`
(`config_file` names the file the config exports as, such as `scheduler_config.json`; `None`
keeps a reopened or base component's, else `config.json`),
`component_writer(name)`, `set_metadata(key, Value)`, `has(component, key)`, `committed()`
and `abandon(self)`. `component_writer` reopens a declared component without changing
its class or configuration. `ComponentWriter` supports `put(key, Dtype, shape, &mut Read)`,
`copy(key, &Component, as_key)`,
`asset(name, bytes)`, `order(keys)` and `has(key)`. These operations return `Result`, except
`committed`, which returns `Option<Checkpoint>`.

Plain input streams are consumed by the write; `base`/`copy` reuse existing checkpoint
content. Raw encoded-stream producers can use the low-level writer interface described
below; normal model consumers do not need it.

## Rust: options and returned objects

| Type | Fields / use |
|---|---|
| `Config` | `endpoint: String`, `cache_dir: PathBuf`, `mirror: Option<PathBuf>`, `token: Option<Arc<dyn TokenProvider>>`, `ca_der: Option<Vec<u8>>` (a private CA trusted on top of the system roots, for every operation of the client). `Config::from_env()` returns configuration; `Client::new` validates it. Debug output redacts access material. |
| `Ref` | `repo_id`, optional `revision`, `variant`, `endpoint`; implements `FromStr`. Use explicit fields/options when names contain shorthand separators. |
| `Open` | Optional `revision`, `variant`, `endpoint`, plus `local_files_only` and `lazy`; defaults to `lazy: true`. |
| `Selection` | `All`, `Components(Vec<String>)`, `Tensors { component, keys }`, `Part { component, key, role: Option<String>, range: Option<Range<u64>> }`, or `Assets { component, names: Option<Vec<String>> }`. Part ranges are stored byte ranges; absent asset names selects all assets. |
| `Download` | `selection`, `priority: u64`, optional borrowed progress callback; defaults to all. |
| `Fetched` | Byte counts `local`, `volume`, `origin`. |
| `VerificationReport` | Separate `missing` and `corrupt` object-reference vectors; `is_ok()` checks both are empty. |
| `Convert` | Optional `converter`, `variant`, borrowed conversion-progress callback. |
| `WriterOptions` | Optional `dest`, `variant` (only with `repo_id`), `base`, `resume`, `repo_id`, `endpoint`, `token`; implements `Default`. |
| `ReadInto` | `role: Option<&str>`, `decode: bool`, `offset: u64` for `Component::read_into`; implements `Default`. |
| `Format` | `Safetensors` or native portable `Cozytensors` export. |

`Progress` is a borrowed `Fn(u64, u64) -> Result<()> + Sync` (held bytes, total bytes).
`ConvertProgress` additionally receives a stage string. A callback can stop remaining work
by returning an error. The final `upload_folder` progress callback runs after publication:
its error is returned to the caller but does not undo the repository update. Configured
providers supply ordinary credentials; construction does not create a separate
worker-permission mechanism.

### Checkpoint and component handles

| Checkpoint method | Result |
|---|---|
| `id()` | `&ObjectRef` for the header. |
| `header()` | `&format::Header`. |
| `metadata()` | `&Metadata`. |
| `chunks()` | `&[ObjectRef]`. |
| `variant()` | `Option<&str>` for the selected variant. |
| `tree()` | `Option<&ObjectRef>`. |
| `named()` | `Option<&Ref>`. |
| `pipeline()` | `Option<&format::Pipeline>`. |
| `components()` | `&[Component]`. |
| `component(name)` | `Result<&Component>`. |
| `verify(&Selection)` | `Result<VerificationReport>`, separating absent objects from length/digest failures. |
| `spans(regions)` | `Result<Vec<io::Span>>`: the stored byte spans of the named parts, for placement. |
| `stored_part(name)` | `Option<io::StoredPart>`: a part's stored dtype and shape. |
| `source(direct_io)` | An `io::source::Source` over the chunks for the read engine; absent chunks are fetched on the process runtime. |
| `missing(&Selection)` | `Result<Vec<ObjectRef>>`. |
| `download(&Download).await` | `Result<Fetched>`. |
| `materialize(directory, Format, dequantize).await` | `Result<PathBuf>` for the complete checkpoint folder. |

`download` completes when the selected chunks and supporting files are verified and locally
available; the open checkpoint retains them. `missing` is optional availability inspection.
Use `verify` for an explicit local audit, not after every download: admission already checks
incoming bytes. Verification always checks the header, then the selected complete objects,
and never fetches, repairs or deletes data. `Selection::All` covers the checkpoint's
declared chunks and component assets, not its enclosing repository files; checking a
shared chunk hashes that whole object. Other local I/O failures remain
typed errors rather than being classified as corruption.

A `Component` provides `name()`, `class()`, `config()`, `config_file()`, `tensors()` and
`assets()` metadata views. `config_file()` is the file name the config exports as
(`config.json` unless the writer named another, such as `scheduler_config.json`). Its
`read(key)` returns the tensor as stored, as Python's `decode=False`: `Read::Plain(Vec<u8>)`,
or `Read::Encoded { dtype, shape, encoding, parts }` with the logical dtype and shape and
each stored part's bytes by role. `read_decoded(key)` returns logical bytes in the tensor's
dtype, as `decode=True`. `asset(name)` returns `Vec<u8>`.

`read_into(key, &mut out, &ReadInto { role, decode, offset })` fills `out` from byte
`offset`, as Python's `read_into`. A plain tensor gives its data. A `role` selects that stored
part, and `decode: true` an encoded tensor's logical bytes. An encoded tensor with neither
returns `Invalid` naming its roles; a role with `decode` is also `Invalid`.

These direct configuration, file and tensor reads are the normal cozytensors integration
path. `export_files(Some(directory))` is only a convenience for libraries that require
filenames: it writes configuration and supporting files, never weights, and returns
`PathBuf`. `export_files(None)` writes them to a folder the cache manages and evicts like its
other entries. Native consumers, including our Runtime, do not need that step. The I/O calls
return `Result`.
Selections control exactly which component/tensor/part/asset bytes are fetched, including
when other model data happens to be present in the cache.

### Advanced repository and authorization use

`Endpoint::new(url, token)` constructs a reference client; `Endpoint::with_config(url,
&Config)` also supplies explicit cache/CA/provider configuration. `repo(repo_id).await`
returns `Result<Option<(Repo, String)>>`, containing the document and ETag.
`update_repo(repo_id, change).await` returns `Result<Repo>` and can retry the change callback
on a conflict. `Endpoint::new("local", ...)` addresses local repository references.

`bearer(token)` returns `Arc<dyn TokenProvider>`. A custom provider implements
`headers(&Request).await -> Result<Vec<(String, String)>>`; optional
`retry(&Request, status, &HeaderMap) -> Result<bool>` defaults to `Ok(false)`.
`Request` contains borrowed `method`, `url`, existing `headers`, and optional body-digest
`payload`. The public result uses the same error families as the other client operations.

`ObjectRef` has `sha256: String` and `length: u64`; `of(bytes)` hashes content, `id()` returns
`sha256:…` and `key()` gives its object-storage key. `Metadata` is `BTreeMap<String, Value>`.
`format::Pipeline` contains `class` and JSON `config`; `format::Tensor` contains the key,
logical dtype/shape and plain location or encoding/parts. `Encoding` holds scheme and
parameters. Format metadata structures are described in the [format spec](spec/cozytensors.md).
The lower-level I/O/codec interfaces used by in-tree integrations are separate from this
application-facing session API.

### Low-level encoded-stream producers

For converters or custom encoders that stream already encoded bytes,
`ComponentWriter::put_encoded(key, dtype, shape, encoding, parts)` accepts
`Vec<raw::PartData>`. Each entry identifies a named array with `role`, `dtype`, `shape`
and a borrowed `data: &mut dyn Read` stream. This low-level input does not require the
caller to choose disk chunks or byte offsets; TensorFS assigns those. Physical format
structures stay in the format/inspection interface, outside ordinary model-loading code.

## Advanced: decode existing host or GPU buffers

Ordinary checkpoint reads select and decode their stored arrays for you. An integration
that already owns encoded arrays in RAM or GPU memory can instead use
`raw::check_decode` and `unsafe raw::decode_into`. These operations use the same format
registry and decode engine; they do not open a checkpoint, fetch chunks or allocate the
output. `decode_into` returns `Result<()>` after writing into your existing destination.

Supply an `Encoding`, the logical tensor's dtype and shape, and one `raw::DecodePart` per
named encoded array. Each part holds its format dtype/shape and a contiguous `io::Src`
byte span. The actual carrier can differ: packed four-bit values may live in a byte array,
while their format descriptor still describes packed values. `raw::DecodeOutput` holds
the actual destination dtype, shape and `io::Dst`; they must match the requested logical
tensor exactly. There is no implicit final dtype conversion.

`check_decode` accepts `format::decode::PartMetadata` (role, dtype, shape and byte length)
and validates the descriptors without reading bytes or initializing CUDA. It does not
validate pointers or establish buffer ownership. The call to `decode_into` also checks
placement, lengths, alignment and input/output overlap before writing.

All-host spans use the CPU decoder. All-CUDA spans must share one device and use a
supported resident-input scheme; the initial target covers `comfy.asym_w4a8_int8`,
`comfy.w6a8_int8` and `comfy.convrot_w4a4`. Mixed host/device input, file/zero sources and
implicit transfers are refused. Keep input allocations alive, the output exclusively
writable, and producer/output streams valid in that device's primary context on the
calling thread. Rust's raw pointers do not enforce these requirements for you.

Success establishes completion. After an error, keep **all** input and output owners
until `io::drain` for the device and every producer stream report checked completion;
an error alone does not establish quiescence. Descriptor and overlap refusals precede
writes; after a runtime or value error the output contents are unspecified.

The [exact declarations](API-TYPES.md#advanced-resident-buffer-decoding) list the types.

## Errors and value types

Operations return `Result<T, Failure>`. Inspect `Failure.kind` and `Failure.detail`.
The `Kind` variants are `Invalid`, `Format`, `NotFound`, `Integrity`, `UnknownScheme`,
`Decode`, `Io`, `Full`, `Network`, `Denied`, `Unsupported` and `Conflict`.
`Kind::code()` returns the stable family string. Provider callbacks use this same error type.

`Invalid` covers malformed requests/documents; `Format` identifies a different document
kind or major version. Unknown encodings can still be read as stored data; interpreting
or decoding them may fail.

Loading metadata and encoding parameters accept booleans, integers in the signed-64 through
unsigned-64 range, strings, lists and string-keyed maps. They do not accept floats, null
or bytes. Pipeline/component configuration is ordinary JSON, including finite floats and
null values.
