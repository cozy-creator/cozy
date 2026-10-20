# TensorFS Rust API

Create a client, perform an operation, then use the checkpoint, component or byte data
it returns. This guide covers the complete planned application-facing Rust API by task.

This is the **0.13 target**, currently being integrated. The implementation on `master`
can still expose the preceding interface. The [Python API](../tensorfs-py/API.md) and
[technical file-format specification](spec/cozytensors.md) are separate documents.

- [Create a client and read a model](#rust-create-a-client-and-read-a-model)
- [Write, convert and publish](#rust-write-convert-and-publish)
- [Options and returned objects](#rust-options-and-returned-objects)
- [Errors and value types](#errors-and-value-types)

The [temporary type-declaration dump](API-TYPES.md) is available for signature and naming review.

## Rust: create a client and read a model

Inside an async Rust function, create one client and reuse it:

```rust
use tensorfs::{Client, Config, Fetch, Open, Selection};

let client = Client::new(Config::from_env())?;
let checkpoint = client.open("me/model", &Open {
    revision: Some("1.0".into()),
    variant: Some("bf16".into()),
    ..Default::default()
}).await?;

checkpoint.fetch(&Fetch {
    selection: Selection::Components(vec!["unet".into()]),
    ..Default::default()
}).await?;

let unet = checkpoint.component("unet")?;
let bytes = unet.read("conv_in.weight")?;
println!("{} bytes", bytes.len());
```

`Client::open` also accepts a local .cozytensors file, a portable folder, or an immutable reference. It returns a checkpoint
with metadata and source access. Rust network operations use your runtime: explicitly fetch
before synchronous `read`, `asset`, `chunk` or `export_files` operations. There is no hidden core
Tokio runtime.

A borrowed component follows Rust's ordinary borrow rules. Clone the `Component` if it
must outlive the parent checkpoint; owned child handles retain their source and data holds.

## Rust: write, convert and publish

```rust
use std::path::Path;
use tensorfs::{Dtype, WriterOptions};

let mut writer = client.writer(WriterOptions {
    dest: Some(Path::new("./model.cozytensors")),
    ..Default::default()
})?;
let bytes: Vec<u8> = [1.0_f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
let mut input = bytes.as_slice();
{
    let mut model = writer.component("model", "", Default::default())?;
    model.put("weight", Dtype::F32, &[2], &mut input)?;
}
let saved = writer.commit().await?;
println!("saved {}", saved.id().id());
```

A file destination writes only the header; data is retained in the shared content-addressed
store before the header is atomically published. Store roots and endpoint configuration are
not embedded as paths in the file. A native portable export writes a complete folder with
chunks; ordinary file saving does not.

`dest` is the exact filename; TensorFS does not append a variant or create a directory.
Leave `variant` unset for a file destination. Variant labels belong to repository trees;
setting both `dest` and `variant` is invalid. With no destination, the writer retains the
checkpoint in the shared store and may assign a repository variant.

A standalone header identifies one checkpoint. It does not contain its enclosing tree,
release label, README or LICENSE. Moving or copying it preserves that checkpoint identity;
repository context is available only when separately known. Deleting or replacing the
file does not release its saved data automatically.

The component borrows its writer while being filled. `commit(&mut self)` is async for both
local and remote writes. After an upload failure, `writer.committed()` can return the strong
local checkpoint; the operation still returns an error. Successful publication records the
exact remote tree.

| Client operation | Inputs and result |
|---|---|
| `Client::new(config)` | Validate configuration and create a reusable client; returns `Result<Client>`. |
| `open(name_or_path, &Open)` | Metadata/source access; returns `Result<Checkpoint>`. |
| `snapshot_download(&Ref, &Fetch, local_dir, local_files_only)` | Fetch/materialize selected repository content; returns `Result<PathBuf>`. |
| `hub_download(&Ref, filename, local_dir, local_files_only)` | Fetch one repository file; returns `Result<PathBuf>`. |
| `upload_folder(&Ref, folder, progress)` | Publish the folder; returns `Result<ObjectRef>`. |
| `convert(&[String], &Convert)` | Convert pinned sources; returns `Result<Checkpoint>`. |
| `resolve_source(source)` | Resolve an upstream source; returns `Result<String>`. |
| `writer(WriterOptions)` | Begin a write; returns `Result<Writer>`. |

For example, parse a repository reference and use the returned folder or tree:

```rust
use tensorfs::{Format, Ref};

let model: Ref = "me/model@1.0/bf16".parse()?;
let folder = client.snapshot_download(&model, &Fetch::default(), None, false).await?;
println!("downloaded to {}", folder.display());

tensorfs::export(&saved, Path::new("./portable"), Format::Cozytensors, None)?;
let destination: Ref = "me/output".parse()?;
let tree = client.upload_folder(&destination, Path::new("./portable"), None).await?;
println!("published {}", tree.id());
```

Native export includes the selected checkpoint and any known repository files. It is not
a full multi-variant tree snapshot; use `snapshot_download` for repository materialization.
Exporting a standalone header cannot recover unknown repository files or release metadata.

All operations involving network/source resolution above are async except construction and
`writer`. Export uses `tensorfs::export(&checkpoint, directory, format, dequantize)` and
returns `Result<Vec<String>>`; choose `Format::Safetensors` or `Format::Cozytensors`, and
ensure required bytes are available first.

A `Writer` also has `pipeline(class, config)`, `component(name, class, config)`,
`set_metadata(key, Value)`, `has(component, key)`, `committed()` and `abandon(self)`. Its
`ComponentWriter` supports `put(key, Dtype, shape, &mut Read)`,
`copy(key, &Component, as_key)`,
`asset(name, bytes)`, `order(keys)` and `has(key)`. These operations return `Result`, except
`committed`, which returns `Option<Checkpoint>`.

Plain input streams are consumed by the write; `base`/`copy` reuse existing checkpoint
content. Raw encoded-stream producers can use the low-level writer interface described
below; normal model consumers do not need it.

## Rust: options and returned objects

| Type | Fields / use |
|---|---|
| `Config` | `endpoint: String`, `cache_dir: PathBuf`, `mirror: Option<PathBuf>`, `token: Option<Arc<dyn TokenProvider>>`, `ca_der: Option<Vec<u8>>`. `Config::from_env()` returns configuration; `Client::new` validates it. Debug output redacts access material. |
| `Ref` | `repo_id`, optional `revision`, `variant`, `endpoint`; implements `FromStr`. Use explicit fields/options when names contain shorthand separators. |
| `Open` | Optional `revision`, `variant`, `endpoint`, plus `local_files_only`; implements `Default`. |
| `Selection` | `All`, `Components(Vec<String>)`, `Tensors { component, keys }`, `Part { component, key, role: Option<String>, range: Option<Range<u64>> }`, or `Assets { component, names: Option<Vec<String>> }`. Part ranges are stored byte ranges; absent asset names selects all assets. |
| `Fetch` | `selection`, `priority: u64`, optional borrowed progress callback; defaults to all. |
| `Fetched` | Byte counts `local`, `volume`, `origin`. |
| `VerificationReport` | Separate `missing` and `corrupt` object-reference vectors; `is_ok()` checks both are empty. |
| `Convert` | Optional `converter`, `variant`, borrowed conversion-progress callback. |
| `WriterOptions` | Optional `dest`, `variant`, `base`, `resume`, `repo_id`, `endpoint`, `token`, `ca_der`; implements `Default`. |
| `Format` | `Safetensors` or native portable `Cozytensors` export. |

`Progress` is a borrowed `Fn(u64, u64) -> Result<()> + Sync` (held bytes, total bytes).
`ConvertProgress` additionally receives a stage string. Returning an error cancels the
operation. Configured providers supply ordinary credentials; construction does not create
a separate worker-permission mechanism.

### Checkpoint and component handles

| Checkpoint method | Result |
|---|---|
| `id()` | `&ObjectRef` for the header. |
| `header()` | `&format::Header`. |
| `metadata()` | `&Metadata`. |
| `chunks()` | `&[ObjectRef]`. |
| `tree()` | `Option<&ObjectRef>`. |
| `named()` | `Option<&Ref>`. |
| `pipeline()` | `Option<&format::Pipeline>`. |
| `components()` | `&[Component]`. |
| `component(name)` | `Result<&Component>`. |
| `verify(&Selection)` | `Result<VerificationReport>`, separating absent objects from length/digest failures. |
| `view()` | An owned `io::view::View` for byte-engine/placement integration. |
| `chunk(index)` | `Result<File>`; the returned file owns its descriptor. |
| `missing(&Selection)` | `Result<Vec<ObjectRef>>`. |
| `fetch(&Fetch).await` | `Result<Fetched>`. |

`fetch` completes when the selected chunks and supporting files are verified and locally
available; the open checkpoint retains them. `missing` is optional availability inspection.
Use `verify` for an explicit local audit, not after every fetch: admission already checks
incoming bytes. Verification always checks the header, then the selected complete objects,
and never fetches, repairs or deletes data. `Selection::All` covers the checkpoint's
declared chunks and component assets, not its enclosing repository files; checking a
shared chunk hashes that whole object. Other local I/O failures remain
typed errors rather than being classified as corruption.

A `Component` provides `name()`, `class()`, `config()`, `tensors()` and
`assets()` metadata views. Its `read(key)` returns decoded `Vec<u8>`;
`read_part_into(key, role, offset, &mut [u8])` fills a stored byte range;
`asset(name)` returns `Vec<u8>`. These direct configuration, file and tensor reads are the
normal cozytensors integration path. `export_files(directory)` is only a convenience for
legacy libraries that require filenames: it explicitly writes configuration and supporting
files and returns `PathBuf`, without exporting weights. Native consumers, including our
Runtime, do not need that step. The I/O calls return `Result`.
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

`DPoP` is a cloneable `TokenProvider` for callers using proof-of-possession credentials.
`DPoP::token(key, access_token, resource, endpoint)` wraps an already issued token;
`DPoP::grant(&config, key, JwtBearer { token_endpoint, client_id, resource, claims })`
configures a lazy JWT-bearer exchange. The key is a private P-256 JWK. Clones share the
same native token state. `issued_token().await` returns `Option<String>` without starting
an exchange. These provider operations use the common `Failure` families.

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
