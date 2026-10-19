# TensorFS usage

TensorFS is a library for opening, reading, writing and sharing cozytensors checkpoints.
Use the Rust crate `tensorfs` or the Python package `tensorfs`.
The [full API reference](API.md) lists the planned functions, types, properties and methods.

These examples describe the **0.13 target interface**. The public Rust `Client` and the
Python interface are being integrated; the current master implementation still exposes
the preceding interface.

## Python

Open a model and read its tensors. A local folder works in place of the repository name.
Opening fetches metadata; reads fetch the tensor bytes they need.

```python
import tensorfs

with tensorfs.open("me/model", revision="1.0", variant="bf16") as checkpoint:
    unet = checkpoint.components["unet"]
    print(unet.cls, unet.config)
    weight = unet.tensors.read("conv_in.weight")
    weights = unet.tensors.read_all()
```

Arrays support DLPack: `torch.from_dlpack(weight)` gives PyTorch a tensor.
Use `read(..., decode=False)` to keep encoded weights and inspect their scheme and parts.

Write a checkpoint into a normal folder, then optionally upload it:

```python
from array import array
import tensorfs

with tensorfs.Writer("./output", variant="f32") as writer:
    model = writer.component("model", cls="", config={"width": 2})
    model.put("weight", array("f", [1, 2]), dtype="F32", shape=[2])

tensorfs.upload_folder(repo_id="me/example", folder_path="./output", token="...")
```

Public reads need no credential. For private reads or uploads, supply `token=` or configure
`TENSORFS_TOKEN`; `endpoint=` selects a server other than the default `https://tensorhub.com`.

## Rust

Network operations are async on your Tokio runtime. Fetch the selected bytes before using
the synchronous tensor reader. This example reads the folder written above:

```rust
use tensorfs::{Client, Config, Fetch, Open};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(Config::from_env())?;
    let checkpoint = client.open("./output", &Open::default()).await?;
    checkpoint.fetch(&Fetch::default()).await?;

    let model = checkpoint.component("model")?;
    let bytes = model.read("weight")?;
    println!("{} bytes", bytes.len());
    Ok(())
}
```

`Client::writer` creates a writer, with an optional base checkpoint to reuse unchanged
tensors. Rust finalizes it with `writer.commit().await?`; the same operation is synchronous
from Python. `Client::upload_folder` publishes a folder and returns its immutable tree reference.

## Operations

| Task | Python | Rust |
|---|---|---|
| Open a checkpoint | `tensorfs.open` | `Client::open` |
| Fetch a whole repository | `tensorfs.snapshot_download` | `Client::snapshot_download` |
| Fetch one repository file | `tensorfs.hub_download` | `Client::hub_download` |
| Convert another format | `tensorfs.convert` | `Client::convert` |
| Write a checkpoint | `tensorfs.Writer` | `Client::writer` |
| Upload a folder | `tensorfs.upload_folder` | `Client::upload_folder` |
| Export another format | `tensorfs.export` | `tensorfs::export` |
| Resolve an upstream source | `tensorfs.resolve_source` | `Client::resolve_source` |
| Set endpoint, cache and credentials | `tensorfs.configure` | `Config` / `Client::new` |

Generate reference docs for the checked-in Rust implementation with
`cargo doc -p tensorfs --no-deps --open`.

The [longer Python guide](../tensorfs-py/README.md) covers variants, downloads, conversion
and storage. The [format specification](spec/cozytensors.md),
[distribution protocol](spec/distribution.md) and [readable documents](spec/examples/)
describe the bytes and protocol rather than ordinary library usage.
