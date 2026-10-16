# Cozy machine

TensorD (executable `tensord`) is the Rust daemon that owns the machine API, execution journal,
scheduling, memory policy and executor supervision. It uses embedded TensorFS to make selected
model weights and metadata available to Runtime and to fill, retain and release shared CPU weight buffers.
Python cozy-runtime executors run package code, construct models and use their own TensorFS
weight plane for GPU transfers and mappings. TensorFS is a library on both sides, with no
separate daemon.

- Architecture and module owners: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Links TensorFS from this workspace (`crates/tensorfs`).
- Linux 6.5+ (`SO_PEERPIDFD`). TensorD has no CUDA context. It can retain GPU allocation
  handles exported by executors; Runtime performs the device operations. The CPU service
  does not load NVML; a configured GPU pool samples it on its GPU sampler threads.

## Build and test

```sh
cargo build -p tensord --locked
cargo clippy -p tensord --all-targets -- -D warnings
cargo test -p tensord --locked -- --test-threads=2
uv run --locked --extra test pytest -q
```

## Release

TensorD releases nothing on its own while it moves into the `cozy` binary as `cozy machine serve`
(xs-043 T). Until then cozy-runtime's Linux wheel bundles the TensorD of `cozy-creator/tensord`
at its `tensord.pin`.

## Run

```sh
target/debug/tensord version --json
target/debug/tensord serve --state PATH [--generations PATH] [--host-bytes N] \
  [--cpu-parallelism N] [--gpu-config FILE] [--machine-config FILE --listen ADDR]
```

`--machine-config` with `--listen` starts the authenticated TLS/gRPC API the Cozy CLI uses.
`--gpu-config` adds the GPU pool ([GPU service](docs/GPU-SERVICE.md)).
