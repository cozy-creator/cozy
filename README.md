# Cozy

Cozy runs generative-media packages on your own machines and on rented GPUs. One `cozy` binary
carries the command line, Calcifer (the local controller), TensorD (the machine) and TensorFS
(the weight store).

## Install

Linux and macOS:

```sh
curl -LsSf https://tensorhub.com/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://tensorhub.com/install.ps1 | iex
```

Both put `cozy` in `~/.cozy/bin`; `cozy update` moves to the newest release.

## Use

```sh
cozy login you@example.com
cozy run fidika/sdxl prompt='a lighthouse at dusk' seed=7
cozy machine rent            # what is for rent; then cozy machine rent <sku>
cozy model search sdxl
```

`cozy --help` and `cozy <verb> --help` are the reference.

## Build

```sh
cargo build --release -p cozy
```

The workspace's crates live in `crates/`: `cozy` (the binary), `cozy-cli`, `calcifer`, `tensord`,
`tensorfs`, `hub-client`, `cozy-proto`, `cozy-platform` and `migrate` (the Go home importer). The
browser player is `web/player`.
