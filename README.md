# Cozy

Cozy runs generative-media packages on your own machines and on rented GPUs. One `cozy` binary
carries the command line, Calcifer (the local controller), TensorD (the machine) and TensorFS
(the weight store).

## Install

Linux and macOS:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cozy-creator/cozy/releases/latest/download/cozy-installer.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/cozy-creator/cozy/releases/latest/download/cozy-installer.ps1 | iex"
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
