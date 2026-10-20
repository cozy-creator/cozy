# Cozy

Cozy runs generative-media packages (image, video and captioning models) on your own NVIDIA GPU or
on one you rent by the hour. One `cozy` binary carries the command line, Calcifer (the local
controller), TensorD (the machine) and TensorFS (the weight store).

## Install

Linux and macOS:

```sh
curl -LsSf https://tensorhub.com/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://tensorhub.com/install.ps1 | iex
```

It runs on Linux (x86_64 and arm64, glibc 2.28 or newer), macOS and Windows (x86_64 and arm64),
and puts `cozy` in `~/.cozy/bin` and on your PATH: open a new terminal, or run the line the
installer prints. `cozy update` moves to the newest release.

## Find a model

```sh
cozy package search                         # the packages, e.g. fidika/sdxl
cozy run fidika/sdxl                        # what it does, and its functions
cozy run fidika/sdxl/generate --describe    # one function's inputs
```

## Run on this computer

On Linux with an NVIDIA GPU:

```sh
cozy machine enable
cozy run fidika/sdxl/generate prompt='a lighthouse at dusk' seed=7
```

The first run installs the package and downloads its model; outputs land in `~/.cozy/outputs`.
Without an NVIDIA GPU a run says so and uses the CPU, slowly. Published packages are built for
Linux, so on macOS and Windows rent them; your own scripts (`cozy run ./script.py`) run here on any
of them.

## Rent a GPU

```sh
cozy login you@example.com
cozy run fidika/sdxl/generate prompt='a lighthouse at dusk' seed=7 --on new
```

`cozy login` signs you up on first use, with a code it emails; a new account starts with $2 of
credit (`cozy credits`). `--on new` rents the cheapest card that holds the model, shows its price
and asks first. Scripts and agents, with no terminal to ask on, pass `--yes`; without it nothing is
rented and the refusal (`quote_unagreed`) carries the price. The rental takes more runs with `--on <name>`, ends
itself after 15 minutes with nothing to run, and `cozy machine end <name>` ends it at once.
`cozy machine list` shows your rentals and `cozy machine rent` what is for rent.

`cozy --help` and `cozy <verb> --help` are the reference.

## Uninstall

```sh
cozy uninstall        # lists what it removes, with sizes, and asks; --yes skips the question
```

It stops Calcifer and the machine and removes their services, `~/.cozy`, `~/.config/cozy` and the
installer's PATH lines. Downloaded models stay unless `--all`.

## Build

```sh
cargo build --release -p cozy
```

The workspace's crates live in `crates/`: `cozy` (the binary), `cli`, `calcifer`, `tensord`,
`tensorfs`, `varena`, `tensorfs-py`, `hub-client`, `protocol` and `platform`.
`crates/tensorfs` is the Rust library and `tfs` CLI; `crates/tensorfs-py` contains its PyO3 bindings,
Python package and wheel configuration together. The browser player is `web/player`.
