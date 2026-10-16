# cozy

Run generative-media models on your own GPU, on a rented one, or on another computer you
control. One binary: the `cozy` command, its background controller (Calcifer) and the machine
agent (TensorD).

> **This build:** verbs marked *coming* are in the grammar but answer `not_ported`. Every
> example here is from `cozy <verb> --help`.

## Install

```sh
curl -LsSf https://github.com/cozy-creator/cozy/releases/latest/download/cozy-installer.sh | sh
```

```powershell
irm https://github.com/cozy-creator/cozy/releases/latest/download/cozy-installer.ps1 | iex
```

Both put `cozy` in `~/.cozy/bin` (`%USERPROFILE%\.cozy\bin`) and on your PATH. No Python is
needed to drive machines. Tab completion: `cozy completion bash|zsh|fish|powershell`, for
example `source <(cozy completion zsh)` in `~/.zshrc`.

`cozy update` replaces the binary with the newest release (`cozy update 0.9.1` picks one),
after checking its checksum and GitHub attestation, then restarts Calcifer. Restarting Calcifer
never stops work that a machine has accepted.

## Sign in

```sh
cozy login you@example.com     # a code is emailed; type it when asked
cozy whoami
cozy logout                    # --everywhere also signs out your other computers
```

Login keeps one device key per Hub in `~/.cozy/keys`. The default Hub is `tensorhub`
(https://tensorhub.com). More Hubs work like kubectl contexts:

```sh
cozy hub add local http://127.0.0.1:8819
cozy hub use local             # where new work goes
cozy hub list
cozy package search sdxl --tensorhub tensorhub   # one command on another Hub
```

Lists show every Hub; `--tensorhub` narrows them to one.

## Run

```sh
cozy run fidika/sdxl                                      # the package's functions
cozy run fidika/sdxl/generate --describe                  # its inputs
cozy run fidika/sdxl/generate prompt="a red fox in snow" seed=7
```

- **Target:** `org/pkg[/fn][@release]`. Local code (`./dir[/fn]`, `./script.py`) is *coming*.
- **Inputs:** `key=value`, `key:=json`, `model.<slot>=<ref>`, one positional value, or files.
  `--input FILE` takes the whole payload from JSON or YAML (inline inputs override it).
  `--asset [NAME=]FILE` attaches a file. `--lora slot:component=ref[,strength]` applies a LoRA.
- **Options:** `--bits 16|8|4` sets weight precision (by default the machine picks per GPU).
  `--gpus N` caps the GPUs. `--out DIR` changes where outputs land (default
  `~/.cozy/outputs`).
- **Following:** a run streams its progress until it ends; Ctrl-C stops watching, never the
  run. `--detach` prints the run's record at once. Resume watching with `cozy run watch <run>`.
  Stop a run with `cozy run cancel <run>`.
- **Warm:** `--warm installed|downloaded|imported|host|gpu` keeps a function ready instead of
  running it; `--warm off` takes it out.
- **Retries:** `--idempotency-key K` names one submission forever, so a retry never runs it
  twice.
- *Coming:* `cozy run list`, `run show`, `run play` (plays an output in the browser, straight
  from its machine) and `run upload`.

`--json` prints JSON instead of tables; while following, it prints NDJSON events
(`run.created`, `run.in_progress`, `output_item.done`, `run.completed`, …).

## Machines

A run goes where `--on` says:

| `--on` | Where |
|---|---|
| `local` (default) | this computer's machine |
| a name | a rented or added machine |
| `any` | any ready machine |
| `new[:sku]` | a new rental: the Hub places it, or rents that SKU |

```sh
cozy machine rent              # what is for rent, with prices
cozy machine rent <sku>        # coming: --gpus N --disk-gb N, --detach
cozy machine list              # coming: this computer's, rentals on every Hub, added ones
cozy machine end <name>        # coming: billing stops once its Hub confirms
```

- *Coming:* `machine show`, `machine logs`, `machine keepalive` (resets a rental's 15-minute
  idle clock) and `machine update`.
- *Coming:* `machine enable` and `disable` (makes this computer a machine).
- *Coming:* `machine add <name> <host:port> --fingerprint sha256:…` and, on that other
  computer, `machine authorize <public-key>`.
- *Coming:* `cozy doctor [--on M]` (driver, GPUs, toolchain and kernels).

A rental is released after 15 minutes with nothing queued or running.

## Packages

```sh
cozy package search sdxl
cozy package info fidika/sdxl
```

*Coming:* `package install|update|remove|list [--on M]`, which act on a machine, and
`package lock|publish|yank` for authors.

## Models

```sh
cozy model search sdxl
cozy model info org/name                  # releases, lanes, sizes, checkpoints
cozy model publish org/name --release 1.0 --lane bf16=sha256:…
```

- `family`, `retarget`, `yank` and `delete` manage your own models.
- *Coming:* `model list` and `model download <model> [local/alias]`, which act on a machine.
- *Coming:* `model upload <source> <org/name>`, where the source can be `hf://…`,
  `civitai://…`, `https://…` or `./file.safetensors`.
- *Coming:* `model quantize --int8|--fp8|--nvfp4 <model>`.

Model references: `org/name[@release[/lane]]`, `org/name#sha256:…`, `hf://…`, `civitai://…`,
`https://…`.

## Credits

```sh
cozy credits                   # balance, held, available, burn per hour
cozy credits history
cozy credits buy 25            # opens a payment page; --no-browser, --no-wait
```

## Calcifer

Calcifer is the controller in the background. Any command starts it. It stops by itself
once it owes nothing.

```sh
cozy status                    # Calcifer and unfinished runs; never starts it
cozy up                        # start it now
cozy down                      # stop it; runs on machines continue, the next one resumes
cozy logs -f                   # its log; `cozy logs machine` is coming
```

## Where things live

```
~/.cozy/
  bin/cozy          the binary
  config.toml       Hubs, the current Hub, settings
  keys/             device key per Hub (0600)
  calcifer/         Calcifer's state.db, calcifer.log, lock and record
  machine/          this computer's machine: state, identity, logs
  outputs/          what runs made
~/.tensorfs/        model weights (TensorFS store)
```

`config.toml`:

```toml
hub = "local"                      # the current Hub: a name below, or a URL

[hubs]                             # tensorhub = https://tensorhub.com is built in
local = "http://127.0.0.1:8819"

[calcifer]
port = 0                           # loopback port; 0 takes any free one
```

Keys this build does not read are named once and ignored. The environment carries only
credentials and locations:

| Variable | What it is |
|---|---|
| `HF_TOKEN` | Hugging Face credential |
| `CIVITAI_TOKEN` | Civitai credential |
| `COZY_DOWNLOAD_URL` | where `cozy update` and the installers fetch releases |

## Platforms

| | Linux x86_64 | Linux arm64 | Windows x86_64 | macOS arm64 |
|---|---|---|---|---|
| Controller (drive local, rented and added machines) | yes | yes | yes | yes |
| NVIDIA machine role | yes¹ | coming (GH200, GB10, GB300) | coming (single GPU) | — |
| Apple machine role (MPS) | — | — | — | coming |

¹ On rented machines. `machine enable`, which makes this computer a machine, is *coming*.

## Scripting

- **Exit codes:** 0 success, 1 a refusal or a run that did not succeed, 2 a usage error.
- **Errors:** with `--json`, every error is one envelope on stdout,
  `{"error": {"kind", "code", "message", "remedy", "next"}}`, with at most two next steps.
- **Columns:** human tables never truncate when stdout is not a terminal.

## Coming from the Go cozy

Run the Go `cozy down` first; accepted runs continue on their machines. Install this binary.
Then:

```sh
cozy migrate --from ~/.cozy --dry-run     # what would be imported, archived and kept
cozy migrate --from ~/.cozy
```

The import carries over your Hubs, keys, live rentals, owed runs and run history. Outputs stay
in place.
