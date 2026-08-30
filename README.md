# Cozy

Cozy is a local-first command-line application for generative media. It installs
package code, downloads models, runs package callables on your machine or a private rented
worker, and keeps your local execution records and outputs under your control.

The command is `cozy`.

## Install

Binary releases are not published yet. Building from source currently requires Go 1.26 or
newer. Package installation also uses `uv`; model download and publication use the `tfs`
executable from TensorFS.

```sh
git clone https://github.com/cozy-creator/cozy.git
cd cozy
go build -o cozy .
install -m 0755 ./cozy ~/.local/bin/cozy
cozy -v
```

On Windows, build `cozy.exe` and place it in a directory on `PATH`:

```powershell
go build -o cozy.exe .
```

## Discover commands

Running Cozy without arguments shows its complete launch surface. Help and version never load
configuration or start a background process.

```sh
cozy
cozy package
cozy help package install
cozy help invoke run
cozy -v
```

There is no container “stack.” Start the persistent local product explicitly when you want its
localhost UI available:

```sh
cozy up
# open http://127.0.0.1:8818/ (or the returned fallback URL when 8818 is occupied)
```

`up` backgrounds one lightweight per-user Cozy daemon: web UI, local API, durable records, local
worker manager, and private-rental sessions. It does not attach a log stream or load a package or
model. Repeating `up` returns the same healthy URL with `changed: false`; concurrent callers
converge on one daemon. A startup failure is returned directly as a bounded diagnostic and does
not create a persistent log. Commands that require the daemon may ensure the same process is
running automatically.

## Packages

Search the Tensorhub catalog, install a package, and inspect local installations:

```sh
cozy package search video
cozy package search org/name
cozy package install org/name
cozy package install org/name@1.2.3
cozy package list
```

Remove local package generations with:

```sh
cozy package remove org/name
```

Publishing builds the current working tree with `uv build --wheel`; Git, commits, and a clean tree
are not publication inputs. Standard `[project].dependencies` remain the runtime authority. Cozy
recursively builds referenced local `[tool.uv.sources]` paths/workspace members as separate exact
wheels in the same publication, including local dependencies activated through requested extras.
It first asks Tensorhub whether the release is already committed, so a replay skips all wheel
builds and uploads. Builds and uploads stop when they repeatedly make no byte progress; uploads run
at no more than 16 files concurrently.
Creator does not guess which names a base worker image owns: it uploads local candidates, and
Tensorhub's exact per-profile inventory chooses a compatible base distribution instead of overlaying
it. Tensorhub resolves indexed requirements and freezes their exact wheels. Development dependency
groups are not published. The package name and
release come from `[project]`; `[tool.cozy]` supplies the Tensorhub organization:

```toml
[project]
name = "marco-polo-package"
version = "1.0.0"

[tool.cozy]
organization = "paul"
```

From that project directory, publishing is simply:

```sh
cozy package publish
```

This publishes `paul/marco-polo-package@1.0.0`; Cozy never invents a `v` prefix. The destination
comes from `tensorhub_url`. An enrolled machine authenticates automatically; an operator may still
configure `tensorhub_token` or `TENSORHUB_TOKEN` explicitly.

Publication succeeds even when no compatible base worker image is currently active. The package
appears in the catalog immediately, and Tensorhub qualifies the same immutable release when a
compatible base becomes available. Until then, invocation remains unavailable.

A pending retry preserves every file already accepted by Tensorhub and uploads only missing slots
from the current tree. If the tree changed after a partial publication, use a new project version
instead of relying on a retry to replace already accepted bytes.

Tensorhub owns serving promotion; publication does not move traffic implicitly.
See [package publication](docs/package-publication.md) for the release contract.

## Models

Model checkpoints live in the local TensorFS store:

```sh
cozy model search flux
cozy model download org/model@release --lane task=text-to-image
cozy model list
cozy model remove org/model
```

Publish an existing canonical TensorFS snapshot. The remote model name is created automatically
when absent:

```sh
cozy model publish org/model sha256:<snapshot>
```

Download and publication are resumable and verify content identities before making a local or
remote root visible.

## Invoke packages and jobs

Serving entrypoints and bounded jobs use the same command. Cozy reads the installed package descriptor
to determine the callable lifecycle:

```sh
cozy invoke run org/package/v1/generate \
  prompt="a watercolor lighthouse at dusk" \
  --out ./outputs

cozy invoke run org/package/v1/train epochs=3 --detach
cozy invoke list
cozy invoke cancel <invocation-or-job-id>
```

An attached invocation follows progress and returns its terminal result. `--detach` returns after
durable acceptance. Reusing an explicit `--idempotency-key` safely returns the same recorded work.

Local Runtime workers start on demand. Successful serving workers may remain resident for warm
reuse; job workers are reclaimed at terminal.

## Private rentals

With Tensorhub configured, rent a private worker for an exact package:

```sh
cozy rental new                    # Cozy GPUs, VRAM, and retail hourly prices
cozy rental new h200 org/package/v1/generate \
  --idempotency-key <unique-key>

cozy rental list
cozy invoke run org/package/v1/generate --worker <rental-id> prompt="moonlit lake"
cozy rental end <rental-id>
```

GPU names and prices come from Tensorhub's Cozy-owned rental catalog. Creator never exposes or
reads RunPod SKU names or provider prices.

Each rental has one local Ed25519 Creator key and one separate media bearer. Creator signs the
exact worker/boot/TLS identity it claims and sends only signed package/model intent over
WorkerControl. The worker resolves and downloads those artifacts directly from Tensorhub; Creator
never requests or relays package/model download URLs. Both credentials are removed when the rental
ends, and a lost Creator key requires a new rental.

Rentals can continue billing until Tensorhub confirms their termination. `rental end` and
`down --all` keep the Cozy daemon alive when remote absence cannot be confirmed.

## Release GPU memory or stop Cozy

These commands have deliberately different scopes:

```sh
cozy unload       # stop idle local Runtime workers and release their GPU models
cozy down         # stop locally; refuses while invocations or rentals are active
cozy down --all   # cancel all work, end all rentals, then stop the daemon
```

None of them deletes installed package or model bytes. A failed partial `down --all` leaves the
daemon running so cancellation and paid-resource reconciliation can continue.

The launch web UI is currently a stub rooted in [`web/`](web/). The daemon already serves
opaque output media and a bounded authenticated content-addressed upload API; full browser
authentication, upload-to-invocation binding, and media-history UX come with the real frontend.

## Configuration

The default local root is `~/.cozy`. Configuration is read once from
`~/.cozy/config.yaml`, then credential/location environment variables override it:

```yaml
tensorhub_url: https://tensorhub.example
tensorhub_token: replace-with-your-token
tfs: /usr/local/bin/tfs
port: 8818
local_rate_micro_usd_per_hour: 250000
```

Without a configured `port`, Cozy prefers `127.0.0.1:8818` and falls back to an available
loopback port when another process owns 8818. A configured nonzero port is strict; `port: 0`
explicitly asks the OS to select any available port.

Without `tensorhub_url`, Cozy uses the standing local Tensorhub at `http://127.0.0.1:8819`.

The YAML schema is strict: unknown keys, duplicate keys, nested structures, and multiple documents
are refused. Cozy does not load a working-directory `.env` file.

Supported environment variables are limited to:

- `COZY_HOME`
- `COZY_TFS`
- `TENSORHUB_URL`
- `TENSORHUB_TOKEN`

The daemon launcher also uses a private per-process bootstrap credential. Secrets are never
accepted as command-line values.

## Authentication

Register or recover this Creator installation directly from the CLI:

```sh
cozy auth login person@example.com
```

Tensorhub emails a one-time code. After it is entered, Cozy stores only this installation's
Ed25519 machine key under its mode-0700 home and mode-0600 credential file. Short AuthKit access
tokens stay in memory. Later authenticated commands sign a one-time challenge automatically; there
is no refresh token or repeated login command.

## Output and automation

Commands print concise human-readable results by default. `--json` emits stable structured output
for scripts and other software integrations. Interactive terminals add restrained color to errors;
redirected and JSON output never contain ANSI escapes. Progress goes to stderr.

```sh
cozy package search
# - cozy/marco-polo-derived
# - cozy/marco-polo-cu130
# - cozy/marco-polo-launch1

cozy package list --fields package,version,disk
cozy invoke list --json
cozy model search flux --full
```

A one-field list uses `- value` bullets; multiple selected fields become a compact table. The
collection length is already its count. `omitted` appears only when a limit withheld rows, and
diagnostic fields appear only under `--full` or an explicit `--fields` selection. Human errors
state the problem and repair directly; `--json` retains the stable class, code, message, remedy,
and contextual next action.

Shell exits are intentionally small: `0` success or idempotent no-op, `2` invocation/configuration
error, and `1` operational failure.

Billing management, datasets, passkey recovery, and the full web UI are planned later; Cozy does
not advertise placeholder commands for features that do not exist yet.
