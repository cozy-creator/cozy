# Cozy Creator

Cozy Creator is a local-first command-line application for generative media. It installs
package code, downloads models, runs package callables on your machine or a private rented
worker, and keeps your local execution records and outputs under your control.

The command is `cozy`.

## Install

Binary releases are not published yet. Building from source currently requires Go 1.26 or
newer. Package installation also uses `uv`; model download and publication use the `tfs`
executable from TensorFS.

```sh
git clone https://github.com/cozy-creator/cozy-creator.git
cd cozy-creator
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
# open the returned http://127.0.0.1:<port>/ URL
```

`up` backgrounds one lightweight per-user controller: web UI, local API, durable records, local
worker manager, and private-rental sessions. It does not attach a log stream or load a package or
model. Repeating `up` returns the same healthy URL with `changed: false`; concurrent callers
converge on one controller. A startup failure is returned directly as a bounded diagnostic and does
not create a persistent log. Commands that require the controller may ensure the same process is
running automatically.

## Packages

Search the Tensorhub catalog, install a package, and inspect local installations:

```sh
cozy package search video
cozy package search org/name
cozy package install org/name@release \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --major v1 \
  --reason "local install"
cozy package list
```

For local package development, install a source tree explicitly:

```sh
cozy package install org/name --dir ./my-package --allow-unsigned
```

Remove local package generations with:

```sh
cozy package remove org/name
```

Publishing builds the current working tree with `uv build --wheel`; Tensorhub accepts only one
bounded pure project wheel. Git, commits, and a clean tree are not publication inputs. Tensorhub derives
compatible base worker profiles from wheel metadata, `uv.lock`, and its image inventory. The
package name is created automatically when absent:

```sh
cozy package publish org/name \
  --release 1.0.0 \
  --dir . \
  --reason "release 1.0.0"
```

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
cozy model publish org/model sha256:<snapshot> --reason "initial release"
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
cozy rental new org/package@release \
  --accelerator "NVIDIA H200" \
  --reason "private generation" \
  --idempotency-key <unique-key>

cozy rental list
cozy invoke run org/package/v1/generate --worker <rental-id> prompt="moonlit lake"
cozy rental end <rental-id>
```

Rentals can continue billing until Tensorhub confirms their termination. `rental end` and
`down --all` keep the local controller alive when remote absence cannot be confirmed.

## Release GPU memory or stop Cozy

These commands have deliberately different scopes:

```sh
cozy unload       # stop idle local Runtime workers and release their GPU models
cozy down         # stop locally; refuses while invocations or rentals are active
cozy down --all   # cancel all work, end all rentals, then stop the controller
```

None of them deletes installed package or model bytes. A failed partial `down --all` leaves the
controller running so cancellation and paid-resource reconciliation can continue.

The launch web UI is currently a stub rooted in [`web/`](web/). The controller already serves
opaque output media and a bounded authenticated content-addressed upload API; full browser
authentication, upload-to-invocation binding, and media-history UX come with the real frontend.

## Configuration

The default local root is `~/.cozy`. Configuration is read once from
`~/.cozy/config.yaml`, then credential/location environment variables override it:

```yaml
tensorhub_url: https://tensorhub.example
tensorhub_token: replace-with-your-token
tfs: /usr/local/bin/tfs
port: 2699
local_rate_micro_usd_per_hour: 250000
```

The YAML schema is strict: unknown keys, duplicate keys, nested structures, and multiple documents
are refused. Cozy does not load a working-directory `.env` file.

Supported environment variables are limited to:

- `COZY_HOME`
- `COZY_TFS`
- `TENSORHUB_URL`
- `TENSORHUB_TOKEN`

The controller launcher also uses a private per-process bootstrap credential. Secrets are never
accepted as command-line values.

## Output and automation

Command results and errors are one typed document. TOON is the concise default; `--json` changes
only the encoding. Success output contains the domain answer without an `ok/kind/data` envelope.
Progress goes to stderr.

```sh
cozy package search
# packages[#3]:
#   cozy/marco-polo-derived
#   cozy/marco-polo-cu130
#   cozy/marco-polo-launch1

cozy package list --fields package,version,disk
cozy invoke list --json
cozy model search flux --full
```

A one-field list is a scalar collection; multiple selected fields become a compact table. The
collection length is already its count. `omitted` appears only when a limit withheld rows, and
diagnostic fields appear only under `--full` or an explicit `--fields` selection. Errors retain
their stable class, code, message, remedy, and contextual repair action.

Shell exits are intentionally small: `0` success or idempotent no-op, `2` invocation/configuration
error, and `1` operational failure. The structured error document retains the detailed stable code.

Authentication, billing management, datasets, and the full web UI are planned later; Cozy does
not advertise placeholder commands for features that do not exist yet.
