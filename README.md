# Cozy Creator

Cozy Creator is a local-first command-line application for generative media. It installs
endpoint code, downloads models, runs endpoint callables on your machine or a private rented
worker, and keeps your local execution records and outputs under your control.

The command is `cozy`.

## Install

Binary releases are not published yet. Building from source currently requires Go 1.26 or
newer. Endpoint installation also uses `uv`; model download and publication use the `tfs`
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
cozy help endpoint install
cozy help invoke run
cozy -v
```

There is no public stack, `up`, or `down` command. Commands that need durable coordination
automatically start one lightweight per-user controller. Installed endpoints and downloaded
models remain files on disk until an invocation needs them.

## Endpoints

Search the Tensorhub catalog, install an endpoint, and inspect local installations:

```sh
cozy endpoint search video
cozy endpoint search org/name
cozy endpoint install org/name@release \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --major v1 \
  --reason "local install"
cozy endpoint list
```

For local endpoint development, install a source tree explicitly:

```sh
cozy endpoint install org/name --dir ./my-endpoint --allow-unsigned
```

Remove local endpoint generations with:

```sh
cozy endpoint remove org/name
```

Publishing packages one deterministic pure project wheel and an explicit compatibility-profile
set. The endpoint name is created automatically when absent:

```sh
cozy endpoint publish org/name \
  --release 1.0.0 \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --dir . \
  --reason "release 1.0.0"
```

Tensorhub owns hardware qualification and serving promotion; publication does neither implicitly.
See [endpoint publication](docs/endpoint-publication.md) for the release contract.

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

## Invoke endpoints and jobs

Serving entrypoints and bounded jobs use the same command. Cozy reads the installed descriptor
to determine the callable lifecycle:

```sh
cozy invoke run org/endpoint/v1/generate \
  prompt="a watercolor lighthouse at dusk" \
  --out ./outputs

cozy invoke run org/endpoint/v1/train epochs=3 --detach
cozy invoke list
cozy invoke cancel <invocation-or-job-id>
```

An attached invocation follows progress and returns its terminal result. `--detach` returns after
durable acceptance. Reusing an explicit `--idempotency-key` safely returns the same recorded work.

Local Runtime workers start on demand. Successful serving workers may remain resident for warm
reuse; job workers are reclaimed at terminal.

## Private rentals

With Tensorhub configured, rent a private worker for an exact endpoint:

```sh
cozy rental new org/endpoint@release \
  --accelerator "NVIDIA H200" \
  --reason "private generation" \
  --idempotency-key <unique-key>

cozy rental list
cozy invoke run org/endpoint/v1/generate --worker <rental-id> prompt="moonlit lake"
cozy rental end <rental-id>
```

Rentals can continue billing until Tensorhub confirms their termination. `rental end` and
`exit --all` keep the local controller alive when remote absence cannot be confirmed.

## Release GPU memory or exit

These commands have deliberately different scopes:

```sh
cozy unload       # stop idle local Runtime workers and release their GPU models
cozy exit         # stop locally; refuses while invocations or rentals are active
cozy exit --all   # cancel all work, end all rentals, then stop the controller
```

None of them deletes installed endpoint or model bytes. A failed partial `exit --all` leaves the
controller running so cancellation and paid-resource reconciliation can continue.

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
only the encoding. Progress goes to stderr.

```sh
cozy endpoint list --fields endpoint,version,disk
cozy invoke list --json
cozy model search flux --full
```

Shell exits are intentionally small: `0` success or idempotent no-op, `2` invocation/configuration
error, and `1` operational failure. The structured error document retains the detailed stable code.

Authentication, billing management, datasets, and a web UI are planned later; Cozy does not
advertise placeholder commands for features that do not exist yet.
