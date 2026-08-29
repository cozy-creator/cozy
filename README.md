# Cozy Creator

Cozy Creator is a local-first command-line application for running generative-media
endpoints. Install an endpoint once, run it on your own machine or a rented GPU, and keep
the resulting media and execution history under your control.

The command is named `cozy`.

## What you can do

- Install versioned endpoint releases with locked Python environments.
- Run generative-media functions locally.
- Rent a remote GPU through Tensorhub and send work to that exact worker.
- Search the public endpoint and model catalogs.
- Download and publish verified model checkpoints.
- Compose multi-shot videos and submit durable workflows or jobs.
- Inspect logs, outputs, resource fit, costs, and execution status.
- Use stable JSON output from scripts and other applications.

Cozy Creator consists of a small CLI and one local background service. The service owns
endpoint processes and durable execution records; normal `cozy` commands talk to it over
localhost.

## Project status

Cozy Creator is currently early-access software. There are no published binary releases
yet, so installation currently requires access to this source repository. Several
cloud-facing commands are also still marked as planned. Run `cozy commands` to see
exactly what your installed build supports. Planned commands have a `*` beside their
names.

## Install

### Requirements

- Go 1.26 or newer while binary releases are unavailable.
- [`uv`](https://docs.astral.sh/uv/) to install endpoint environments.
- An NVIDIA GPU and compatible driver only for endpoints that require CUDA.
- The `tfs` executable only when downloading or publishing model checkpoints.

Clone the repository using your GitHub access, then install the current build:

```sh
git clone git@github.com:cozy-creator/cozy-creator.git
cd cozy-creator
go build -o cozy .
mkdir -p ~/.local/bin
install -m 0755 ./cozy ~/.local/bin/cozy
```

Make sure `~/.local/bin` is on `PATH`, then confirm the installation:

```sh
cozy version
```

On Windows, build `cozy.exe` and move it into a directory on `PATH`:

```powershell
go build -o cozy.exe .
```

If you have received a release archive and its `SHA256SUMS`, use the included installer:

```sh
scripts/install.sh --asset ./cozy-<version>-linux-amd64.tar.gz
```

On Windows, use `scripts/install.ps1 -Asset <archive>` instead. Both installers verify
the checksum before replacing an existing installation.

## Start Cozy

Start the local service in the background:

```sh
cozy up -d
```

Then inspect the current state:

```sh
cozy
cozy doctor
```

Running `cozy` with no command shows the command overview, global flags, and examples.
`cozy status` is the operational dashboard for the service, installed endpoints, workers,
jobs, and workflows.

Stop the service when you are finished:

```sh
cozy down
```

Stopping Cozy drains its endpoint processes. It does not delete installed endpoints,
execution records, or outputs.

## Install an endpoint

An endpoint release is a `.tar.gz` archive containing the endpoint code, its descriptor,
and a locked dependency environment. Install it using the digest supplied by its
publisher:

```sh
cozy install org/endpoint \
  --from ./endpoint-1.0.0.tar.gz \
  --digest sha256:<digest>
```

Inspect what was installed:

```sh
cozy ls
cozy describe org/endpoint
cozy fit org/endpoint
```

`cozy install` verifies the archive before executing anything from it. The
`--allow-unsigned` option exists for local development, but should not be used for an
endpoint obtained from someone else.

## Run an endpoint

Invoke a function by its full reference:

```sh
cozy run org/endpoint/v1/generate "a watercolor lighthouse at dusk" \
  --out ./outputs
```

Arguments can be supplied as `key=value` pairs or as one JSON document:

```sh
cozy run org/endpoint/v1/generate prompt="a red bicycle" --seed 7
cozy run org/endpoint/v1/generate --in ./request.json --out ./outputs
```

Local execution is the default. Cozy starts the endpoint when it is first needed. To
prewarm it explicitly:

```sh
cozy start org/endpoint -d
```

Useful runtime commands:

```sh
cozy status
cozy logs org/endpoint
cozy stop org/endpoint
cozy media ls
```

Use `--stream` on `cozy run` for newline-delimited progress and `--timeout <duration>` to
set a request deadline.

## Use a rented GPU

Configure Tensorhub first, then request a worker for an endpoint:

```sh
cozy rent org/endpoint \
  --accelerator <gpu-model> \
  --reason "video generation" \
  --idempotency-key <unique-key>
```

List the rentals attached to this machine and verify a ready worker:

```sh
cozy rent ls
cozy rent probe <rental-id>
```

Send an invocation to that worker:

```sh
cozy run org/endpoint/v1/generate "a moonlit mountain lake" \
  --worker <rental-id> \
  --out ./outputs
```

Rentals can incur charges until they are explicitly released. Preview the release, then
confirm it:

```sh
cozy rent release <rental-id>
cozy rent release <rental-id> --yes
```

A timeout while waiting for a rental does not destroy the pod. Check `cozy rent ls` and
release it when it is no longer needed.

## Endpoint publication and profiles (cl-039/cl-043)

`cozy endpoint publish <org/name> --release <id> --profile <profile> ...` snapshots one
clean committed source subtree, creates a deterministic provenance archive and pure
project wheel, canonicalizes the reviewed descriptor/config, and drives Tensorhub's exact
`begin` → presigned missing-role PUTs → `finalize` route pair. The declaration carries no
path, credential, URL, image, or command. Its explicit sorted profile set has no mutable
default. Exact replay sends the same canonical declaration and lets Tensorhub return the
same pending/committed result; Creator keeps no publication journal.

`endpoint.release.json` is optional only for a genuinely weightless endpoint. It contains
the sorted compatible accelerator models, exact `{org,name,checkpoint_id}` model roots,
and complete path-free model bindings: binding path, checkpoint, exact config document/
asset refs, construction-order execution layout, and hardware variant. A descriptor with
model inputs and no bindings refuses before upload. `endpoint.evaluated-config.json` is
canonicalized when present; its absent weightless spelling is `{}`.

The project wheel is exactly one `py3-none-any` wheel. Repeatable
`--custom-wheel <profile>=<path>` values are separately inspected immutable prebuilt
wheels, aggregated by exact digest, and uploaded only under Tensorhub-requested
`custom_wheel:<distribution>:<digest>` roles. Creator runs no backend/compiler, embeds no
custom wheel into the project wheel, and never repairs, renames, or retags one.

`cozy endpoint qualify <org/name>@<release> --profile ... --gpu ... --max-cost ...`
is the separate cost-bounded hardware act. Publication never rents implicitly.
`cozy endpoint promote <org/name> <release> --serve <vN/function> ...` atomically moves
one or more explicit qualified serving pointers; a bare major has no guessed function.
`datasets push|pull` waits on th-035.

See [docs/endpoint-publication.md](docs/endpoint-publication.md) for the release file, custom-wheel,
qualification, promotion, and qualified managed-local install contracts.

## Tensorhub catalog and models

Catalog reads are public:

```sh
cozy hub status
cozy endpoint search video
cozy endpoint show org/endpoint
cozy model search flux
cozy model show org/model
```

Model transfers require the `tfs` executable. Downloads are verified and resumable:

```sh
cozy model download org/model@release --lane task=text-to-video
```

Creating catalog entries, publishing models, and renting workers require the configured
Tensorhub token. Account login is not part of the current early-access release.

## Videos, workflows, and jobs

Cozy can turn an editable `cozy.video/1` YAML file into a durable workflow:

```sh
cozy video compose ./film.cozy-video.yaml --h3 org/h3 --out ./film.plan.json
cozy video submit ./film.cozy-video.yaml \
  --h3 org/h3 \
  --idempotency-key <unique-key>
```

See [docs/cozy-video.md](docs/cozy-video.md) for the source format.

General workflows and bounded jobs use the same local service:

```sh
cozy workflow submit --in ./plan.json --idempotency-key <unique-key>
cozy workflow status <workflow-id>

cozy job submit org/endpoint/v1/train epochs=3 --follow
cozy job status <job-id>
```

Use an idempotency key for paid or long-running work. Retrying the same operation with the
same key returns the same request instead of creating another one.

## Configuration

The default data directory is `~/.cozy`. Put user configuration in
`~/.cozy/config.yaml`:

```yaml
tensorhub_url: http://127.0.0.1:8080
tensorhub_token: replace-with-your-token
tfs: /usr/local/bin/tfs
port: 2699
yield: smart
local_rate_micro_usd_per_hour: 250000
```

Only add the settings you need. `yield` can be `smart`, `always`, or `never`. The local
rate is optional and is used only to estimate the cost of work on your own machine.

The following environment variables override the file when needed:

- `COZY_HOME`
- `COZY_TFS`
- `COZY_LOCAL_RATE_MICRO_USD_PER_HOUR`
- `TENSORHUB_URL`
- `TENSORHUB_TOKEN`

Run `cozy hub status` or `cozy hub config` to inspect the active Tensorhub configuration
without printing the raw token.

## Storage and cleanup

Cozy keeps its database, installed environments, retained outputs, and rental credentials
under `~/.cozy` by default.

```sh
cozy ls
cozy media ls
cozy gc
```

`cozy gc` prints a reclaim plan without changing anything. Add `--yes` to execute it.
Removing an endpoint also requires explicit confirmation:

```sh
cozy rm org/endpoint --yes
cozy gc --yes
```

There are no interactive confirmation prompts, which keeps commands safe and predictable
in terminals and scripts.

## Scripting and help

Most commands support structured output:

```sh
cozy status --json
cozy ls --fields endpoint,version,disk
cozy commands --full
```

Use the built-in command reference for the exact surface supported by your binary:

```sh
cozy --help
cozy commands
cozy help run
cozy help rent
```

Errors are typed and use stable exit codes. See [docs/exit-matrix.md](docs/exit-matrix.md)
when integrating Cozy into another application.
