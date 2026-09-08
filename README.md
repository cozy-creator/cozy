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

Install the host Runtime with its supported Python interpreter:

```sh
uv tool install --force --python 3.12 'cozy-runtime[media,model-execution]'
cozy-runtime version
```

Keep `--python 3.12` when reinstalling or upgrading. uv
[ignores dependency Python upper bounds](https://docs.astral.sh/uv/pip/compatibility/#requires-python-upper-bounds),
so an unqualified tool installation can select a newer interpreter that Runtime cannot use.

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
cozy help run
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
cozy package install org/name --version 1.2.3
cozy package install .
cozy package list
```

An explicit directory (`.`, `..`, `./project`, `../project`, or an absolute path) creates a
local-only editable install after a bounded source scan. While the daemon runs it watches that
tree: an edit atomically prepares a new install and re-prepares every worker holding the
package — the local one and each attached rental — so the next run is warm; every invocation
still checks the tree itself. A failed rebuild (a syntax error mid-edit) keeps the last good
install pinned, is logged typed in `cozy daemon log`, and refuses an invocation only if the tree
is still broken. `cozy package list --full` shows `synced` or `stale <error>` for the tree.
Editable installs are not published releases. Model bindings resolve the exact release and lane
declared in `package.toml` from the local TensorFS store; they never synthesize Hub release records.

Remove local package installs with:

```sh
cozy package remove org/name
```

Package authors may permanently yank an immutable release without freeing its package/version
coordinate for reuse:

```sh
cozy package yank org/name --version 1.2.3
```

Publishing builds the current working tree with `uv build --wheel`; Git, commits, and a clean tree
are not publication inputs. Standard `[project].dependencies` remain the runtime authority. Cozy
recursively builds referenced local `[tool.uv.sources]` paths/workspace members as separate exact
wheels in the same publication, including local dependencies activated through requested extras.
It first asks Tensorhub whether the release is already committed, so a replay skips all wheel
builds and uploads. Builds and uploads stop when they repeatedly make no byte progress; uploads run
at no more than 16 files concurrently.
Creator omits only Python/Torch/Runtime/TensorFS platform families and their small declared native
closure from its registry-wheel export; it has no worker-image inventory. It uploads exact
pure-Python registry and local dependency wheels, and the actual worker compares platform
requirements with its selected base before importing the package. Development dependency groups
are not published. The package name and release come from `[project]`; the owner is the
authenticated user's Tensorhub account:

```toml
[project]
name = "marco-polo-package"
version = "1.0.0"
```

From that project directory, publishing is simply:

```sh
cozy package publish
```

If the logged-in account is `paul`, this publishes `paul/marco-polo-package@1.0.0`; Cozy never
invents a `v` prefix. The destination comes from `tensorhub_url`, and an enrolled machine
authenticates automatically.

Publication succeeds even when no base worker image is active. The package appears in the catalog
immediately. On invocation, the selected worker verifies the immutable wheels against its actual
pinned base before installing them; Tensorhub does not prequalify packages against image profiles.

A pending retry preserves every file already accepted by Tensorhub and uploads only missing slots
from the current tree. If the tree changed after a partial publication, use a new project version
instead of relying on a retry to replace already accepted bytes.

Tensorhub owns serving promotion; publication does not move traffic implicitly.
See [package publication](docs/package-publication.md) for the release contract.

## Models

Search lists `MODEL`, `FAMILY`, `RELEASE`, and `LANES`, with one row per available
release. Long columns are shortened in the compact human table; `--full` and JSON
retain complete values and lane lists.
`cozy model info org/name` shows all available releases, while `@release` selects
one exact tag. Info shows full lanes and immutable checkpoint refs. Unavailable
Hub timestamps remain unavailable; model creation is distinct from release creation.

Model releases live in the local TensorFS store:

```sh
cozy model search flux
cozy model info org/model
cozy model info org/model@release
cozy model download org/model@release local/flux --lane task=text-to-image
cozy model list
cozy model remove local/flux
cozy model gc
```

`model remove` deletes the repository name and reclaims its bytes in the same act, printing
`reclaimed: 5.2GiB`; `model list` shows `unreferenced 0` after it. `model gc` reclaims what no
local model names for any other reason (a crashed download, an abandoned ingest), and the daemon
runs the same pass on `maintenance.gc_cron` (default `0 3 * * *`, while it is up and manages
nothing). What is reclaimed is TensorFS's decision from its filesystem census — repos, manifests,
blobs — never a database's. A pass never runs beside an active request (a download's bytes are
unnamed until its commit): `remove` defers reclamation to `model gc`, `model gc` refuses, the
cron logs `gc: deferred`. A live worker holding a model refuses by name (`cozy unload` first).

Upload an already canonical local alias, or ingest a provider source with `model download`.
Run quantization and other weight-producing jobs through the ordinary package command.
The job receives prepared model inputs and its typed payload; TensorFS owns provider
conversion. The destination owner must match the account shown by `cozy auth`:

```sh
cozy model upload local/model org/model

cozy run paul/minimax-h3-tools/four-lane \
  --model.dits=hf://MiniMaxAI/MiniMax-H3@<full-commit> \
  --model.shared=hf://MiniMaxAI/MiniMax-H3@<full-commit> \
  --source-profile dits=hf/minimax-h3/native-dual-bf16/1 \
  --source-profile shared=hf/minimax-h3/shared-bf16/1 \
  --publish-to paul/minimax-h3 --rental-only --await

cozy run org/quantize/convert --model.source=org/model@release/bf16 \
  --in quantize.json --publish-to org/quantized --rental-only
```

`--publish-to` retains each declared weight output as an owner-only immutable checkpoint;
it does not create public lane pointers. `--source-profile slot=profile` narrows each foreign
input to a reviewed TensorFS profile. Currently every foreign input must name the same
provider source and all model slots must be foreign; mixed inputs and multiple independent
sources refuse before acquisition. Tensorhub inputs use the ordinary exact model resolver.
The Hub measures their shared closure when sizing a rental; a manifest's small metadata
length is never a disk-size estimate.

`cozy run ... --dry-run` resolves the selected release, typed payload, exact inputs, and
foreign conversion headers without starting a daemon, queueing work or renting. It reads
provider metadata and headers, never downloads the model bodies. Submission records an
ordinary `job-*` run; `--await` watches it and `cozy run watch <job-id>` attaches later.
Use `--idempotency-key` to replay the same request. Existing source operations continue
through their recorded intent after an upgrade. `--producer` on model upload/download is
removed; jobs have one invocation surface.

A successful producer with blocked publication retains its receipts and worker bytes.
`cozy run retry-publication <run-id>` retries publication without rerunning the producer;
explicit cancellation abandons unfinished output custody through the same finalizer.

Publish, repoint, add, or remove release lanes separately. Omitted lanes stay unchanged:

```sh
cozy model publish org/model --release 1.0.0 \
  --lane bf16=sha256:<checkpoint> \
  --lane fp8=sha256:<checkpoint>
cozy model publish org/model --release 1.0.0 --remove-lane defective
cozy model yank org/model --release 1.0.0
```

Checkpoint IDs are immutable. Release labels and their lane maps are mutable owner pointers. Ordinary
consumers follow a release lane so fixes take effect; accepted runs freeze the checkpoint they resolved.

With `--rental`, job packages resolve directly to their latest non-yanked immutable
Tensorhub releases. The transfer intent pins `(package, release)`; downloaded execution files carry
their own exact refs, including `metadata/package-interface.json`.
They do not need to be installable in the laptop's local Python environment. Local production still
uses the ordinary installed package. `--rental-only` skips local capacity and requires an
external attempt. Downloads that could rent currently refuse until Creator wires the negotiated
weights-read return plane; local-only sources under `--rental` remain local.

## Run packages and jobs

Private scripts define ordinary `main()` or `main(ctx)`, synchronous or asynchronous.
They need no App, request/result class, published package, or separate install command.
Dependencies may be declared with PEP 723 inline metadata:

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///

async def main(ctx):
    ctx.log("Starting my experiment")
    ctx.raise_if_cancelled()
    # Call your imported operations and write ordinary Python control flow here.
    ctx.progress(1.0, stage="complete")
```

The optional context supplies cancellation, logs, progress and metrics. Discovery checks the
script's syntax without importing it on the client; its captured code executes on the worker.
Generator entrypoints are refused rather than reported as completed without running their body.

```sh
cozy run ./experiment.py --rental-only --await
# Edit the script or its declared local libraries, then start a fresh run:
cozy run ./experiment.py --rental-only --await
```

To explicitly continue development on an earlier run's retained machine and record that
relationship, use the optional retry handle:

```sh
cozy run ./experiment.py --retry <prior-run-id> --rental-only --await
```

Each execution captures its source and dependency closure. A corrected retry creates a new
record linked to the previous run; it never rewrites the old attempt's code or result.
Retained source preparation can be adopted when its exact source/profile still matches.
Code and unpublished wheels go directly to the authenticated private worker, without a
Tensorhub package publication or intermediate upload. Published dependencies can still come
from their package repositories. Ordinary Python edits do not rebuild the worker image.

Failed private jobs keep their resumable work and rental. `cozy run pause <id>` stops execution
while retaining that state; `cozy run resume <id>` runs the exact captured revision again.
Changed code or parameters are captured by a new run, optionally linked with `--retry`.
`cozy run cancel <id>` abandons the retained
work, releasing only resources with no other owner. Retained rentals continue billing.
An unchanged deterministic failure is not automatically retried. `--await` returns when a
transaction blocks or pauses, and the ordinary run view explains the stopped state.

An invocable dependency exposes an ordinary typed Python call. Its implementation is
captured separately; the parent imports a generated interface containing signatures and
result types. Calling it creates an ordinary managed child job. A reusable computation
opts in with `@invocable(memoize=True)` and registers through `app.job(function)`.
Arbitrary helper functions and external effects do not become cached operations.

On either a local or rented worker, a memoized child can acquire an earlier successful result when its
exact implementation closure, entrypoint, canonical inputs, resolved model checkpoints,
and measured numerical environment match. A fresh script can reuse this work without a retry
ID. Run IDs, the parent's script digest and call position are not computation cache keys.
Native model results travel as `ModelArtifact`
references; reuse acquires independent native custody before the new result is visible.
The reference names its original producer and receipt without copying tensor bytes.

The implementation boundary is a package and its dependencies, not an individual Python
function body. Editing B inside a package that also implements A invalidates that package's
reusable calls. Keep expensive independent operations in independently captured packages.
Source ingestion already has its own exact source/profile identity, so changing an H3
producer does not invalidate its retained source conversion. No system can infer that
arbitrary Python side effects are reversible or safe to repeat.

Scalar and tensor compositions use the same Runtime workspace journal locally and on a
private rental. The journal belongs to the configured TensorFS store and survives package
worker and daemon restarts. The remote Host authenticates access to that same service.
A completed result stored only on a rented machine continues holding the rental until its
owner releases it; completion alone does not make those bytes remotely durable.
Parent and child attempts retain separate history.
The worker may retain reusable results under an independent cache hold. Canceling a producing
run releases its own holds; cache or other consumer holds may keep those bytes available.
Cache holds alone do not keep paid capacity alive, and evicted or missing results compute again.
The cache records original result provenance without inventing another compute attempt.
Use `cozy cache prune` for unused local operation results, or `cozy rental prune <rental>`
on a rented machine. Both preserve results still owned by retained runs or other consumers.

For editable libraries, declare paths explicitly in the same script metadata:

```python
# /// script
# dependencies = ["my-algorithm"]
# [tool.uv.sources]
# my-algorithm = {path = "./my-algorithm", editable = true}
# ///
```

Each run captures current declared source bytes, including library edits made without a
version bump. Source changes never mutate an already active or captured attempt. An ordinary
library call runs inside its caller; only operations participating in the managed memoization
contract reuse completed computation.

Serving entrypoints and bounded jobs use the same command. Cozy reads the installed package interface
to determine the callable lifecycle:

```sh
cozy run org/package/generate \
  prompt="a watercolor lighthouse at dusk" \
  --out ./outputs \
  --await

cozy run org/package/train epochs=3
cozy run list
cozy run watch <run-id>
cozy run cancel <run-id>
```

Every run is durable. Cozy observes its first three seconds so a warm, weightless function can still
return its result directly; otherwise it prints the run id, live status, and authoritative queue
position, then returns while the daemon continues. `--await` stays attached through the terminal and
shows named pipeline stages, measured step speed, elapsed time, and a whole-run estimate only when
the package reports `overall_fraction`; stage-local fractions are labeled as stage progress.
`--await --json` writes typed JSONL progress events to stderr and one final JSON result to stdout.
Human progress is automatic with `--await`; no separate progress flag is needed.
Model overrides accept `--model.<param>=org/model@release/lane` after the target;
the existing `model.<param>=...` payload spelling has the same meaning.
Without an override, Cozy uses the owner's Tensorhub binding, then the selected function's
published model default ladder. Authors can declare that ladder with
`@invocable(defaults={"model": [{"gpu": "H100", "lane": "org/model@release/fp8"}]})`.
Each slot's rungs must use the same model and release; the GPU patterns select its lanes.
These authored defaults stay in the package interface and are never copied into Hub bindings.
Use `cozy package unbind org/package generate.models.model` to remove an owner override
and return that slot to its authored default.
`cozy run watch <run-id>` attaches to that same progress stream later; interrupting a watcher
detaches without canceling the durable run.

If the package is missing, Cozy installs the newest compatible release from Tensorhub before
starting; the download is visible progress, not an interactive prompt. Reusing an explicit
`--idempotency-key` safely returns the same recorded work. Before submission, Cozy materializes an
omitted integer `seed`, so an unseeded run draws fresh entropy.

Every run says where its files are. Result files land in the package's own store,
`~/.cozy/outputs/<org>-<package>/`, or under `--out DIR`; either way each file is named by its own
content digest, `<sha256>.<ext>`, so regenerating the same bytes lands on the same file and two
different results never collide. Nothing but result files is ever written there, and nothing is
staged anywhere first: the worker is granted that directory and writes each file into it under its
digest name, the request payload rides the grant itself, and input assets are read from the
original input paths without copying them. Keep those files available and unchanged until the
request finishes; missing or modified inputs are refused before execution. Downloaded or resized
media lives in the system temporary directory under `cozy/`, named by its content hash.
`saved:` lists the absolute paths (`saved[].path` under `--json`). The
daemon probes the destination writable at submit and refuses typed (`output_destination_unwritable`)
before any execution, and records the destination with the request, so closing the command or
restarting Cozy does not lose where the files are.

Local Runtime workers start on demand. Successful serving workers may remain resident for warm
reuse; job workers are reclaimed at terminal.

## Private rentals

With Tensorhub configured, rent generic private capacity:

```sh
cozy rental new                    # Cozy GPUs, VRAM, and retail hourly prices
cozy rental new h200                # prints e.g. otter
cozy rental new h200 --model paul/minimax-h3@1.0.0/bf16
cozy rental new h200 \
  --idempotency-key <unique-key>

cozy rental list                   # current rented machines, live on a terminal
cozy run org/package/generate --rental prompt="moonlit lake"
cozy rental end otter
```

GPU names and prices come from Tensorhub's Cozy-owned rental catalog. Creator never exposes or
reads RunPod SKU names or provider prices. `cozy run` is local-only by default, while
`--rental` permits Creator to reuse or acquire remote capacity only when ready local capacity cannot
run the request. `--rental-only` deliberately bypasses local capacity and requires an external
rental. Both modes remain under the configured fleet ceiling. Creator names every private rental
with one memorable word, unique among this host's live rentals (a released word is drawn again);
Tensorhub's identity for it is its `pr-…` id, which is what the provider-side pod is named after.
The name carries no workload facts.

Use `--model org/model@release/lane` to size a manual rental for known checkpoints; repeat
`--model` for several models. Creator resolves and pins each manifest, and Tensorhub measures
their shared object closure and required disk headroom. This declares capacity needs; packages
and models are prepared when requests run. Retrying the same idempotency key reuses the pinned
set, including when `--model` is omitted on the retry.

Every rental the daemon owns — bought for a request or started with `cozy rental new` — ends on
its own once nothing has been queued, running, or owed on it for `rentals.idle_release_s`
(default 300). Running work on it is what keeps it: the clock restarts at each settled attempt,
and a rental that never ran anything counts from the moment Tensorhub first reported it `ready`,
so a pod still booting is never ended. A managed rental whose job is done goes at once.
`cozy rental list` shows each machine's idle time and when it will be released; `cozy rental end`
ends one now. A release Tensorhub does not confirm is retried until it does. Set
`idle_release_s: 0` to leave every rental to `cozy rental end`.

Rental creation sends only that private machine name, the SKU, and introduction credential material—never a package, model,
request, profile, image, or placement. Tensorhub readiness means the worker location and TLS identity
are attachable. Creator then claims that worker directly and sends signed exact package and model
release refs; no WorkerControl frame is relayed through Tensorhub. Creator learns the derived
placement/binding and exact environment identities from the worker's existing
observed-state stream before dispatch. Tensorhub's exact PackageInterface validates request/result
shape; Creator never supplies a remote profile, CUDA choice, or precomputed PlacementSet.

Each rental has one local Ed25519 Creator key and one separate media bearer. Both credentials are
removed when the rental ends, and a lost Creator key requires a new rental.

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

The daemon's own words — the orchestrator's frame-by-frame account — are in
`$COZY_HOME/daemon.log`, rotated once at 32 MiB (`daemon.log.1`). `cozy daemon log` prints it;
`cozy daemon log -f` follows it.

The daemon also leaves on its own once it has had nothing to manage for `daemon.idle_shutdown_s`
(default 900): no rental it owns, no request or attempt it owes, no transfer, export, launch or
teardown in flight, no event stream attached. User interaction is not the signal, and an idle
warm local worker is not work — the exit drains it exactly as `down` does. A rental holds the
daemon until its idle release, or `cozy rental end`, has confirmed the pod gone. Set
`idle_shutdown_s: 0` to keep the daemon up until `cozy down`. The next command that needs the
daemon starts it again.

The launch web UI is currently a stub rooted in [`web/`](web/). The daemon already serves
opaque output media and a bounded authenticated content-addressed upload API; full browser
authentication, upload-to-invocation binding, and media-history UX come with the real frontend.

## Configuration

The default local root is `~/.cozy`. The TensorFS store is its own independent home —
`~/.tensorfs` by default, moved with `TENSORFS_HOME` or `tensorfs_root` — written only by
`tfs`. What lives under `~/.cozy`:

| path | what it is |
|---|---|
| `installs/` | one immutable environment per installed package, plus the writer's `.lock` |
| `outputs/<org>-<package>/` | result files, named by their own content digest; never removed automatically |
| `inputs/` | uploaded/streamed request bodies only, created on demand; a local CLI file input stays at its caller path |
| `workers/<instance>/` | a local worker's `run/` state and `worker.log`; removed when it exits, the log stays |
| `tmp/<request-id>/` | bytes in flight for a model transfer; removed when the request settles |
| `creator.sqlite`, `daemon.log` | the one local lifecycle database (mode 0600) and the daemon's own log |
| `daemon.lock` | the daemon's lifetime ownership record: address, pid, and the per-launch CLI token (mode 0600) |

Nothing is kept on disk without a live reason; the daemon's start-time sweeps (`cozy up` prints
them) are only the backstop for a process that crashed and report `reclaimed 0` on a healthy box.

Configuration is read once from `~/.cozy/config.yaml`, then credential/location environment
variables override it:

```yaml
tensorhub_url: https://tensorhub.example
tensorhub_token: replace-with-your-token
huggingface_token: hf_replace-with-your-token
civitai_token: replace-with-your-token
tfs: /usr/local/bin/tfs
port: 8818
local_rate_micro_usd_per_hour: 250000
rentals:
  max_hourly_spend_usd: 4.00
  idle_release_s: 300
daemon:
  idle_shutdown_s: 900
maintenance:
  gc_cron: "0 3 * * *"
```

Without a configured `port`, Cozy prefers `127.0.0.1:8818` and falls back to an available
loopback port when another process owns 8818. A configured nonzero port is strict; `port: 0`
explicitly asks the OS to select any available port.

Without `tensorhub_url`, Cozy uses the standing local Tensorhub at `http://127.0.0.1:8819`.

The YAML schema is strict: unknown keys, duplicate keys, undeclared nested structures, and multiple
documents are refused. Cozy does not load a working-directory `.env` file.
When either provider token is present, `config.yaml` must be an owner-owned regular file with mode
`0600`; symlinks are refused.

Supported environment variables are limited to:

- `COZY_HOME`
- `COZY_TFS`
- `HF_TOKEN`
- `CIVITAI_TOKEN`
- `TENSORHUB_URL`
- `TENSORHUB_TOKEN`

The daemon launcher also uses a private per-process bootstrap credential. Secrets are never
accepted as command-line values.

## Authentication

Register or recover this Creator installation directly from the CLI:

```sh
cozy auth login person@example.com
cozy auth
cozy auth revoke-other-machines
cozy auth logout
```

Tensorhub emails a one-time code. After it is entered, Cozy stores only this installation's
Ed25519 machine key under its mode-0700 home and mode-0600 credential file. Short AuthKit access
tokens stay in memory. A new user also chooses one immutable Tensorhub account name during this
login; existing accounts skip that prompt. Later authenticated commands sign a one-time challenge
automatically; there is no refresh token or repeated login command. `cozy auth` verifies the stored
key, shows the email and Tensorhub account name, or reports `not logged in` when this installation
has no usable key. Revoking other machines asks for a fresh email code. Logout revokes this
machine before erasing its local key. Once enrolled, the machine key takes precedence over a stale
configured `tensorhub_token`; that operator token remains a fallback only for unenrolled automation.

## Output and automation

Commands print concise human-readable results by default. `--json` emits stable structured output
for scripts and other software integrations. Interactive terminals add restrained color to errors;
redirected and JSON output never contain ANSI escapes. Progress goes to stderr.

```sh
cozy package search
# - cozy/marco-polo-derived
# - cozy/marco-polo-cu130
# - cozy/marco-polo-launch1

cozy package list --fields package,version,size
cozy run list --json
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
