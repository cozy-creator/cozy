# Cozy

Cozy is a local-first command-line application for generative media. It installs
package code, downloads models, runs package callables on your machine or a private rented
worker, and keeps your local execution records and outputs under your control.

The command is `cozy`.

## Install

Binary releases are not published yet. Building requires Go 1.26 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/cozy-creator/cozy.git
cd cozy
go build -o cozy .
scripts/install.sh --binary ./cozy     # Windows: scripts\install.ps1 -Binary .\cozy.exe
```

The installer puts `cozy` in `~/.local/bin` and, in one uv tool environment, the host tools
Cozy drives: `cozy-runtime` and TensorFS's `tfs`. A release asset installs the same way with
`--asset <cozy-*.tar.gz>`, verified against its `SHA256SUMS` before anything is replaced.
Rerun it to upgrade. The host-tool step alone is:

```sh
uv tool install --force --python 3.12 --with-executables-from tensorfs 'cozy-runtime[media,model-execution]>=0.18.14'
```

Keep `--python 3.12`: uv
[ignores dependency Python upper bounds](https://docs.astral.sh/uv/pip/compatibility/#requires-python-upper-bounds),
so an unqualified install can select an interpreter Runtime cannot use. Runtime 0.18.14 is the
controller minimum; packages keep their own declared ranges (0.18.0 minimum).

Packages use Runtime's explicit rolling window of CPython 3.12, 3.13, and 3.14.
Creator asks Runtime to select or install an interpreter satisfying `Requires-Python`
and an explicit project `.python-version`, when present. Local execution and rented
workers use the same Runtime provisioning policy. Production images preseed Python
3.12; other supported interpreters are installed on demand when the image explicitly
advertises that capability. The worker's control interpreter is independent of package
executors; one worker can execute packages using different supported minors.
Captured environments retain their exact Python patch, native wheel ABI, and dependency
closure, including Torch. Provisioning capability is reported separately from installed
interpreters; it does not substitute a different captured patch or dependency version.
`cozy package list` distinguishes supported installed Python from Python that can be
provisioned on demand. A package outside the active window
is unusable until upgraded; advancing the window is an explicit Runtime release change.

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

Declare dependencies as compatibility ranges, for example `pydantic-core>=2.46.4`
or `pydantic-core>=2.46.4,<3`. Major and minor ranges such as `==2.*`, `==2.46.*`
and `~=2.46.4` are supported. Package capture and publication reject exact versions
and patch upper bounds in project dependencies, optional dependencies, and the
project wheel's `Requires-Dist`. Script dependencies follow the same rule. Keep
exact resolved versions and artifact hashes in `uv.lock`; third-party dependency
metadata and explicit local wheel sources retain their normal behavior.

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

Search lists `MODEL`, `FAMILY`, `RELEASE`, and `LANES`, with one row per model at its
latest available release in Tensorhub's catalog order. Long columns are shortened
in the compact human table; `--full` and JSON retain complete values and lane lists
for that release.
`cozy model info org/name` shows all available releases, while `@release` selects
one exact tag. Info shows every lane's full stored byte size and immutable checkpoint
ref; JSON also includes components and their sizes when available. Stored size is
not a VRAM requirement or a measurement of bytes missing from the local cache. Unavailable
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
  --input quantize.json --publish-to org/quantized --rental-only
```

`cozy package update-all` upgrades installed Tensorhub packages to newer published
releases, without downloading model weights. It keeps local/editable packages,
development versions, and versions newer than the registry unchanged. Each package
is reported as updated, current, failed, or skipped; a failure leaves its previous
install active and does not stop other packages.

`--input=request.json` reads the whole payload, including nested lists such as `shots`,
from a JSON file. Inline arguments override file values; use `field:=<json>` for a
nested inline value. `--in` remains an alias. `--input-tree` separately binds a job
input directory.

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

With `--rental-only` or a named `--rental`, job packages resolve directly to their latest non-yanked immutable
Tensorhub releases. The transfer intent pins `(package, release)`; downloaded execution files carry
their own exact refs, including `metadata/package-interface.json`.
They do not need to be installable in the laptop's local Python environment. Local production still
uses the ordinary installed package. `--rental-only` skips local capacity and requires an
external attempt. Downloads that could rent currently refuse until Creator wires the negotiated
weights-read return plane; local-only sources under `--rental` remain local.

## Run packages and jobs

Inspect a function with `cozy run org/package/function --describe`; its request,
models and output have separate sections. A call missing required arguments
prints their names and the argument list before resolving or downloading a model.
Functions whose arguments all have defaults may be invoked without values.
Payload arguments and flags may be interleaved, including repeated `--asset` flags.

For local execution, a matching GPU ladder rung selects the preferred lane. If
the local GPU is not listed, Cozy uses the first declared lane instead. The ladder
does not prohibit local execution; actual encoding support, model compatibility
and memory capacity are still checked by Runtime. An explicit model/lane override
takes precedence. Rental placement retains its existing capacity rules.

Local scripts define ordinary `main()` or `main(ctx)`, synchronous or asynchronous.
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
Unchanged callers reuse their completed local installation after source, local dependency,
lock, Python, and capture-tool identities are checked. Each invocation still creates a new
run. One completed capture is retained per caller; a failed preparation keeps the previous
one. Script dependency locking still runs through uv on each command. Build backends remain
the versions selected by the completed installation until source, dependencies, the lock,
or capture tools change. Explicit `cozy package install ./project` also refreshes that
package's capture on its next run.
Retained source preparation can be adopted when its exact source/profile still matches.
Code and unpublished wheels go directly to the authenticated private worker, without a
Tensorhub package publication or intermediate upload. Published dependencies can still come
from their package repositories. Ordinary Python edits do not rebuild the worker image.

An **unpublished package** runs from captured source without a Tensorhub package release.
A **local script** is its single-file form; **editable** describes a dependency's installation
mode. These terms say where code comes from, not whether its model inputs must be downloaded
or whether it is allowed to perform managed child calls.

The existing worker protocol still uses `PreparePrivatePlacement` and related `Private*`
messages. Persisted `private-revision` markers, historical SQLite column names, and stable
`private_*` error codes also keep their serialized spellings so existing captures and workers
remain readable. They are compatibility identifiers, not another package mode. Private rentals
and access controls retain their security meaning.

Failed unpublished package jobs keep their resumable work and rental. `cozy run pause <id>` stops execution
while retaining that state; `cozy run resume <id>` runs the exact captured revision again.
Changed code or parameters are captured by a new run, optionally linked with `--retry`.
`cozy run cancel <id>` abandons the retained
work, releasing only resources with no other owner. Retained rentals continue billing.
An unchanged deterministic failure is not automatically retried. `--await` returns when a
transaction blocks or pauses, and the ordinary run view explains the stopped state.

An invocable dependency exposes an ordinary typed Python call. Its implementation is
captured separately; the parent keeps its ordinary helpers and resources, with generated
callable interfaces for managed exports. Calling one creates an ordinary managed child job. A reusable computation
opts in with `@invocable(memoize=True)` and registers through `app.job(function)`.
Arbitrary helper functions and external effects do not become cached operations.
An editable source library declares its App in `package.toml` (`[application] object =
"my_algorithm:app"`) and the matching `pyproject.toml` entry point
(`[project.entry-points."cozy.application"] default = "my_algorithm:app"`).
A local or PyPI wheel needs the single `cozy.application` entry point in its wheel
metadata; no `package.toml` is required inside the wheel. Declare it in the script's
ordinary Python dependencies, using `[tool.uv.sources]` for a local wheel path.
The calling single-file script needs neither App declaration.

Installed callable wheels require Runtime 0.12 or newer and a qualified uv dependency
graph version (0.12.7 or 0.12.11). Creator retains the downloaded original and creates
an executable with the exact selected dependencies pinned in its metadata; implementation
and resource bytes stay unchanged. An incompatible private worker base refuses before
execution. This uses the same `cozy run` command and private transfer path as source libraries.

On either a local or rented worker, a memoized child can acquire an earlier successful result when its
exact implementation closure, entrypoint, canonical inputs, resolved model checkpoints,
and measured numerical environment match. A fresh script can reuse this work without a retry
ID. Run IDs, the parent's script digest and call position are not computation cache keys.
Native model results travel as `ModelArtifact`
references; reuse acquires independent native custody before the new result is visible.
The reference names its original producer and receipt without copying tensor bytes.

The script starts at `main()` on every run; Python lines and call stacks are not replayed.
Completed operations return their retained results. An interrupted operation can also
adopt compatible progress that its implementation checkpointed: download prefixes,
conversion groups, or complete quantized tensor data/scale groups. Only completed,
verified units can be reused; an unfinished group runs again. Changed operation code,
inputs or encoding starts new work while independently retained preparation remains
available. Ordinary inference starts again after interruption.

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
different results never collide. This includes top-level job images, audio, and videos. Jobs keep
their internal publication custody and export independent user copies; child artifacts and native
checkpoints stay internal. Text-only results create no output directory. Input assets are read from
the original input paths without copying them. Keep those files available and unchanged until the
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

`cozy rent` is an alias for `cozy rental`. An odd `--gpus` above 1 is allowed; the worker
runs at the largest parallel degree the package supports and leaves the rest idle.

```sh
cozy rental new                   # Cozy GPUs, their GPU counts, VRAM, and hourly prices
cozy rental new h100-sxm5-80gb     # prints e.g. otter
cozy rental new h100-sxm5-80gb --gpus 2   # one machine with 2 GPUs; keep counts even
cozy rental new h100-sxm5-80gb --model paul/minimax-h3@1.0.0/bf16
cozy rental new h100-sxm5-80gb \
  --idempotency-key <unique-key>

cozy rental list                   # current rented machines, live on a terminal
cozy run org/package/generate --rental=otter prompt="moonlit lake"
cozy rental end otter
```

GPU names and prices come from Tensorhub's Cozy-owned rental catalog. Creator never exposes or
reads RunPod SKU names or provider prices. `cozy run` is local-only by default, while
`--rental=<name-or-id>` uses only the selected existing rental. It does not permit a
purchase or fallback onto another machine. `--rental-only` automatically reuses or acquires
remote capacity under the configured fleet ceiling. Creator names every private rental
with one memorable word, unique among this host's live rentals (a released word is drawn again);
Tensorhub's identity for it is its `pr-…` id, which is what the provider-side pod is named after.
The name carries no workload facts.

Use `--model org/model@release/lane` to size a manual rental for known checkpoints; repeat
`--model` for several models. Creator resolves and pins each manifest, and Tensorhub measures
their shared object closure and required disk headroom. This declares capacity needs; packages
and models are prepared when requests run. Retrying the same idempotency key reuses the pinned
set, including when `--model` is omitted on the retry.

Every rental has a fixed 15-minute idle shutdown. Active model preparation and work
assigned to that machine keep it busy. The idle clock begins at readiness and starts
again when real work settles. Paused or failed retained files, an open connection,
status polling, and unassigned work elsewhere in the fleet do not keep it alive.
Creator performs idle cleanup while connected; the pod independently enforces the
same fixed policy and asks Tensorhub to release it even if Creator is offline.

`cozy rental keepalive <name>` explicitly resets that rental's deadline once, to
15 minutes after the worker acknowledges it. The command reports the acknowledged
deadline. There is no duration option, automatic renewal, or setting to disable
shutdown. A failed or unanswered keepalive does not extend Creator's deadline.

`cozy rental list` shows each known machine's idle time and release deadline;
`cozy rental end <name>` ends one now. A release Tensorhub does not confirm is retried.
Activity is unknown for machines absent from this controller's history and for an
explicit preparation interrupted by controller restart; their pod guard still applies.
A managed rental whose job is completed and collected may be released immediately.

Rental creation sends only that private machine name, the SKU, and introduction credential material—never a package, model,
request, profile, image, or placement. Tensorhub readiness means the worker location and TLS identity
are attachable. Creator then claims that worker directly and sends signed exact package and model
release refs; no WorkerControl frame is relayed through Tensorhub. Creator learns the derived
placement/binding and exact environment identities from the worker's existing
observed-state stream before dispatch. Tensorhub's exact PackageInterface validates request/result
shape; Creator never supplies a remote profile, CUDA choice, or precomputed PlacementSet.

Each rental has one local Ed25519 Creator key and one separate media bearer. Both credentials are
removed when the rental ends, and a lost Creator key requires a new rental.

Rentals continue billing until Tensorhub confirms their termination. `cozy rental end`
and `cozy down --all` request termination; normal or forced client disconnect does not.

## Release GPU memory or stop Cozy

These commands have deliberately different scopes:

```sh
cozy unload       # stop idle local Runtime workers and release their GPU models
cozy down         # disconnect; guards work that still needs this daemon online
cozy down --force # disconnect anyway; never cancel jobs or end rentals
cozy down --all   # cancel all work, end all rentals, then stop the daemon
```

None deletes installed package or model bytes. Normal and forced disconnect preserve
retained paused/blocked/completed work and rental records. Idle rented machines still
shut down after 15 minutes; `cozy rental keepalive <name>` explicitly resets that
deadline once. Accepted detached local and remote Runtime executions continue and
reconnect on `cozy up`. Unfinished handoffs,
controls, and daemon-owned execution can still block normal shutdown. `--force` may
interrupt legacy daemon-owned local work; it uses existing recovery on restart.
`--all` remains explicit cancellation and rental teardown. See
[daemon shutdown](docs/daemon-shutdown.md).

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
variables override it. `--tensorhub=URL` overrides both for one command:

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
daemon:
  idle_shutdown_s: 900
maintenance:
  gc_cron: "0 3 * * *"
```

Without a configured `port`, Cozy prefers `127.0.0.1:8818` and falls back to an available
loopback port when another process owns 8818. A configured nonzero port is strict; `port: 0`
explicitly asks the OS to select any available port.

Without `tensorhub_url`, Cozy uses the standing local Tensorhub at `http://127.0.0.1:8819`.

```sh
cozy --tensorhub=https://tensorhub.com model search minimax
cozy package publish --tensorhub=http://127.0.0.1:8819
```

The override does not rewrite `config.yaml`. Machine login records are scoped to the selected
Tensorhub. A running daemon keeps its launch-time Tensorhub; a run selecting a different Hub
is refused before submission. Finish that daemon's work before restarting it for another Hub.
Catalog and publication commands can select another Hub without restarting the daemon.

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


Choose existing capacity explicitly with `cozy run org/package/function --rental=name`
(or its rental ID). This constraint survives retries and daemon restarts. If that
rental becomes unavailable, the run fails or retains its work; it never buys a
replacement or moves onto another machine. `--rental-only` retains automatic remote
allocation. Create capacity with `cozy rental new <machine-slug>`, such as `cozy rental new h100-sxm5-80gb --gpus 2`;
`cozy rental new` alone lists the current catalog.
