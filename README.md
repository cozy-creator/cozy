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
local-only editable install after a bounded source scan. Every invocation checks that live tree;
a change atomically prepares a new install and restarts stale execution state. A failed refresh
keeps the last good install pinned and refuses the invocation. Editable installs are neither
published releases nor rentable deployments. Model bindings resolve the exact release and lane
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

Model releases live in the local TensorFS store:

```sh
cozy model search flux
cozy model download org/model@release local/flux --lane task=text-to-image
cozy model list
cozy model remove local/flux
```

Upload a pinned source directly, or execute one package-reviewed producer job whose named outputs
become owner-only immutable checkpoints. `--lane` on upload selects only an existing input release;
producers do not choose public lane names. A local alias can be uploaded without copying its
already-canonical objects. The destination owner must match the account shown by `cozy auth`:

```sh
cozy model upload local/model org/model

cozy model upload hf://MiniMaxAI/MiniMax-H3@<full-commit> \
  tensorhub/minimax-h3 \
  --producer tensorhub/minimax-h3-tools@v2/four-lane \
  --rental-only --await
```

`--dry-run` resolves the immutable source, producer release, resource floors, and named outputs
without moving model bodies or authorizing rental spend. Submission records an ordinary `job-*` run
and returns immediately; `--await` watches it, and `cozy run watch <job-id>` attaches later. Repeating
the exact normalized transfer safely returns the same durable request. A checkpoint retained before a
later destination failure remains visible in `model_outputs`, while the request reports the failure.
Upload never makes checkpoints public.

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

With `--rental`, producer and job packages resolve directly to their latest non-yanked immutable
Tensorhub releases; their exact release and descriptor identities are pinned in the transfer intent.
They do not need to be installable in the laptop's local Python environment. Local production still
uses the ordinary installed package. `--rental-only` skips local capacity and requires an
external attempt. Downloads that could rent currently refuse until Creator wires the negotiated
weights-read return plane; local-only sources under `--rental` remain local.

## Run packages and jobs

Serving entrypoints and bounded jobs use the same command. Cozy reads the installed package descriptor
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
shows named pipeline stages, measured step speed, elapsed time, and an estimate while step telemetry
is available. `--stream` is the corresponding typed event stream and requires `--await`.
`cozy run watch <run-id>` attaches to that same progress stream later; interrupting a watcher
detaches without canceling the durable run.

If the package is missing, Cozy installs the newest compatible release from Tensorhub before
starting; the download is visible progress, not an interactive prompt. Reusing an explicit
`--idempotency-key` safely returns the same recorded work. Before submission, Cozy materializes an
omitted integer `seed` and hashes the finalized input payload. Files saved under `--out` use that full
lowercase hash as their filename, so unseeded runs can share one directory without replacing one
another. The same explicit payload intentionally resolves to the same filename; multi-output results
append each output field name. The daemon records the destination with the request and publishes the
verified media after terminal, so closing the command or restarting Cozy does not abandon `--out`.

Local Runtime workers start on demand. Successful serving workers may remain resident for warm
reuse; job workers are reclaimed at terminal.

## Private rentals

With Tensorhub configured, rent generic private capacity:

```sh
cozy rental new                    # Cozy GPUs, VRAM, and retail hourly prices
cozy rental new h200                # prints e.g. bright-otter-4e81b938cf114e25
cozy rental new h200 \
  --idempotency-key <unique-key>

cozy rental                        # current rented machines
cozy run org/package/generate --rental prompt="moonlit lake"
cozy rental end bright-otter-4e81b938cf114e25
```

GPU names and prices come from Tensorhub's Cozy-owned rental catalog. Creator never exposes or
reads RunPod SKU names or provider prices. `cozy run` is local-only by default, while
`--rental` permits Creator to reuse or acquire remote capacity only when ready local capacity cannot
run the request. `--rental-only` deliberately bypasses local capacity and requires an external
rental. Both modes remain under the configured fleet ceiling. Creator gives every private rental
a safe semantic name and RunPod shows that exact same name; the name carries no workload facts.

Every rental the daemon owns — bought for a request or started with `cozy rental new` — ends on
its own once nothing has been queued, running, or owed on it for `rentals.idle_release_s`
(default 300). Running work on it is what keeps it: the clock restarts at each settled attempt,
and a rental that never ran anything counts from the moment Tensorhub first reported it `ready`,
so a pod still booting is never ended. A managed rental whose job is done goes at once.
`cozy rental` shows each machine's idle time and when it will be released; `cozy rental end`
ends one now. A release Tensorhub does not confirm is retried until it does. Set
`idle_release_s: 0` to leave every rental to `cozy rental end`.

Rental creation sends only that private machine name, the SKU, and introduction credential material—never a package, model,
request, profile, image, or placement. Tensorhub readiness means the worker location and TLS identity
are attachable. Creator then claims that worker directly and sends signed exact package and model
release refs; no WorkerControl frame is relayed through Tensorhub. Creator learns the derived
placement/binding and exact invocation identities from the worker's existing
observed-state stream before dispatch. Tensorhub's exact release descriptor validates request/result
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

The default local root is `~/.cozy`. Configuration is read once from
`~/.cozy/config.yaml`, then credential/location environment variables override it:

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

cozy package list --fields package,version,disk
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
