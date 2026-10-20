# Architecture

TensorD (executable `tensord`) is the persistent Rust daemon that owns the API, execution journal,
node scheduling, memory budgets and executor supervision. Package code runs in Python
executors (cozy-runtime) inside each package environment. TensorFS is a library embedded in
TensorD and used inside Runtime; it is not a separate daemon. TensorD has no CUDA context:
executors own device contexts and operations. NVML is read on each configured GPU's sampler
thread (`memory`).

Model loading spans storage, shared CPU buffers and executor GPU memory:

| Representation | TensorD's responsibility | Runtime's responsibility |
| --- | --- | --- |
| Stored weight bytes and metadata | Use TensorFS to acquire/retain manifests, headers, assets and tensor objects; bind available data to the selected model components | Request the selected header/assets, inspect tensor layouts, and construct Python/PyTorch models |
| Shared CPU weight buffers | Fill and retain sealed layouts or streaming windows through TensorFS; coordinate host budgets and release unused buffers | Adopt granted layouts and descriptors through its TensorFS weight plane; read ready regions and manage its pinned staging within the host budget |
| GPU-resident tensor storage | Coordinate device budgets and eviction; retain eligible executor-exported allocation handles and their leases | Allocate, transfer, map and unmap through TensorFS's plane/device APIs in the executor's CUDA context; attach those tensors to the model |

Runtime reports each rank's needs and measured use. TensorD decides budgets and eviction
across workloads; Runtime and its TensorFS weight plane perform the concrete device operations
within those grants. A model source carries access to selected weight bytes, metadata and assets, not
just a repository name. A host-buffer fill does not construct a PyTorch model, and retaining a
GPU handle does not create a CUDA context in TensorD.

TensorFS `Store::ensure_owned` retains the store claim through live handles and read leases,
coordinates GC pins and excludes outside collectors. Cooperating ordinary processes can
still share its store for reads and writes. Its
[TensorFS storage guide](../../tensorfs/README.md#storage-and-ownership)
describes storage and retention. TensorD retains the node-specific sealed memfd cache, host
tiers, GPU-region custody and peer/process leases. [Recovery](RECOVERY.md) distinguishes
executor replacement, daemon restart and software activation guarantees.

Plan of record and workstream IDs (A–F):
`~/cozy_v2/outputs/cozy-machine-takeover-20261002/PLAN.md`.

## Process picture

```
Calcifer ──HTTPS (Cozy-Cap)──> api::http ─> machine_api::NativeBackend
                                                        │
                                                        ▼
                                          service::Service (dispatch policy)
                                           │ journal + supervision: execution::Engine / journal::Journal
                                           │
                          gpu_service::Pool ─> device_executor::DeviceExecutor
                            (retained executors per plan: deviceless, one GPU or a group,
                             via the Runtime trampoline)
                            memory::GpuMemory: admission, process caps, eviction, floor
                            answers budget-cell, device-room, model-source and sealed-tier requests
                          calls::Calls: child calls and each attempt's weights grant
```

A request is accepted durably (`Engine::submit_public`), then `Service::dispatch_ready` resolves
its immutable generation (`catalog`) and its plan, and starts it on the pool. Every run takes one
path (Start, Load, Activate, PrepareRequest, Invoke): a deviceless call (`accelerator=False`, or a
machine without GPUs) on a pooled deviceless executor within CPU room, a GPU call on the GPUs the
pool places it on. Progress, outputs and terminal state are written
by `Engine`; outputs are kept in native custody until the client acknowledges collection.

Tenancy: a pod has one owner, a package one publisher, and many users submit. A GPU plan (and so
its executor and its memory-policy tenant) is the package's generation, entrypoint, bound models
and group, never the submitter: every submitter of a construction is served by one executor, one
request at a time. Degree 2 holdings are per GPU and layout, shared by every executor on the pod.
What belongs to a submitter is in the journal and the API: installations, preparations, runs,
inputs, outputs, triage and controls are read by actor; a run's spool lives only for its call,
and a failure quotes only the stderr its own request wrote.

The HTTPS machine API is the only way in; there is no local control socket. Same-UID package
code is still not sandboxed.

## Module map

| Module | Owns | Main types | Owner |
|---|---|---|---|
| `main.rs`, `serve.rs` | `serve`/`version` commands, the state root's owner lock and store claim, wiring | — | D2 (rental config) |
| `api/server.rs` | The pinned-TLS listener (h2 and http/1.1) and the readiness self-probe | `MachineIdentity`, `serve` | D1 |
| `api/http.rs` | The machine API's Router (`MACHINE-API.md`): auth, refusals as the error envelope, SSE, ranged outputs, object parts | `router`, `Refusal` | H |
| `api/auth.rs` | Admitted keys (a rental's Hub lease) and each open stream's authority | `Authority`, `Keys`, `StreamAuthority` | D1 |
| `api/identity.rs` | Persistent P-256 TLS identity, typed machine config, readiness secret | `MachineConfig`, `AuthorizedKeys`, `ReadinessSecret` | D2 |
| `api/domain.rs` | Current typed native execution/backend values; no RPC or version identity | Native queries, state, model and custody values | APIM |
| `archive.rs` | Private stored model/product/intake/terminal codecs, preserving existing byte tags | Stored records and codec functions | APIM |
| `api/backend.rs` | The one backend trait the server calls | `MachineBackend`, `InputTreeReceiver`, `Observation` | D1 |
| `api/workspaces.rs` | Resumable package uploads, scoped by owner key | `WorkspaceUploads`, `UploadSession`, `UploadedPackage` | D1 |
| `api/machine_runs.rs` | Runs: submit (idempotent per signer), the engine's pages as a run's events | `submit`, `follow` | G |
| `api/machine_status.rs` | Status: identity and receipt to anyone, the whole machine to a machine cap as it changes | `frame`, `watch` | D2 |
| `api/machine_update.rs` | Runs of `kind: update`: a software cohort from versions or held wheel objects; its log is the update's state history | `submit`, `events`, `owns` | D2 |
| `api/capability.rs` | `Cozy-Cap` grants (the Go agent's token) for run outputs and maintenance | `Grant`, `verify`, `mint` | G |
| `api/install.rs` | Materialize uploaded packages, run the uv installer helper | `InstallerConfig`, `PreparedGeneration` | D1 |
| `machine_api.rs` | `MachineBackend` implementation: submit, events, collect, list, inventory | `NativeBackend` | D1 |
| `products.rs` | Run output log: `Outputs.publish` custody and `product` events (SET/APPEND, composite parts) | `publish`, `retain`, `document` | G |
| `triage.rs` | One bounded triage bundle per failed attempt, named by its outcome | `TriageRef`, `Facts` | G |
| `hub.rs` | Hub access (th-245): the package catalog read anonymously; models through TensorFS's distribution `Endpoint` under the owner bearer a request carries, else anonymously | `Source`, `Catalog`, `Refusal` | D1 |
| `published.rs` | Package and model preparation: releases from the Hub, each model's checkpoint chosen (choice or declared default, name+release+variant → tree/header) and landed as its structure only, provider-source models via TensorFS `source_model`, held per release and resolution | `Publisher`, `Request`, `Prepared` | D1 |
| `precision.rs` | The lane rule (xs-042): a slot's bits and lane order from the GPUs' compute capability, the request's bits and the slot's `min_bits`, `phase` and `encoded_leaves`; its fallback warnings | `Row`, `Gpus`, `Slot` | D1 |
| `access.rs` | What a call's weights are fetched under: origins and their Hub access (the request's owner bearer, for the owner's checkpoint by hash), readers' waits (Now) and a warm member's `downloaded` ticket, all ended with the call or its run's cancel | `Fetch` | D1 |
| `adapter.rs` | Caller LoRA adapters as a zero-copy TensorFS component-writer composition | `compose` | D1 |
| `runs.rs` | Run sources and preparation inside a run: accepted at once, install/download as its progress, the owner bearer in memory only. A warm run with no entrypoint installs and fetches its choices; one with no code makes provider sources and uploads to its weights destination | `Runs`, `Spec`, `Source` | D1 |
| `objects.rs` | Uploaded objects: byte ranges in any order, resumable, verified once whole into the store, per signer | `Objects`, `Upload` | D1 |
| `local_source.rs` | A run's local source: each remote manifest owns its source tree and editable environment; explicit local paths read the directory in place | `LocalSources`, `Manifest` | D1 |
| `machine/` | Launch grant, lifetime identity, readiness receipt, rental lifecycle, supervision, SSH, runtime update, the direct player endpoint (`player.rs`), the embedded Python client and installer helper (`client.rs`) | `Grant`, `Readiness`, `Lifecycle` | D2 |
| `native_inputs.rs` | Native input custody into TensorFS + journal | `SourceIntake`, `IntakeJournal` | D1 |
| `catalog.rs` | Immutable package environment generations and their holds | `Catalog`, `Generation`, `HeldGeneration` | D1 |
| `service.rs` | Sole dispatch policy (CPU parallelism, one GPU slot, startup GPU fences) | `Service` | B2 |
| `calls.rs` | Child calls (the seam broker): each call a run, canceled with its parent; each attempt's weights grant | `Calls`, `Caller` | G |
| `execution.rs` | Acceptance, attempt supervision, cancellation, progress coalescing, output custody, reconcile | `Engine` | E |
| `journal.rs` | SQLite journal: executions, installations, preparations, receipts, process births | `Journal`, `Execution`, `State`, `ProcessBirth` | E |
| `gpu_service.rs` | Executor pool (degree 0, 1 or n): published package/model mapping, executor retention, request callbacks; GPU groups (a degree-K plan on the first K GPUs, one memory decision per GPU), one call slot per GPU and one-GPU replicas and multi-slot plans | `Pool`, `GpuConfig`, `Plan`, `PlanSlot`, `PlanSource`, `ModelGrant` | B2 (admission, grants), E (spawn/fencing), D1 (published mapping), I (groups, slots) |
| `memory/` | Per-GPU ledger and decisions (`policy`), NVML sampler thread (`nvml`), floor watchdog, per-host pinned budgets (`host`) | `GpuMemory`, `policy::Gpu`, `Step`, `Decision` | B2 |
| `device_executor.rs` | Typed control seam to the Runtime device executor, per-request output encoding | `DeviceExecutor`, `ExecutorConfig`, `DeviceCommand`, `Frame`, `Answer` | E |
| `launch_identity.rs` | Runtime trampoline command, the executor environment seal, optional UID/GID | `LaunchIdentity`, `Seal`, `trampoline` | E |
| `process.rs` | Exact process births, kill (scope and group), the progress meter and watch (leader plus process-group members), reaping (`reap_group`) | `Exact`, `Pace`, `Watching`, `Reaped` | E, I |
| `scope.rs` | Each executor's own scope (cgroup-v2, else an inherited token): kill, count, end; sweep of unclaimed and earlier runs' scopes | `Scope` | E |
| `reclaim.rs` | Self-managing caches: TTL and storage pressure for spools, logs, collected results, generations | `Disk`, `Swept`, `sweep` | E |
| `child_launcher.rs` | Pool-owned spawn thread (PDEATHSIG follows the creating thread) | `ChildLauncher` | E |
| `os.rs` | memfd, seals, peer credentials (`SO_PEERPIDFD`, `pidfd_open` fallback), pidfd exit | — | E |
| `protocol.rs` | Length-prefixed frames of the executor seam | `read_body`, `write_body` | E |
| `host_tier.rs` | Degree 1 host tier: machine-filled sealed layouts, adopted read-only, sized by live headroom; a region's absent objects are fetched (Now) and copied as they land | `HostTier`, `HostGrant`, `TierLimit`, `HostTierFacts` | B1 |
| `host_memory.rs` | Live host headroom (cgroup v1/v2 path, `MemAvailable`) | `HostMemory` | B1 |
| `model_sources.rs` | Selected model headers/assets as verified bytes and sealed read-only descriptors; component byte facts for planning | `ModelSources`, `SelectedManifest` | B1 |
| `resident_custody.rs` | Degree 2: executor-exported GPU regions kept as driver fds (no CUDA), leases, revocation | `ResidentCustody`, `HoldingKey`, `SharedRegion` | C |
| `boundary_json.rs` | Strict JSON parse (no duplicate keys) for boundary records | — | D1 |


## Python package `tensord_client`

| Module | Role |
|---|---|
| `packages.py`, `captured_packages.py`, `installer.py`, `package_records.py`, `runtime_describe.py` | Static description and uv generation install (`python -m tensord_client.installer`) |
| `device_codec.py` | Output encoder, embedded into `device_executor.rs` with `include_str!` |

## Development tools (not proof; see PLAN decision 5)

| Path | Use | Owner |
|---|---|---|
| `src/bin/degree2-pilot.rs` | Degree 2 on a rental: stock executors share GPU weights through `ResidentCustody` (replacement, count-once, revoke, output equality) | C |
| `src/bin/model-source-check.rs`, `scripts/check-model-source-descriptor.py` | Check a selected snapshot's sources | B1 |
| `src/bin/install-capture.rs` | Run the installer path on a captured archive (used by Python tests) | D1 |
| `scripts/gate/` | Matched ComfyUI vs TensorD gate through ordinary `cozy run` | F |
| `scripts/benchmarks/permission_probe.py` | CPU executor permission probe | E |

## Contracts

Runtime reads and writes weights directly through public TensorFS in the shared store
sealed as `TENSORFS_HOME`. TensorD passes immutable source references, declared output byte
bounds, actor/operation identity and destination. Optional `tensorfs` connection settings
carry an endpoint, an already supplied token and optional CA data; TensorD does not acquire
credentials. Supplied tokens remain in the request's memory context and diagnostics redact them.

Runtime owns keyed Writer replay, local checkpoint commit and ordinary upload. TensorD has
no weights writer/source/fetch channel server or derived-output transaction journal. It records
the checkpoint reference Runtime reports; failed uploads leave Runtime's local commit available.

- [Durable execution](DURABLE-EXECUTION.md): journal states, acceptance, cancellation, custody.
- [Recovery](RECOVERY.md): executor and daemon lifetimes, update publication and rollback.
- [Front door](FRONT-DOOR.md): the listener, TLS identity, install.
- [Package bridge](PACKAGE-BRIDGE.md): generation description and install.
- [Device executor](DEVICE-EXECUTOR.md): executor launch and control seam.
- [GPU service](GPU-SERVICE.md): GPU pool, config, launch identity.
- [Host tier](HOST-TIER.md), [model sources](MODEL-SOURCES.md): Degree 1.
- [Resident custody](RESIDENT-CUSTODY.md): Degree 2 foundation.
- [Matched gate](../scripts/gate/README.md): ComfyUI vs TensorD, ordinary CLI.

## Build and test

```sh
export CARGO_TARGET_DIR=~/cozy/.cargo-target/tensord
L=~/cozy_v2/outputs/cozy-machine-takeover-20261002/locks
flock $L/rust-build.lock nice -n 19 cargo clippy --all-targets -j 2 -- -D warnings
flock $L/rust-build.lock nice -n 19 cargo test -j 2 -- --test-threads=2
nice -n 19 uv run --locked --extra test pytest -q
```

The test extra installs cozy-runtime; tests fail without it. Ignored Rust tests need installed SDK
generations (paths or `TENSORD_CPU_TEST_PYTHON`); run them with `--ignored` on purpose.
