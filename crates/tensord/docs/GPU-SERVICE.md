# GPU service

`gpu_service.rs` (`GpuPool`) coordinates published, cached GPU callables through retained
Runtime executors. TensorD supplies model data, shared CPU buffers and each GPU's varena ledger
(`memory::GpuLedger`, design memory-varena §5.6). Varena in each executor decides that
executor's GPU bytes and makes room through the ledger's asks. TensorD admits starts, reaps
exited executors' rows, and ends the ones the ledger names stalled.
Acceptance, journal, progress and output custody stay in `Engine`; scheduling stays in `service.rs`.

## Configuration

`tensord serve --gpu-config <json>` loads a root-sealed `GpuConfig`:
- `devices`: the GPU envelope, `"0"` or `"0,1,..."`. A plan of degree K > 1 runs on the first K;
  a one-GPU plan on any of them. An executor is sealed to its lane (`CUDA_VISIBLE_DEVICES`).
- `models: [ModelGrant { package, slot, repository, release, lane, manifest, components }]`: the only
  authority over cached model bytes.
- `packages: [{ package, release, distribution, generation }]`: published-package mapping.
- `authorized_device_limit_bytes`: a GPU's ledger capacity where neither NVML nor the inventory
  says it.
- `memory`: `{ floor_bytes }`: raise the free floor TensorD keeps on each GPU.
- `environment`: keys containing `TOKEN`, `SECRET` or `PASSWORD` are refused.
- `host`: optional `{fill_threads, ttl_seconds}` for the always-on `HostTier`. `identity`: optional `{uid, gid}`.

## Preparation

`prepare_root` reads only the installed static interface.
- Every model the entrypoint declares at `<entrypoint>.models.<param>` gets a slot (H3 turbo:
  base and LoRA). Several slots load as one many-model construction (`Load.models`).
- Choices with `source`, `profiles` or `adapters` are unsupported.
- Exactly one `ModelGrant` must match each slot. Declared `component_use` must be within its components.
- Degree: the widest group every slot declares (`sequence_parallel.degrees`, one always)
  within the machine envelope and `RunSpec.gpus` when positive: a cap the run's children inherit,
  never a refusal. Model lanes/checkpoint overrides never change it; the lane rule reads the
  selected group's GPUs. Busy GPUs cause queuing, not a smaller group.
- Degree 0 (deviceless): `accelerator=False`, or a machine without GPUs. Such a call runs on a
  pooled deviceless executor within CPU room; idle ones are ended by host pressure, least recently
  used first, before any GPU executor. Degree 1: a call that constructs nothing, on one GPU, an
  ordinary tenant of its GPU's ledger.
- Allocations (`gpu_allocation.rs`), as Slurm's: a deviceless root (a request with no parent)
  holds a contiguous lane, `--gpus` wide or as wide as the widest group its environment's models
  run on here. It is registered when the root dispatches and bound when its first GPU step is
  ready; its steps (child calls, any depth) run only inside it, one call per GPU. It ends at
  `Context.release_gpus()` (in-flight calls keep their GPUs until they return) or when the root's
  call ends. A GPU call is its own root for the call's duration, and its family (its calls, any
  depth) shares its GPUs. Children carry `root` and `family`, stamped at acceptance.
- Order: ready runs dispatch by (root acceptance, id). A root takes its place only with a GPU
  step ready: one still downloading weights or in CPU start-up claims nothing, so younger ready
  roots use the GPUs meanwhile. A root's first waiting step claims, for the pass, a lane no
  older root holds; its younger holders yield at their next step boundary: they start no new
  step (calls in flight and their own children finish; nothing is preempted),
  then their lanes end (a `release` naming the root it went to) and they queue again at their
  own place. A root that released or yielded comes back at its place among the waiting roots
  and never displaces a holder: holders keep their lanes between their steps. A waiting step
  keeps free GPUs from younger requests. Each `grant`/`wait`/`release` is a `gpu` entry in the
  root's log (`RunEvent.gpu`); a wait is logged again only when its blocker (the oldest other
  request claiming, holding or calling on its lane) changes.
- A `Plan`'s construction hashes {generation, App, entrypoint, constructed slots, degree}: its
  executors' key. Its id adds the `Weights[M]` sources (`Invoke.sources`). A plan binding a model
  is journaled as a `Preparation`, and `submit` records `preparation_id`; one binding none is
  derived at dispatch.
- An empty `installation_id` binds a `published-<hash>` installation from `packages`.

## Dispatch

- One call slot per GPU: a call takes every slot of its lane. GPU dispatch is fenced only by a
  startup fence or an active GPU record this machine run does not supervise. CPU dispatch continues.
- Placement (`GpuPool::place`), inside the root's lane (else on unallocated, unclaimed GPUs): a
  group takes a free contiguous lane of K GPUs. A one-GPU call takes, in order, a free GPU with
  its plan's idle executor; a free GPU where what its executors hold elsewhere (else a new
  context) fits in that GPU's ledger once others' cold and idle bytes are asked for (a replica
  when the plan runs elsewhere too); or, when the plan runs nowhere or inside its root's lane,
  any free one. A waiting standalone call claims the GPUs it waits for. A prefetch loads only
  inside its root's bound lane, after queued requests of that root and older ones, and ends at
  its release. A new executor's GPU is logged as `{"event": "place"}` in the memory log;
  `loads.jsonl` and `invokes.jsonl` name each line's `gpus`.
- Memory events (admissions, asks, reaps, returns, stalls, placements) are lines of the store's
  `logs/memory.log` (rotated once past 4 MiB), machine log `memory`
  (`cozy machine logs <name> memory`; `local` names this computer's). Each call's own memory is
  the run's `memory` measurements (`cozy run show`): torch's peaks and the executor's `stats`
  (bytes paged in per model per step, evictions with rung and reason, activation peaks per
  shape, leftover), which the merge gate reads as `measurements.memory.stats`.
- Ending idle executors for room runs under that GPU's call slot; host pressure ends idle
  executors only on GPUs no call holds.
- Startup: every journaled birth still alive from a previous machine run (GPU births including
  completed requests', and any nonterminal run's) is killed, since nothing can adopt it. GPU
  dispatch stays fenced until each exit is observed. Leftover executor scopes of this machine
  (cgroups, tokens) are killed and removed. A leader still tearing down (no pidfd yet) is watched by sampling.
- Groups (degree K > 1): one executor sealed to the K GPUs with the NCCL seal (`NCCL_NVLS_ENABLE=0`,
  `NCCL_P2P_LEVEL=NVL`); `Start`/`Load` carry `sequence_parallel_degree: K`. Rank 0 spawns, dials back
  and forms its followers inside `Start` (Runtime `RankGroup`); the machine never talks to a follower.
  Every command's watch meters rank 0 plus its process-group members, so formation ends only on
  measured lack of progress. Each GPU of the group is admitted by its own ledger (device order).
  `Start` carries each GPU's ledger memfd and eventfd in lane order; rank 0 hands follower r
  its pair, and each rank claims its own row. Followers' pids from `Start` are their GPUs' tenants: a stalled one
  is ended by pid.
  Teardown waits for every group member's exit. Startup fences GPU dispatch until a previous
  machine's leaders and their followers are gone. A refusal or poisoned group call fails the run
  with the executor's own code (a group's first fault names its GPU). A group forks from the generation's import-only parent like a
  single GPU does: the child takes the lane and NCCL seal with its environment; followers are spawned
  by rank 0.
- TensorD handles no byte count: it asks each GPU's ledger `fits(package, shape)` (design
  memory-varena §5.6), from the peaks varena measured per `<construction>|<shape>` (the shape:
  PrepareRequest's features, hashed) beside every row's unevictable bytes and every running
  request's remaining peak; never beside a dead or reaping row, and a shape never measured only
  alone. A replica goes to a GPU where the plan's widest measured shape fits with a new process.
  Before a cold session spawns, and again once its request's shape is known, while it does not
  fit and nothing runs on a GPU of its lane, exited executors' bytes are waited for while they
  come back, then idle executors end, least recently used first, outside every warm set first.
  A request is never refused: varena in its executor makes what room it can. The request is
  named to varena under its key (the ledger's dispatch table) until it ends, so its peak is
  recorded. It is spawned by the pool's `ChildLauncher`, which journals the birth, then sent
  Start (with its ledgers), Load and Activate once. Each request sends PrepareRequest and Invoke.
- Warming: a warm member's executor is loaded, then `Map{construction, level}` has it load its
  models at `gpu` or `ram`; varena fills them, and its priority is in use > warm > idle.
- Each GPU's ledger: its capacity is NVML's total; TensorD holds a row of its own whose context
  is the display floor (shared mode: 512 MiB or 1/16 of the card) plus every byte NVML shows that
  no row explains (other programs, executors still exiting). A watch thread reads NVML once a
  second, and wakes on the ledger's eventfd: a dead row whose process is gone is reaped, and
  returned once NVML charges the pid no more (in a container, at a reading after the exit; what it
  still shows stays in TensorD's row). Once a wedge floor it ends the rows asked for bytes whose
  progress did not move. When bytes free (an executor's `freed`, a returned row) the engine's
  dispatch is woken (`notify_activity`): the queue asks `fits` again and never polls. A restarted
  TensorD makes fresh ledgers: its old executors die with it (`PDEATHSIG`), and what they hold
  until then is unexplained bytes.
- An executor's failed terminal (its own refusal before entry, or a failed invoke) first keeps a
  triage bundle (`triage.rs`: terminal, traceback, executor pid and stderr tail; canonical JSON
  under `execution/triage/`, fsynced) and binds it in the journal; the outcome body names it as
  `triage_bundle`, and `ReadMachineExecutionTriage` returns its bytes.
- Start happens only after `authorize_managed`.
- Invoke must be quiescent. Then `postprocess` and `managed_result` run. A custody failure is
  `failed`.
- On error the session is terminated (exit observed, wedge killed) before the run is settled:
  CANCELED if a cancel was journaled, otherwise FAILED with the reason, including a pre-start exit
  with its stderr tail; nothing is requeued to fail again. A retained session can lose its channel (EPIPE,
  ECONNRESET, EOF) at PrepareRequest, which enters no handler: a killed executor closes its
  socket before it is a zombie, so the pre-reuse check can miss it. That attempt stays with its
  dispatch (journal `redeliver`: back to awaiting its first executor) and starts once more on a
  fresh executor, whose own failure is FAILED. An unprovable exit leaves the run charged and
  nonterminal.

Executor requests:
- `read_grant {manifest, layout}`: a weight set's bytes. The sealed host layout an executor
  deposited for that manifest and layout (one fd, `held`); else every chunk file of its selected
  components, sorted by digest, with `objects_sha256` (`descriptors` fds). TensorD never fills.
- `deposit {manifest, layout}` + the memfd: a layout varena sealed in the executor, kept as an
  opaque handle (`deposits.rs`) for the next executor of the same weights (D9's refill path);
  under host pressure the least recently granted goes first.
- `model_source` goes to `ModelSources` for selected headers/assets in sealed descriptors.
- `publish` (`Outputs.publish`) goes to `products.rs`: the spool file is retained as a native
  one-file tree bound to the run's actor, then journaled as a `product` (list outputs APPEND,
  single outputs SET; re-publishing identical SET bytes is a no-op). Answers `Published`.
- Progress counts only `advance > 0`.

## Launch identity

With `identity`:
- A non-root owner may only name its own UID and GID.
- The trampoline gets `--uid/--gid`. Peer credentials must match.
- The pool root and its parent become 0710, keeping their owner. Executor dirs and per-request
  spools become identity-owned 0700.
- The interface file and cancel marker become 0440. Journal, results and generations are never
  changed.
- The socket path is `<root>/executor` and at most 107 bytes. Without an identity it is
  `/proc/<pid>/fd/<dirfd>/executor`.

`ChildLauncher` spawns from one pool-owned thread, because Linux ties `PDEATHSIG` to the creating
thread.

## Known gaps

- Groups always take the first K envelope GPUs. Warm members and import-only parents live on the
  first GPU; executors never share GPU weights (the host cache is shared). No adapters.
- One Python post helper per request.
- Each executor gets its own scope: a cgroup-v2 where the host delegates one (laptop systemd
  scopes, rootful hosts), else a token (containers with a read-only or v1 hierarchy, RunPod).
- Seal: `threads` config field (torch's allocator is set in Runtime code); `<root>/home`, `<root>/kernels` (per UID) and
  `<root>/jit/<run>/<generation>`. Earlier runs' JIT scopes, and executor roots (logs) older than a day, are removed at pool start.
- `PDEATHSIG` retention after a UID drop is unverified.
