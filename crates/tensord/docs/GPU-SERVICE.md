# GPU service

`gpu_service.rs` (`GpuPool`) coordinates published, cached GPU callables through retained
Runtime executors. TensorD supplies model data and grants each executor its GPUs' varena ledgers
(`memory.rs`, design varena-slim). Varena in each executor decides that executor's bytes and
makes room through the ledger's asks; kept-page custody runs in the ledger and the RAM tier in
TensorD. TensorD places and admits calls (`fits`), reaps exited executors' rows, and ends the
ones the ledger names stalled. It holds no byte count and decides no eviction.
Acceptance, journal, progress and output custody stay in `Engine`; scheduling stays in `service.rs`.

## Configuration

`tensord serve --gpu-config <json>` loads a root-sealed `GpuConfig`:
- `devices`: the GPU envelope, `"0"` or `"0,1,..."`. A plan of degree K > 1 runs on the first K;
  a one-GPU plan on any of them. An executor is sealed to its lane (`CUDA_VISIBLE_DEVICES`).
- `models: [ModelGrant { package, slot, repository, release, lane, manifest, components }]`: the only
  authority over cached model bytes.
- `packages: [{ package, release, distribution, generation }]`: published-package mapping.
- `inventory`: each GPU's UUID, model and memory (NVML); a GPU it does not name has no ledger.
  `gpu_memory_limit` caps every ledger's capacity.
- `environment`: keys containing `TOKEN`, `SECRET` or `PASSWORD` are refused.
- `host`: optional `{mode}`, `shared` (default) or `dedicated`. `identity`: optional `{uid, gid}`.

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
  pooled deviceless executor; idle ones end with their generation. Degree 1: a call that constructs nothing, on one GPU, an
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
- Placement (`GpuPool::place`), inside the root's lane (else on unallocated, unclaimed GPUs), one
  path for every degree K over the contiguous lanes of K GPUs, in order: a free lane with its
  plan's idle executor; a free lane where it fits on every GPU (`fits` before its shape is known,
  a replica when the plan runs elsewhere too); or, when the plan runs nowhere or inside its root's
  lane, any free one. A warm member loads where a call of it would be placed. A waiting
  standalone call claims the lanes it waits for. A prefetch loads only
  inside its root's bound lane, after queued requests of that root and older ones, and ends at
  its release. A new executor's GPU is logged as `{"event": "place"}` in the memory log;
  `loads.jsonl` and `invokes.jsonl` name each line's `gpus`.
- Memory events (ledgers, placements, shortfalls, stalls) are lines of the store's
  `logs/memory.log` (rotated once past 4 MiB), machine log `memory`
  (`cozy machine logs <name> memory`; `local` names this computer's). Each call's own memory is
  the run's `memory` measurements (`cozy run show`): torch's peaks and varena's record of the
  request (`stats`, kept whole), which the merge gate reads as `measurements.memory.stats`.
- Admission names the call's family (`Execution::family`), which Invoke also carries: while a
  parent waits on a child call it yields under its family (varena `Request::yielding`), so its
  working set is left out for its family's child and counts for every stranger.
- The RAM tier's host-short event (it gave back all it may, the host is still short) ends one
  idle process, least recently used first: a deviceless executor, else an import-only parent
  with no live child, else an idle GPU executor (outside every warm set first) whose lane's call
  slots are free.
- Startup: every journaled birth still alive from a previous machine run (GPU births including
  completed requests', and any nonterminal run's) is killed, since nothing can adopt it. GPU
  dispatch stays fenced until each exit is observed. Leftover executor scopes of this machine
  (cgroups, tokens) are killed and removed. A leader still tearing down (no pidfd yet) is watched by sampling.
- Groups (degree K > 1): one executor sealed to the K GPUs with the NCCL seal (`NCCL_NVLS_ENABLE=0`,
  `NCCL_P2P_LEVEL=NVL`); `Start`/`Load` carry `sequence_parallel_degree: K`. Rank 0 spawns, dials back
  and forms its followers inside `Start` (Runtime `RankGroup`); the machine never talks to a follower.
  Every command's watch meters rank 0 plus its process-group members, so formation ends only on
  measured lack of progress. A group is admitted on all of its GPUs or none (`Ledger::admit`).
  `Start` carries each GPU's grant (its table and its row's lock description, granted to rank
  0's pid) in lane order; rank 0 hands follower r its grant, and each rank claims its own row. Followers' pids from `Start` are their GPUs' tenants: a stalled one
  is ended by pid.
  Teardown waits for every group member's exit. Startup fences GPU dispatch until a previous
  machine's leaders and their followers are gone. A refusal or poisoned group call fails the run
  with the executor's own code (a group's first fault names its GPU). A group forks from the generation's import-only parent like a
  single GPU does: the child takes the lane and NCCL seal with its environment; followers are spawned
  by rank 0.
- TensorD handles no byte count: admission is `memory::admit` alone, `Ledger::admit` on every GPU
  of the lane or none, from the peaks varena measured per `<construction>|<shape>` (the shape:
  PrepareRequest's features as `axis=value,…`, escaped; `-` with none; `""` at placement, before
  it is known: the package's widest measured shape, and with none measured it fits only on an
  idle GPU). `Ledger::admit` names the key, so the request's peak is recorded. Refused, it waits,
  with no timer, for the ledgers' events (the engine's activity) and asks again: running
  requests end, and a GPU nothing runs on always admits. A cancel ends the wait; the request
  never started. It is spawned by the pool's `ChildLauncher`, which journals the birth, then sent
  Start (with its ledgers), Load and Activate once. Each request sends PrepareRequest and Invoke.
- Warming: a warm member's executor is loaded, then `Map{construction, level}` has it load its
  models at `gpu` or `ram` (varena's `load(level)`), for any degree.
- Each GPU's ledger (`Ledger::create`, capacity min(card, `gpu_memory_limit`)) is granted to an
  executor at its spawn and reaps its rows once its exit is proven. One thread blocks on every
  ledger's events (room freed, a row died, a peak recorded), with no timeout: each wake has the
  engine's dispatch ask `fits` again (`notify_activity`), ends each row the ledger newly names
  stalled (once), and saves changed peaks (varena's records, whole) to
  `<pool>/peaks/<GPU model>.json`, restored when a ledger is created.
  A device fault (an unsettled call) drops the GPU's kept pages (`Ledger::fault`). The machine's
  RAM tier is served from TensorD on the host's pressure trigger (`varena::Tier`): TensorD
  computes no host room; it ends idle processes only on the tier's host-short event.
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
- `checkpoint_item` records a weights output's TensorFS reference, which the Runtime committed.
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

- Import-only parents live on the first GPU. No adapters.
- One Python post helper per request.
- Each executor gets its own scope: a cgroup-v2 where the host delegates one (laptop systemd
  scopes, rootful hosts), else a token (containers with a read-only or v1 hierarchy, RunPod).
- Seal: `threads` config field (torch's allocator is set in Runtime code); `<root>/home`, `<root>/kernels` (per UID) and
  `<root>/jit/<run>/<generation>`. Earlier runs' JIT scopes, and executor roots (logs) older than a day, are removed at pool start.
- `PDEATHSIG` retention after a UID drop is unverified.
