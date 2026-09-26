# Native pull cache convergence audit

Baseline: TensorFS 4146fe9, including identical-pull repair #197. The fixed native wheel and all three local worker defaults are already qualified and activated independently of this audit.

## Questions and falsifiable checks

1. Independent checkpoints or lanes in the same repository should merge under SQLite coordination; an identical pull should converge; an incompatible update of the same lane must have explicit semantics. Authored repository CAS must continue refusing stale writers.
2. A warm or cold pull must keep its required objects resident through its result, with no gap between verification and repository retention. Exercise real GC from the plan observer to demonstrate whether existing native guards protect this transition.
3. The only retryable catalog failure is demonstrated contention. SQLite busy/locked must be distinguished from corruption, readonly filesystem, disk failure, and ordinary permission failures.
4. Validate each distinct checkpoint closure once per pull, retain digest/length/header/declared-closure checks, and avoid revalidating the same manifest for both checkpoint addition and lane assignment.
5. Preserve local cache ownership receipts, authored-root protection, filesystem-first durability, SQLite rebuildability, and Runtime checkpoint/read leases. Separate NFS root-squash platform qualification (#195) from pull correctness.

## Initial source findings

- `Store::apply_cached_repository` uses the generic mutation list, stale observed repository body, generic expected release revisions, and an exact-intent predicate. Independent compatible updates still reach a whole-body `REPOSITORY_CONFLICT`.
- `apply_repository_locked` validates the same manifest independently for PutCheckpoint and UpdateRelease; pull already completed a FetchPlan proof before this call.
- `while_store_busy` retries every `IO_FAILED`; the catalog maps every rusqlite error to that code.
- `transport::pull` owns no lifetime guard for the entire presence/transfer/recording transition. Its pinned warm fast path returns after `on_plan` without reacquiring or revalidating residency.

Native reproductions and consumer/retention evidence are in progress. No source behavior change has been made.

## Reviewed implementation boundary

PR198 fixes two reproduced defects: compatible concurrent lane/checkpoint additions merge under SQLite coordination, and a shared native writer guard keeps each pull GC-safe through completion. Authored whole-body CAS remains strict. Same-lane conflicting updates, yanks, deleted repositories, local aliases, corrupt manifests, and cache ownership rules remain protected. Checkpoint validation is consolidated into one helper, without removing checks.

The two desired-behavior probes fail against 4146fe9 and pass with this change. The warm-GC probe uses real native collect_cached at the plan observer; GC returns STORE_BUSY during the guarded pull and successfully reclaims after it returns. Existing cache concurrency, authored CAS, provenance/eviction and cold/warm/pinned transport tests pass on one CPU. Logs are retained at `/home/fidika/cozy_v2/outputs/tfs-cache-pull-audit-20260926/`.

## Findings deliberately not hidden by this patch

- **Runtime scope versus retained-root completeness:** real native reproduction confirms ordinary snapshot siblings omitted by a runtime-only Hub closure cause repository recording to refuse. Skipping that check causes the next GC to fail. Tracker #231 carries the design and required negative proofs; PR198 retains a safety regression requiring refusal before root retention.
- **Global census under the repository write lock:** `Census::open_metadata` reads every repository and manifest before projecting one repository. A pure single-repository projection is a follow-up opportunity; full GC census remains necessary.
- **Broad local retry classification:** `catalog::sql` maps every SQLite error to IO_FAILED, and `while_store_busy` retries that entire code. A targeted busy/locked classification would separate contention from filesystem/corruption failures. No error-code contract was changed here.
- **Pinned-local error fallback:** local manifest/repository failures are converted to an absent local fast path, followed by a Hub request. Preserve fallback for missing cache data; review whether malformed metadata or permission failures should retain their typed diagnosis.
- **Optional cache write-through completeness:** only objects fetched from the origin are offered. The existing native `only_origin_fetched_objects_are_backfilled` test explicitly proves that after an unavailable mount recovers, already-held objects remain absent from the external cache even when the retry and queue drain succeed. Queue-drain success is not a whole-model cache residency certificate. This is separate from NFS #195's scoped filesystem-identity proposal, which remains unqualified by this audit.

## Consumer boundaries

Host `pod-supervisor/fetch.go` at ed70bf2a calls the native CLI `fetch`/`fetch bounded` for model materialization. Python `tensorfs.pull` is another direct binding to transport::pull. Runtime `machine_model_defaults.py` materializes through it, then verifies and acquires explicit tensor leases. Runtime `store_gc.py` uses native cache eviction and translates STORE_BUSY to nonfatal progress. Creator's Python `replace_local` and native CLI repo commands use authored `apply_repository`, so their compare-and-swap semantics cannot be removed as a cache simplification. Checkpoint-root retention and reader leases still own post-pull execution custody.
