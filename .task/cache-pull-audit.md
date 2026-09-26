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
