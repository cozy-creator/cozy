# One-off H3 payload staging plan

This branch contains a plan and metadata-only contract tests, not a new fetcher or a running transfer. Product API source is exact25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2. C3's metadata client/store remain untouched. No weights, GPU, provider, service mutation, or local native build is authorized by this file.

## Supported existing path

Public Rust already exposes every required transport primitive:

- `fetch::TensorhubOrigin::new(&Client, base)` and `.named(name, release, lane)` (`fetch/tensorhub.rs:98,134`) issue metadata requests and return the resolved manifest reference without parsing/downloading it as a D10 tree.
- `Origin::locate(&[ObjectRef])` calls existing `/v1/tensorfs/closure` and `/v1/tensorfs/presign`, binding presigns to that resolved old manifest; it returns locations without downloading object bodies.
- `fetch::Fetcher::new(Arc<dyn ObjectStore>, client, None, Config)` plus `.objects(&origin, &objects, &Want)` (`fetch/mod.rs:161,255`) downloads exactly the supplied object set into normal store admission; `.settled()` awaits outstanding writes.
- `store2::Store` implements ObjectStore; admission verifies object bytes. No cross-filesystem hardlinks or hand-written object-path writes are needed.

The old collector already used POST `/v1/tensorfs/presign` with `{digests:[hex],manifest:old_manifest_hex}`, then its immutable GET. This plan reuses the compiled transport and admission instead of extending that collector into a new downloader. The new-format `tfs pull` command uses `pull_tree`, so it is not the route for this old manifest. Python's normal `open/download` API does not expose an old-manifest Origin override. A small Rust caller is therefore the correct boundary; no public API/product change is required.

## Smallest client to implement after this plan is reviewed

One file under this same owned prefix, installed as a temporary Cargo example only in a fresh hosted Product25 checkout. Default mode is metadata-only; explicit `--fetch` is required for payload movement.

1. Read bounded input containing sourceHub `https://cozy-e2e-6.ngrok.app`, repo `paul/minimax-h3`, release `1.0.0-rc.3`, lane `int8`, old manifest `bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff`/743bytes, current-header path/hash, and independent destination store. Reject any changed pins, current-header hash mismatch, count other than1676, duplicate/conflicting chunk refs, or total physical bytes other than75,535,069,920. Parse the current header with the existing format reader; never read source tensor bytes for this validation.
2. Inside `fetch::runtime()`, create the existing Client with normal HTTPS policy (no local-network bypass for the real source; no new credentials/minting), and the existing TensorhubOrigin. Call `named` and require its returned manifest equals the pinned OLD manifest, model/release/lane match, and public istrue for the anonymous plan. This resolves only the old closure; the new header/tree must never be sent as a remote ref.
3. For metadata-only preflight, call `locate` on one known retained chunk, require a location for that exact reference, report success without requesting the URL's bytes. Never log presigned query strings. This confirms target-machine closure/presign reachability; the controller's previous metadata success is insufficient.
4. Only in explicitly authorized `--fetch` mode, open a new independent current Store and call the SAME existing Fetcher with all1676 header chunk ObjectRefs. No header/tree pull, reference update, publication, encoding conversion, other lane, or HF fallback. Keep metadata bytes outside this store immutable; valid partially admitted objects may remain after a failed transfer.
5. Await `fetcher.settled()` even after objects() fails, then propagate failure. Use the caller's existing bounded process deadline/stop handling; do not add a daemon, detached process, or another retry policy. Record exact accepted object counts/bytes, destination root, source/header/manifest hashes, source-control identity and existing fetch stats, excluding signed URLs/credentials.
6. Verify the full local checkpoint closure against the unchanged local D10 header, then run the already-prepared h3_load.py. No benchmark starts on a partial cache.

Proposed call sequence (existing signatures, not a new API):

```rust
let origin = Arc::new(TensorhubOrigin::new(&client, source_hub)?);
let resolved = origin.named(repo, Some(release), Some(lane)).await?;
require_exact_old_identity(&resolved, &old_manifest)?; // one-off input check
let origin: Arc<dyn Origin> = origin;
let _location = origin.locate(&header.chunks[..1]).await?; // metadata-only mode stops here
let fetcher = Fetcher::new(Arc::new(own_store), client, None, Config::default());
let result = fetcher.objects(&origin, &header.chunks, &Want::default()).await;
fetcher.settled().await;
result?;
```

`require_exact_old_identity` is proposed local validation, not a claimed library function. No transfer executable is supplied yet: review this supported boundary before implementing/compiling the tiny caller. An existing C2 driver that exposes the same exact operations can be used instead without another implementation.

## Contract tests supplied now

`transport_contract.rs` reuses `crates/tensorfs/tests/common`'s existing real loopback old-manifest Hub. No server/HTTP implementation is copied or added.

- `current_origin_resolves_and_presigns_old_manifest_without_reading_payloads`: confirms the old manifest identity can be resolved and a retained chunk URL located without interpreting a tree; closure/presign each occur once; object GET count stays0.
- `wrong_old_manifest_refuses_before_presigning_or_fetching`: a wrong old-manifest pin gets REF_NOT_FOUND, with no presign or object GET.

They are prepared source only, not locally compiled or executed. On a fresh Product25 hosted checkout, copy this one test beside the existing common module and run the existing targeted cargo test command. No native TensorFS wheel/AOT build or real source endpoint is needed. After the contract passes, a separately authorized one-object real-Hub metadata preflight still precedes any75GB payload transfer.
