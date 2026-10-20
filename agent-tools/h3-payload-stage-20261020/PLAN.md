# One-off H3 payload staging plan

This branch contains a plan, a one-off caller of the existing transport, and metadata-only contract tests; no transfer has run. Product API source is exact25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2. C3's metadata client/store remain untouched. No weights, GPU, provider, service mutation, or local native build is authorized by this file.

## Supported existing path

Public Rust already exposes every required transport primitive:

- `fetch::TensorhubOrigin::new(&Client, base)` and `.named(name, release, lane)` (`fetch/tensorhub.rs:98,134`) issue metadata requests and return the resolved manifest reference without parsing/downloading it as a D10 tree.
- `Origin::locate(&[ObjectRef])` calls existing `/v1/tensorfs/closure` and `/v1/tensorfs/presign`, binding presigns to that resolved old manifest; it returns locations without downloading object bodies.
- `fetch::Fetcher::new(Arc<dyn ObjectStore>, client, None, Config)` plus `.objects(&origin, &objects, &Want)` (`fetch/mod.rs:161,255`) downloads exactly the supplied object set into normal store admission; `.settled()` awaits outstanding writes.
- `store2::Store` implements ObjectStore; admission verifies object bytes. No cross-filesystem hardlinks or hand-written object-path writes are needed.

The old collector already used POST `/v1/tensorfs/presign` with `{digests:[hex],manifest:old_manifest_hex}`, then its immutable GET. This plan reuses the compiled transport and admission instead of extending that collector into a new downloader. The new-format `tfs pull` command uses `pull_tree`, so it is not the route for this old manifest. Python's normal `open/download` API does not expose an old-manifest Origin override. A small Rust caller is therefore the correct boundary; no public API/product change is required.

## One-off caller

`agent-tools/h3-chunk-stage/h3_chunk_stage.rs` is installed as a temporary Cargo example only in a fresh hosted Product25 checkout. Default mode is metadata-only; explicit `--download` is required for payload movement.

1. Read bounded input containing sourceHub `https://cozy-e2e-6.ngrok.app`, repo `paul/minimax-h3`, release `1.0.0-rc.3`, lane `int8`, old manifest `bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff`/743bytes, current-header path/hash, and independent destination store. Reject any changed pins, current-header hash mismatch, count other than1676, duplicate/conflicting chunk refs, or total physical bytes other than75,535,069,920. Parse the current header with the existing format reader; never read source tensor bytes for this validation.
2. Inside `fetch::runtime()`, create the existing Client with normal HTTPS policy (no local-network bypass for the real source; no new credentials/minting), and the existing TensorhubOrigin. Call `named` and require its returned manifest equals the pinned OLD manifest, model/release/lane match, and public istrue for the anonymous plan. This resolves only the old closure; the new header/tree must never be sent as a remote ref.
3. For metadata-only preflight, call `locate` on one known retained chunk, require a location for that exact reference, report success without requesting the URL's bytes. Never log presigned query strings. This confirms target-machine closure/presign reachability; the controller's previous metadata success is insufficient.
4. Only in explicitly authorized `--download` mode, open the independent current Store, admit the exact local header bytes with `Store::put(..., Some(&header_ref))`, and create an owned `Store::pin` containing `Pin::new(vec![header_ref])`. Pinning the header retains the chunk closure as objects arrive and across the handoff; it does not change the external header or publish anything. Refuse a nonempty destination without this exact owned pin. Call the SAME existing Fetcher with all1676 header chunk ObjectRefs. No tree pull, reference update, publication, encoding conversion, other lane, or HF fallback. Valid partially admitted objects and the owned pin remain after a failed transfer.
5. Await `fetcher.settled()` even after objects() fails, then `Store::sync`. Use the caller's existing bounded process deadline/stop handling; do not add a daemon, detached process, or another retry policy. Record exact accepted object counts/bytes, destination root, source/header/manifest hashes, source-control identity and existing fetch stats, excluding signed URLs/credentials. The destination uses raw `Store::open`, without automatic cache eviction; the explicit pin is not a TTL promise against a caller explicitly removing it.
6. Verify the full local checkpoint closure against the unchanged local D10 header, then run the already-prepared h3_load.py. No benchmark starts on a partial cache.

Proposed call sequence (existing signatures, not a new API):

```rust
let origin = Arc::new(TensorhubOrigin::new(&client, source_hub)?);
let resolved = origin.named(repo, Some(release), Some(lane)).await?;
require_exact_old_identity(&resolved, &old_manifest)?; // one-off input check
let origin: Arc<dyn Origin> = origin;
let _location = origin.locate(&header.chunks[..1]).await?; // metadata-only mode stops here
own_store.put(header_bytes, Some(&header_ref))?;
own_store.pin(owned_pin_id, &Pin::new(vec![header_ref]))?;
let fetcher = Fetcher::new(Arc::clone(&own_store), client, None, Config::default());
let result = fetcher.objects(&origin, &header.chunks, &Want::default()).await;
fetcher.settled().await;
own_store.sync()?;
result?;
```

`require_exact_old_identity` above describes local validation, not a claimed library function. The one-off source now implements those checks and uses `verify_selected(None)` afterward; it never calls the old destructive `Store::verify`. Its script prepares tests and the executable only. No live call or payload transfer is automatic.

## Contract tests supplied now

`transport_contract.rs` reuses `crates/tensorfs/tests/common`'s existing real loopback old-manifest Hub. No server/HTTP implementation is copied or added.

- `current_origin_resolves_and_presigns_old_manifest_without_reading_payloads`: confirms the old manifest identity can be resolved and a retained chunk URL located without interpreting a tree; closure/presign each occur once; object GET count stays0.
- `wrong_old_manifest_refuses_before_presigning_or_fetching`: a wrong old-manifest pin gets REF_NOT_FOUND, with no presign or object GET.

They are prepared source only, not locally compiled or executed. On a fresh Product25 hosted checkout, copy this one test beside the existing common module and run the existing targeted cargo test command. No native TensorFS wheel/AOT build or real source endpoint is needed. After the contract passes, a separately authorized one-object real-Hub metadata preflight still precedes any75GB payload transfer.
