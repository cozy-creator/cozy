# Exact H3 chunk staging, one-off

This source uses Product25's public Rust transport and store. It is not a new TensorFS API, downloader, service, or benchmark. It has not been compiled or executed yet.

Product source is `25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2`; the accepted local header remains `0f03804af2280bac6c81c633cd058f4dfbb5fd71f5db0a3c5829907e4ef21f2e`. Only the old manifest `bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff` is used at the source Hub. Exactly 1,676 original chunks / 75,535,069,920 physical bytes are requested. No remote request names the new header/tree, and no HF fallback exists.

In a fresh hosted checkout at Product25, fetch the reviewed client commit without moving HEAD, extract only `agent-tools/h3-chunk-stage`, export its full SHA as `H3_STAGE_SOURCE_COMMIT`, then run:

```sh
bash agent-tools/h3-chunk-stage/check-hosted.sh
```

This runs four metadata/contract tests and compiles the Rust example. It neither builds a TensorFS wheel/AOT catalog nor invokes the client against the real Hub. Its loopback test uses the existing old-manifest Hub fixture and asserts zero object GETs. A local metadata-only test checks that an unchanged header pin can retain yet-absent chunks without reading them.

After the metadata bundle and binary reach the benchmark host, use the existing `h3-current-source.json` contract with absolute local header/cache paths. The cache must be independent of the metadata store. The source-controlled program fixes Hub/model/release/lane and verifies header identity, all five components, 3,858 tensors, 4,834 stored parts, and byte totals before any network request.

```sh
# Metadata-only: closure + presign, no cache creation or chunk GET.
target/debug/examples/h3_chunk_stage /absolute/path/h3-current-source.json > metadata-receipt.json

# Only after separate approval of transfer and capacity/deadline:
target/debug/examples/h3_chunk_stage /absolute/path/h3-current-source.json --download > staging-receipt.json
```

The download creates a named owned pin for an identical admitted copy of the local header before receiving chunks. An empty destination or a previous matching owned stage store is accepted. The source metadata store, refs, and local header remain unchanged. It retains partial admitted bytes for a later explicit retry. After the existing Fetcher settles, it syncs and calls the read-only `verify_selected(None)` over header/chunks/component assets. Missing, corrupt, transfer, or sync failure exits nonzero; signed URLs are never included in its receipt. README/LICENSE are repository files and are outside checkpoint verification.

Use an existing parent process deadline and stop policy. This client does not start a detached process, new retry policy, or GPU work. A successful receipt proves staged bytes only; it does not prove H3 load timings, decoding, VRAM capacity, or GPU performance. Run the separately reviewed H3 benchmark only after `status=staged_verified` and matching byte/provenance checks.
