# One selected H3 A11 metadata proof

This one-off fixture stages only `paul/minimax-h3@1.0.0-rc.3`, revision7, int8. It does not change the published library or live data. Input is the immutable collected bundle SHA256 `891aeb30592b74beedba96583b7bd130b6730b07fdba8977a23f4d0248811dff` and its ten metadata objects: .12 manifest/header, five original component configs, pinned upstream listing, README and LICENSE. No tensor chunk is included.

The compiled client reuses the existing pure `Migrator::repo` and approved UPSTREAMS policy. It accepts that exact bundle, source-base25 and one repo; only its metadata allowlist is readable. The existing replay controller was reduced to its stage-only functions: no collector, credential option or external HTTP request path. The sole server is bounded in-process loopback replay of those fixed files, and the Rust client writes a fresh owned destination. The independent cbor2 comparator checks all1676chunk SHA/length pairs AND order, encoding table and3858complete tensor records; original configs, class declarations, pinned docs and source/filled provenance; chunks must remain absent.

The source branch starts at freshly fetched origin/master `dc0d4d27a21bb42a5f6d8e7a2c959aeab24d0a82`; it changes only this agent-tools prefix. Hosted native compilation/execution uses PRODUCT `25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2`, separately recording the PROOF commit. The native workflow checks out PRODUCT25, fetches the exact PROOF through the existing public Git remote, and extracts only this prefix. No product overlay or broad cohort merge is used.

The parent dispatches after source/file-manifest review. In the PRODUCT25 hosted checkout with this prefix already extracted:

```bash
H3_PROOF_SHA=<immutable-proof-commit> bash agent-tools/h3-a11-metadata-20261020/run-hosted.sh
```

The command checks production paths unchanged, builds a temporary example, runs the metadata-only stage and independent comparison, then exports `dist/testbins/h3-a11` through the existing workflow artifact step. Its portable metadata archive excludes compiled clients, request/partial logs and tensor payloads. Controller work is source/AST/hash/metadata preparation only; no native TensorFS/Torch/GPU/model import or native client execution runs there. This proof does not qualify payload availability, GPU capacity/performance or inference. No provider/catalog/rental/service operation is involved.
