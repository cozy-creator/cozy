# Retained sources and interrupted download recovery

The native integration API accepts an already pinned provider URI and reviewed carrier
selection. `Store.resolve_source` delegates to the provider resolver and its index/shard
closure. `Store.materialize_source` consumes those exact rows without re-listing. These
are deployment-adapter APIs: Creator still owns moving-ref acceptance, credentials and
permission. An author-facing facade should pass a SourceArtifact capability, not URLs,
credentials, or local paths.

`resolve_source(uri, carriers, *, credential="", huggingface=None, civitai=None,
allow_local=False)` returns the selected `(member, sha256, exact_length, URL)` rows,
provider policy, accepted source spelling/selection digest, and content manifest identity.
Endpoint overrides are for privileged deployment/tests. The content identity is the
ordinary-file Manifest: logical names, raw SHA-256 and exact lengths. URL, origin and
credential do not appear in that manifest or retained root.

`materialize_source(owner, members, *, allowed_hosts, credential_hosts=[], credential="",
allow_local=False)` returns `artifact` plus per-call `transferred_bytes`. Artifact fields
are owner, manifest_digest, manifest_length, complete and released. Owner IDs are prefixed
SHA-256 identities assigned by the caller's operation journal. They are not user retry IDs.

The default source downloader retains a contiguous prefix every 64 MiB. It fsyncs bytes
before atomically recording the prefix extent and digest. Restart rehashes retained bytes,
truncates an uncommitted suffix and continues from the verified extent. Corrupt retained
prefixes are discarded with an explicit OBJECT_CORRUPT refusal; the following retry
starts that object at zero. The complete object
must match the pinned length and SHA-256 before the same native no-clobber CAS admission
used by ordinary writes. The prefix file becomes the CAS object; there is no second full
staging write. Actual prefix bytes count in Store occupancy.

A source owner fences its writer with kernel locks. `adopt_source_progress(source_owner,
owner)` refuses a live or released predecessor, independently retains completed raw objects,
and copies verified partial bytes into an independently writable recipient prefix. This
copy is bounded by retained progress and is charged to disk occupancy. Releasing the source
then leaves the recipient intact. A plain retry of the same owner performs no such copy.
Ignored Range responses restart the object and count repeated bytes. Expired delivery
URLs preserve progress for a refreshed adapter call. The common HTTP opener refuses
transformed Content-Encoding before any body bytes reach the prefix.

`retain_tree_root(source_owner, owner)` acquires independent custody of a completed tree.
`release_tree_root(owner)` is permanent and idempotent after creation, cleans owned partial
bytes, and never releases another owner. Native filesystem roots participate in GC even
when no Python process is alive. `create_tree_root` can retain an already admitted ordinary
file Manifest; it does not accept a CozyTensors checkpoint disguised as a source tree.

`prepare_source_artifact(source_owner, operation_id, profiles, *, checkpoints=[],
adopt_from_operation_id=None)` builds the existing native converter's roster from retained
members. Carrier paths stay inside native code. It returns the existing prepare_model_source
result, including native progress heads; callers repeat incomplete calls until complete.
Its `spent` observations do not authorize deletion of the separately retained raw artifact.
New tensor groups charge the current Store occupancy, including retained raw inputs.

Conversion preparation and tensor-operation journals are keyed by the request and the
re-derived conversion plan (converter name, ordered ops, carrier headers), never by the
TensorFS build. An upgraded or differently built process replays completed preparations and
resumes partial ones. A change to a converter's output bytes renames the converter or bumps
the `tensorfs.conversion-journal` domain tag.
`Store.implementation_sha256` remains an informational build fingerprint.

Qualification runs `cargo test --workspace`, plus `scripts/source-artifacts.py` against the
compiled Python extension (included in wheel smoke). Subprocess tests use real SIGKILLs
around prefix fsync/checkpoint/admission, verify request offsets, full readback and moved
bytes, and cover custody, tampering, URL replacement, ignored Range and GC. No GPU or paid
provider is needed for these contracts.
