# Retained sources and interrupted download recovery

TensorD resolves a pinned provider URI and its reviewed carrier selection, then retains the
selected members as a source tree. The content identity is the ordinary-file Manifest:
logical names, raw SHA-256 and exact lengths. URL, origin and credential do not appear in
that manifest or retained root.

The downloader moves up to 16 ranged GETs at once across all members, fewer when the
memory this process can use (cgroup limit less unreclaimable usage, or MemAvailable) does not
hold 16 streams' buffers. Each object is one
sparse file of verified runs; a run is asked for with one GET from its reached byte to its
end and retains progress every 64 MiB: bytes are fdatasynced before the run's extent and
digest are recorded by atomic rename. When no whole run is left to start, an idle stream
splits the largest range in flight (never below 64 MiB per half), so one large member still
uses every stream. The object digest is computed in offset order as the run at zero absorbs
each finished run behind it. A delivery URL the source redirected to is reused until the
origin refuses it. A 429/5xx is re-asked after its stated `Retry-After`, which spends no
attempt, or after the ledger's patience, which does. Restart rehashes retained runs and
continues each from its verified extent. Corrupt retained runs are discarded with an
explicit OBJECT_CORRUPT refusal; the following retry fetches those ranges again. The
complete object
must match the pinned length and SHA-256 before the same native no-clobber CAS admission
used by ordinary writes. The prefix file becomes the CAS object; there is no second full
staging write. Actual prefix bytes count in Store occupancy.

A source owner fences its writer with kernel locks. An origin that ignores a Range restarts
the object as one body from zero and counts repeated bytes. Expired delivery URLs preserve
progress for a refreshed adapter call. The common HTTP opener refuses
transformed Content-Encoding before any body bytes reach the prefix.

`source_artifact::release` is permanent and idempotent after creation, cleans owned
partial bytes, and never releases another owner. Native filesystem roots participate in GC.

Conversion preparation and tensor-operation journals are keyed by the request and the
re-derived conversion plan (converter name, ordered ops, carrier headers), never by the
TensorFS build. An upgraded or differently built process replays completed preparations and
resumes partial ones. A change to a converter's output bytes renames the converter or bumps
the `tensorfs.conversion-journal` domain tag.

Qualification runs `cargo test --workspace`. Subprocess tests use real SIGKILLs
around prefix fsync/checkpoint/admission, verify request offsets, full readback and moved
bytes, and cover custody, tampering, URL replacement, ignored Range and GC. No GPU or paid
provider is needed for these contracts.
