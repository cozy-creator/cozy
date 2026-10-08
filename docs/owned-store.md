# Embedded store ownership

`Store::ensure_owned(root)` opens or initializes one TensorFS store and claims its
exclusive owner role. It is for an embedding daemon, such as TensorD. TensorFS
remains a library; authentication, execution records, scheduling and memory budget
decisions remain with the daemon.

This owner role covers the managed Store's serialized objects and metadata. Shared CPU
buffers and exported GPU allocations have separate descriptor/mapping lifetimes in
`tensorfs-plane`. TensorD coordinates their retention; Runtime constructs models and
performs executor-side GPU operations using the library. A Store claim alone does not
construct a model or make its tensors GPU-resident.

The store's `owner.lock` is retained by the returned Store, its clones, and every
live owner read lease. Stores reopened normally in that same process also retain
the existing claim. A second owner receives `STORE_BUSY` before catalog recovery.
Dropping the embedding daemon's wrapper cannot release ownership while one of
those handles still serves bytes. Normal process exit or death closes the lock;
the next owner opens the persisted store using normal TensorFS recovery.

`Store::ensure`, `Store::open`, and `Store::init` keep their ordinary shared-access
behavior. Ownership does not bar other processes from using the store. A reader
in the owner process pins individual objects so its own GC can run; other
processes' read leases continue to exclude GC with the recovery lock. The owner
registration is scoped to the live owner handles, rather than lasting forever.

GC, catalog rebuild and destructive reclamation require the owner role or a
shared hold on `owner.lock` for the whole operation. An outside process cannot
see the daemon's in-memory object pins, so its destructive operation returns
`STORE_BUSY` while that daemon owns the store. A shared destructive hold also
prevents a daemon from claiming ownership partway through a collection; it does
not register that outside collector as an in-process owner. Ordinary Store reads
and writes remain available.

Older full-store maintenance binaries do not consult `owner.lock` and cannot
see a daemon's private read pins. Their destructive GC/reclaim/rebuild commands
must run only after the owner and all leases have ended; online maintenance uses
the owning daemon or the updated library. This restriction does not apply to
ordinary reads or to Hub catalog planning on a separate census directory. There
is no peer-version admission check.

The daemon no longer needs to know the ownership lock filename or separately call
`meta::own`. GPU custody, executor supervision and daemon restart adoption are
outside this API.

Implementation and CPU process/storage regression proofs are tracked in
[tracker #338](https://github.com/cozy-creator/tracker/issues/338).

`cargo test -p tensorfs-core --test owned_store` exercises process death, path aliases,
interoperation with the daemon's earlier direct lock, clone and lease lifetimes,
GC beside owner leases, GC exclusion for another process's readers, and ownership
refusal before catalog recovery. These are CPU storage proofs, not inference or
GPU restart qualification.
