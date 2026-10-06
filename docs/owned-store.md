# Embedded store ownership

`Store::ensure_owned(root)` opens or initializes one TensorFS store and claims its
exclusive owner role. It is for an embedding daemon, such as TensorD. TensorFS
remains a library; authentication, execution records, scheduling and memory budget
decisions remain with the daemon.

The store's `owner.lock` is retained by the returned Store, its clones, and every
live owner read lease. A second owner receives `STORE_BUSY` before catalog recovery.
Dropping the embedding daemon's wrapper cannot release ownership while one of
those handles still serves bytes. Normal process exit or death closes the lock;
the next owner opens the persisted store using normal TensorFS recovery.

`Store::ensure`, `Store::open`, and `Store::init` keep their ordinary shared-access
behavior. Ownership does not bar other processes from using the store. A reader
in the owner process pins individual objects so its own GC can run; other
processes' read leases continue to exclude GC with the recovery lock. The owner
registration is scoped to the live owner handles, rather than lasting forever.

The daemon no longer needs to know the ownership lock filename or separately call
`meta::own`. GPU custody, executor supervision and daemon restart adoption are
outside this API.

Implementation and CPU process/storage regression proofs are tracked in
[tracker #338](https://github.com/cozy-creator/tracker/issues/338).
