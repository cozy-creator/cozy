# Optional cache backfill after native acquisition

The live H100 pull proved the problem: all16 origin workers were waiting in optional
FUSE cache writes, no sockets open. Unbinding the cache and retrying only the fetch
process preserved411local objects and restored network acquisition. The first fix is
bounded to backfill; cache-hit reads are still synchronous.

One process-scoped pool in existing repo_cache owns four optional writers. Each origin
worker offers only cache/source paths, kind and ObjectRef after local admission. The
128-entry queue is an internal memory policy, about tens ofKB for typical paths and
roughly1MB with two4KBpaths per work item. try_send never waits; full/disconnected queues
can lose offers. There is no persistent backlog, payload-memory queue, cache watermark,
new identity or user knob. Repeated pulls and multipleStores share the same worker bound.

Workers reuse RepoObjectCache::backfill_store exactly. It reopens a fresh unobserved
local Store, requires exact verified source bytes, hashes while writing the ordinary
cache temporary, fsyncs, no-clobber links the canonical path, and cleans its own temporary.
A queued source reclaimed before opening is a missed cache opportunity. Once its verified
file is open, the descriptor preserves the inode through concurrent unlink. No StoreGC
hold is kept while waiting onNFS. Local canonical checkout custody remains the pull's
ordinary finalizer's responsibility, independent of cache completion.

Origin completion never joins these workers. Process exit can leave at most four active
noncanonical cache temporaries; they are ignored by readers, as with the existing
publication path after processdeath. No canonical partial file is published, no unfinished
work is claimed as durable, and no unrelated NFS temp sweep is introduced. Source-ingest
Mirror and its explicit drain/watermark/checkpoint semantics remain unchanged.

Tests exercise actualHTTP origins and localStoreverification with four blocked backfill
opens,26object completion, zerobody warmreplay, GCwhileblocked, anotherStore/cachepair,
512nonblockingoffers within128queuecapacity, exactlyfourworkers, then verified cache
completion afterblockrelease. The old synchronouscall must fail the same bounded test.
No live process/configuration/pod changes are made by this task.
