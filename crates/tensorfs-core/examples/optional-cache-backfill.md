# Optional cache backfill after native acquisition

A live H100 pull exposed the problem: all 16 origin workers were waiting in optional
FUSE cache writes, with no sockets open. Unbinding the cache and retrying only the
fetch process preserved 411 local objects and restored network acquisition. This fix
is bounded to backfill; cache-hit reads remain synchronous.

One process-scoped pool in `repo_cache` owns four optional writers. Each origin worker
offers only cache/source paths, kind and ObjectRef after local admission. The 128-entry
queue is an internal memory policy: tens of KB for typical paths, roughly 1 MB with two
4 KB paths per item. `try_send` never waits. A full or disconnected queue loses optional
offers. There is no persistent backlog, payload queue, cache watermark or user knob.
Repeated pulls and multiple Stores share the same worker bound.

A worker reopens an unobserved local Store and reuses `RepoObjectCache::backfill_store`.
That path requires exact verified source bytes, hashes while writing the cache temporary,
fsyncs, links the canonical path without clobbering, and cleans its own temporary. A
queued source reclaimed before opening is a missed cache opportunity. Once its verified
file is open, the descriptor preserves the inode through concurrent unlink. No Store GC
hold is kept while waiting on NFS. The pull's ordinary finalizer owns local checkpoint
custody independently of cache completion.

Origin completion never joins these workers. Process exit can leave up to four active
noncanonical temporaries from this pool. Readers ignore them, as with the existing
publication path after process death. No partial canonical object is published, no
unfinished work is claimed durable, and no unrelated temporary sweep is introduced.
Source-ingest `Mirror` retains its explicit drain, watermark and checkpoint semantics.

The real HTTP regression blocks four backfill opens, then proves all 26 objects complete,
local replay fetches zero bodies, and GC returns. Another Store/cache pair acquires its
own exact objects while the shared pool is blocked; 512 offers obey the 128-item queue
and four-worker bounds. Released cache writes verify exact bytes. The old synchronous
call fails the same completion test.

Additional controls prove that GC unlinks an unrooted source during the actual streaming
copy while its verified descriptor finishes the exact cache object; an injected read
failure cleans only that copy's temporary; and a subprocess exits normally with all four
writers still blocked after completing and retaining its local checkpoint. No live
process, configuration, volume or pod is changed by these tests.
