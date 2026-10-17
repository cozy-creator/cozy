# Downloaded repository cache roots

A successful native Hub pull may mark a newly created repository as a disposable
cache root. The marker binds the exact repository bytes, under the same native
repository compare-and-set. An existing unmarked or mixed repository is never
retroactively marked. Ordinary local repository mutations invalidate the marker,
including no-op writes. Missing, mismatched, or interrupted markers preserve roots.

Pressure collection is an explicit managed-store action. TensorD supplies a keep list
covering live executors, in-progress acquisitions, and model inputs/preparations of
unfinished runs. The embedded TensorFS library chooses eligible roots and reclaims bytes
using its existing dedup-aware planner, retaining native source, derive, and operation
holds. The native exclusive recovery lock spans eligibility, root removal, and GC;
live readers or writers return STORE_BUSY before any root is changed. Cache roots
are never removed just because their age exceeds a threshold.

Existing roots without provenance stay protected. This intentionally makes an
upgrade conservative; a successful fetch is not permission to discard an existing
user-owned repository.
