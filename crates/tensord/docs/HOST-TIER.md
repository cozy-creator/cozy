# Library host cache

Varena's `HostCache` is shared across the machine's per-GPU ledgers. It stores raw source
regions before fill transforms; GPU custody uses its separate device/fill identity. TensorFS
exports selected verified read-only files and performs every transfer through its existing I/O
path. TensorD forwards grants and admission facts, and never opens a CUDA context.

RAM warming is acknowledged background work. Each region is independently ready, failed or
absent. GPU demand uses an actually ready region or reads disk immediately; it never waits for
optional host warming. Reader leases protect only regions currently being consumed. Idle
regions can be shed independently, including while another region is live.

Admission reads current cgroup-aware host room. Unknown or large demand sheds optional RAM
before admission; host pressure does not impose a RAM floor or block GPU admission. Warming
and release follow actual state changes and event notifications rather than a polling timer.

Fresh mapped-host outputs reserve private bytes through this same authority before allocation.
Pending bytes reduce admission room; committed physical bytes are already in the host reading
and are not subtracted twice. The reservation survives row closure and stays live until the
last Array/DLPack alias and its consumer work are done. Commit, final close and process death
wake existing eventfd/host waiters through the private lease channel.

The implementation and process/pressure tests are in `varena/src/ledger.rs`, its `HostCache`
and `HostReservation`, and `varena/tests/ledger.rs`. No executor-supplied sealed-layout deposit
creates another host-cache policy.
