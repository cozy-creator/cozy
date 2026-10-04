# Machine telemetry and durable transitions

A progress/log sample carries no authority to settle, cancel, admit or replay a run. Writing
and synchronizing every v1 sample on the records database's single connection blocks all
readers and the terminal consumer behind that disk write. A busy disk has produced 12–75
second fsync waits in the previous delivery trace.

Creator retains a bounded sample set per observed run: the first measured fraction and the
latest progress, preparation and log sample. Samples enter memory without opening a database
transaction. Status/readers obtain the latest progress there, and SSE emits new samples with
sequence_number zero and no resumable id. A newly attached live subscriber gets the latest
sample immediately. These samples never advance either the durable SSE or remote run cursor.

State/product transitions coalesce pending samples before their lifecycle events within the
existing transaction. A terminal coalesces them before its absorbing outcome, in that same
FULL transaction. Product identity, lifecycle state, terminal result and durable cursors retain
their existing persistence. A crash may lose telemetry since the last real transition and
therefore re-observe those samples; it cannot skip a product because of a telemetry cursor.

Completed runs and detached controller loops release live samples. Multiplexed SSE retains
only currently live sample keys. No telemetry writer, background timer or weaker synchronization
mode is introduced. Other worker/API paths retain their existing records behavior.

CPU database checks cover 2,000 samples while the sole database connection is held, bounded
retention, terminal ordering and restart cursor recovery. A separate SSE check covers a latest
sample at subscribe with sequence zero. Ordinary CLI progress/reconnect and congested-disk
outcome-to-exit timing remain product gates; these checks do not qualify GPU inference.
