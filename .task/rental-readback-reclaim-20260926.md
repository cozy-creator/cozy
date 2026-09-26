Owner: /root/h3_reference_astra
Purpose: explicit rental claim/update retries a transient incomplete hardware readback through a fresh authenticated claim and snapshot, without treating missing observations as idle.
Branch: fix/rental-readback-reclaim-20260926
Base: c899707a (fresh origin/master, includes PR692)

Yoonho's direct ClaimAck reported empty hardware and the owner latched it permanently. The same installed hostfacts probe subsequently reports both GPUs, including under the supervisor's closed environment. The initial ~11 second claim delay is consistent with the probe's 10 second timeout, but the original probe error is discarded and its exact cause is not proved. Runtime activity reports no active execution; recovery still requires fresh authenticated snapshot reconciliation.

No tests, CI, rentals or live mutations by this task. Root owns merge/build/install.
