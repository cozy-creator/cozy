# Retained checkpoint recovery operator

This explicit support command banks successful producer receipts when the original
Runtime cannot answer upload calls. It is not a package execution hook and does not
repair request lifecycle, restart a worker, acknowledge outcomes, or release capacity.

Build with `go build -o retainedcustody ./tests/support/retainedcustody`. With the
existing Creator home selected, `--request`, `--rental`, and `--boot` inspect the exact
retained successful outcome and native receipts. Add `--execute --reader <program>
<args...>` only for the reviewed recovery. `--proof-object sha256:...` limits the first
joined proof to one exact object. The reader program receives bounded JSON lines on
stdin, each with up to four exact manifest/object references and short-lived grants;
its stdout must contain the matching ordered results. No grant belongs in argv,
environment variables, logs, or a permanent credential file.

`reader.py --store /absolute/native/root --tfs /absolute/tfs` is the native adapter.
It validates manifest membership through public TensorFS, then runs independent
`tfs push --grant-file /dev/stdin` children. It performs no Python lease body reads.
Batch preflight is serial; at most four native push children run together. SSH/SFTP
launch and key management are operator-owned. If the transport requires a PTY, turn
off echo and canonical line processing before sending any grant, and isolate shell
startup output from the reader protocol. No persistent account keys reach the worker.

The existing Hub publication owner opens and verifies the exact closure. Objects the
Hub already holds need no invented worker statuses. Fresh objects require native
upload, Hub verification, and a real signed PodHost HELD response. Before finalization,
a complete Hub grant census must say HELD for every declared object. The normal Hub
checkpoint finalizer and identity validator remain the authority; a failed request can
bank output FinalIDs while whole-request lifecycle recovery remains explicitly pending.

`prove_reader.py` and `prove_reader_batch.py` run with an installed public TensorFS
Python/CLI pair. They use actual native writer/ADOPT, HTTP uploads and checksum/readback,
412 replay, unknown-member refusal, secret-output checks, and a barrier requiring four
simultaneous HTTP bodies. The live joined recovery additionally exercised pinned TLS,
signed PodHost custody, Creator records, and ordinary Hub checkpoint finalization.
