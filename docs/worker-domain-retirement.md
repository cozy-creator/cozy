# CLI domain and worker-protocol retirement

Current machine transport is `cozy.machine.v1`. Remove unused old worker submission,
control, claim and checkpoint producers; current cancellation and observation remain native.
Capture, model choices, byte/model custody and measured device observations keep their owner
and become current domain/native values. Preserve active outboxes and historical readback in
a small private archive reader that carries only the fields its consumers use.

The `cozy.capture/1` document belongs to `internal/localpackage/execution_capture.go`.
It records the immutable installation graph and interface selected at intake; the controller
stores and hashes its JCS bytes for request replay. It is not sent as a machine protocol or
used to compare client/server builds. Native model selections use `cozy.machine.v1` directly.

Local launchers use the existing owner's `COZY_AUTHORIZED_KEYS`. Native rental attachment
uses the retained Creator key and certificate pin; an obsolete media-token projection is no
longer an attachment gate. Hub's still-required request field stays until its owner cuts it.

Keep D2 software following and the current native run/status/rental/update/output consumer
tests. Delete generated worker bindings, their vendoring/build/corpus, and obsolete reflection
helpers after current consumers reach zero. No renamed controller, compatibility RPC wrapper,
peer equality gate, image admission, rental, GPU operation, deployment or publication.

Trackers: [331](https://github.com/cozy-creator/tracker/issues/331),
[334](https://github.com/cozy-creator/tracker/issues/334).

## Proof and review status

The source/type cut is validated on fetched master52b2b159. Product compilation and
all-package compilation pass; go vet passes. Focused archive/capture/custody/recovery
and native dispatch-marker checks pass. Actual native CPU consumers pass with merged
machine9636781: nested callees, delayed acceptance/cancel, client/watcher exit, daemon
restart, missing source reattach and immutable input readback after original edits.
The frozen original binding-producer fixtures read without worker bindings, including
additive archived fields. GPU observations and existing outbox/archive rows are preserved.

Sent native work stays canceling when its machine is stopped, and acceptance-unknown
cannot advertise retry from empty old blob columns. Old worker submissions/receipts are
abandoned locally with an explicit uncertainty reason, never resent as native runs.

This is source/CPU proof. Independent PR review and root merge checking still precede
merge; candidate3 and the final real-machine hardcut condition12 are separate gates.
No deployment, publication, provider rental or GPU qualification is claimed here.
