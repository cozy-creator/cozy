# CLI domain and worker-protocol retirement

Current machine transport is `cozy.machine.v1`. Remove unused old worker submission,
control, claim and checkpoint producers; current cancellation and observation remain native.
Capture, model choices, byte/model custody and measured device observations keep their owner
and become current domain/native values. Preserve active outboxes and historical readback in
a small private archive reader that carries only the fields its consumers use.

Local launchers use the existing owner's `COZY_AUTHORIZED_KEYS`. Native rental attachment
uses the retained Creator key and certificate pin; an obsolete media-token projection is no
longer an attachment gate. Hub's still-required request field stays until its owner cuts it.

Keep D2 software following and the current native run/status/rental/update/output consumer
tests. Delete generated worker bindings, their vendoring/build/corpus, and obsolete reflection
helpers after current consumers reach zero. No renamed controller, compatibility RPC wrapper,
peer equality gate, image admission, rental, GPU operation, deployment or publication.

Trackers: [331](https://github.com/cozy-creator/tracker/issues/331),
[334](https://github.com/cozy-creator/tracker/issues/334).
