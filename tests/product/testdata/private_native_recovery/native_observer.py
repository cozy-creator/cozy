"""Trusted, owned-worker fault observer; never imported by submitted package code.

The operator supplies a private directory in COZY_RECOVERY_OBSERVER. An arm.json
file selects one native fault. Observations and a durable claimed marker precede
the fault, so a Runtime restart cannot inject the same fault twice. No workspace
or TensorFS state is edited. Remove the launcher instrumentation after proof.
"""

import json
import os
import signal
from pathlib import Path


def install():
    import tensorfs.derived as derived

    root = Path(os.environ["COZY_RECOVERY_OBSERVER"])
    original = derived.serve_derived

    def event(kind, **fields):
        raw = json.dumps({"kind": kind, "pid": os.getpid(), **fields}, sort_keys=True)
        with (root / "events.jsonl").open("a") as stream:
            stream.write(raw + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def fault(kind, operation, facts):
        arm_file = root / "arm.json"
        if not arm_file.exists():
            return
        arm = json.loads(arm_file.read_text())
        if arm["kind"] != kind:
            return
        # The test's dedicated worker has no other requests. Optional operation
        # matching allows a source stage to pass before candidate checkpointing.
        if arm.get("operation") and arm["operation"] != operation:
            return
        marker = root / (arm["id"] + ".claimed")
        try:
            with marker.open("x") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "facts": facts,
                                         "operation": operation, "kind": kind}))
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            return
        event("fault", fault=kind, operation=operation, facts=facts, arm=arm["id"])
        if arm.get("action", "kill") == "raise":
            raise RuntimeError("owned qualification failure after native checkpoint")
        os.kill(os.getpid(), signal.SIGKILL)

    class Writer:
        def __init__(self, writer, operation):
            self.writer, self.operation = writer, operation

        def __getattr__(self, name):
            return getattr(self.writer, name)

        def completed_parts(self):
            parts = self.writer.completed_parts()
            event("completed", operation=self.operation, parts=parts)
            return parts

        def add_part(self, component, key, role, reader):
            result = self.writer.add_part(component, key, role, reader)
            event("part", operation=self.operation, component=component, key=key, role=role)
            return result

    def observed(writer, fd, **kwargs):
        operation = kwargs["operation_id"]
        receipt = kwargs["record_receipt"]
        checkpoint = kwargs["record_checkpoint"]

        def record_receipt(facts):
            event("receipt_before_ack", operation=operation, facts=facts)
            fault("receipt_before_ack", operation, facts)
            return receipt(facts)

        def record_checkpoint(facts):
            result = checkpoint(facts)
            event("checkpoint", operation=operation, facts=facts)
            fault("checkpoint", operation, facts)
            return result

        kwargs.update(record_receipt=record_receipt, record_checkpoint=record_checkpoint)
        return original(Writer(writer, operation), fd, **kwargs)

    derived.serve_derived = observed
    event("installed")
