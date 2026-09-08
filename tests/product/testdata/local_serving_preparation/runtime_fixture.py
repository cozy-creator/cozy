"""Record real Runtime preparation RPC results from an otherwise ordinary local worker."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from cozy_runtime.internal.worker import session

AUDIT = Path(__file__).with_name("serving_fixture_audit.jsonl")


def record(method, worker, request, result=None, error=None, **facts):
    row = {
        "method": method,
        "worker_pid": os.getpid(),
        "worker_boot_id": worker.fence.worker_boot_id if worker is not None else "",
        "owner": worker.fence.record_owner_id if worker is not None else "",
        "operation_id": request.operation_id if request is not None else "",
        "birth": Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
    }
    if result is not None:
        row["placement"] = base64.b64encode(
            result.placement_set.placement_set_canonical_bytes
        ).decode()
        row["digest"] = result.placement_set.placement_set_digest.hex()
    if error is not None:
        row["error"] = str(error)
    row.update(facts)
    with AUDIT.open("a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")


if "serve" in sys.argv[1:]:
    record("WorkerProcess", None, None)


class FixtureWorker(session.Worker):
    def _acquire(self, placement, revision):
        acquired = super()._acquire(placement, revision)
        if acquired is not None and acquired.installed is not None:
            record(
                "Acquire", self, None,
                environment=str(acquired.installed.python),
                dependency_environment=str(self.options.environment_python),
                receipt=acquired.installed.receipt_digest,
            )
        return acquired

    def prepare_local_package(self, request):
        try:
            result = super().prepare_local_package(request)
        except Exception as error:
            record("PrepareLocalPackage", self, request, error=error)
            raise
        record("PrepareLocalPackage", self, request, result=result)
        return result

    def prepare_private_placement(self, request):
        try:
            result = super().prepare_private_placement(request)
        except Exception as error:
            record("PreparePrivatePlacement", self, request, error=error)
            raise
        record("PreparePrivatePlacement", self, request, result=result)
        return result


session.Worker = FixtureWorker
from cozy_runtime.cli.main import main

raise SystemExit(main(sys.argv[1:]))
