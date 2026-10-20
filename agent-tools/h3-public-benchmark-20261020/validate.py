#!/usr/bin/env python3
"""Validate full H3 receipts and the actual public run that returned them."""
import argparse
import json
from pathlib import Path

from contract import (HARNESS_SHA, check_benchmark, check_capacity, check_spec,
                      check_stage, deadline_budget, finite, require)
from public_binding import physical, public_tree, sha


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def validate(root, manifest, run, machine, launch):
    root = Path(root)
    spec = manifest["spec"]
    check_spec(spec)
    require(read(root / "manifest.json") == spec, "manifest differs")
    files = read(root / "files.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(set(files) == actual - {"files.json"}, "receipt inventory differs")
    for name, digest in files.items():
        path = root / name
        require(not path.is_symlink() and sha(path) == digest, "receipt hash differs: " + name)
    allowed = {"manifest.json", "files.json", "identity.json", "source/payload.py"}
    allowed.update("source/" + name for name in spec["sources"])
    allowed.update("logs/" + name + suffix for name in ("metadata", "capacity", "stage", "benchmark") for suffix in (".stdout", ".stderr"))
    allowed.update({"reports/capacity.json", "reports/stage.json", "reports/h3.json"})
    require(actual == allowed, "Tree has missing receipts or unexpected/model files")
    require(sha(root / "source/payload.py") == manifest["payload_sha256"], "payload source differs")
    for name, digest in spec["sources"].items():
        require(sha(root / "source" / name) == digest, "frozen source differs: " + name)
    require(spec["sources"]["h3_load.py"] == HARNESS_SHA, "drifted H3 implementation")
    identity = read(root / "identity.json")
    require(identity.get("ok") is True, "worker failed")
    deadline = identity.get("deadline_unix")
    deadline_budget(identity["started_unix"], deadline)
    require(finite(identity.get("finished_unix"), True) and identity["started_unix"] < identity["finished_unix"] <= deadline,
            "wrong/exceeded deadline")
    require(type(launch.get("deadline_unix")) is int and launch["deadline_unix"] == deadline and
            launch.get("argv", []).count(f"deadline_unix={deadline}") == 1, "launch budget differs")
    require(identity.get("runtime_source") == spec["runtime"]["source"] and
            identity.get("runtime_version") == spec["runtime"]["version"] and
            identity.get("runtime_files") == spec["runtime"]["files"], "Runtime identity differs")
    require(identity.get("dependencies") == spec["dependencies"]["versions"], "dependency identity differs")
    operations = identity.get("operations", [])
    require([op.get("name") for op in operations] == ["metadata", "capacity", "stage", "benchmark"], "incomplete operation sequence")
    for op in operations:
        require(op.get("ok") is True and type(op.get("rc")) is int and op["rc"] == 0 and
                op.get("timed_out") is False and op.get("reaped") is True and
                op.get("lingering_descendants") is False and op.get("signals") == [], "child did not finish cleanly")
        require(type(op.get("pid")) is int and op["pid"] > 0 and op.get("pgid") == op["pid"], "child ownership missing")
        require(finite(op.get("elapsed_s"), True) and finite(op.get("timeout_s"), True) and
                op["timeout_s"] <= deadline_budget(op["started_unix"], deadline) and
                op["started_unix"] <= op["finished_unix"] <= deadline, "child budget differs")
    require(all(left["finished_unix"] <= right["started_unix"] for left, right in zip(operations, operations[1:])),
            "capacity/staging/benchmark ordering differs")
    require(len(operations[0]["command"]) == 2 and "--download" not in operations[0]["command"], "metadata command changed")
    require(len(operations[2]["command"]) == 3 and operations[2]["command"][-1] == "--download", "stage command changed")
    require(operations[1]["command"].count("--deadline") == 1 and str(deadline) in operations[1]["command"], "capacity deadline differs")
    require("--prepare-only" not in operations[3]["command"], "prepared-only result is not a benchmark")
    capacity = read(root / "reports/capacity.json")
    check_capacity(capacity)
    require(capacity["cuda_visible_devices"] == identity.get("cuda_visible_devices"), "sealed visibility differs")
    stage = read(root / "reports/stage.json")
    check_stage(stage)
    benchmark = read(root / "reports/h3.json")
    require(stage.get("cache_dir") == benchmark.get("source", {}).get("cache_dir"), "benchmark did not use staged cache")
    result = check_benchmark(benchmark, spec, capacity["gpu_uuid"])
    require(identity.get("benchmark") == result, "worker benchmark summary differs")
    require(launch.get("payload_sha256") == manifest["payload_sha256"], "public payload differs")
    bound_gpu = physical(run, machine, [capacity], launch, manifest["payload_sha256"])
    bound_tree = public_tree(root, run, identity)
    return {"ok": True, **result, "physical": bound_gpu, "public_tree": bound_tree,
            "r4_acceptance": False, "r4_link_review_required": True, "r4_lifecycle_review_required": True,
            "scope": "full raw-byte H3 load comparison; no inference or decoder qualification"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    for name in ("manifest", "run-show", "machine-show", "launch"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(validate(args.root, read(args.manifest), read(args.run_show), read(args.machine_show), read(args.launch)), indent=2))
