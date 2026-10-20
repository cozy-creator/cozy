"""Synthetic public receipt binding tests; no runtime/native/device execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import contract
from public_binding import tree_document
from test_contract import benchmark, UUID
from validate import validate

HERE = Path(__file__).resolve().parent


def fixture(root):
    value, spec, chunk_hash = benchmark()
    (root / "source").mkdir()
    (root / "reports").mkdir()
    (root / "logs").mkdir()
    harness = subprocess.check_output(["git", "-C", str(HERE), "show",
        contract.HARNESS_SOURCE + ":crates/tensorfs-py/gates/speed/h3_load.py"])
    sources = {"h3_load.py": harness, "probe.py": (HERE / "probe.py").read_bytes(),
               "contract.py": (HERE / "contract.py").read_bytes(), "requirements.txt": b"torch==2.14.0+cu130\n",
               "owned_child.py": subprocess.check_output(["git", "-C", str(HERE), "show",
                   "1a221c31d6f38ba4628caf340d48522e1a2bf102:crates/tensorfs-py/gates/consumers/owned_child.py"])}
    spec["sources"] = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    for name, data in sources.items(): (root / "source" / name).write_bytes(data)
    payload = b"# explicitly synthetic public receipt fixture\n"
    (root / "source/payload.py").write_bytes(payload)
    manifest = {"spec": spec, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    capacity = {"visible_count": 1, "device": "cuda:0", "name": "NVIDIA H100", "compute_capability": [9, 0],
                "gpu_uuid": UUID, "cuda_visible_devices": UUID, "same_work_filesystem": True,
                "host_available": contract.HOST, "gpu_available": contract.GPU,
                "disk_available": contract.DISK_BASE + contract.HEADROOM,
                "physical": {"gpu": 1, "uuid": UUID, "name": "NVIDIA H100", "driver": "synthetic"}}
    stage = {"status": "staged_verified", "client_source": contract.CLIENT_SOURCE,
             "product_source": contract.CLIENT_PRODUCT, "benchmark_run": False, "gpu_used": False,
             "header": {"sha256": contract.HEADER, "length": 433449},
             "old_manifest": {"sha256": contract.MANIFEST, "length": 743}, "chunk_count": 1676,
             "verified_chunk_count": 1676, "physical_chunk_bytes": contract.PHYSICAL,
             "verified_chunk_bytes": contract.PHYSICAL, "closure_verified": True, "download_requested": True,
             "payload_transfer_called": True, "missing": [], "corrupt": [], "sync_error": None,
             "transfer_error": None, "cache_dir": "/synthetic/owned/source-store"}
    value["source"]["cache_dir"] = stage["cache_dir"]
    operations = []
    commands = [["/synthetic/stage-client", "/synthetic/source.json"],
                ["python", "probe.py", "--deadline", "1200"],
                ["/synthetic/stage-client", "/synthetic/source.json", "--download"],
                ["python", "h3_load.py", "--device", "cuda:0"]]
    for i, (name, command) in enumerate(zip(("metadata", "capacity", "stage", "benchmark"), commands)):
        operations.append({"name": name, "command": command, "started_unix": 1001 + i * 10,
                           "finished_unix": 1002 + i * 10, "ok": True, "rc": 0, "timed_out": False,
                           "reaped": True, "lingering_descendants": False, "signals": [], "pid": 100 + i,
                           "pgid": 100 + i, "elapsed_s": 1.0, "timeout_s": 50.0})
        for suffix in ("stdout", "stderr"): (root / f"logs/{name}.{suffix}").write_text("")
    identity = {"request_id": "17", "context_device": "cuda:0", "cuda_visible_devices": UUID,
                "deadline_unix": 1200, "started_unix": 1000, "finished_unix": 1100, "ok": True,
                "runtime_source": spec["runtime"]["source"], "runtime_version": spec["runtime"]["version"],
                "runtime_files": spec["runtime"]["files"], "dependencies": spec["dependencies"]["versions"],
                "operations": operations, "benchmark": {"measurement_complete": True, "parity_pass": False, "rows": 60}}
    for name, data in {"manifest.json": spec, "identity.json": identity, "reports/capacity.json": capacity,
                       "reports/stage.json": stage, "reports/h3.json": value}.items():
        (root / name).write_text(json.dumps(data))
    inventory = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    (root / "files.json").write_text(json.dumps(inventory))
    doc = tree_document(root)
    run = {"id": "run-synthetic", "number": 7, "status": "completed", "machine": "synthetic-machine",
           "measurements": {"execution": {"degree": 1, "gpus": [{"gpu": 1, "uuid": UUID, "pid": 105,
                                  "start_us": 1, "end_us": 2, "arch": "sm_90"}]}},
           "output": [{"index": 0, "status": "completed", "name": "out", "id": "7/out",
                       "media_type": "application/vnd.cozy.tree-manifest", "sha256": "sha256:" + hashlib.sha256(doc).hexdigest(),
                       "length": len(doc)}]}
    # A two-physical-GPU machine is valid when only one assigned/executed UUID is used.
    machine = {"name": "synthetic-machine", "status": {"gpus": [{"index": 0, "name": "unused", "driver": "synthetic"},
                  {"index": 1, "name": "NVIDIA H100", "driver": "synthetic"}]}}
    launch = {"payload_sha256": manifest["payload_sha256"], "deadline_unix": 1200,
              "argv": ["cozy", "run", "payload.py", "deadline_unix=1200", "--gpus", "1", "--on", "synthetic-machine"]}
    return manifest, run, machine, launch, chunk_hash


class PublicValidator(unittest.TestCase):
    def test_complete_slower_bound_tree_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, run, machine, launch, anchor = fixture(root)
            with patch.object(contract, "CHUNKS_SHA", anchor): result = validate(root, spec, run, machine, launch)
            self.assertTrue(result["ok"])
            self.assertFalse(result["parity_pass"])

    def test_rewritten_receipt_cannot_reuse_public_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, run, machine, launch, anchor = fixture(root)
            run["output"][0]["sha256"] = "sha256:" + "0" * 64
            with patch.object(contract, "CHUNKS_SHA", anchor), self.assertRaisesRegex(ValueError, "public run Tree"):
                validate(root, spec, run, machine, launch)

    def test_wrong_budget_or_public_gpu_identity_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, run, machine, launch, anchor = fixture(root)
            launch["deadline_unix"] = 1201
            with patch.object(contract, "CHUNKS_SHA", anchor), self.assertRaisesRegex(ValueError, "budget"):
                validate(root, spec, run, machine, launch)
            launch["deadline_unix"] = 1200
            run["measurements"]["execution"]["gpus"][0]["uuid"] = "f" * 32
            with patch.object(contract, "CHUNKS_SHA", anchor), self.assertRaisesRegex(ValueError, "physical GPU"):
                validate(root, spec, run, machine, launch)


if __name__ == "__main__": unittest.main()
