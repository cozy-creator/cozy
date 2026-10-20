"""Synthetic metadata/validator controls only; never imports Torch or TensorFS."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import contract as c
import render

HERE = Path(__file__).resolve().parent
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def spec():
    value = c.template(json.loads((HERE / "metadata-files.json").read_text()))
    value.update(runtime={"source": "a" * 40, "sha256": "a" * 64, "bytes": 1, "version": "1",
                          "files": {"cozy_runtime/__init__.py": "a" * 64}},
                 native={"source": "b" * 40, "sha256": "b" * 64, "bytes": 1, "version": "1",
                         "files": {"tensorfs/__init__.py": "b" * 64}},
                 dependencies={"versions": {"torch": "2.14.0+cu130", "safetensors": "0.8.0",
                                            "fastsafetensors": "0.5.0", "numpy": "2.5.4"}},
                 pair_receipt_sha256="c" * 64, proof_source="d" * 40,
                 sources={name: "d" * 64 for name in ("contract.py", "probe.py", "owned_child.py", "h3_load.py", "requirements.txt")})
    return value


def benchmark():
    s = spec()
    initial_free = 100_000_000_000
    initial_floor, floor = initial_free - (64 << 20), initial_free - (2 << 20)
    def recovered(at):
        return {"floor": at, "settled_free": initial_free, "before_free": initial_free,
                "deadline_seconds": 10, "seconds": 0.001}
    parts = [{"component": "transformer", "key": f"part{i}", "role": "data",
              "length": c.PAYLOAD // 4834, "sha256": "d" * 64} for i in range(4834)]
    parts[-1]["length"] += c.PAYLOAD - sum(p["length"] for p in parts)
    chunks = [{"sha256": f"{i:064x}", "length": c.PHYSICAL // 1676} for i in range(1676)]
    chunks[-1]["length"] += c.PHYSICAL - sum(p["length"] for p in chunks)
    rows = [{**r, "bytes": c.PAYLOAD, "verified": True, "seconds": 2.0 if r["reader"] == "tensorfs" else 1.0,
             "cache_control": "POSIX_FADV_DONTNEED requested" if r["cold"] else "untimed full read",
             "warmup_recovery": None if r["cold"] else recovered(floor), "recovery": recovered(floor),
             "metrics": {"kind": "h2d", "measured": True, "bytes": c.PAYLOAD}} for r in c.schedule()]
    cells = {f"{dst} {'cold' if cold else 'warm'}": {
        "complete": True, "parity_pass": False,
        "medians_seconds": {reader: 2.0 if reader == "tensorfs" else 1.0 for reader in readers},
        "best_control": readers[1]} for dst, readers in c.READERS.items() for cold in (True, False)}
    value = {"measurement_complete": True, "parity_pass": False, "status": "complete_slower", "cold_is_advisory": True,
             "environment": {"harness_sha256": c.HARNESS_SHA, "actual_visible_gpus": 1,
                             "expected_visible_gpus": 1, "device": "cuda:0", "device_uuid": UUID},
             "cohort": {"declared_native_source": s["native"]["source"], "wheel_sha256": s["native"]["sha256"],
                        "installed_file_sha256": s["native"]["files"]},
             "versions": {key: s["dependencies"]["versions"][key] for key in ("torch", "safetensors", "fastsafetensors")},
             "source": {"header_sha256": c.HEADER, "legacy_manifest_sha256": c.MANIFEST},
             "capacity": {"host_required": c.HOST, "gpu_required": c.GPU, "available": c.HOST, "gpu_free": c.GPU},
             "gpu_recovery": {"initial_allowance": 64 << 20, "steady_allowance": 2 << 20,
                              "initial_free": initial_free, "initial_floor": initial_floor,
                              "baseline_free": initial_free, "fixed_floor": floor},
             "initialization": [{"destination": dst, "reader": reader, "recovery": recovered(initial_floor)}
                                for dst, readers in c.READERS.items() for reader in readers],
             "fixture": {"component_tensors": c.COUNTS, "tensor_count": 3858, "part_count": 4834,
                         "stored_bytes": c.PAYLOAD, "parts": parts, "source_files": chunks},
             "rows": rows, "summary": {"complete": True, "parity_pass": False, "observations_complete": True, "cells": cells,
                                       "r4_link_review_required": True, "r4_lifecycle_review_required": True}}
    chunks_sha = hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value, s, chunks_sha


class Contract(unittest.TestCase):
    def check(self, value, spec, chunks_sha):
        # Synthetic source anchor only; this is validator behavior, never H3/device evidence.
        with patch.object(c, "CHUNKS_SHA", chunks_sha):
            return c.check_benchmark(value, spec, UUID)

    def test_inventory_is_sixty_rotated_full_cells(self):
        rows = c.schedule()
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({tuple(r.values()) for r in rows}), 60)
        self.assertEqual([r["reader"] for r in rows if r["rep"] == 1 and r["destination"] == "cuda" and r["cold"]],
                         ["fastsafetensors", "safetensors-cuda", "safetensors-cpu-cuda", "tensorfs"])

    def test_slower_complete_result_is_valid_data(self):
        value, s, h = benchmark()
        self.assertEqual(self.check(value, s, h), {"measurement_complete": True, "parity_pass": False, "rows": 60})

    def test_missing_rep_rejected(self):
        value, s, h = benchmark(); value["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing measured row"): self.check(value, s, h)

    def test_wrong_uuid_rejected(self):
        value, s, h = benchmark(); value["environment"]["device_uuid"] = "f" * 32
        with self.assertRaisesRegex(ValueError, "UUID"): self.check(value, s, h)

    def test_partial_read_rejected(self):
        value, s, h = benchmark(); value["rows"][0]["bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "partial"): self.check(value, s, h)

    def test_false_cold_rejected(self):
        value, s, h = benchmark(); value["rows"][0]["cache_control"] = "untimed full read"
        with self.assertRaisesRegex(ValueError, "false cold"): self.check(value, s, h)

    def test_drifted_source_rejected(self):
        value, s, h = benchmark(); value["environment"]["harness_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source drift"): self.check(value, s, h)

    def test_missing_part_hash_rejected(self):
        value, s, h = benchmark(); value["fixture"]["parts"][0].pop("sha256")
        with self.assertRaisesRegex(ValueError, "hash missing"): self.check(value, s, h)

    def test_wrong_chunk_hash_rejected(self):
        value, s, h = benchmark(); value["fixture"]["source_files"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source chunks drift"): self.check(value, s, h)

    def test_partial_h2d_telemetry_rejected(self):
        value, s, h = benchmark()
        next(r for r in value["rows"] if r["destination"] == "cuda" and r["reader"] == "tensorfs")["metrics"]["bytes"] = 1
        with self.assertRaisesRegex(ValueError, "H2D"): self.check(value, s, h)

    def test_unsettled_release_rejected(self):
        value, s, h = benchmark(); value["rows"][0]["recovery"]["settled_free"] = 0
        with self.assertRaisesRegex(ValueError, "ownership"): self.check(value, s, h)

    def test_budget_and_unresolved_candidate_rejected(self):
        s = spec(); c.check_spec(s); s["limits"]["headroom"] = 0
        with self.assertRaisesRegex(ValueError, "budget drift"): c.check_spec(s)
        with self.assertRaises(ValueError): c.check_spec(c.template({}))
        for deadline in (1, 5000, True):
            with self.assertRaises(ValueError): c.deadline_budget(1000, deadline)
        self.assertEqual(c.deadline_budget(1000, 1100), 87)

    def test_capacity_never_pools_or_offloads(self):
        good = {"visible_count": 1, "device": "cuda:0", "name": "NVIDIA H100", "compute_capability": [9, 0],
                "gpu_uuid": UUID, "cuda_visible_devices": "GPU-" + "a" * 32, "same_work_filesystem": True,
                "host_available": c.HOST, "gpu_available": c.GPU, "disk_available": c.DISK_BASE + c.HEADROOM}
        c.check_capacity(good)
        for key, value in (("visible_count", 2), ("host_available", c.HOST - 1),
                           ("gpu_available", c.GPU - 1), ("disk_available", c.DISK_BASE + c.HEADROOM - 1)):
            bad = {**good, key: value}
            with self.assertRaises(ValueError): c.check_capacity(bad)

    def test_renderer_refuses_unresolved_before_opening_wheels(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = root / "review.json"; receipt.write_text("{}")
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({"pair_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                              "runtime": None, "native": None}))
            args = SimpleNamespace(candidate=candidate, pair_receipt=receipt, output=root / "payload.py")
            with self.assertRaisesRegex(ValueError, "unresolved runtime"): render.generate(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
