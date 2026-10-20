"""Stdlib-only controls for h3_load; these tests do not execute a benchmark."""

from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("h3_load", Path(__file__).with_name("h3_load.py"))
h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3)


def observations(destination):
    return [{"kind": "disk" if destination == "cpu" else "h2d", "bytes": 100,
             "ns": 20, "ceiling": 10, "link": "Gen4 x16" if destination == "cuda" else ""}]


def complete_rows():
    return [{"rep": rep, "destination": dst, "reader": reader, "cold": cold,
             "seconds": 1 if reader == "tensorfs" else 2, "verified": True, "bytes": 100,
             "metrics": h3.copy_metrics(observations(dst), dst, 100) if reader == "tensorfs" else None}
            for rep, dst, reader, cold in h3.schedule()]


class PlanAndCapacity(unittest.TestCase):
    def test_source_hash_and_fixed_h3_identity(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "header").write_bytes(b"small metadata fixture; not a native model")
            plan = {"source_label": h3.SOURCE_LABEL, "legacy_manifest_sha256": h3.LEGACY_MANIFEST,
                    "header": "header", "header_sha256": h3.digest(root / "header"), "cache_dir": ".",
                    "tensor_count": 3858, "stored_bytes": 100}
            path = root / "source.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(h3.source_plan(path)["header"], str(root / "header"))
            for key, bad in (("tensor_count", 3857), ("legacy_manifest_sha256", "0" * 64),
                             ("header_sha256", "0" * 64), ("source_label", "different-model")):
                with self.subTest(key=key):
                    path.write_text(json.dumps({**plan, key: bad}))
                    with self.assertRaises(ValueError):
                        h3.source_plan(path)

    def test_all_components_and_exact_count_bytes_are_required(self):
        parts = [{"component": c, "key": "w", "role": "", "length": 20} for c in h3.COMPONENTS]
        counts = dict.fromkeys(h3.COMPONENTS, 1)
        counts["transformer"] = 3854
        h3.validate_geometry(parts, counts, 100)
        for broken, nbytes in (({k: v for k, v in counts.items() if k != "transformer_ref"}, 100),
                               ({**counts, "other": 1}, 100), ({**counts, "transformer": 3853}, 100),
                               (counts, 99)):
            with self.subTest(counts=broken, nbytes=nbytes), self.assertRaises(ValueError):
                h3.validate_geometry(parts, broken, nbytes)

    def test_duplicate_part_identity_is_not_extra_progress(self):
        counts = dict.fromkeys(h3.COMPONENTS, 1)
        counts["transformer"] = 3854
        parts = [{"component": "transformer", "key": "w", "role": "", "length": 50}] * 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            h3.validate_geometry(parts, counts, 100)

    def test_capacity_cannot_fall_back_to_cpu_window_or_offload(self):
        payload, part = 70 * h3.GIB, h3.GIB
        required = h3.capacity(payload, part, 200 * h3.GIB, 80 * h3.GIB, 200 * h3.GIB, 100)
        self.assertGreater(required["host_required"], 2 * payload)
        for host, gpu, disk in ((80, 80, 200), (200, 24, 200), (200, 80, 100)):
            with self.subTest(host=host, gpu=gpu, disk=disk), self.assertRaises(ValueError):
                h3.capacity(payload, part, host * h3.GIB, gpu * h3.GIB, disk * h3.GIB, 100)


class Completeness(unittest.TestCase):
    def test_five_rotated_full_repetitions_for_every_control(self):
        rows = list(h3.schedule())
        self.assertEqual(len(rows), 60)
        for dst, readers in h3.READERS.items():
            for rep in range(5):
                current = [(reader, cold) for n, d, reader, cold in rows if n == rep and d == dst]
                self.assertEqual(current[0][0], readers[rep % len(readers)])
                self.assertEqual(set(current), {(reader, cold) for reader in readers for cold in (True, False)})

    def test_full_proof_is_complete_and_fastest_control_is_used(self):
        summary = h3.summarize(complete_rows(), 100)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["parity_pass"])
        self.assertTrue(summary["observations_complete"])
        self.assertEqual(set(summary["cells"]), {"cpu cold", "cpu warm", "cuda cold", "cuda warm"})
        self.assertIsNone(summary["link_threshold"])
        self.assertTrue(summary["r4_lifecycle_review_required"])

    def test_missing_duplicate_extra_or_corrupt_row_cannot_pass(self):
        rows = complete_rows()
        damaged = [rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]]
        for change in ({"bytes": 99}, {"verified": False}, {"seconds": 0}):
            damaged.append([{**row, **change} if i == 0 else row for i, row in enumerate(rows)])
        for candidate in damaged:
            with self.subTest(candidate=candidate[0]):
                self.assertFalse(h3.summarize(candidate, 100)["parity_pass"])

    def test_faster_control_failure_is_not_hidden_by_slower_controls(self):
        rows = complete_rows()
        for row in rows:
            if row["reader"] == "safetensors-cuda":
                row["seconds"] = 0.5
        summary = h3.summarize(rows, 100)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["parity_pass"])

    def test_missing_engine_observations_do_not_qualify(self):
        rows = complete_rows()
        first_gpu = next(r for r in rows if r["reader"] == "tensorfs" and r["destination"] == "cuda")
        first_gpu["metrics"] = h3.copy_metrics([], "cuda", 100)
        self.assertFalse(h3.summarize(rows, 100)["observations_complete"])
        record = observations("cuda")[0]
        for change in ({"ceiling": 0}, {"ns": 0}, {"link": ""}, {"bytes": 1}):
            self.assertFalse(h3.copy_metrics([{**record, **change}], "cuda", 100)["measured"])

    def test_no_fake_disk_telemetry_required_for_raw_cpu_reads(self):
        rows = complete_rows()
        for row in rows:
            if row["reader"] == "tensorfs" and row["destination"] == "cpu":
                row["metrics"] = h3.copy_metrics([], "cpu", 100)
        self.assertTrue(h3.summarize(rows, 100)["observations_complete"])

    def test_summary_rechecks_full_telemetry_coverage(self):
        rows = complete_rows()
        row = next(r for r in rows if r["reader"] == "tensorfs" and r["destination"] == "cuda")
        row["metrics"].update(measured=True, bytes=1)
        summary = h3.summarize(rows, 100)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["observations_complete"])


class CleanupAndNamespace(unittest.TestCase):
    def test_exact_one_gpu_and_cuda_zero(self):
        h3.device_namespace(1, "cuda:0")
        for count, device in ((0, "cuda:0"), (2, "cuda:0"), (1, "cuda:1"), (1, "cpu")):
            with self.subTest(count=count, device=device), self.assertRaises(ValueError):
                h3.device_namespace(count, device)

    def test_delayed_native_release_recovers_before_next_read(self):
        now, reads, waits = [0.0], iter((10, 90, 100)), []
        def pause(seconds):
            waits.append(seconds)
            now[0] += seconds
        result = h3.recover_memory(lambda: next(reads), 100, clock=lambda: now[0], pause=pause)
        self.assertEqual(result["settled_free"], 100)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(len(waits), 2)
        self.assertEqual(result["floor"], 100)

    def test_timeout_cannot_ratchet_down_the_fixed_baseline(self):
        now = [0.0]
        def pause(_):
            now[0] += 1
        with self.assertRaisesRegex(ValueError, "fixed floor=100"):
            h3.recover_memory(lambda: 99, 100, clock=lambda: now[0], pause=pause)
        self.assertEqual(now[0], h3.RECOVERY_SECONDS)


class FakeArray:
    dtype = "U8"
    shape = (4,)
    nbytes = 4

    def __init__(self, data=b"data", device=(1, 0)):
        self.data, self.device = data, device

    def tobytes(self):
        return self.data

    def __dlpack_device__(self):
        return self.device


class OutputValidation(unittest.TestCase):
    def fixture(self):
        fixture = object.__new__(h3.Fixture)
        fixture.payload = 4
        fixture.parts = [{"control_key": "transformer/part000000", "length": 4,
                          "sha256": hashlib.sha256(b"data").hexdigest()}]
        return fixture

    def test_entire_output_hash_is_checked_not_only_length(self):
        fixture = self.fixture()
        key = fixture.parts[0]["control_key"]
        self.assertEqual(fixture.validate({key: FakeArray()}, "cpu", "tensorfs", None, 0), 4)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fixture.validate({key: FakeArray(b"nope")}, "cpu", "tensorfs", None, 0)

    def test_missing_part_and_wrong_actual_device_fail(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture.validate({}, "cpu", "tensorfs", None, 0)
        with self.assertRaisesRegex(ValueError, "device"):
            fixture.validate({fixture.parts[0]["control_key"]: FakeArray()}, "cuda", "tensorfs", None, 0)

    def test_cpu_control_clones_all_five_components(self):
        clones = []
        class Tensor:
            def clone(self):
                clones.append(self)
                return object()
        class File:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def get_tensor(self, _):
                return Tensor()
        class SafeTensors:
            @staticmethod
            def safe_open(*_, **kwargs):
                self.assertEqual(kwargs["device"], "cpu")
                return File()
        fixture = object.__new__(h3.Fixture)
        fixture.parts = [{"component": c, "control_key": c + "/part000000"} for c in h3.COMPONENTS]
        fixture.st_files = list(h3.COMPONENTS)
        with patch.object(h3.importlib, "import_module", return_value=SafeTensors), ExitStack() as stack:
            arrays = fixture.read("cpu", "safetensors-copy", stack, None, "cuda:0")
        self.assertEqual(len(clones), 5)
        self.assertEqual(len(arrays), 5)
        self.assertTrue(all(array not in clones for array in arrays.values()))


if __name__ == "__main__":
    unittest.main()
