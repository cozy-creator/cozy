"""Public wheel selection and immutable transport proof; no network or worker mutation."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from packaging.markers import default_environment

MODULE = Path(__file__).parents[3] / "internal/cli/runtime_update_transport.py"


class PublishedRuntimeUpdates(unittest.TestCase):
    def setUp(self):
        self.module = runpy.run_path(str(MODULE))
        self.api = self.module["resolve"].__globals__
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.observed = {
            "runtime": {"distribution": "0.18.20", "supports_guarded_restart": True},
            "tensorfs": "0.3.49", "python": "3.12.12", "torch": "2.14.0",
            "tags": ["cp312-cp312-manylinux_2_28_x86_64", "py3-none-any"],
            "markers": {**default_environment(), "python_version": "3.12", "python_full_version": "3.12.12"},
            "updater": True, "durable_updates": True,
        }
        self.documents = {name: {"releases": {}} for name in ("cozy-runtime", "tensorfs")}
        self.data = {}
        self.fetched = []
        self.api["fetch"] = self.fetch
        self.runtime = self.release("cozy-runtime", "0.18.21", requirements=["tensorfs>=0.3.50,<0.4"])
        self.tensorfs = self.release("tensorfs", "0.3.51")

    def fetch(self, url, limit):
        self.fetched.append(url)
        if url.startswith("https://pypi.org/pypi/"):
            return json.dumps(self.documents[url.split("/")[-2]]).encode()
        return self.data[url]

    def release(self, name, version, *, tag="cp312-cp312-manylinux_2_28_x86_64", requirements=(), requires_python=">=3.12", yanked=False, metadata_name=None):
        filename = f"{name.replace('-', '_')}-{version}-{tag}.whl"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as wheel:
            metadata = f"Metadata-Version: 2.3\nName: {metadata_name or name}\nVersion: {version}\nRequires-Python: {requires_python}\n"
            metadata += "".join(f"Requires-Dist: {value}\n" for value in requirements)
            wheel.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA", metadata)
        data = buffer.getvalue()
        url = "https://files.pythonhosted.org/packages/fixture/" + filename
        row = {"filename": filename, "url": url, "size": len(data), "digests": {"sha256": hashlib.sha256(data).hexdigest()}, "yanked": yanked, "packagetype": "bdist_wheel", "requires_python": requires_python}
        self.documents[name]["releases"].setdefault(version, []).append(row)
        self.data[url] = data
        return row

    def resolve(self):
        return self.api["resolve"](self.observed, self.directory)

    def test_latest_native_pair_obeys_runtime_requirements_without_an_image(self):
        self.release("tensorfs", "0.4.0")
        self.release("tensorfs", "0.3.99", yanked=True)
        self.release("tensorfs", "0.3.98", requires_python=">=3.13")
        self.release("tensorfs", "0.3.97", tag="cp312-cp312-manylinux_2_28_aarch64")
        self.release("tensorfs", "0.3.96", tag="py3-none-any")
        self.release("tensorfs", "0.3.95rc1")
        target, wheels = self.resolve()
        self.assertEqual([row["version"] for row in wheels], ["0.18.21", "0.3.51"])
        self.assertEqual(set(target), {"runtime_update"})
        for row in wheels:
            published = target["runtime_update"]["runtime" if row["distribution"] == "cozy-runtime" else "tensorfs"]
            content = Path(row["path"]).read_bytes()
            self.assertEqual(len(content), published["length"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), row["sha256"])
        self.assertEqual(len([url for url in self.fetched if url.endswith(".whl")]), 2)

    def test_requires_dist_markers_use_worker_not_client(self):
        self.release("cozy-runtime", "0.18.22", requirements=[
            'tensorfs<0.3.51; python_version == "3.12"',
            'tensorfs>=0.4; python_version == "3.13"',
            'tensorfs>=0.3.50; extra == "model-execution"',
        ])
        self.release("tensorfs", "0.3.50")
        _, wheels = self.resolve()
        self.assertEqual(wheels[1]["version"], "0.3.50")

    def test_refuses_downgrade_or_missing_compatible_native_release(self):
        self.observed["tensorfs"] = "0.3.52"
        with self.assertRaisesRegex(ValueError, "without downgrading 0.3.52"):
            self.resolve()
        self.observed["tensorfs"] = "0.3.49"
        self.observed["tags"] = ["cp313-cp313-manylinux_2_28_x86_64"]
        with self.assertRaisesRegex(ValueError, "No published native cozy-runtime"):
            self.resolve()

    def test_tampered_download_never_becomes_selection(self):
        self.data[self.runtime["url"]] += b"changed"
        with self.assertRaisesRegex(ValueError, "digest verification"):
            self.resolve()
        self.assertFalse((self.directory / self.runtime["filename"]).exists())

    def test_metadata_identity_and_reverse_dependency_must_match(self):
        self.release("cozy-runtime", "0.18.22", metadata_name="other-project")
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            self.resolve()
        del self.documents["cozy-runtime"]["releases"]["0.18.22"]
        self.release("tensorfs", "0.3.52", requirements=["cozy-runtime>=0.19"])
        with self.assertRaisesRegex(ValueError, "selected Runtime .* incompatible"):
            self.resolve()

    def test_plan_records_verified_pair_and_unchanged_readback(self):
        self.api["inspect"] = lambda arguments: self.observed
        request = {"ssh_arguments": [], "action": "plan", "directory": str(self.directory)}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
            plan = self.api["main"]()
        self.assertFalse(plan["unchanged"])
        self.observed["runtime"]["distribution"] = "0.18.21"
        self.observed["tensorfs"] = "0.3.51"
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
            current = self.api["main"]()
        self.assertTrue(current["unchanged"])
        self.assertEqual(plan["target"], current["target"])

    def test_reconcile_uses_recorded_pair_without_reselecting_new_release(self):
        _, wheels = self.resolve()
        expected = [row["sha256"] for row in wheels]
        stage = "a" * 32
        calls = []
        self.api["ssh"] = lambda arguments, command: calls.append(command) or json.dumps({"operation": stage, "state": "queued"})
        self.api["fetch"] = lambda *args: self.fail("recovery must not resolve another release")
        for action in ("resume", "status"):
            request = {"ssh_arguments": [], "action": action, "stage": stage, "selection": {"wheels": wheels}}
            with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
                result = self.api["main"]()
            self.assertEqual(result["update"]["operation"], stage)
        self.assertEqual(calls[0], "/opt/cozy/python/bin/python3 /opt/cozy/dev/update.py apply " + stage + " " + " ".join(expected))
        self.assertEqual(calls[1], "/opt/cozy/python/bin/python3 /opt/cozy/dev/update.py status " + stage)

    def test_probe_does_not_rely_on_ssh_path(self):
        calls = []
        self.api["ssh"] = lambda arguments, command: calls.append(command) or '{}'
        self.api["inspect"]([])
        self.assertTrue(calls[0].startswith("/opt/cozy/python/bin/python3 -I -c "))
        self.assertIn("sys.executable", self.api["PROBE"])
        self.assertNotIn("['cozy-runtime'", self.api["PROBE"])
        self.assertNotIn("['python3'", self.api["PROBE"])

    def test_index_cannot_redirect_wheel_to_another_host_or_name(self):
        self.runtime["url"] = "https://untrusted.example/" + self.runtime["filename"]
        with self.assertRaisesRegex(ValueError, "exact artifact"):
            self.resolve()

    def test_runtime_wire_is_read_from_the_selected_wheel(self):
        for body, want in [
            ("WIRE_MINOR = 61\nMIN_COMPATIBLE_WIRE_MINOR = 61\n", {"wire_minor": 61, "minimum_wire_minor": 61}),
            ("WIRE_MINOR = 61\n", None), (None, None),
        ]:
            path = self.directory / f"wire-{len(str(body))}.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                if body is not None:
                    wheel.writestr("cozy/worker/v1/wire_version.py", '"""Generated."""\n\n' + body)
            self.assertEqual(self.api["runtime_wire"]({"path": path}), want)

if __name__ == "__main__":
    unittest.main()
