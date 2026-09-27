"""Public wheel selection and immutable transport proof; no network or worker mutation."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from packaging.markers import default_environment

MODULE = Path(__file__).parents[3] / "internal/cli/runtime_update_transport.py"
PROBE = MODULE.with_name("runtime_update_probe.py").read_text()


class PublishedRuntimeUpdates(unittest.TestCase):
    def setUp(self):
        self.module = runpy.run_path(str(MODULE))
        self.api = self.module["resolve"].__globals__
        self.discover_python = self.api["worker_python"]
        self.api["worker_python"] = lambda arguments: "/opt/cozy/python/bin/python3"
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
        self.api["inspect"] = lambda arguments, probe, python=None: self.observed
        request = {"ssh_arguments": [], "probe": PROBE, "action": "plan", "directory": str(self.directory)}
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
            request = {"ssh_arguments": [], "probe": PROBE, "action": action, "stage": stage, "selection": {"wheels": wheels}}
            with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
                result = self.api["main"]()
            self.assertEqual(result["update"]["operation"], stage)
        self.assertEqual(calls[0], "/opt/cozy/python/bin/python3 /opt/cozy/dev/update.py status " + stage)
        self.assertEqual(calls[1], "/opt/cozy/python/bin/python3 /opt/cozy/dev/update.py apply " + stage + " " + " ".join(expected))
        self.assertEqual(calls[2], calls[0])

    def test_missing_update_retransfers_only_the_frozen_verified_pair(self):
        _, wheels = self.resolve()
        stage = "b" * 32
        calls = []
        self.api["inspect"] = lambda arguments, probe, python=None: self.observed
        self.api["fetch"] = lambda *args: self.fail("recovery must not select another release")
        def remote(arguments, command):
            calls.append(command)
            return json.dumps({"operation": stage, "state": "missing" if " status " in command else "queued"})
        self.api["ssh"] = remote
        request = {"ssh_arguments": [], "probe": PROBE, "sftp_arguments": [], "host": "fixture",
                   "action": "resume", "stage": stage, "directory": str(self.directory),
                   "selection": {"wheels": wheels}}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))), patch.object(subprocess, "run") as transfer:
            result = self.api["main"]()
        self.assertEqual(result["update"]["state"], "queued")
        transfer.assert_called_once()
        batch = (self.directory / "transfer.batch").read_text()
        for row in wheels:
            self.assertIn(str(row["path"]), batch)
            self.assertIn(stage + "/" + row["file"], batch)
            self.assertIn(row["sha256"], calls[-1])
        # A failed transfer cannot make changed local bytes admissible on retry.
        Path(wheels[0]["path"]).write_bytes(b"changed")
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))), patch.object(subprocess, "run") as transfer:
            with self.assertRaisesRegex(ValueError, "changed after verification"):
                self.api["main"]()
            transfer.assert_not_called()

    def test_resume_never_retransfers_when_status_names_another_operation(self):
        _, wheels = self.resolve()
        self.api["ssh"] = lambda arguments, command: json.dumps({"operation": "b" * 32, "state": "missing"})
        request = {"ssh_arguments": [], "probe": PROBE, "action": "resume", "stage": "a" * 32,
                   "selection": {"wheels": wheels}}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
            with self.assertRaisesRegex(ValueError, "recorded operation"):
                self.api["main"]()

    def local(self, **kwargs):
        row = self.release(kwargs.pop("name", "cozy-runtime"), kwargs.pop("version", "0.18.24+dev.h123"), **kwargs)
        directory = self.directory / "frozen"
        directory.mkdir(exist_ok=True)
        path = directory / row["filename"]
        path.write_bytes(self.data[row["url"]])
        return {"filename": row["filename"], "path": str(path),
                "digest": "sha256:" + row["digests"]["sha256"], "length": row["size"]}

    def test_local_candidate_keeps_exact_hash_and_public_tensorfs_dependency_selection(self):
        local = self.local(requirements=["tensorfs>=0.3.51,<0.4"])
        self.release("cozy-runtime", "9.0.0")
        target, wheels = self.api["resolve"](self.observed, self.directory, local)
        self.assertEqual([row["version"] for row in wheels], ["0.18.24+dev.h123", "0.3.51"])
        self.assertEqual(wheels[0]["sha256"], local["digest"].removeprefix("sha256:"))
        self.assertEqual(target["runtime_update"]["runtime"]["path"], local["path"])
        self.assertNotIn("https://pypi.org/pypi/cozy-runtime/json", self.fetched)
        Path(local["path"]).unlink()
        # Recovery depends on the verified operation copy, not the source path.
        request = {"ssh_arguments": [], "probe": PROBE, "action": "resume", "stage": "a" * 32,
                   "selection": {"wheels": wheels}}
        self.api["fetch"] = lambda *args: self.fail("resume reselected a public wheel")
        self.api["ssh"] = lambda arguments, command: json.dumps({"operation": "a" * 32, "state": "queued"})
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
            self.assertEqual(self.api["main"]()["update"]["state"], "queued")

    def test_local_pair_keeps_installed_dev_tensorfs_without_network_and_recovers(self):
        runtime = self.local(requirements=["tensorfs>=0.3.50,<0.4"])
        self.observed["tensorfs"] = "0.3.54+dev.g9d02fc3"
        tensorfs = self.local(name="tensorfs", version=self.observed["tensorfs"])
        self.api["fetch"] = lambda *args: self.fail("explicit pair must not contact PyPI")
        target, wheels = self.api["resolve"](self.observed, self.directory, runtime, tensorfs)
        self.assertEqual(wheels[1]["version"], self.observed["tensorfs"])
        self.assertEqual(wheels[1]["sha256"], tensorfs["digest"].removeprefix("sha256:"))
        self.assertEqual(target["runtime_update"]["tensorfs"]["path"], tensorfs["path"])
        for candidate in (runtime, tensorfs):
            Path(candidate["path"]).unlink()
        stage = "c" * 32
        self.api["ssh"] = lambda arguments, command: json.dumps({"operation": stage, "state": "queued"})
        for action in ("resume", "status"):
            request = {"ssh_arguments": [], "probe": PROBE, "action": action, "stage": stage,
                       "selection": {"wheels": wheels}}
            with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
                self.assertEqual(self.api["main"]()["update"]["operation"], stage)

    def test_local_tensorfs_refuses_wrong_identity_platform_and_constraints(self):
        runtime = self.local(requirements=["tensorfs>=0.3.50,<0.4"])
        for kwargs, error in [
            ({"name": "cozy-runtime"}, "must be tensorfs"),
            ({"metadata_name": "other-project"}, "metadata does not match"),
            ({"tag": "py3-none-any"}, "native wheel"),
            ({"tag": "cp312-cp312-manylinux_2_28_aarch64"}, "platform"),
            ({"version": "0.3.48"}, "without downgrading"),
            ({"version": "0.4.0"}, "incompatible"),
            ({"requirements": ["cozy-runtime>=9"]}, "incompatible"),
        ]:
            with self.subTest(kwargs=kwargs):
                tensorfs = self.local(**{"name": "tensorfs", "version": "0.3.54+dev.g9d02fc3", **kwargs})
                with self.assertRaisesRegex(ValueError, error):
                    self.api["resolve"](self.observed, self.directory, runtime, tensorfs)
        with self.assertRaisesRegex(ValueError, "requires a local Runtime"):
            self.api["resolve"](self.observed, self.directory, None, tensorfs)

    def test_local_metadata_platform_and_version_refusals(self):
        for kwargs, error in [
            ({"tag": "py3-none-any"}, "native wheel"),
            ({"tag": "cp312-cp312-manylinux_2_28_aarch64"}, "platform"),
            ({"metadata_name": "other-project"}, "metadata does not match"),
            ({"requires_python": ">=3.13"}, "different worker Python"),
            ({"version": "0.18.19"}, "without downgrading its public release"),
        ]:
            with self.subTest(kwargs=kwargs):
                local = self.local(**kwargs)
                with self.assertRaisesRegex(ValueError, error):
                    self.api["resolve"](self.observed, self.directory, local)

    def test_local_tampering_is_refused_before_transfer(self):
        local = self.local()
        Path(local["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest verification"):
            self.api["resolve"](self.observed, self.directory, local)
        local = self.local()
        _, wheels = self.api["resolve"](self.observed, self.directory, local)
        Path(wheels[0]["path"]).write_bytes(b"changed after plan")
        self.api["inspect"] = lambda arguments, probe, python=None: self.observed
        request = {"ssh_arguments": [], "probe": PROBE, "action": "apply", "directory": str(self.directory),
                   "stage": "a" * 32, "selection": {"wheels": wheels}}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
            with self.assertRaisesRegex(ValueError, "changed after verification"):
                self.api["main"]()

    def test_python_selection_supports_both_fixed_image_layouts(self):
        previous = self.directory / "previous-python"
        current = self.directory / "current-python"
        current.write_text("#!/bin/sh\nexit 0\n")
        current.chmod(0o700)
        self.api["WORKER_PYTHONS"] = (str(previous), str(current))
        self.api["ssh"] = lambda arguments, command: subprocess.check_output(
            ["/bin/sh", "-c", command], text=True
        )
        self.assertEqual(self.discover_python([]), str(current))
        previous.write_text("#!/bin/sh\nexit 0\n")
        previous.chmod(0o700)
        self.assertEqual(self.discover_python([]), str(previous))
        previous.chmod(0o600)
        self.assertEqual(self.discover_python([]), str(current))

    def test_python_selection_rejects_an_unexpected_response(self):
        self.api["ssh"] = lambda arguments, command: "/tmp/untrusted-python"
        with self.assertRaisesRegex(ValueError, "supported SDK interpreter"):
            self.discover_python([])

    def test_probe_does_not_rely_on_ssh_path(self):
        calls = []
        self.api["ssh"] = lambda arguments, command: calls.append(command) or '{}'
        self.api["inspect"]([], PROBE)
        self.assertTrue(calls[0].startswith("/opt/cozy/python/bin/python3 -I -c "))
        self.assertIn("sys.executable", PROBE)
        self.assertNotIn("['cozy-runtime'", PROBE)
        self.assertNotIn("['python3'", PROBE)

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
