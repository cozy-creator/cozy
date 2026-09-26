"""Real private-wheel preparation and admission for a Creator-emitted invocation."""

from __future__ import annotations

import copy
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cozy_runtime.internal import (
    canonical,
    package_environment,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.config import read_config
from cozy_runtime.internal.worker.attempts import AttemptRefusal
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.package_prepare import prepare_local_package
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def run(*args: str) -> bytes:
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=sys.stderr)
    return result.stdout


def prepare(root: Path) -> None:
    python = Path(sys.executable)
    dependency_requirements = (root / "dependency-requirements.txt").read_bytes()
    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("""[project]
name="weightless"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime"]
[project.entry-points."cozy.application"]
default="private_environment_fixture:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["private_environment_fixture.py"]
""")
    source = """import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    size: int
class Result(msgspec.Struct):
    value: int
@app.entrypoint
def tile(ctx: Context, payload: Request) -> Result:
    ctx.raise_if_cancelled()
    return Result(payload.size)
"""
    (project / "private_environment_fixture.py").write_text(source)
    (project / "package.toml").write_text(
        '[application]\nobject="private_environment_fixture:app"\n'
    )
    interface = package_interface.canonical_bytes(static_interface.build(project))
    operation = "private-environment-proof"
    install = root / "installed"
    wheels = install / ".stage" / operation / "wheels"
    wheels.mkdir(parents=True)
    run("uv", "build", "--wheel", "--out-dir", str(wheels), str(project))
    (wheel,) = wheels.glob("*.whl")
    sdk_wheel = (root / "sdk-wheel.txt").read_text()
    if sdk_wheel:
        shutil.copyfile(sdk_wheel, wheels / Path(sdk_wheel).name)
    rows = [
        pb.LocalPackageFile(
            digest=documents.digest_of(path.read_bytes()),
            filename=path.name,
            path=str(path),
            length=path.stat().st_size,
        )
        for path in sorted(wheels.glob("*.whl"))
    ]
    rows.sort(key=lambda row: row.digest)
    installation_id = "private-environment-proof"

    def describe(installed, distribution):
        return run(
            str(installed.python),
            "-I",
            "-c",
            "import sys; from cozy_runtime.internal import package_interface; "
            "from cozy_runtime.internal.discovery import discover_distribution; "
            f"sys.stdout.buffer.write(package_interface.canonical_bytes(package_interface.build(discover_distribution({distribution!r}))))",
        )

    request = pb.PrepareLocalPackageRequest(
        operation_id=operation,
        package=pb.DevelopmentPackage(
            package="local/weightless",
            release="1.0.0",
            installation_id=installation_id,
        ),
        files=rows,
        dependency_requirements=dependency_requirements,
        install_root=str(install),
    )
    (root / "prepare-request.bin").write_bytes(request.SerializeToString())
    result = prepare_local_package(
        request,
        artifact_cache=root / "artifacts",
        install_root=install,
        python=python,
        describe=describe,
        verified=lambda *_: None,
        base=package_environment.observe_base(python),
        job_plan_root=root / "jobs",
    )
    (root / "prepared.json").write_bytes(
        result.placement_set.placement_set_canonical_bytes
    )
    (root / "installation.json").write_text(json.dumps({
        "ID": installation_id, "Package": "local/weightless", "Release": "1.0.0",
        "PackageInterface": base64.b64encode(interface).decode(),
        "Files": [{"Digest": documents.spell(row.digest), "Filename": row.filename,
                   "Kind": "project" if row.filename == wheel.name else "dependency",
                   "Path": row.path, "Length": row.length} for row in rows],
    }))
    print(json.dumps({"prepared": True, "installation": installation_id}))



def accept(root: Path) -> None:
    raw = (root / "prepared.json").read_bytes()
    desired = pb.DesiredPlacementSet(
        placement_set_digest=documents.digest_of(raw),
        placement_set_canonical_bytes=raw,
    )
    original = pb.AttemptOffer.FromString((root / "offer.bin").read_bytes())
    assert (
        original.invocation_spec_canonical_bytes
        == (root / "invocation.json").read_bytes()
    )
    config = read_config({**os.environ, "COZY_HOME": str(root / "home")})
    worker = Worker(
        config,
        WorkerOptions(
            root=root / "worker",
            python=sys.executable,
            install_root=root / "installed",
            artifact_cache=root / "artifacts",
            tensorfs_root=root / "tensorfs",
            devices="",
            accelerator_backend="none",
        ),
        InMemoryControlHost(),
    )
    try:
        preparation = worker.prepare_local_package(
            pb.PrepareLocalPackageRequest.FromString(
                (root / "prepare-request.bin").read_bytes()
            )
        )
        assert preparation.placement_set.placement_set_canonical_bytes == raw
        worker._apply_desired_state(
            pb.DesiredWorkerState(
                revision=1,
                placement_set=desired,
                wire_minor=62,
            )
        )
        worker.converge_placement(1)
        assert worker.prepared_models, (worker.latched, worker.failed_bindings)
        assert worker.bindings
        declared = next(iter(worker.bindings.values()))
        assert declared.development is False and declared.installation_id
        # The positive arm is the exact Go-emitted offer, including its grant.
        attempt = worker.engine.offer(original)
        worker._bind_attempt_slot(attempt)
        accepted = worker.engine.prepare_and_accept(attempt)
        assert accepted.invocation_spec_digest == original.invocation_spec_digest
        assert attempt.state == "accepted", attempt.state
        refused = []
        # Reissue fresh owner-authored negative offers. Their canonical document
        # and grant subject both name the altered request; no identity is forced.
        original_spec = documents.read(
            original.invocation_spec_canonical_bytes, pb.InvocationSpec
        )
        for index, value in enumerate((None, "other-installation"), 1):
            spec = copy.deepcopy(original_spec)
            if value is None:
                spec.pop("installation_id", None)
            else:
                spec["installation_id"] = value
            body = canonical.write(spec)
            offer = pb.AttemptOffer()
            offer.CopyFrom(original)
            offer.request_id += f"-negative-{index}"
            offer.invocation_spec_canonical_bytes = body
            offer.invocation_spec_digest = documents.digest_of(body)
            offer.grant.invocation_spec_digest = offer.invocation_spec_digest
            negative = worker.engine.offer(offer)
            worker._bind_attempt_slot(negative)
            try:
                worker.engine.prepare_and_accept(negative)
            except AttemptRefusal as error:
                assert error.code == "environment_mismatch", str(error)
                refused.append(error.code)
            else:
                raise AssertionError("mismatched installation admitted")
        result = {
            "accepted": True,
            "refused": refused,
            "installation_id": declared.installation_id,
        }
        (root / "admission.json").write_text(json.dumps(result))
        print(json.dumps(result))
    finally:
        worker.shutdown()


if __name__ == "__main__":
    mode, directory = sys.argv[1:]
    {"prepare": prepare, "accept": accept}[mode](Path(directory).resolve())
