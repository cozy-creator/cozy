"""Prepare a real installed weightless package and retain its exact worker result."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from cozy.worker.v1 import worker_pb2 as pb
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_distribution
from cozy_runtime.internal.worker.package_prepare import prepare_package_set
from cozy_runtime.protocol import documents

root = Path(sys.argv[1])
project = root / "project"
project.mkdir()
(project / "pyproject.toml").write_text("""[project]
name="published-identity-fixture"
version="0.0.1"
requires-python=">=3.12,<3.13"
[project.entry-points."cozy.application"]
default="published_identity_fixture:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["published_identity_fixture.py"]
""")
(project / "published_identity_fixture.py").write_text("""import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    value: int = 3
class Result(msgspec.Struct):
    value: int
@app.job
def inspect(ctx: Context, payload: Request) -> Result:
    return Result(payload.value)
""")
subprocess.run(
    ["uv", "build", "--wheel", "--out-dir", str(root / "wheels"), str(project)],
    check=True,
)
(wheel,) = (root / "wheels").glob("*.whl")
subprocess.run(
    ["uv", "pip", "install", "--no-deps", "--python", sys.executable, str(wheel)],
    check=True,
)
locked = (
    "--index-url https://pypi.org/simple\n"
    "published-identity-fixture==0.0.1 --hash=sha256:"
    + hashlib.sha256(wheel.read_bytes()).hexdigest()
    + "\n"
).encode()
(root / "locked-requirements.txt").write_bytes(locked)
selection = documents.canonical_bytes(
    pb.DownloadDelegation(
        packages=[
            pb.DownloadPackageRef(
                package="proof/published-identity-fixture", release="0.0.1"
            )
        ]
    )
)
result = prepare_package_set(
    pb.PreparePackageSetRequest(
        install_root=str(root),
        download_delegation=selection,
        application="published_identity_fixture:app",
        locked_requirements=locked,
    ),
    artifact_cache=root / "artifact-cache",
    tensorfs_root=root / "tensorfs",
    install_root=root,
    python=Path(sys.executable),
    preinstalled_python=Path(sys.executable),
    describe=lambda _installed, distribution: package_interface.canonical_bytes(
        package_interface.build(discover_distribution(distribution))
    ),
    verified=lambda _digest, _length, _path: None,
    job_plan_root=root / "job-plans",
)
(root / "prepared.json").write_bytes(result.placement_set.placement_set_canonical_bytes)
