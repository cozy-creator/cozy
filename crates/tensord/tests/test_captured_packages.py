"""Real captured uv environments; no implementation doubles. The helper captures a package's own
closure; the machine's pair, client, check, description and record are the machine's
(`environments.rs`, tested there and through the Rust installer below). Inference in these
environments runs on the machine's executors (`machine_surface.rs`)."""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

from tensord_client import layout
from tensord_client.captured_packages import capture
from tensord_client.packages import PackageError
from test_package_bridge import FIXTURE, ROOT


def versions(python: str) -> dict[str, str]:
    """Every distribution an environment holds, as uv lists it."""
    listed = subprocess.check_output(["uv", "pip", "list", "--python", python, "--format", "json"])
    return {row["name"]: row["version"] for row in json.loads(listed)}


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-captured-package")
    source = root / "source"
    shutil.copytree(FIXTURE, source)
    subprocess.run(["uv", "lock", "--project", str(source), "--python", "3.12"], check=True)
    frozen = root / "frozen"
    frozen.mkdir()
    result = capture(root=frozen, project=source, wheels=[], requirements=None, distribution="tensord-cpu-classifier",
                     release="0.1.0", python_requires=">=3.12", python_version="3.12", python="3.12")
    yield root, source, result


def locked_sdk(source):
    """The Runtime the capture's lock selected."""
    lock = tomllib.loads((source / "uv.lock").read_text())
    return next(row["version"] for row in lock["package"] if row["name"] == "cozy-runtime")


def test_frozen_source_keeps_locked_sdk(captured):
    root, source, result = captured
    assert (result.package, result.version, result.application) == ("tensord-cpu-classifier", "0.1.0", "cpu_classifier:app")
    assert versions(result.python)["cozy-runtime"] == locked_sdk(source)
    subprocess.run([result.python, "-I", "-c", "import cpu_classifier"], check=True)


def test_real_wheel_roster_and_hashed_requirements_preserve_dependency_versions(captured):
    root, source, frozen = captured
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "wheels"), str(source)], check=True)
    wheel = next((root / "wheels").glob("*.whl"))
    requirements = root / "requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(source), "--output-file", str(requirements)], check=True)
    carried = root / "carried"
    carried.mkdir()
    result = capture(root=carried, project=None, wheels=[wheel], requirements=requirements,
                     distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12",
                     python_version="3.12", python="3.12")
    assert versions(result.python) == versions(frozen.python)


def test_incoherent_source_lock_overlay_and_python_mismatch_refuse_only_installation(captured, tmp_path):
    root, source, frozen = captured
    carried = source / "pyproject.toml"  # any carried file: a frozen capture takes none
    with pytest.raises(PackageError, match="cannot overlay"):
        capture(root=tmp_path, project=source, wheels=[carried], requirements=None, distribution="tensord-cpu-classifier",
                release="0.1.0", python_requires="", python_version="", python="3.12")
    with pytest.raises(PackageError, match="does not match"):
        capture(root=tmp_path, project=source, wheels=[], requirements=None, distribution="tensord-cpu-classifier",
                release="0.1.0", python_requires="", python_version="3.11", python="3.12")


def test_rust_tensorfs_materialization_and_trusted_installer(captured, tmp_path):
    """The machine's own path, end to end: the helper captures, the machine adds its client,
    checks, describes and publishes the typed record (`install-capture`)."""
    root, source, frozen = captured
    archive = tmp_path / "source.tar"
    subprocess.run(["tar", "-cf", str(archive), "-C", str(source), "pyproject.toml", "uv.lock", "package.toml", "cpu_classifier"], check=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path / "client"), str(ROOT)], check=True)
    client = next((tmp_path / "client").glob("*.whl"))
    output = tmp_path / "rust-owner-install"
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT.parents[1] / "target"))
    subprocess.run([str(target / "debug/install-capture"), "--archive", str(archive), "--output", str(output),
        "--helper-python", sys.executable, "--client-wheel", str(client), "--python", "3.12",
        "--package", "local/tensord-cpu-classifier", "--release", "0.1.0"], check=True)
    generation = json.loads((output / "prepared-generation.json").read_bytes())
    assert json.loads((output / "package-interface.json").read_bytes()) == generation["interface"]
    assert {row["name"]: row["version"] for row in generation["dependencies"]}["cozy-runtime"] == locked_sdk(source)
    assert generation["runtime_build"] == "" and generation["sdk_fallback"] == ""
    subprocess.run([generation["python"], "-I", "-c", "import cpu_classifier, tensord_client.runtime_describe"], check=True)


def test_installer_only_helper_captures_with_its_own_interpreter(captured, tmp_path):
    """The machine's helper holds the client and packaging alone, never the Runtime."""
    root, source, frozen = captured
    helper = tmp_path / "installer-only-helper"
    subprocess.run(["uv", "venv", "--python", "3.14", str(helper)], check=True)
    python = layout.interpreter(helper)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path / "client"), str(ROOT)], check=True)
    client = next((tmp_path / "client").glob("*.whl"))
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(client), "packaging"], check=True)
    subprocess.run([str(python), "-c", "import importlib.util;assert importlib.util.find_spec('cozy_runtime') is None"], check=True)
    generation = tmp_path / "generation"
    generation.mkdir()
    process = subprocess.run([str(python), "-m", "tensord_client.installer", "capture", "--root", str(generation),
        "--project", str(source), "--distribution", "tensord-cpu-classifier", "--release", "0.1.0", "--python", "3.12"],
        check=True, capture_output=True)
    result = json.loads(process.stdout)
    assert result["kind"] == "captured" and result["python"] == str(layout.interpreter(generation / "env"))
    assert versions(result["python"])["cozy-runtime"] == locked_sdk(source)


def pypi_pair(directory: Path) -> list[Path]:
    """PyPI's newest Runtime and the TensorFS it accepts, wheels for this platform: a machine's pair."""
    download = ["uvx", "--from", "pip", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "--python-version", "3.12",
                "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_17_x86_64", "-d", str(directory)]
    subprocess.run([*download, "cozy-runtime"], check=True)
    runtime = next(directory.glob("cozy_runtime-*.whl"))
    with zipfile.ZipFile(runtime) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))).decode()
    accepted = next(line[15:].split(";")[0].replace(" ", "") for line in metadata.splitlines()
                    if re.match(r"Requires-Dist: tensorfs[^A-Za-z0-9._-]", line))
    subprocess.run([*download, accepted], check=True)
    return [runtime, next(directory.glob("tensorfs-*.whl"))]


def test_a_captured_lock_leaves_the_pair_to_the_machine_without_fetching_its_own(captured, tmp_path):
    """The machine's pair is installed once, by the machine: a captured lock's own Runtime and
    TensorFS, made unreachable here, are never fetched to be replaced; restored, they are."""
    root, source, frozen = captured
    pair = pypi_pair(tmp_path / "pair")
    project = tmp_path / "source"
    shutil.copytree(source, project)
    lock = (project / "uv.lock").read_text()
    # Each file keeps its name (uv reads a locked wheel's tags) on a host nothing answers.
    unreachable = lambda row: re.sub(r'url = "[^"]*/([^"/]+)"', r'url = "https://127.0.0.1:9/\1"', row.group(0))
    lock = re.sub(r'\[\[package\]\]\nname = "(cozy-runtime|tensorfs)"\n.*?(?=\n\[\[package\]\]|\Z)', unreachable, lock, flags=re.S)
    (project / "uv.lock").write_text(lock)
    generation = tmp_path / "generation"
    generation.mkdir()
    result = capture(root=generation, project=project, wheels=[], requirements=None, distribution="tensord-cpu-classifier",
                     release="0.1.0", python_requires=">=3.12", python_version="3.12", python="3.12", sdk=pair)
    held = versions(result.python)
    assert "cozy-runtime" not in held and "tensorfs" not in held
