"""Real captured uv installations; no implementation doubles. Inference in these environments
runs on the machine's executors (`machine_surface.rs`)."""
import base64
import hashlib
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
from tensord_client.captured_packages import conflicts, install_captured, installed_description
from tensord_client.packages import GenerationHold, PackageError
from tensord_client.package_records import GENERATION_DECODER
from test_package_bridge import FIXTURE, ROOT


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-captured-package")
    source = root / "source"
    shutil.copytree(FIXTURE, source)
    subprocess.run(["uv", "lock", "--project", str(source), "--python", "3.12"], check=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "client"), str(ROOT)], check=True)
    client = next((root / "client").glob("*.whl"))
    frozen = install_captured(project=source, wheels=[], requirements=None,
        distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12", python_version="3.12",
        generations=root / "frozen", client_wheel=client, python="3.12")
    yield root, source, client, frozen


def assert_installed(generation):
    """The environment imports the package and the machine client, and keeps its description."""
    with GenerationHold(Path(generation.python).parents[2]):
        subprocess.run([generation.python, "-I", "-c", "import cpu_classifier, tensord_client.runtime_describe"],
                       check=True)
        assert b"classify" in bytes(generation.interface)


def locked_sdk(source):
    """The Runtime the capture's lock selected."""
    lock = tomllib.loads((source / "uv.lock").read_text())
    return next(row["version"] for row in lock["package"] if row["name"] == "cozy-runtime")


def test_frozen_source_keeps_locked_sdk(capture):
    root, source, client, generation = capture
    versions = {dependency.name: dependency.version for dependency in generation.dependencies}
    lock = tomllib.loads((source / "uv.lock").read_text())
    sdk = next(row["version"] for row in lock["package"] if row["name"] == "cozy-runtime")
    assert versions["cozy-runtime"] == sdk
    assert_installed(generation)


def test_real_wheel_roster_and_hashed_requirements_preserve_dependency_versions(capture):
    root, source, client, frozen = capture
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "wheels"), str(source)], check=True)
    wheel = next((root / "wheels").glob("*.whl"))
    requirements = root / "requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(source), "--output-file", str(requirements)], check=True)
    generation = install_captured(project=None, wheels=[wheel], requirements=requirements,
        distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12", python_version="3.12",
        generations=root / "wheel-generations", client_wheel=client, python="3.12")
    before = {dependency.name: dependency.version for dependency in frozen.dependencies}
    after = {dependency.name: dependency.version for dependency in generation.dependencies}
    assert after == before
    assert_installed(generation)


def test_incoherent_source_lock_overlay_and_python_mismatch_refuse_only_installation(capture):
    root, source, client, generation = capture
    with pytest.raises(PackageError, match="cannot overlay"):
        install_captured(project=source, wheels=[client], requirements=None,
            distribution="tensord-cpu-classifier", release="0.1.0", python_requires="", python_version="",
            generations=root / "unsupported", client_wheel=client, python="3.12")
    with pytest.raises(PackageError, match="does not match"):
        install_captured(project=source, wheels=[], requirements=None,
            distribution="tensord-cpu-classifier", release="0.1.0", python_requires="", python_version="3.11",
            generations=root / "unsupported", client_wheel=client, python="3.12")
    assert_installed(generation)


def test_rust_tensorfs_materialization_and_trusted_installer(capture):
    root, source, client, frozen = capture
    archive = root / "source.tar"
    subprocess.run(["tar", "-cf", str(archive), "-C", str(source), "pyproject.toml", "uv.lock", "package.toml", "cpu_classifier"], check=True)
    output = root / "rust-owner-install"
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT.parents[1] / "target"))
    subprocess.run([str(target / "debug/install-capture"), "--archive", str(archive), "--output", str(output),
        "--helper-python", sys.executable, "--client-wheel", str(client), "--python", "3.12",
        "--package", "local/tensord-cpu-classifier", "--release", "0.1.0"], check=True)
    generation = GENERATION_DECODER.decode((output / "prepared-generation.json").read_bytes())
    assert json.loads((output / "package-interface.json").read_bytes()) == json.loads(bytes(generation.interface))
    assert {row.name: row.version for row in generation.dependencies}["cozy-runtime"] == locked_sdk(source)
    assert_installed(generation)


def test_installer_only_helper_uses_locked_sdk_and_static_reader_skips_pth(capture):
    root, source, client, frozen = capture
    helper = root / "installer-only-helper"
    subprocess.run(["uv", "venv", "--python", "3.14", str(helper)], check=True)
    python = layout.interpreter(helper)
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(client), "packaging"], check=True)
    subprocess.run([str(python), "-c", "import importlib.util;assert importlib.util.find_spec('cozy_runtime') is None"], check=True)
    process = subprocess.run([str(python), "-m", "tensord_client.installer", "install-captured",
        "--project", str(source), "--generations", str(root / "minimal-helper-generations"), "--client-wheel", str(client),
        "--distribution", "tensord-cpu-classifier", "--release", "0.1.0", "--python", "3.12"], check=True, capture_output=True)
    generation = GENERATION_DECODER.decode(process.stdout)
    assert {row.name: row.version for row in generation.dependencies}["cozy-runtime"] == locked_sdk(source)
    site, = layout.site_packages(Path(generation.python).parents[1])
    sentinel = root / "pth-must-not-run-during-static-description"
    pth = site / "authored-startup.pth"
    pth.write_text(f"import pathlib;pathlib.Path({str(sentinel)!r}).write_text('executed')\n")
    installed_description(generation.package, Path(generation.python))
    assert not sentinel.exists()
    pth.unlink()  # uniquely owned authored test fixture, not user source/work
    assert_installed(generation)


def app_wheel(directory: Path, module: str, source: str, requires: str = "") -> Path:
    """A one-module cozy package wheel, written as wheel bytes (nothing builds or runs it)."""
    name, info = module.replace("_", "-"), f"{module}-0.1.0.dist-info"
    members = {f"{module}.py": source,
               f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n{requires}",
               f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
               f"{info}/entry_points.txt": f"[cozy.application]\ndefault = {module}:app\n"}
    rows = "".join(f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b'=').decode()},{len(text.encode())}\n"
                   for path, text in members.items())
    wheel = directory / f"{module}-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in members.items():
            archive.writestr(path, text)
        archive.writestr(f"{info}/RECORD", rows + f"{info}/RECORD,,\n")
    return wheel


def test_an_installed_app_the_reader_refuses_fails_the_installation_naming_it(capture):
    """Left out of the callee rows, its callers would call its raw function instead (run 4777)."""
    root, source, client, frozen = capture
    wheels = root / "refused-wheels"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(source)], check=True)
    requirements = root / "refused-requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(source), "--output-file", str(requirements)], check=True)
    refused = app_wheel(wheels, "refused_app", "from cozy_runtime.author import App\napp = App() if True else None\n")
    with pytest.raises(PackageError, match="installed package refused-app 0.1.0 cannot be described"):
        install_captured(project=None, wheels=[next(wheels.glob("tensord_cpu_classifier-*.whl")), refused],
            requirements=requirements, distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12",
            python_version="3.12", generations=root / "refused-generations", client_wheel=client, python="3.12")


def test_a_dependency_conflict_is_the_runs_warning_never_a_refusal(capture):
    """The conflict uv's check finds is named in the generation's warning; it installs and runs."""
    root, source, client, frozen = capture
    wheels = root / "conflicted-wheels"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(source)], check=True)
    requirements = root / "conflicted-requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(source), "--output-file", str(requirements)], check=True)
    conflicted = app_wheel(wheels, "conflicted_app", "from cozy_runtime.author import App\napp = App()\n", "Requires-Dist: numpy==1.0.0\n")
    generation = install_captured(project=None, wheels=[next(wheels.glob("tensord_cpu_classifier-*.whl")), conflicted],
        requirements=requirements, distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12",
        python_version="3.12", generations=root / "conflicted-generations", client_wheel=client, python="3.12")
    assert re.search(r"^uv pip check: .*conflicted-app.* requires .numpy==1\.0\.0.", generation.sdk_fallback), generation.sdk_fallback
    assert_installed(generation)


def test_a_wheel_tagged_for_another_platform_stands_and_a_missed_requirement_refuses(tmp_path):
    """uv's real check of an environment holding both (NVIDIA's aarch64 cusparseLt wheel says sbsa)."""
    subprocess.run(["uv", "venv", "--python", "3.12", str(tmp_path / "env")], check=True)
    site, = layout.site_packages(tmp_path / "env")
    for name, version, tag, requires in [("numpy", "2.0.0", "cp312-cp312-manylinux_2_17_x86_64", ""),
                                         ("conflicted_app", "0.1.0", "py3-none-any", "Requires-Dist: numpy==1.0.0\n"),
                                         ("nvidia_cusparselt_cu12", "0.6.3", "py3-none-manylinux2014_sbsa", "")]:
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requires}")
        (info / "WHEEL").write_text(f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n")
        (info / "RECORD").write_text(f"{info.name}/METADATA,,\n")
    python = layout.interpreter(tmp_path / "env")
    check = subprocess.run(["uv", "pip", "check", "--python", str(python)], capture_output=True, text=True)
    assert "The package `nvidia-cusparselt-cu12` was built for a different platform" in check.stdout + check.stderr
    assert conflicts(python) == {"The package `conflicted-app` requires `numpy==1.0.0`, but `2.0.0` is installed"}


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


def test_a_captured_lock_takes_the_machines_pair_without_fetching_its_own(capture, tmp_path):
    """The machine's pair is installed once: a captured lock's own Runtime and TensorFS, made
    unreachable here, are never fetched to be replaced."""
    root, source, client, _ = capture
    pair = pypi_pair(tmp_path / "pair")
    project = tmp_path / "source"
    shutil.copytree(source, project)
    lock = (project / "uv.lock").read_text()
    # Each file keeps its name (uv reads a locked wheel's tags) on a host nothing answers.
    unreachable = lambda row: re.sub(r'url = "[^"]*/([^"/]+)"', r'url = "https://127.0.0.1:9/\1"', row.group(0))
    lock = re.sub(r'\[\[package\]\]\nname = "(cozy-runtime|tensorfs)"\n.*?(?=\n\[\[package\]\]|\Z)', unreachable, lock, flags=re.S)
    (project / "uv.lock").write_text(lock)
    generation = install_captured(project=project, wheels=[], requirements=None,
        distribution="tensord-cpu-classifier", release="0.1.0", python_requires=">=3.12", python_version="3.12",
        generations=tmp_path / "generations", client_wheel=client, python="3.12", sdk=pair)
    versions = {dependency.name: dependency.version for dependency in generation.dependencies}
    assert versions["cozy-runtime"] == pair[0].name.split("-")[1]
    assert versions["tensorfs"] == pair[1].name.split("-")[1]
    assert_installed(generation)


def test_normal_sdk_contradictions_refuse_and_explicit_development_remains_visible(tmp_path):
    """Real uv checks a package bound against the selected SDK; no inference substitute."""
    from tensord_client.captured_packages import checked
    subprocess.run(["uv", "venv", "--python", "3.12", str(tmp_path / "env")], check=True)
    site, = layout.site_packages(tmp_path / "env")
    for name, version, requires in [("cozy_runtime", "0.31.0", ""),
                                    ("bounded_app", "1.0", "Requires-Dist: cozy-runtime<0.31\n")]:
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requires}")
        (info / "WHEEL").write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    interpreter = layout.interpreter(tmp_path / "env")
    with pytest.raises(PackageError, match="bounded-app") as refused:
        checked(interpreter, "", sdk=True)
    assert refused.value.code == "package_sdk_incompatible"
    warning = checked(interpreter, "", "0.31.0+explicit", sdk=True)
    assert "bounded-app" in warning and "development Runtime 0.31.0+explicit" in warning


def test_sdk_dependency_resolution_cannot_silently_fall_back_to_no_deps(tmp_path, monkeypatch):
    from tensord_client.captured_packages import checked, inventory, machine_sdk
    subprocess.run(["uv", "venv", "--python", "3.12", str(tmp_path / "env")], check=True)
    site, = layout.site_packages(tmp_path / "env")
    info = site / "synthetic_dep-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: synthetic-dep\nVersion: 1.0\n")
    (info / "WHEEL").write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    wheel = app_wheel(tmp_path, "cozy_runtime", "", "Requires-Dist: synthetic-dep>=2\n")
    interpreter = layout.interpreter(tmp_path / "env")
    monkeypatch.setenv("UV_NO_INDEX", "1")
    with pytest.raises(PackageError) as refused:
        machine_sdk(interpreter, [wheel])
    assert refused.value.code == "package_sdk_incompatible"
    assert all(row.name != "cozy-runtime" for row in inventory(interpreter))
    note = machine_sdk(interpreter, [wheel], "0.1.0+explicit")
    assert "captured lock" in note
    assert "synthetic-dep" in checked(interpreter, note, "0.1.0+explicit", sdk=True)
