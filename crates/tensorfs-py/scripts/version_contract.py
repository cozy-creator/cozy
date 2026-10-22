#!/usr/bin/env python3
"""Prove that one TensorFS source tree produces one coherent package version."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import pathlib
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = "tensorfs"
# A cp311-abi3 wheel loads on 3.11 and every later CPython; 3.10 has its own cp310 wheel
# (the limited API's buffer protocol starts at 3.11). The floor is real; there is no ceiling.
PYTHON_SPEC = ">=3.10"
WHEEL_ABIS = (["cp311", "abi3"], ["cp310", "cp310"])
CURRENT_CARGO_VERSION = "0.13.1"
CURRENT_PYTHON_VERSION = "0.13.1"


def _toml(path: pathlib.Path) -> dict[str, object]:
    with path.open("rb") as source:
        return tomllib.load(source)


def _python_version_from_cargo(version: str) -> str:
    public, local_separator, local = version.partition("+")
    base, separator, prerelease = public.partition("-")
    if not separator:
        return version
    phase, dot, number = prerelease.partition(".")
    if phase != "dev" or dot != "." or not number.isdigit():
        raise ValueError(f"Cargo development version {version!r} is not <release>-dev.<number>")
    return f"{base}.dev{number}{local_separator}{local}"


def source_versions() -> tuple[str, str]:
    pyproject = _toml(ROOT / "pyproject.toml")
    cargo = _toml(ROOT / "Cargo.toml")
    python_version = pyproject["project"]["version"]  # type: ignore[index]
    cargo_version = cargo["package"]["version"]  # type: ignore[index]
    requires_python = pyproject["project"]["requires-python"]  # type: ignore[index]
    if requires_python != PYTHON_SPEC:
        raise ValueError(f"pyproject.toml requires-python {requires_python!r} != {PYTHON_SPEC!r}")
    if not isinstance(python_version, str) or not isinstance(cargo_version, str):
        raise ValueError("pyproject and Cargo versions must be strings")
    if cargo_version != CURRENT_CARGO_VERSION:
        raise ValueError(
            f"Cargo version {cargo_version!r} != current release identity "
            f"{CURRENT_CARGO_VERSION!r}"
        )
    if python_version != CURRENT_PYTHON_VERSION:
        raise ValueError(
            f"pyproject version {python_version!r} != current release identity "
            f"{CURRENT_PYTHON_VERSION!r}"
        )
    if _python_version_from_cargo(cargo_version) != python_version:
        raise ValueError(
            f"Cargo version {cargo_version!r} does not normalize to Python version "
            f"{python_version!r}"
        )

    for manifest in (ROOT.parent / crate / "Cargo.toml" for crate in ("tensorfs", "tensorfs-py", "varena")):
        package = _toml(manifest).get("package")
        version = package.get("version") if isinstance(package, dict) else None
        if version != cargo_version:
            raise ValueError(f"{manifest.relative_to(ROOT.parent)} version {version!r} != {cargo_version!r}")
    return cargo_version, python_version


def source_version() -> str:
    return source_versions()[1]


def _binary_version(tfs: pathlib.Path, expected_cargo: str) -> None:
    """One wheel carries one version (tfs-051): the shipped binary must SAY the version
    the wheel claims, and the digest it prints must be the digest of its own bytes."""
    said = subprocess.run([tfs, "version"], capture_output=True, text=True, check=False)
    if said.returncode != 0:
        raise ValueError(f"shipped tfs refused `version`: {said.stderr.strip()!r}")
    fields = said.stdout.split()
    if len(fields) != 3 or fields[0] != "tfs" or not fields[2].startswith("sha256:"):
        raise ValueError(f"shipped tfs printed {said.stdout!r}, not `tfs <version> sha256:<hex>`")
    if fields[1] != expected_cargo:
        raise ValueError(f"shipped tfs is version {fields[1]!r}, wheel claims {expected_cargo!r}")
    digest = hashlib.sha256(tfs.read_bytes()).hexdigest()
    if fields[2] != f"sha256:{digest}":
        raise ValueError(f"shipped tfs prints digest {fields[2]!r}, its bytes hash to {digest}")


def wheel_binary(path: pathlib.Path, expected_cargo: str, expected_python: str) -> None:
    member = f"tensorfs-{expected_python}.data/scripts/tfs"
    with zipfile.ZipFile(path) as wheel:
        try:
            info = wheel.getinfo(member)
        except KeyError:
            raise ValueError(f"{path.name} ships no {member}: the wheel IS the tfs channel")
        if not (info.external_attr >> 16) & 0o111:
            raise ValueError(f"{member} is not executable in {path.name}")
        with tempfile.TemporaryDirectory(prefix="tensorfs-wheel-tfs-") as scratch:
            extracted = pathlib.Path(wheel.extract(member, scratch))
            extracted.chmod(0o755)
            _binary_version(extracted, expected_cargo)


def wheel_version(path: pathlib.Path) -> str:
    cargo_expected, expected = source_versions()
    platform = subprocess.run(
        ["uv", "run", "--script", str(ROOT / "scripts/wheel-platform.py"), str(path)],
        capture_output=True, text=True, check=False,
    )
    if platform.returncode != 0:
        raise ValueError(platform.stderr.strip() or platform.stdout.strip())
    wheel_binary(path, cargo_expected, expected)
    with zipfile.ZipFile(path) as wheel:
        members = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError(f"{path.name} carries {len(members)} METADATA files, not one")
        metadata = BytesParser(policy=default).parsebytes(wheel.read(members[0]))
        wheel_members = [name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(wheel_members) != 1:
            raise ValueError(f"{path.name} carries {len(wheel_members)} WHEEL files, not one")
        wheel_metadata = BytesParser(policy=default).parsebytes(wheel.read(wheel_members[0]))
    name, version = metadata["Name"], metadata["Version"]
    if name != PACKAGE:
        raise ValueError(f"wheel distribution is {name!r}, expected {PACKAGE!r}")
    if version != expected:
        raise ValueError(f"wheel version {version!r} != source version {expected!r}")
    tags = wheel_metadata.get_all("Tag", [])
    if not tags or any(tag.split("-")[:2] not in WHEEL_ABIS for tag in tags):
        raise ValueError(f"wheel tags {tags!r} are neither CPython 3.11 abi3 nor CPython 3.10")
    return expected


def installed_version(path: pathlib.Path) -> str:
    expected = wheel_version(path)
    metadata_version = importlib.metadata.version(PACKAGE)
    if metadata_version != expected:
        raise ValueError(
            f"installed METADATA version {metadata_version!r} != wheel version {expected!r}"
        )
    import tensorfs
    from tensorfs import _ext

    if tensorfs.__version__ != expected:
        raise ValueError(
            f"tensorfs.__version__ {tensorfs.__version__!r} != wheel version {expected!r}"
        )
    for name in ("open", "Writer", "upload_folder", "Tensors"):
        if not callable(getattr(tensorfs, name, None)):
            raise ValueError(f"tensorfs must install the full public component API: missing {name}")
    if hasattr(tensorfs, "Store"):
        raise ValueError("the internal Store is exposed by the public package")
    for obsolete in ("Closure", "ReadPlan", "parse_header", "plan", "regions"):
        if hasattr(tensorfs, obsolete) or hasattr(_ext, obsolete):
            raise ValueError(f"removed legacy utility {obsolete} is exposed by the package")
    if importlib.util.find_spec("tensorfs.records") is not None:
        raise ValueError("the removed legacy header projection remains importable")
    installed_tfs = pathlib.Path(sysconfig.get_path("scripts")) / "tfs"
    if not os.access(installed_tfs, os.X_OK):
        raise ValueError(f"install put no executable tfs at {installed_tfs}")
    _binary_version(installed_tfs, source_versions()[0])
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("source", "wheel", "installed"))
    parser.add_argument("wheel", nargs="?", type=pathlib.Path)
    args = parser.parse_args()
    if args.mode == "source":
        if args.wheel is not None:
            parser.error("source takes no wheel")
        version = source_version()
    else:
        if args.wheel is None:
            parser.error(f"{args.mode} requires a wheel")
        version = wheel_version(args.wheel) if args.mode == "wheel" else installed_version(args.wheel)
    print(f"version contract green: {PACKAGE} {version} ({args.mode})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"version contract red: {error}", file=sys.stderr)
        raise SystemExit(1) from error
