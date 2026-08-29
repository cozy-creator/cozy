#!/usr/bin/env python3
"""Prove that one TensorFS source tree produces one coherent package version."""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = "tensorfs"
PRELAUNCH_MAJOR = 0
V1_GATE = "proto-014"


def _toml(path: pathlib.Path) -> dict[str, object]:
    with path.open("rb") as source:
        return tomllib.load(source)


def source_version() -> str:
    pyproject = _toml(ROOT / "pyproject.toml")
    cargo = _toml(ROOT / "Cargo.toml")
    python_version = pyproject["project"]["version"]  # type: ignore[index]
    cargo_version = cargo["workspace"]["package"]["version"]  # type: ignore[index]
    if not isinstance(python_version, str) or not isinstance(cargo_version, str):
        raise ValueError("pyproject and Cargo workspace versions must be strings")
    if python_version != cargo_version:
        raise ValueError(
            f"pyproject version {python_version!r} != Cargo workspace version {cargo_version!r}"
        )

    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        package = _toml(manifest).get("package")
        version = package.get("version") if isinstance(package, dict) else None
        if version != {"workspace": True}:
            raise ValueError(f"{manifest.relative_to(ROOT)} must inherit version.workspace")

    try:
        major = int(python_version.partition(".")[0])
    except ValueError as error:
        raise ValueError(f"package version {python_version!r} has no numeric major") from error
    if major != PRELAUNCH_MAJOR:
        raise ValueError(
            f"TensorFS remains 0.x until {V1_GATE} cuts the first public v1; got {python_version}"
        )
    return python_version


def wheel_version(path: pathlib.Path) -> str:
    expected = source_version()
    with zipfile.ZipFile(path) as wheel:
        members = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError(f"{path.name} carries {len(members)} METADATA files, not one")
        metadata = BytesParser(policy=default).parsebytes(wheel.read(members[0]))
    name, version = metadata["Name"], metadata["Version"]
    if name != PACKAGE:
        raise ValueError(f"wheel distribution is {name!r}, expected {PACKAGE!r}")
    if version != expected:
        raise ValueError(f"wheel version {version!r} != source version {expected!r}")
    return expected


def installed_version(path: pathlib.Path) -> str:
    expected = wheel_version(path)
    metadata_version = importlib.metadata.version(PACKAGE)
    if metadata_version != expected:
        raise ValueError(
            f"installed METADATA version {metadata_version!r} != wheel version {expected!r}"
        )
    import tensorfs

    if tensorfs.__version__ != expected:
        raise ValueError(
            f"tensorfs.__version__ {tensorfs.__version__!r} != wheel version {expected!r}"
        )
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
