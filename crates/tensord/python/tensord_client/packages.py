"""A local package's declared metadata, and the helper's typed refusal."""
from __future__ import annotations

import tomllib
from pathlib import Path

import msgspec

from .package_records import PackageMetadata, Pyproject


class PackageError(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def read_metadata(project: Path) -> PackageMetadata:
    parsed = msgspec.convert(tomllib.loads((project / "pyproject.toml").read_text()), type=Pyproject)
    metadata = parsed.project
    return PackageMetadata(metadata.name, metadata.version, metadata.entry_points.application.default)
