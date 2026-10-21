#!/usr/bin/env python3
"""The store format this tree speaks, against the newest tensorfs on PyPI (design §2).

The `cozy` binary's TensorFS and a machine's tensorfs wheel share stores, so both must speak one
STORE_FORMAT:

    format_contract.py binary [--wheel W]  the newest published wheel (or wheel file W) speaks this tree's
                                           integers; prints its version
    format_contract.py wheel               integers unlike the newest published wheel's need a newer minor

Read-only: PyPI's JSON and the wheel itself (`uv`), nothing uploaded.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tomllib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAMES = ("STORE_FORMAT",)


def source() -> tuple[int, ...]:
    text = (ROOT.parent / "tensorfs/src/lib.rs").read_text()
    found = [re.search(rf"^pub const {name}: u32 = (\d+);", text, re.MULTILINE) for name in NAMES]
    if not all(found):
        raise SystemExit(f"tensorfs's lib.rs no longer declares `pub const {'/'.join(NAMES)}: u32 = N;`")
    return tuple(int(match.group(1)) for match in found if match)


def speaks(requirement: str) -> tuple[str, tuple[int, ...] | None]:
    """The version and integers of the tensorfs that `requirement` (a pin or a wheel file) installs."""
    probe = "import tensorfs, tensorfs._ext as e; print(tensorfs.__version__, *(getattr(e, n, '') for n in %r))" % (NAMES,)
    said = subprocess.run(
        ["uv", "run", "--no-project", "--isolated", "--python", "3.12", "--with", requirement, "python", "-c", probe],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return said[0], tuple(map(int, said[1:])) if said[1:] else None


def published() -> tuple[str, tuple[int, ...] | None]:
    with urllib.request.urlopen("https://pypi.org/pypi/tensorfs/json") as answer:
        return speaks(f"tensorfs=={json.load(answer)['info']['version']}")


def minor(version: str) -> tuple[int, int]:
    major, minor_, *_ = version.split(".")
    return int(major), int(minor_)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) in (2, 4) else ""
    wheel = sys.argv[3] if len(sys.argv) == 4 and sys.argv[2] == "--wheel" and mode == "binary" else None
    if len(sys.argv) == 4 and wheel is None:
        mode = ""
    ours = source()
    version, theirs = speaks(str(pathlib.Path(wheel).resolve())) if wheel else published()
    said = lambda ints: "nothing" if ints is None else f"store format {ints[0]}"
    if mode == "binary":
        if theirs != ours:
            which = f"the wheel {wheel}" if wheel else "the newest on PyPI"
            print(f"this tree speaks {said(ours)}; tensorfs {version}, {which}, speaks {said(theirs)}:"
                  " publish its wheel first (publish-tensorfs.yaml)", file=sys.stderr)
            return 1
        print(version)
        return 0
    if mode == "wheel":
        with (ROOT / "pyproject.toml").open("rb") as pyproject:
            declared = tomllib.load(pyproject)["project"]["version"]
        if theirs is not None and theirs != ours and minor(declared) <= minor(version):
            print(f"tensorfs {declared} speaks {said(ours)} but {version} speaks {said(theirs)}:"
                  " a changed format or protocol needs a new minor", file=sys.stderr)
            return 1
        print(f"tensorfs {declared}: {said(ours)}; newest published {version}: {said(theirs)}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
