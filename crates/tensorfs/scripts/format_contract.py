#!/usr/bin/env python3
"""The store format and claim protocol this tree speaks, against the newest tensorfs on PyPI (design §2).

The `cozy` binary's TensorFS and a machine's tensorfs wheel share stores, so both must speak one
STORE_FORMAT and CLAIM_PROTOCOL:

    format_contract.py binary  the newest published wheel speaks this tree's integers; prints its version
    format_contract.py wheel   integers unlike the newest published wheel's need a newer minor than it

Read-only: PyPI's JSON and the published wheel itself (`uv`), nothing uploaded.
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
NAMES = ("STORE_FORMAT", "CLAIM_PROTOCOL")


def source() -> tuple[int, ...]:
    text = (ROOT / "crates/tensorfs-core/src/lib.rs").read_text()
    found = [re.search(rf"^pub const {name}: u32 = (\d+);", text, re.MULTILINE) for name in NAMES]
    if not all(found):
        raise SystemExit(f"tensorfs-core's lib.rs no longer declares `pub const {'/'.join(NAMES)}: u32 = N;`")
    return tuple(int(match.group(1)) for match in found if match)


def published() -> tuple[str, tuple[int, ...] | None]:
    with urllib.request.urlopen("https://pypi.org/pypi/tensorfs/json") as answer:
        version = json.load(answer)["info"]["version"]
    probe = "import tensorfs._ext as e; print(*(getattr(e, n, '') for n in %r))" % (NAMES,)
    said = subprocess.run(
        ["uv", "run", "--no-project", "--isolated", "--python", "3.12", "--with", f"tensorfs=={version}",
         "python", "-c", probe],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return version, tuple(map(int, said)) if said else None


def minor(version: str) -> tuple[int, int]:
    major, minor_, *_ = version.split(".")
    return int(major), int(minor_)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    ours = source()
    version, theirs = published()
    speaks = lambda ints: "nothing" if ints is None else f"store format {ints[0]}, claim protocol {ints[1]}"
    if mode == "binary":
        if theirs != ours:
            print(f"this tree speaks {speaks(ours)}; tensorfs {version}, the newest on PyPI, speaks {speaks(theirs)}:"
                  " publish its wheel first (tensorfs-release.yaml)", file=sys.stderr)
            return 1
        print(version)
        return 0
    if mode == "wheel":
        with (ROOT / "pyproject.toml").open("rb") as pyproject:
            declared = tomllib.load(pyproject)["project"]["version"]
        if theirs is not None and theirs != ours and minor(declared) <= minor(version):
            print(f"tensorfs {declared} speaks {speaks(ours)} but {version} speaks {speaks(theirs)}:"
                  " a changed format or protocol needs a new minor", file=sys.stderr)
            return 1
        print(f"tensorfs {declared}: {speaks(ours)}; newest published {version}: {speaks(theirs)}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
