#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["auditwheel==6.8.2"]
# ///
"""Check the whole wheel's platform promise, including bundled executables.

Maturin audits the extension; it does not infer the floor of a CLI staged through
wheel-data. Linux: auditwheel owns ELF/library/ISA policy and legacy manylinux aliases.
macOS: every Mach-O must match the tag's architecture and deploy no later than its OS.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import struct
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser

from auditwheel.policy import WheelPolicies
from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc


ELF = b"\x7fELF"
MACHO_64 = b"\xcf\xfa\xed\xfe"
MACHO_FAT = (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")
MACHO_CPU = {0x01000007: "x86_64", 0x0100000C: "arm64"}
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32


def macho_floor(name: str, data: bytes) -> tuple[str, tuple[int, int]]:
    if data[:4] in MACHO_FAT:
        raise ValueError(f"{name} is a universal Mach-O; each wheel ships one architecture")
    cpu, _, _, ncmds = struct.unpack_from("<iiII", data, 4)
    offset, floor = 32, None
    for _ in range(ncmds):
        command, size = struct.unpack_from("<II", data, offset)
        if command == LC_BUILD_VERSION:
            (minos,) = struct.unpack_from("<I", data, offset + 12)
            floor = (minos >> 16, (minos >> 8) & 0xFF)
        elif command == LC_VERSION_MIN_MACOSX and floor is None:
            (minos,) = struct.unpack_from("<I", data, offset + 8)
            floor = (minos >> 16, (minos >> 8) & 0xFF)
        offset += size
    if cpu not in MACHO_CPU or floor is None:
        raise ValueError(f"{name} names no known architecture and macOS deployment floor")
    return MACHO_CPU[cpu], floor


def check_native(wheel: Path, platforms: set[str], natives: dict[str, bytes]) -> str:
    """macOS: the tag names one OS floor and one architecture for every binary."""
    if not natives:
        raise ValueError("wheel carries no native binary")
    for platform in sorted(platforms):
        macos = re.fullmatch(r"macosx_(\d+)_(\d+)_(x86_64|arm64)", platform)
        if not macos:
            raise ValueError(f"unsupported platform tag {platform}")
        tagged = (int(macos[1]), int(macos[2]))
        for name, data in natives.items():
            if data[:4] not in (MACHO_64, *MACHO_FAT):
                raise ValueError(f"{name} is not Mach-O in a {platform} wheel")
            arch, floor = macho_floor(name, data)
            if arch != macos[3] or floor > tagged:
                raise ValueError(
                    f"{wheel.name} advertises {platform}, but {name} is {arch} for macOS "
                    f"{floor[0]}.{floor[1]}; set MACOSX_DEPLOYMENT_TARGET for cargo and maturin alike"
                )
    return ".".join(sorted(platforms))


def check(wheel: Path) -> str:
    platforms = set(wheel.name.removesuffix(".whl").rsplit("-", 1)[1].split("."))
    linux = all(p.startswith(("manylinux", "musllinux", "linux_")) for p in platforms)
    natives: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1:
            raise ValueError("wheel must contain exactly one WHEEL metadata record")
        tags = BytesParser().parsebytes(archive.read(metadata[0])).get_all("Tag", [])
        if {tag.rsplit("-", 1)[1] for tag in tags} != platforms:
            raise ValueError("filename and WHEEL platform tags disagree")
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("wheel must contain exactly one RECORD")
        recorded = {row[0] for row in csv.reader(io.StringIO(archive.read(records[0]).decode()))}
        for name in archive.namelist():
            with archive.open(name) as member:
                magic = member.read(4)
            if magic == ELF and name not in recorded:
                raise ValueError(f"bundled ELF {name} is absent from RECORD and auditwheel's inventory")
            if magic in (ELF, MACHO_64, *MACHO_FAT):
                if name not in recorded:
                    raise ValueError(f"bundled binary {name} is absent from RECORD")
                if not linux:
                    natives[name] = archive.read(name)
    if not linux:
        return check_native(wheel, platforms, natives)
    inspected = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", "--json", str(wheel)],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(inspected.stdout)
    if report.get("version") != 1 or report.get("pure") is not False:
        raise ValueError("auditwheel did not report a versioned native wheel inspection")
    architecture = get_wheel_architecture(wheel.name)
    if platforms == {f"linux_{architecture}"}:
        return report["overall_tag"]  # Explicit host-only build makes no manylinux promise.
    policies = WheelPolicies(libc=get_wheel_libc(wheel.name), arch=architecture)
    required = policies.get_policy_by_name(report["overall_tag"])
    for platform in sorted(platforms):
        if policies.get_policy_by_name(platform) > required:
            raise ValueError(
                f"{wheel.name} advertises {platform}, but all bundled ELF files require "
                f"{required.name}; build the CLI and extension inside the existing manylinux "
                "CI/release environment, or explicitly use --compatibility linux for a host-only build"
            )
    return required.name


def prove(wheel: Path, host_elf: Path) -> None:
    """Real red arms: keep the extension unchanged and introduce a newer-host ELF."""
    native = host_elf.read_bytes()
    if native[:4] != b"\x7fELF":
        raise ValueError("--prove requires the host-built ELF executable")
    for mode in ("bundled CLI", "opaque extra ELF", "unrecorded ELF"):
        extra = mode != "bundled CLI"
        with tempfile.TemporaryDirectory(prefix="tensorfs-platform-red-") as directory:
            changed = Path(directory) / wheel.name
            replaced = False
            with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(changed, "w") as target:
                records = []
                record_name = ""
                for info in source.infolist():
                    if info.filename.endswith(".dist-info/RECORD"):
                        record_name = info.filename
                        continue
                    data = source.read(info)
                    if not extra and info.filename.endswith(".data/scripts/tfs"):
                        data, replaced = native, True
                    target.writestr(info, data)
                    records.append((info.filename, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), len(data)))
                if extra:
                    name = "tensorfs/data/opaque-resource"
                    target.writestr(name, native)
                    if mode != "unrecorded ELF":
                        records.append((name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(native).digest()).rstrip(b"=").decode(), len(native)))
                record = io.StringIO()
                csv.writer(record).writerows([*records, (record_name, "", "")])
                target.writestr(record_name, record.getvalue())
            if not extra and not replaced:
                raise ValueError("portable wheel did not contain its bundled tfs")
            try:
                check(changed)
            except ValueError as error:
                expected = "absent from RECORD" if mode == "unrecorded ELF" else "all bundled ELF files require"
                if expected not in str(error):
                    raise
                print(f"platform red observed ({mode}): {error}")
            else:
                raise ValueError("newer-host ELF did not falsify the portable tag; red arm did not run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--prove", type=Path, metavar="HOST_ELF")
    args = parser.parse_args()
    required = check(args.wheel)
    print(f"wheel platform green: every bundled binary satisfies the tags; required {required}")
    if args.prove:
        prove(args.wheel, args.prove)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"wheel platform red: {error}", file=sys.stderr)
        raise SystemExit(1) from error
