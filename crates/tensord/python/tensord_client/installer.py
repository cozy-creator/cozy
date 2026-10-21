"""The trusted installer helper's command line: stdout carries exactly one typed record.

`capture` builds one environment of a local package's own closure; the machine does the rest
(`environments.rs`). It is its own module so `PackageError` is one class: run as `-m
...packages`, that module would be `__main__`, and the errors `captured_packages` raises would
escape its handler untyped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import msgspec

from .captured_packages import capture, capture_editable
from .package_records import InstallFailed
from .packages import PackageError


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    captured = commands.add_parser("capture")
    # The generation root the machine made: the environment is built at <root>/env.
    captured.add_argument("--root", type=Path)
    captured.add_argument("--project", type=Path)
    captured.add_argument("--wheel", action="append", type=Path, default=[])
    captured.add_argument("--requirements", type=Path)
    captured.add_argument("--distribution", required=True)
    captured.add_argument("--release", required=True)
    captured.add_argument("--python-requires", default="")
    captured.add_argument("--python-version", default="")
    # The machine's Runtime/TensorFS pair, which it installs over the closure next.
    captured.add_argument("--sdk-wheel", action="append", type=Path, default=[])
    # The package's own pair again, where the machine's could not stand.
    captured.add_argument("--restore", action="store_true")
    captured.add_argument("--python", default=sys.executable)
    # A live project installed editable in this environment, kept while its lock stands.
    captured.add_argument("--editable-environment", type=Path)
    # An index the project's sources name without declaring (name=url).
    captured.add_argument("--index", action="append", default=[])
    args = parser.parse_args()
    try:
        shared = dict(distribution=args.distribution, release=args.release, python_requires=args.python_requires,
                      python_version=args.python_version, python=args.python, sdk=args.sdk_wheel, restore=args.restore)
        if args.editable_environment:
            result = capture_editable(project=args.project, environment=args.editable_environment, wheels=args.wheel,
                                      indexes=args.index, **shared)
        elif args.root is None:
            raise PackageError("invalid_request", "a captured environment is built under --root")
        else:
            result = capture(root=args.root, project=args.project, wheels=args.wheel, requirements=args.requirements,
                             **shared)
    except PackageError as exc:
        print(msgspec.json.encode(InstallFailed(exc.code, str(exc))).decode())
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        # The machine attaches this helper's last stderr lines: the operation's own words.
        operation = " ".join(map(str, exc.cmd[:3]))
        print(msgspec.json.encode(InstallFailed("package_dependency_operation_failed", f"{operation} exited {exc.returncode}")).decode())
        raise SystemExit(1)
    print(msgspec.json.encode(result).decode())


if __name__ == "__main__":
    main()
