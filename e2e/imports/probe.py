"""Imports one installed package release the way an executor loads it: its `cozy.application` entry points,
then every module under them. Prints the Runtime and TensorFS it imported with; exits 1 naming each failure.
Runs in the release's environment (e2e/imports/run.sh): probe.py <distribution>."""

import importlib
import importlib.metadata as md
import pkgutil
import sys
import traceback

name = sys.argv[1]
failed: list[str] = []


def load(what: str, call) -> object | None:
    try:
        return call()
    except BaseException:
        failed.append(f"{what}: {traceback.format_exc(limit=-2).strip().splitlines()[-1]}")
        print(traceback.format_exc(), file=sys.stderr)
        return None


print(f"cozy-runtime {md.version('cozy-runtime')}, tensorfs {md.version('tensorfs')}")
points = [p for p in md.distribution(name).entry_points if p.group == "cozy.application"]
if not points:
    failed.append(f"{name}: no cozy.application entry point")
for point in points:
    load(point.value, point.load)
for top in sorted({p.module.split(".")[0] for p in points}):
    package = load(top, lambda: importlib.import_module(top))
    for found in pkgutil.walk_packages(getattr(package, "__path__", []), top + ".", onerror=lambda n: None):
        load(found.name, lambda: importlib.import_module(found.name))
for line in failed:
    print(f"FAIL {line}")
print(f"{name}: {len(points)} entry point(s), {'ok' if not failed else f'{len(failed)} failure(s)'}")
sys.exit(1 if failed else 0)
