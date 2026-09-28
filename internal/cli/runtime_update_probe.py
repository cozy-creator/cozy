"""Read-only worker inventory used by the fixed Creator maintenance transport.

Go embeds this file and supplies it directly; a package or API caller cannot
replace it. Keeping it as source lets mypy and the language server inspect it.
"""

from __future__ import annotations

import fcntl
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import TypeGuard

from packaging.markers import default_environment
from packaging.tags import sys_tags


def object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def mapping(value: object) -> dict[str, object]:
    if not object_mapping(value):
        raise ValueError("worker inventory must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("worker inventory keys must be strings")
        result[key] = item
    return result


def main() -> dict[str, object]:
    lock = Path("/var/lib/cozy/dev/update.lock")
    if lock.exists():
        with lock.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream, fcntl.LOCK_UN)
            except BlockingIOError:
                return {"update_in_progress": True}
    runtime = mapping(json.loads(subprocess.check_output(
        [str(Path(sys.executable).parent / "cozy-runtime"), "version", "--json"],
        text=True,
    )))
    base = Path("/var/lib/cozy/dev/base.json")
    current = Path("/var/lib/cozy/dev/current")
    updater = Path("/opt/cozy/dev/update.py")
    capabilities: dict[str, object] = {}
    if updater.is_file():
        capability_probe = subprocess.run(
            [sys.executable, str(updater), "capabilities"], text=True, capture_output=True
        )
        if capability_probe.returncode == 0:
            capabilities = mapping(json.loads(capability_probe.stdout))
    durable = capabilities.get("durable_updates", False)
    if not isinstance(durable, bool):
        raise ValueError("worker durable-update capability must be a boolean")
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = ""
    return {
        "runtime": runtime, "tensorfs": metadata.version("tensorfs"),
        "python": platform.python_version(), "tags": [str(tag) for tag in sys_tags()],
        "updater": base.is_file() and updater.is_file(), "durable_updates": durable,
        "selection": str(current.resolve()) if current.exists() else "",
        "torch": torch_version, "markers": default_environment(),
    }


if __name__ == "__main__":
    print(json.dumps(main()))
