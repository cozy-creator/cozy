#!/usr/bin/env python3
"""A development pod's `ssh` and `sftp` for the Runtime update transport, over a local root.

Installed as both names in a bin directory; its parent holds the pod's state:
pod/ (the pod filesystem), observed.json (the probe's answer), ops/ (the updater's
journal), update-outcome (present: the guardian stops for recovery), sftp-mode
(ok | drop | partial:<bytes> | hold) and sftp.log (every batch).
"""

import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(sys.argv[0]).parent.parent
POD = HOME / "pod"
DEV = "/var/lib/cozy/dev"
PYTHON = "/opt/cozy/python/bin/python3"


def local(path: str) -> Path:
    return POD / path.lstrip("/")


def updater(action: str, stage: str, expected: list[str]) -> int:
    journal = HOME / "ops" / (stage + ".json")
    journal.parent.mkdir(exist_ok=True)
    if action == "status":
        print(journal.read_text() if journal.exists() else json.dumps({"operation": stage, "state": "missing"}))
        return 0
    staged = local(DEV) / "staged" / stage
    wheels = sorted(staged.glob("*.whl"))
    if sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in wheels) != sorted(expected):
        print("transferred wheel digest differs from operator-selected bytes", file=sys.stderr)
        return 1
    if (HOME / "update-outcome").exists():
        # The guardian stopped part-way; only an operator's resume can finish it.
        journal.write_text(json.dumps({"operation": stage, "expected": sorted(expected), "state": "running",
                                       "phase": "recovery_required", "error": "the guardian lost its operation lock",
                                       "updater_running": False}))
        print(json.dumps({"operation": stage, "state": "running"}))
        return 0
    observed = json.loads((HOME / "observed.json").read_text())
    for path in wheels:
        name, version = path.name.split("-")[:2]
        if name == "cozy_runtime":
            observed["runtime"]["distribution"] = version
        else:
            observed["tensorfs"] = version
    (HOME / "observed.json").write_text(json.dumps(observed))
    journal.write_text(json.dumps({"operation": stage, "expected": sorted(expected), "state": "succeeded",
                                   "phase": "done", "updater_running": False}))
    print(json.dumps({"operation": stage, "state": "queued"}))
    return 0


def ssh(command: str) -> int:
    if command.startswith("for python in"):
        print(PYTHON, end="")
        return 0
    words = shlex.split(command)
    if words[:2] == [PYTHON, "/opt/cozy/dev/update.py"]:
        return updater(words[2], words[3], words[4:])
    if words[:3] == [PYTHON, "-I", "-c"]:
        if "Read-only worker inventory" in words[3]:
            print((HOME / "observed.json").read_text())
            return 0
        rooted = str(local(DEV))
        return subprocess.run([sys.executable, "-I", "-c", *(word.replace(DEV, rooted) for word in words[3:])]).returncode
    print("unexpected pod command: " + command, file=sys.stderr)
    return 127


def sftp(arguments: list[str]) -> int:
    batch = Path(arguments[arguments.index("-b") + 1]).read_text()
    with (HOME / "sftp.log").open("a") as log:
        log.write(batch + "--\n")
    mode = (HOME / "sftp-mode").read_text().strip() if (HOME / "sftp-mode").exists() else "ok"
    if mode == "hold":
        (HOME / "sftp-holding").touch()
        while not (HOME / "sftp-release").exists():
            time.sleep(0.02)
        mode = "drop"
    budget = 0 if mode == "drop" else int(mode.split(":")[1]) if mode.startswith("partial:") else None
    for line in batch.splitlines():
        words = shlex.split(line)
        if words[0].lstrip("-") == "mkdir":
            if local(words[1]).exists():
                print(f'remote mkdir "{words[1]}": Failure', file=sys.stderr)
            else:
                local(words[1]).mkdir()
            continue
        source, target = Path(words[1]), local(words[2])
        if not target.parent.is_dir():
            print(f'remote open "{words[2]}": No such file', file=sys.stderr)
            return 1
        resume = words[0] == "reput" and target.exists()
        data = source.read_bytes()[target.stat().st_size if resume else 0:]
        sent = data if budget is None else data[:budget]
        with target.open("ab" if resume else "wb") as out:
            out.write(sent)
        if len(sent) < len(data):
            print("client_loop: send disconnect: Broken pipe", file=sys.stderr)
            return 255
        if budget is not None:
            budget -= len(sent)
    return 0


if __name__ == "__main__":
    name = Path(sys.argv[0]).name
    sys.exit(ssh(sys.argv[-1]) if name == "ssh" else sftp(sys.argv[1:]))
