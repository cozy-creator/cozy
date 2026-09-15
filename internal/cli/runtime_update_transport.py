"""Fixed developer-worker maintenance transport, embedded in the Creator CLI.

Only first-party Runtime/TensorFS distributions are downloaded. The worker's
existing root updater owns validation, replacement and rollback.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

from packaging.tags import parse_tag
from packaging.utils import parse_wheel_filename


PROBE = """import importlib.metadata as m, json, platform, subprocess, fcntl
from packaging.tags import sys_tags
from pathlib import Path
lock=Path('/var/lib/cozy/dev/update.lock')
if lock.exists():
 with lock.open('rb') as stream:
  try:
   fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
   fcntl.flock(stream,fcntl.LOCK_UN)
  except BlockingIOError:
   print(json.dumps({'update_in_progress':True})); raise SystemExit(0)
runtime=json.loads(subprocess.check_output(['cozy-runtime','version','--json'],text=True))
base=Path('/var/lib/cozy/dev/base.json')
current=Path('/var/lib/cozy/dev/current')
updater=Path('/opt/cozy/dev/update.py')
capability_probe=subprocess.run(['python3',str(updater),'capabilities'],text=True,capture_output=True) if updater.is_file() else None
capabilities=json.loads(capability_probe.stdout) if capability_probe and capability_probe.returncode == 0 else {}
try: torch_version=m.version('torch')
except m.PackageNotFoundError: torch_version=''
print(json.dumps({'runtime':runtime,'tensorfs':m.version('tensorfs'),'python':platform.python_version(),
 'tags':[str(t) for t in sys_tags()], 'updater':base.is_file() and updater.is_file(), 'durable_updates':capabilities.get('durable_updates',False),
 'selection':str(current.resolve()) if current.exists() else '',
 'torch':torch_version}))
"""


def ssh(arguments, command):
    return subprocess.run(
        ["ssh", *arguments, command], capture_output=True, text=True, check=True
    ).stdout


def inspect(arguments):
    # The script is fixed application code, never a package callback or prompt.
    import shlex

    return json.loads(ssh(arguments, "python3 -I -c " + shlex.quote(PROBE)))


def fetch(url, limit):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "pypi.org", "files.pythonhosted.org"
    }:
        raise ValueError("Runtime update artifact URL is outside the first-party release index")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Runtime update artifact exceeds its download bound")
    return data


def wheel(name, selected, tags, directory, native):
    version, filename = selected["version"], selected["filename"]
    if name not in {"cozy-runtime", "tensorfs"} or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
        raise ValueError("Invalid approved Runtime update wheel")
    project, release, _, wheel_tags = parse_wheel_filename(filename)
    supported = {tag for text in tags for tag in parse_tag(text)}
    if project != name or str(release) != version or not supported.intersection(wheel_tags):
        raise ValueError(f"Approved {name} {version} does not support this worker's platform")
    if native and all(tag.platform == "any" for tag in wheel_tags):
        raise ValueError("The worker requires a native wheel from the approved image")
    digest = selected["digest"].removeprefix("sha256:")
    if not re.fullmatch(r"[a-f0-9]{64}", digest) or not 0 < selected["length"] <= 128 << 20:
        raise ValueError("Approved wheel digest or length is invalid")
    path = directory / filename
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        data = fetch(selected["url"], selected["length"])
        if len(data) != selected["length"] or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Runtime update wheel failed approved digest verification")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    return {"distribution": name, "version": version, "file": filename,
            "path": str(path), "sha256": digest}


def main():
    request = json.load(sys.stdin)
    arguments = request["ssh_arguments"]
    action = request["action"]
    if action == "resume":
        stage = request["stage"]
        expected = [row["sha256"] for row in request["selection"]["wheels"]]
        if not re.fullmatch(r"[a-f0-9]{32}", stage) or len(expected) != 2 or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in expected):
            raise ValueError("Invalid recorded update identity")
        return {"update": json.loads(ssh(arguments, "python3 /opt/cozy/dev/update.py apply " + stage + " " + " ".join(expected)))}
    if action == "status":
        stage = request["stage"]
        if not re.fullmatch(r"[a-f0-9]{32}", stage):
            raise ValueError("Invalid update operation identity")
        update = json.loads(ssh(arguments, "python3 /opt/cozy/dev/update.py status " + stage))
        result = {"update": update}
        if update.get("state") in {"succeeded", "rolled_back", "refused"}:
            result["observed"] = inspect(arguments)
        return result
    observed = inspect(arguments)
    if request["action"] == "inspect":
        return {"observed": observed}
    if observed.get("update_in_progress"):
        raise ValueError("The worker is already applying a Runtime update")
    if request["action"] == "plan" and observed["runtime"]["distribution"] == request["target"]["runtime_update"]["runtime"]["version"] and observed["tensorfs"] == request["target"]["runtime_update"]["tensorfs"]["version"]:
        return {"observed": observed, "unchanged": True}
    if not observed["updater"] or not observed.get("durable_updates", False) or not observed["runtime"].get("supports_guarded_restart", False):
        raise ValueError("This worker image does not support safe Runtime updates; select a current maintenance-capable private image")
    directory = Path(request["directory"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if request["action"] == "plan":
        selection = []
        for name in ("cozy-runtime", "tensorfs"):
            selection.append(wheel(name, request["target"]["runtime_update"]["runtime" if name == "cozy-runtime" else "tensorfs"], observed["tags"],
                                   directory, name == "tensorfs" or bool(observed["torch"])))
        return {"observed": observed, "wheels": selection}
    if request["action"] != "apply":
        raise ValueError("Unknown Runtime update operation")
    stage = request["stage"]
    if not re.fullmatch(r"[a-f0-9]{32}", stage):
        raise ValueError("Invalid update staging identity")
    wheels = request["selection"]["wheels"]
    remote = "/var/lib/cozy/dev/staged/" + stage
    commands = ["-mkdir /var/lib/cozy/dev/staged", "-mkdir " + remote]
    for row in wheels:
        path = Path(row["path"])
        if path.parent != directory or path.name != row["file"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Selected update wheel changed after verification")
        # SFTP's quoted path grammar is independent of the remote shell.
        quoted = str(path).replace("\\", "\\\\").replace('"', '\\"')
        commands.append(f'put "{quoted}" {remote}/{path.name}')
    batch = directory / "transfer.batch"
    batch.write_text("\n".join(commands) + "\n")
    subprocess.run(["sftp", *request["sftp_arguments"], "-b", str(batch), request["host"]],
                   check=True, capture_output=True, text=True)
    reply = ssh(arguments, "python3 /opt/cozy/dev/update.py apply " + stage + " " +
                " ".join(row["sha256"] for row in wheels))
    result = json.loads(reply.strip().splitlines()[-1])
    if result.get("operation") != stage or result.get("state") not in {
        "queued", "running", "succeeded", "rolled_back", "refused"
    }:
        raise ValueError("Worker did not acknowledge this durable Runtime update")
    return {"update": result}



if __name__ == "__main__":
    try:
        print(json.dumps(main(), sort_keys=True))
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        # Keep SSH/native tracebacks out of ordinary end-user output. The CLI
        # retains the complete subprocess diagnostics in this update's log.
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
