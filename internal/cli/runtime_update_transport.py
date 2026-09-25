"""Fixed developer-worker maintenance transport, embedded in the Creator CLI.

Only first-party Runtime/TensorFS distributions are downloaded. The worker's
existing root updater owns validation, replacement and rollback.
"""

import email.parser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile
import urllib.parse
import urllib.request

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


WORKER_PYTHON = "/opt/cozy/python/bin/python3"

PROBE = """import importlib.metadata as m, json, platform, subprocess, fcntl, sys
from packaging.markers import default_environment
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
runtime=json.loads(subprocess.check_output([str(Path(sys.executable).parent/'cozy-runtime'),'version','--json'],text=True))
base=Path('/var/lib/cozy/dev/base.json')
current=Path('/var/lib/cozy/dev/current')
updater=Path('/opt/cozy/dev/update.py')
capability_probe=subprocess.run([sys.executable,str(updater),'capabilities'],text=True,capture_output=True) if updater.is_file() else None
capabilities=json.loads(capability_probe.stdout) if capability_probe and capability_probe.returncode == 0 else {}
try: torch_version=m.version('torch')
except m.PackageNotFoundError: torch_version=''
print(json.dumps({'runtime':runtime,'tensorfs':m.version('tensorfs'),'python':platform.python_version(),
 'tags':[str(t) for t in sys_tags()], 'updater':base.is_file() and updater.is_file(), 'durable_updates':capabilities.get('durable_updates',False),
 'selection':str(current.resolve()) if current.exists() else '',
 'torch':torch_version, 'markers':default_environment()}))
"""


def ssh(arguments, command):
    result = subprocess.run(
        ["ssh", *arguments, command], capture_output=True, text=True
    )
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise ValueError(f"Worker maintenance failed (SSH exit {result.returncode}): {result.stderr.strip()[-1024:]}")
    return result.stdout


def inspect(arguments):
    # The script is fixed application code, never a package callback or prompt.
    import shlex

    return json.loads(ssh(arguments, WORKER_PYTHON + " -I -c " + shlex.quote(PROBE)))


def fetch(url, limit):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "pypi.org", "files.pythonhosted.org"
    }:
        raise ValueError("Runtime update artifact URL is outside the first-party release index")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError("Runtime update URL must identify one public release artifact")
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.geturl() != url:
            raise ValueError("Runtime update release URL unexpectedly redirected")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Runtime update artifact exceeds its download bound")
    return data


def wheel(name, selected, tags, directory, native):
    version, filename = selected["version"], selected["filename"]
    if name not in {"cozy-runtime", "tensorfs"} or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
        raise ValueError("Invalid published Runtime update wheel")
    project, release, _, wheel_tags = parse_wheel_filename(filename)
    supported = {tag for text in tags for tag in parse_tag(text)}
    if project != name or str(release) != version or not supported.intersection(wheel_tags):
        raise ValueError(f"Published {name} {version} does not support this worker's platform")
    if native and all(tag.platform == "any" for tag in wheel_tags):
        raise ValueError("The worker requires a published native wheel")
    digest = selected["digest"].removeprefix("sha256:")
    if not re.fullmatch(r"[a-f0-9]{64}", digest) or not 0 < selected["length"] <= 128 << 20:
        raise ValueError("Published wheel digest or length is invalid")
    path = directory / filename
    if not path.exists() or path.stat().st_size != selected["length"] or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        if selected.get("path"):
            # The daemon has already frozen these exact bytes before journaling.
            with Path(selected["path"]).open("rb") as source:
                data = source.read(selected["length"] + 1)
        else:
            data = fetch(selected["url"], selected["length"])
        if len(data) != selected["length"] or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Runtime update wheel failed published digest verification")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    return {"distribution": name, "version": version, "file": filename,
            "path": str(path), "sha256": digest}



def published(name, observed, constraint=""):
    """Choose the newest non-yanked native release for this exact interpreter."""
    document = json.loads(fetch(f"https://pypi.org/pypi/{name}/json", 32 << 20))
    supported = [tag for text in observed["tags"] for tag in parse_tag(text)]
    rank = {tag: index for index, tag in enumerate(supported)}
    installed = observed["runtime"]["distribution"] if name == "cozy-runtime" else observed["tensorfs"]
    candidates = []
    for release, files in document["releases"].items():
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or version < Version(installed):
            continue
        if version not in SpecifierSet(constraint):
            continue
        for row in files:
            if row.get("yanked") or row.get("packagetype") != "bdist_wheel":
                continue
            project, artifact_version, _, tags = parse_wheel_filename(row["filename"])
            matches = set(rank).intersection(tags)
            if (project != name or artifact_version != version or not matches
                or all(tag.platform == "any" for tag in tags)
                or Version(observed["python"]) not in SpecifierSet(row.get("requires_python") or "")):
                continue
            candidates.append((version, -min(rank[tag] for tag in matches), row))
    if not candidates:
        raise ValueError(f"No published native {name}{constraint} wheel supports this worker's Python {observed['python']} and platform without downgrading {installed}")
    version, _, row = max(candidates, key=lambda value: (value[0], value[1], value[2]["filename"]))
    filename = row["filename"]
    location = urllib.parse.urlparse(row["url"])
    if location.hostname != "files.pythonhosted.org" or location.path.rsplit("/", 1)[-1] != filename:
        raise ValueError("Published wheel URL does not identify its exact artifact")
    return {"version": str(version), "filename": filename, "url": row["url"],
            "digest": "sha256:" + row["digests"]["sha256"], "length": row["size"]}


def metadata(row, observed):
    # A wheel filename is not authority for the distribution contained inside it.
    with zipfile.ZipFile(row["path"]) as archive:
        files = [item for item in archive.infolist()
                 if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/METADATA")]
        if len(files) != 1 or files[0].file_size > 1 << 20:
            raise ValueError("Published wheel must contain one bounded METADATA")
        info = email.parser.BytesParser().parsebytes(archive.read(files[0]))
    if canonicalize_name(info.get("Name", "")) != row["distribution"] or info.get("Version") != row["version"]:
        raise ValueError("Published wheel metadata does not match its selected distribution/version")
    if Version(observed["python"]) not in SpecifierSet(info.get("Requires-Python", "")):
        raise ValueError("Published wheel metadata requires a different worker Python")
    requirements = []
    for value in info.get_all("Requires-Dist", []):
        required = Requirement(value)
        if required.marker is None or any(required.marker.evaluate({**observed["markers"], "extra": extra}) for extra in ("", "model-execution")):
            requirements.append(required)
    return requirements


def resolve(observed, directory, local_runtime=None):
    if local_runtime is None:
        runtime = published("cozy-runtime", observed)
    else:
        runtime = dict(local_runtime)
        project, version, _, _ = parse_wheel_filename(runtime["filename"])
        if project != "cozy-runtime" or version <= Version(observed["runtime"]["distribution"]):
            raise ValueError("Local Runtime wheel must be cozy-runtime with a distinct newer version")
        runtime["version"] = str(version)
    runtime_wheel = wheel("cozy-runtime", runtime, observed["tags"], directory, True)
    requirements = metadata(runtime_wheel, observed)
    tensorfs_requirements = [requirement for requirement in requirements if canonicalize_name(requirement.name) == "tensorfs"]
    if any(requirement.url for requirement in tensorfs_requirements):
        raise ValueError("Runtime requires a non-index TensorFS artifact; publish a portable release")
    constraint = ",".join(str(requirement.specifier) for requirement in tensorfs_requirements)
    tensorfs = published("tensorfs", observed, constraint)
    tensorfs_wheel = wheel("tensorfs", tensorfs, observed["tags"], directory, True)
    for required in metadata(tensorfs_wheel, observed):
        if canonicalize_name(required.name) == "cozy-runtime" and (required.url or Version(runtime["version"]) not in required.specifier):
            raise ValueError(f"Published TensorFS {tensorfs['version']} requires {required}; selected Runtime {runtime['version']} is incompatible")
    return {"runtime_update": {"runtime": runtime, "tensorfs": tensorfs}}, [runtime_wheel, tensorfs_wheel]


def main():
    request = json.load(sys.stdin)
    arguments = request["ssh_arguments"]
    action = request["action"]
    if action == "resume":
        stage = request["stage"]
        expected = [row["sha256"] for row in request["selection"]["wheels"]]
        if not re.fullmatch(r"[a-f0-9]{32}", stage) or len(expected) != 2 or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in expected):
            raise ValueError("Invalid recorded update identity")
        return {"update": json.loads(ssh(arguments, WORKER_PYTHON + " /opt/cozy/dev/update.py apply " + stage + " " + " ".join(expected)))}
    if action == "status":
        stage = request["stage"]
        if not re.fullmatch(r"[a-f0-9]{32}", stage):
            raise ValueError("Invalid update operation identity")
        update = json.loads(ssh(arguments, WORKER_PYTHON + " /opt/cozy/dev/update.py status " + stage))
        result = {"update": update}
        if update.get("state") in {"succeeded", "rolled_back", "refused"}:
            result["observed"] = inspect(arguments)
        return result
    observed = inspect(arguments)
    if request["action"] == "inspect":
        return {"observed": observed}
    if observed.get("update_in_progress"):
        raise ValueError("The worker is already applying a Runtime update")
    if not observed["updater"] or not observed.get("durable_updates", False) or not observed["runtime"].get("supports_guarded_restart", False):
        raise ValueError("This worker image does not support safe Runtime updates; select a current maintenance-capable private image")
    directory = Path(request["directory"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if request["action"] == "plan":
        target, selection = resolve(observed, directory, request.get("local_runtime"))
        unchanged = (
            observed["runtime"]["distribution"] == target["runtime_update"]["runtime"]["version"]
            and observed["tensorfs"] == target["runtime_update"]["tensorfs"]["version"]
        )
        return {"observed": observed, "target": target, "wheels": selection, "unchanged": unchanged}
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
    reply = ssh(arguments, WORKER_PYTHON + " /opt/cozy/dev/update.py apply " + stage + " " +
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
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, zipfile.BadZipFile) as error:
        # Keep SSH/native tracebacks out of ordinary end-user output. The CLI
        # retains the complete subprocess diagnostics in this update's log.
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
