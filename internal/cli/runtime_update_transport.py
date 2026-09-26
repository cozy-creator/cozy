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
from collections.abc import Sequence
from typing import TypeGuard, TypedDict

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


type JSONObject = dict[str, object]


class SelectedWheel(TypedDict):
    version: str
    filename: str
    url: str
    digest: str
    length: int
    path: str


class WheelArtifact(TypedDict):
    distribution: str
    version: str
    file: str
    path: str
    sha256: str


class RuntimeVersion(TypedDict):
    distribution: str
    supports_guarded_restart: bool


class Observation(TypedDict):
    runtime: RuntimeVersion
    tensorfs: str
    python: str
    tags: list[str]
    updater: bool
    durable_updates: bool
    markers: dict[str, str]


class RuntimePair(TypedDict):
    runtime: SelectedWheel
    tensorfs: SelectedWheel


class RuntimeTarget(TypedDict):
    runtime_update: RuntimePair


class RuntimeWire(TypedDict):
    wire_minor: int
    minimum_wire_minor: int


class IndexFile(TypedDict):
    filename: str
    url: str
    digest: str
    length: int
    requires_python: str
    yanked: bool
    packagetype: str


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Runtime update metadata field must be a string")
    return value


def integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Runtime update metadata field must be an integer")
    return value


def boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Runtime update metadata field must be a boolean")
    return value


def object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    # This guard establishes only a container of objects. Field readers below
    # still validate every key and value used by the maintenance contract.
    return isinstance(value, dict)


def object_sequence(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def mapping(value: object) -> JSONObject:
    if not object_mapping(value):
        raise ValueError("Runtime update metadata must be an object")
    return {text(key): item for key, item in value.items()}


def sequence(value: object) -> list[object]:
    if not object_sequence(value):
        raise ValueError("Runtime update metadata field must be a list")
    return list(value)


def strings(value: object) -> list[str]:
    return [text(item) for item in sequence(value)]


def selected_wheel(value: object) -> SelectedWheel:
    row = mapping(value)
    return {
        "version": text(row.get("version", "")), "filename": text(row["filename"]),
        "url": text(row.get("url", "")), "digest": text(row["digest"]),
        "length": integer(row["length"]), "path": text(row.get("path", "")),
    }


def optional_wheel(value: object) -> SelectedWheel | None:
    return None if value is None else selected_wheel(value)


def wheel_artifact(value: object) -> WheelArtifact:
    row = mapping(value)
    return {
        "distribution": text(row["distribution"]), "version": text(row["version"]),
        "file": text(row["file"]), "path": text(row["path"]), "sha256": text(row["sha256"]),
    }


def observation(value: object) -> Observation:
    row = mapping(value)
    runtime = mapping(row["runtime"])
    return {
        "runtime": {
            "distribution": text(runtime["distribution"]),
            "supports_guarded_restart": boolean(runtime.get("supports_guarded_restart", False)),
        },
        "tensorfs": text(row["tensorfs"]), "python": text(row["python"]),
        "tags": strings(row["tags"]), "updater": boolean(row["updater"]),
        "durable_updates": boolean(row.get("durable_updates", False)),
        "markers": {key: text(item) for key, item in mapping(row["markers"]).items()},
    }


def index_file(value: object) -> IndexFile:
    row = mapping(value)
    return {
        "filename": text(row["filename"]), "url": text(row["url"]),
        "digest": text(mapping(row["digests"])["sha256"]), "length": integer(row["size"]),
        "requires_python": text(row.get("requires_python") or ""),
        "yanked": boolean(row.get("yanked", False)), "packagetype": text(row["packagetype"]),
    }


WORKER_PYTHONS = ("/opt/cozy/python/bin/python3", "/usr/local/bin/python3")

def ssh(arguments: Sequence[str], command: str) -> str:
    result = subprocess.run(
        ["ssh", *arguments, command], capture_output=True, text=True
    )
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise ValueError(f"Worker maintenance failed (SSH exit {result.returncode}): {result.stderr.strip()[-1024:]}")
    return result.stdout


def worker_python(arguments: Sequence[str]) -> str:
    # Both first-party image layouts have a fixed SDK interpreter. Do not use
    # SSH's PATH, which can select an unrelated system Python without Runtime.
    command = "for python in " + " ".join(WORKER_PYTHONS) + '; do if test -x "$python"; then printf "%s" "$python"; exit 0; fi; done; exit 1'
    selected = ssh(arguments, command).strip()
    if selected not in WORKER_PYTHONS:
        raise ValueError("Worker maintenance did not select a supported SDK interpreter")
    return selected


def inspect(arguments: Sequence[str], probe: str, python: str | None = None) -> JSONObject:
    # The script is fixed application code, never a package callback or prompt.
    import shlex

    return mapping(json.loads(ssh(arguments, (python or worker_python(arguments)) + " -I -c " + shlex.quote(probe))))


def fetch(url: str, limit: int) -> bytes:
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
        data: object = response.read(limit + 1)
    if not isinstance(data, bytes):
        raise ValueError("Runtime update response did not contain bytes")
    if len(data) > limit:
        raise ValueError("Runtime update artifact exceeds its download bound")
    return data


def wheel(name: str, selected: SelectedWheel, tags: Sequence[str], directory: Path,
          native: bool) -> WheelArtifact:
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



def published(name: str, observed: Observation, constraint: str = "") -> SelectedWheel:
    """Choose the newest non-yanked native release for this exact interpreter."""
    document = mapping(json.loads(fetch(f"https://pypi.org/pypi/{name}/json", 32 << 20)))
    supported = [tag for text in observed["tags"] for tag in parse_tag(text)]
    rank = {tag: index for index, tag in enumerate(supported)}
    installed = observed["runtime"]["distribution"] if name == "cozy-runtime" else observed["tensorfs"]
    candidates: list[tuple[Version, int, IndexFile]] = []
    for release, files in mapping(document["releases"]).items():
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or Version(version.public) < Version(Version(installed).public):
            continue
        if version not in SpecifierSet(constraint):
            continue
        for value in sequence(files):
            row = index_file(value)
            if row["yanked"] or row["packagetype"] != "bdist_wheel":
                continue
            project, artifact_version, _, tags = parse_wheel_filename(row["filename"])
            matches = set(rank).intersection(tags)
            if (project != name or artifact_version != version or not matches
                or all(tag.platform == "any" for tag in tags)
                or Version(observed["python"]) not in SpecifierSet(row["requires_python"])):
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
            "digest": "sha256:" + row["digest"], "length": row["length"], "path": ""}


def metadata(row: WheelArtifact, observed: Observation) -> list[Requirement]:
    # A wheel filename is not authority for the distribution contained inside it.
    with zipfile.ZipFile(row["path"]) as archive:
        files = [item for item in archive.infolist()
                 if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/METADATA")]
        if len(files) != 1 or files[0].file_size > 1 << 20:
            raise ValueError("Published wheel must contain one bounded METADATA")
        info = email.parser.BytesParser().parsebytes(archive.read(files[0]))
    if canonicalize_name(text(info.get("Name", ""))) != row["distribution"] or info.get("Version") != row["version"]:
        raise ValueError("Published wheel metadata does not match its selected distribution/version")
    if Version(observed["python"]) not in SpecifierSet(text(info.get("Requires-Python", ""))):
        raise ValueError("Published wheel metadata requires a different worker Python")
    requirements: list[Requirement] = []
    for value in info.get_all("Requires-Dist", []):
        required = Requirement(text(value))
        if required.marker is None or any(required.marker.evaluate({**observed["markers"], "extra": extra}) for extra in ("", "model-execution")):
            requirements.append(required)
    return requirements


def runtime_wire(row: WheelArtifact) -> RuntimeWire | None:
    # The selected Runtime's own declared cozy.worker.v1 range, read from its wheel.
    with zipfile.ZipFile(row["path"]) as archive:
        try:
            text = archive.read("cozy/worker/v1/wire_version.py").decode()
        except KeyError:
            return None
    values = dict(re.findall(r"^(WIRE_MINOR|MIN_COMPATIBLE_WIRE_MINOR) = (\d+)$", text, re.M))
    if set(values) != {"WIRE_MINOR", "MIN_COMPATIBLE_WIRE_MINOR"}:
        return None
    return {"wire_minor": int(values["WIRE_MINOR"]), "minimum_wire_minor": int(values["MIN_COMPATIBLE_WIRE_MINOR"])}


def resolve(observed: Observation, directory: Path,
            local_runtime: SelectedWheel | None = None,
            local_tensorfs: SelectedWheel | None = None) -> tuple[RuntimeTarget, list[WheelArtifact]]:
    if local_tensorfs is not None and local_runtime is None:
        raise ValueError("Local TensorFS requires a local Runtime wheel")
    if local_runtime is None:
        runtime = published("cozy-runtime", observed)
    else:
        runtime = local_runtime.copy()
        project, version, _, _ = parse_wheel_filename(runtime["filename"])
        # Local build labels identify artifacts, not chronological releases.
        # Replacing a development build of the same public release is valid.
        if project != "cozy-runtime" or Version(version.public) < Version(Version(observed["runtime"]["distribution"]).public):
            raise ValueError("Local Runtime wheel must be cozy-runtime without downgrading its public release")
        runtime["version"] = str(version)
    runtime_wheel = wheel("cozy-runtime", runtime, observed["tags"], directory, True)
    requirements = metadata(runtime_wheel, observed)
    tensorfs_requirements = [requirement for requirement in requirements if canonicalize_name(requirement.name) == "tensorfs"]
    if any(requirement.url for requirement in tensorfs_requirements):
        raise ValueError("Runtime requires a non-index TensorFS artifact; publish a portable release")
    constraint = ",".join(str(requirement.specifier) for requirement in tensorfs_requirements)
    if local_tensorfs is None:
        tensorfs = published("tensorfs", observed, constraint)
    else:
        tensorfs = local_tensorfs.copy()
        project, version, _, _ = parse_wheel_filename(tensorfs["filename"])
        if project != "tensorfs" or Version(version.public) < Version(Version(observed["tensorfs"]).public):
            raise ValueError("Local TensorFS wheel must be tensorfs without downgrading the installed version")
        if not SpecifierSet(constraint).contains(version, prereleases=True):
            raise ValueError("Local TensorFS wheel is incompatible with the selected Runtime requirements")
        tensorfs["version"] = str(version)
    tensorfs_wheel = wheel("tensorfs", tensorfs, observed["tags"], directory, True)
    for required in metadata(tensorfs_wheel, observed):
        if canonicalize_name(required.name) == "cozy-runtime" and (required.url or Version(runtime["version"]) not in required.specifier):
            raise ValueError(f"Published TensorFS {tensorfs['version']} requires {required}; selected Runtime {runtime['version']} is incompatible")
    return {"runtime_update": {"runtime": runtime, "tensorfs": tensorfs}}, [runtime_wheel, tensorfs_wheel]


def main() -> JSONObject:
    request = mapping(json.load(sys.stdin))
    arguments = strings(request["ssh_arguments"])
    action = text(request["action"])
    probe = text(request["probe"])
    python = worker_python(arguments)
    if action == "resume":
        stage = text(request["stage"])
        expected = [wheel_artifact(row)["sha256"] for row in sequence(mapping(request["selection"])["wheels"])]
        if not re.fullmatch(r"[a-f0-9]{32}", stage) or len(expected) != 2 or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in expected):
            raise ValueError("Invalid recorded update identity")
        observed_update = mapping(json.loads(ssh(arguments, python + " /opt/cozy/dev/update.py status " + stage)))
        if observed_update.get("operation") != stage:
            raise ValueError("Worker update status does not match the recorded operation")
        if observed_update.get("state") != "missing":
            return {"update": mapping(json.loads(ssh(arguments, python + " /opt/cozy/dev/update.py apply " + stage + " " + " ".join(expected))))}
        # Transfer may have ended before enqueue. No accepted operation exists,
        # so replay the same frozen, hash-checked pair through ordinary apply.
        # Never reselect releases, or overwrite staging owned by a live updater.
        request = {**request, "action": "apply"}
    if action == "status":
        stage = text(request["stage"])
        if not re.fullmatch(r"[a-f0-9]{32}", stage):
            raise ValueError("Invalid update operation identity")
        update = mapping(json.loads(ssh(arguments, python + " /opt/cozy/dev/update.py status " + stage)))
        result: JSONObject = {"update": update}
        if update.get("state") in {"succeeded", "rolled_back", "refused"}:
            result["observed"] = inspect(arguments, probe, python)
        return result
    observed_document = inspect(arguments, probe, python)
    if request["action"] == "inspect":
        return {"observed": observed_document}
    if observed_document.get("update_in_progress"):
        raise ValueError("The worker is already applying a Runtime update")
    observed = observation(observed_document)
    if not observed["updater"] or not observed.get("durable_updates", False) or not observed["runtime"].get("supports_guarded_restart", False):
        raise ValueError("This worker image does not support safe Runtime updates; select a current maintenance-capable private image")
    directory = Path(text(request["directory"]))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if request["action"] == "plan":
        target, selection = resolve(observed, directory, optional_wheel(request.get("local_runtime")), optional_wheel(request.get("local_tensorfs")))
        unchanged = (
            observed["runtime"]["distribution"] == target["runtime_update"]["runtime"]["version"]
            and observed["tensorfs"] == target["runtime_update"]["tensorfs"]["version"]
        )
        return {"observed": observed_document, "target": target, "wheels": selection, "unchanged": unchanged,
                "runtime_wire": runtime_wire(selection[0])}
    if request["action"] != "apply":
        raise ValueError("Unknown Runtime update operation")
    stage = text(request["stage"])
    if not re.fullmatch(r"[a-f0-9]{32}", stage):
        raise ValueError("Invalid update staging identity")
    wheels = [wheel_artifact(row) for row in sequence(mapping(request["selection"])["wheels"])]
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
    try:
        subprocess.run(["sftp", *strings(request["sftp_arguments"]), "-b", str(batch), text(request["host"])],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise
    reply = ssh(arguments, python + " /opt/cozy/dev/update.py apply " + stage + " " +
                " ".join(row["sha256"] for row in wheels))
    result = mapping(json.loads(reply.strip().splitlines()[-1]))
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
