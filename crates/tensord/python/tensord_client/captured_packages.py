"""Capture a local package's own closure with standard uv, preserving its lock/constraints.

This helper builds one environment of what the package itself selects. Everything after is the
machine's, one path for a release and a local package (`environments.rs`): this machine's
Runtime/TensorFS pair over the closure (`without` leaves the pair's distributions to it, and
`restore` installs the package's own instead), the machine client, uv's check, the description
and the generation record.
"""
from __future__ import annotations

import configparser
import email.parser
import itertools
import os
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from . import layout
from .package_records import Captured, PackageMetadata
from .packages import PackageError, read_metadata


def wheel_metadata(path: Path) -> PackageMetadata:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 1 << 20:
            raise PackageError("package_wheel_metadata_invalid", "one bounded wheel metadata record is required")
        headers = email.parser.Parser().parsestr(archive.read(names[0]).decode(), headersonly=True)
        entry_name = names[0].removesuffix("METADATA") + "entry_points.txt"
        if entry_name not in archive.namelist() or archive.getinfo(entry_name).file_size > 1 << 20:
            raise PackageError("package_application_absent", "the selected wheel has no bounded App declaration")
        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(archive.read(entry_name).decode())
        applications = list(entries.items("cozy.application"))
        if len(applications) != 1:
            raise PackageError("package_application_invalid", "one App declaration is required")
        return PackageMetadata(headers["Name"], headers["Version"], applications[0][1])


#: Under symlinks (an image's seeded cache) the code the Runtime describes (an App's wheel, the
#: SDK, the machine client) is copied into the environment: its static reader refuses metadata
#: and sources linked from elsewhere. Only the locked dependencies are linked.
COPY = ("--link-mode", "copy") if os.environ.get("UV_LINK_MODE") == "symlink" else ()


def local_builds(project: Path) -> list[str]:
    """The lock's packages built from a directory on this machine. uv keys a built directory by
    its metadata alone, so an environment built because their code changed builds them again."""
    lock = tomllib.loads((project / "uv.lock").read_text())
    return [row["name"] for row in lock.get("package", ())
            if {"directory", "editable", "path"} & set(row.get("source", {})) and row["source"].get("editable") != "."]


def linked_applications(interpreter: Path) -> list[str]:
    """Installed distributions that declare an App and were linked, not copied: the Runtime's
    static reader describes only copied ones (a callee package among the locked dependencies)."""
    names = []
    for site in layout.site_packages(interpreter.parents[1]):
        for metadata in site.glob("*.dist-info"):
            entries = metadata / "entry_points.txt"
            if entries.is_symlink() and "[cozy.application]" in entries.read_text():
                names.append(metadata.name.removesuffix(".dist-info").rpartition("-")[0])
    return names


def interpreter_for(python: str, python_requires: str, python_version: str) -> str:
    """The machine's interpreter for a package, refused when its declared Python excludes it."""
    python = subprocess.check_output(["uv", "python", "find", "--no-project", "--no-python-downloads", python]).decode().strip()
    version = Version(subprocess.check_output([python, "-I", "-S", "-c", "import platform;print(platform.python_version())"]).decode().strip())
    if python_requires and version not in SpecifierSet(python_requires):
        raise PackageError("package_python_unavailable", "configured interpreter does not satisfy captured Python requirements")
    if python_version and str(version) != python_version and ".".join(str(version).split(".")[:2]) != python_version:
        raise PackageError("package_python_unavailable", "configured interpreter does not match captured Python selection")
    return python


def without(requirements: Path, names: set[str], out: Path) -> Path:
    """`requirements` (hashed, `\\`-continued) less the requirements naming `names`, at `out`."""
    kept, entry = [], []
    for line in requirements.read_text().splitlines():
        entry.append(line)
        if line.rstrip().endswith("\\"):
            continue
        first = entry[0].strip()
        name = "".join(itertools.takewhile(lambda c: c.isalnum() or c in "-_.", first))
        if not (name and not first.startswith(("#", "-")) and canonicalize_name(name) in names):
            kept.extend(entry)
        entry = []
    kept.extend(entry)
    out.write_text("\n".join(kept) + "\n")
    return out


def live_paths(environment: Path) -> None:
    """Spell an editable install's source paths as `import` lines. Python adds them at startup
    all the same, and the Runtime's static reader, which never interprets such a line, reads the
    environment's installed packages without following the project out of it: the project is
    described from its own directory."""
    for path in (pth for site in layout.site_packages(environment) for pth in site.glob("*.pth")):
        lines = path.read_text().splitlines()
        live = [line for line in lines if line and not line.startswith(("#", "import ", "import\t"))
                and not (path.parent / line.rstrip()).resolve().is_relative_to(environment.resolve())]
        if live:
            path.write_text("".join(f"import sys; sys.path.append({line.rstrip()!r})\n" if line in live else line + "\n" for line in lines))


def pair_names(sdk: list[Path]) -> set[str]:
    return {canonicalize_name(wheel.name.split("-")[0]) for wheel in sdk}


def capture_editable(*, project: Path, environment: Path, wheels: list[Path], distribution: str, release: str,
                     python_requires: str, python_version: str, python: str, sdk: list[Path] = (),
                     restore: bool = False, indexes: list[str] = ()) -> Captured:
    """The live project installed editable in the controller's source home and environment: a
    ready environment is reused as it stands while its lock does. Remote captures keep
    separate source homes/environments; direct local paths reuse their live environment.
    `wheels` are locked dependencies the caller carries because this machine cannot fetch them:
    each installs from its file, at the version the lock selected. `indexes` (name=url) are
    those the project's sources name without declaring: a lock already pins where each
    dependency comes from, so only an unlocked project resolves against them. `sdk` is the pair
    the machine installs next, left out of a lock and offered to an unlocked resolution (PyPI's
    may not install here: no macOS TensorFS wheel); `restore` installs the project's own."""
    python = interpreter_for(python, python_requires, python_version)
    project = project.resolve(strict=True)
    metadata = read_metadata(project)
    if canonicalize_name(metadata.name) != canonicalize_name(distribution) or Version(metadata.version) != Version(release):
        raise PackageError("package_metadata_changed", "the live project declares another package release")
    interpreter, ready = layout.interpreter(environment), environment / ".ready"
    if ready.is_file() and not restore:
        return Captured(metadata.name, metadata.version, metadata.application, str(interpreter), ready.read_text())
    if not restore:
        shutil.rmtree(environment, ignore_errors=True)  # a build that never finished
        environment.parent.mkdir(parents=True, exist_ok=True)
    machine = set() if restore else pair_names(sdk)
    if (project / "uv.lock").is_file():
        # Each carried distribution at the version the lock selected, from its own files (uv
        # picks the one this platform takes); the lock installs the rest, every dependency as
        # an ordinary install: only the project itself is live.
        carried = sorted({tuple(wheel.name.split("-")[:2]) for wheel in wheels})
        sync = ["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--no-install-project", "--no-python-downloads",
                "--project", str(project), "--python", python,
                *(f"--no-install-package={name}" for name in sorted({name for name, _ in carried} | machine)),
                *(f"--refresh-package={name}" for name in local_builds(project))]
        placed = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
        subprocess.run(sync, env=placed, check=True)
        linked = linked_applications(interpreter) if COPY else []
        if linked:
            subprocess.run([*sync, *COPY, *(f"--reinstall-package={name}" for name in linked)], env=placed, check=True)
        if carried:
            subprocess.run(["uv", "pip", "install", "--no-config", *COPY, "--python", str(interpreter), "--no-deps",
                            "--no-index", "--find-links", str(wheels[0].parent),
                            *(f"{name}=={version}" for name, version in carried)], check=True)
        # Its dependencies are in place: the project alone, whatever indexes its sources name.
        subprocess.run(["uv", "pip", "install", "--no-config", "--python", str(interpreter), "--no-deps", "--no-sources",
                        "--editable", str(project)], check=True)
    else:
        subprocess.run(["uv", "venv", "--no-config", "--allow-existing", "--python", python, str(environment)], check=True)
        found = ["--find-links", str(wheels[0].parent)] if wheels else []
        named = [f"--index={index}" for index in indexes]
        install = ["uv", "pip", "install", "--no-config", "--python", str(interpreter), *found, *named, "--editable", str(project)]
        offered = [str(wheel) for wheel in sdk] if machine else []
        if not offered or subprocess.run([*install, *offered]).returncode != 0:
            subprocess.run(install, check=True)
    live_paths(environment)
    return Captured(metadata.name, metadata.version, metadata.application, str(interpreter))


def capture(*, root: Path, project: Path | None, wheels: list[Path], requirements: Path | None, distribution: str,
            release: str, python_requires: str, python_version: str, python: str, sdk: list[Path] = (),
            restore: bool = False) -> Captured:
    """The capture's own selection in `root`/env, without the `sdk` pair distributions the
    machine installs next: the pair is installed once, never the capture's first and then
    replaced. An unlocked capture resolves with the pair offered, where its bounds admit it.
    `restore` installs what the machine's pair replaced."""
    python = interpreter_for(python, python_requires, python_version)
    project = project.resolve() if project else None
    wheels = [wheel.resolve(strict=True) for wheel in wheels]
    if project and (project / "uv.lock").is_file() and (wheels or requirements):
        raise PackageError("package_capture_mixed_lock_unsupported", "a frozen source capture cannot overlay an independent wheel/requirements closure")
    if project:
        metadata = read_metadata(project)
    else:
        matches = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 1 << 20:
                    raise PackageError("package_wheel_metadata_invalid", "invalid bounded wheel metadata")
                name = email.parser.Parser().parsestr(archive.read(names[0]).decode(), headersonly=True)["Name"]
                if canonicalize_name(name) == canonicalize_name(distribution):
                    matches.append(wheel_metadata(wheel))
        if len(matches) != 1:
            raise PackageError("package_root_wheel_absent", "one captured wheel must name the requested distribution")
        metadata = matches[0]
    if canonicalize_name(metadata.name) != canonicalize_name(distribution) or Version(metadata.version) != Version(release):
        raise PackageError("package_metadata_changed", "uploaded source/wheel declares another package release")
    interpreter = layout.interpreter(root / "env")
    frozen = project and (project / "uv.lock").is_file()
    source = root / "source"
    machine = set() if restore else pair_names(sdk)
    if not restore:
        if frozen:
            shutil.copytree(project, source)
        else:
            subprocess.run(["uv", "venv", "--no-config", "--python", python, str(root / "env")], check=True)
    if frozen:
        # UV_PROJECT_ENVIRONMENT is a standard destination configuration value; the project's
        # own `[tool.uv]` settings hold, as a release's do.
        environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "env")}
        sync = ["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--no-python-downloads",
                "--project", str(source), "--python", python,
                *(f"--no-install-package={name}" for name in sorted(machine))]
        if COPY:
            # The locked dependencies link; then the project itself, and any App among them (a
            # callee the lock holds by path), is copied.
            subprocess.run([*sync, "--no-install-project"], env=environment, check=True)
            subprocess.run([*sync, *COPY], env=environment, check=True)
            linked = linked_applications(interpreter)
            if linked:
                subprocess.run([*sync, *COPY, *(f"--reinstall-package={name}" for name in linked)], env=environment, check=True)
        else:
            subprocess.run(sync, env=environment, check=True)
    elif requirements and requirements.stat().st_size:
        # uv owns hashed requirements parsing and standard download integrity. The locked
        # dependencies link as the machine says (an image's seeded cache: symlinks). Less the
        # pair it is a whole closure: installed as listed, the pair comes next.
        listed = without(requirements, machine, root / "requirements.txt") if machine else requirements
        command = ["uv", "pip", "install", "--no-config", "--python", str(interpreter), "--require-hashes", "-r", str(listed)]
        if wheels:
            command += ["--find-links", str(wheels[0].parent)]
        subprocess.run([*command, *(["--no-deps"] if machine else [])], check=True)
        linked = linked_applications(interpreter) if COPY else []
        if linked:
            subprocess.run([*command, *COPY, "--no-deps", *(f"--reinstall-package={name}" for name in linked)], check=True)
        if wheels:
            subprocess.run(["uv", "pip", "install", "--no-config", *COPY, "--python", str(interpreter), "--no-deps",
                            *map(str, wheels)], check=True)
    else:
        install = ["uv", "pip", "install", "--no-config", *COPY, "--python", str(interpreter),
                   *([str(project)] if project else []), *map(str, wheels)]
        offered = [str(wheel) for wheel in sdk] if machine else []
        if not offered or subprocess.run([*install, *offered]).returncode != 0:
            subprocess.run(install, check=True)
    return Captured(metadata.name, metadata.version, metadata.application, str(interpreter))
