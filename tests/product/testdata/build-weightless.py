"""Build the editable weightless source tree used by the product test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

FIXTURE = pathlib.Path(__file__).with_name("weightless")


def run(
    *args: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def released_wheel(version: str, into: pathlib.Path) -> pathlib.Path:
    """Download the released pure wheel of cozy-runtime `version`, checked against PyPI's digest."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/cozy-runtime/{version}/json") as reply:
        urls = json.load(reply)["urls"]
    wheels = [u for u in urls if u["filename"].endswith("py3-none-any.whl")]
    if len(wheels) != 1:
        raise RuntimeError(f"cozy-runtime {version} has {len(wheels)} pure wheels on PyPI")
    with urllib.request.urlopen(wheels[0]["url"]) as reply:
        raw = reply.read()
    if hashlib.sha256(raw).hexdigest() != wheels[0]["digests"]["sha256"]:
        raise RuntimeError(f"{wheels[0]['filename']} does not match PyPI's sha256")
    path = into / wheels[0]["filename"]
    path.write_bytes(raw)
    return path


def embedded_commit(wheel: pathlib.Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        text = archive.read("cozy_runtime/_build_provenance.py").decode()
    found = re.search(r'COMMIT = "([0-9a-f]{40})"', text)
    if found is None:
        raise RuntimeError(f"{wheel.name} embeds no source commit")
    return found.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runtime-wheel", help="exact Runtime wheel to vendor")
    source.add_argument("--runtime-version", help="released Runtime version to fetch from PyPI")
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument(
        "--expect-commit",
        help="the released host cozy-runtime's commit; the vendored wheel must embed it",
    )
    provenance.add_argument(
        "--expect-sha256",
        help="explicit development artifact sha256; does not assert a released Git identity",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-out")
    parser.add_argument("--tensorfs-wheel", default="")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    if args.expect_sha256 and (
        not args.runtime_wheel or not re.fullmatch(r"[0-9a-f]{64}", args.expect_sha256)
    ):
        parser.error("--expect-sha256 requires an explicit wheel and one lowercase sha256")

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="weightless-", dir=out) as scratch:
        work = pathlib.Path(scratch)
        tree = work / "tree"
        vendor = tree / "vendor"
        vendor.mkdir(parents=True)

        if args.runtime_wheel:
            wheel = vendor / pathlib.Path(args.runtime_wheel).name
            shutil.copy2(pathlib.Path(args.runtime_wheel).resolve(), wheel)
        else:
            wheel = released_wheel(args.runtime_version, vendor)
        if args.expect_sha256:
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != args.expect_sha256:
                raise RuntimeError("selected Runtime wheel does not match the explicit sha256")
            verified_identity = "sha256:" + args.expect_sha256
        else:
            full_sha = embedded_commit(wheel)
            if not full_sha.startswith(args.expect_commit):
                raise RuntimeError(
                    f"{wheel.name} embeds {full_sha} and the host cozy-runtime is {args.expect_commit}; "
                    "pass -script-runtime-wheel with the wheel the host Runtime was installed from"
                )
            verified_identity = full_sha
        runtime_version = wheel.name.split("-")[1].split("+", 1)[0]
        wheels = [wheel]
        native = pathlib.Path(args.tensorfs_wheel).resolve() if args.tensorfs_wheel else None
        if native is not None:
            shutil.copy2(native, vendor / native.name)

        shutil.copy2(FIXTURE / "weightless.py", tree / "weightless.py")
        shutil.copy2(FIXTURE / "package.toml", tree / "package.toml")
        # The backend is declared, and told exactly which file is the module. Left to
        # guess a flat layout, setuptools shipped a wheel holding only .dist-info (its
        # top_level.txt said `vendor`), which a rented pod refused minutes later. The
        # entry point is the wheel's spelling of package.toml's [application] object: an
        # editable run reads package.toml, a worker reads the installed wheel.
        # Declare image-compatible Runtime support; the vendored wheel source and
        # uv.lock below still select the exact qualified Runtime revision.
        (tree / "pyproject.toml").write_text(
            "[project]\n"
            'name = "cozy-weightless-package"\n'
            f'version = "{args.version}"\n'
            'requires-python = ">=3.12,<3.13"\n'
            f'dependencies = ["cozy-runtime[media]>={runtime_version},<1"]\n\n'
            '[project.entry-points."cozy.application"]\n'
            'default = "weightless:app"\n\n'
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            'only-include = ["weightless.py"]\n\n'
            "[tool.uv.sources]\n"
            f'cozy-runtime = {{ path = "vendor/{wheels[0].name}" }}\n'
            + (f'tensorfs = {{ path = "vendor/{native.name}" }}\n' if native is not None else "")
        )
        run("uv", "lock", "--quiet", cwd=tree)
        run("uv", "sync", "--locked", "--quiet", cwd=tree)
        # The committed PackageInterface every publishable tree carries (cl-175): publication
        # pre-flights it against this host's static reading and uploads it unchanged.
        env = dict(os.environ, PYTHONPATH=str(tree))
        interface = subprocess.check_output(
            [
                str(tree / ".venv/bin/python"),
                "-m",
                "cozy_runtime.cli.main",
                "--dir",
                str(tree),
                "describe",
                "--json",
            ],
            cwd=tree,
            env=env,
        )
        (tree / "metadata").mkdir()
        (tree / "metadata" / "package-interface.json").write_bytes(interface)

        if args.source_out:
            source_out = pathlib.Path(args.source_out).resolve()
            shutil.copytree(
                tree,
                source_out,
                ignore=shutil.ignore_patterns(".venv", "__pycache__", "dist"),
            )
        print(f"source:   {source_out if args.source_out else tree}")
        print(f"runtime: {verified_identity} ({wheel.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
