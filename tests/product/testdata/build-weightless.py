"""Build the weightless package archive used by TestProductPath."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import time

import tomllib

SKIP = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "dist"}
FIXTURE = pathlib.Path(__file__).with_name("weightless")


def run(
    *args: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pack(
    tree: pathlib.Path, package: str, version: str, archive: pathlib.Path
) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(tree.rglob("*")):
        relative = path.relative_to(tree)
        if (
            any(part in SKIP for part in relative.parts)
            or not path.is_file()
            or path.is_symlink()
        ):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": digest(path),
                "size": path.stat().st_size,
            }
        )

    declaration = json.dumps(
        {"package": package, "version": version, "files": files}, indent=2
    ).encode()
    with tarfile.open(archive, "w:gz") as output:
        header = tarfile.TarInfo("release.json")
        header.size, header.mtime, header.mode = (
            len(declaration),
            int(time.time()),
            0o644,
        )
        output.addfile(header, io.BytesIO(declaration))
        for row in files:
            name = str(row["path"])
            output.add(tree / name, arcname=name, recursive=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-repo", default=os.getenv("RUNTIME_REPO", "~/cozy_v2/cozy-runtime")
    )
    parser.add_argument("--runtime-sha", default=os.getenv("RUNTIME_SHA", "HEAD"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--package", default="cozy/weightless")
    args = parser.parse_args()

    runtime_repo = pathlib.Path(args.runtime_repo).expanduser().resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    full_sha = subprocess.check_output(
        ["git", "-C", str(runtime_repo), "rev-parse", args.runtime_sha], text=True
    ).strip()

    with tempfile.TemporaryDirectory(prefix="weightless-", dir=out) as scratch:
        work = pathlib.Path(scratch)
        runtime = work / "runtime"
        tree = work / "tree"
        vendor = tree / "vendor"
        runtime.mkdir()
        vendor.mkdir(parents=True)

        archived = subprocess.check_output(
            ["git", "-C", str(runtime_repo), "archive", full_sha]
        )
        with tarfile.open(fileobj=io.BytesIO(archived)) as source:
            source.extractall(runtime)

        run(
            "uv",
            "build",
            "--wheel",
            "--project",
            str(runtime),
            "--out-dir",
            str(vendor),
        )
        runtime_version = tomllib.loads((runtime / "pyproject.toml").read_text())[
            "project"
        ]["version"]
        wheels = list(vendor.glob("cozy_runtime-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one cozy-runtime wheel, found {len(wheels)}")

        shutil.copy2(FIXTURE / "weightless.py", tree / "weightless.py")
        shutil.copy2(FIXTURE / "package.toml", tree / "package.toml")
        (tree / "pyproject.toml").write_text(
            "[project]\n"
            'name = "cozy-weightless-package"\n'
            f'version = "{args.version}"\n'
            'requires-python = ">=3.11"\n'
            f'dependencies = ["cozy-runtime[media]=={runtime_version}"]\n\n'
            "[tool.uv.sources]\n"
            f'cozy-runtime = {{ path = "vendor/{wheels[0].name}" }}\n'
        )
        run("uv", "lock", "--quiet", cwd=tree)
        run("uv", "sync", "--locked", "--quiet", cwd=tree)
        env = dict(os.environ, PYTHONPATH=str(tree))
        run(
            str(tree / ".venv/bin/python"),
            "-m",
            "cozy_runtime.cli.main",
            "--dir",
            str(tree),
            "describe",
            "--json",
            cwd=tree,
            env=env,
        )

        archive = out / f"{args.package.rsplit('/', 1)[-1]}-{args.version}.tar.gz"
        pack(tree, args.package, args.version, archive)
        print(f"archive:  {archive}")
        print(f"runtime: {full_sha} (git archive, read-only)")
        print(f"digest:  sha256:{digest(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
