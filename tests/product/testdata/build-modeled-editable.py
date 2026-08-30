"""Build the modeled editable package and its exact local TensorFS release."""

from __future__ import annotations

import argparse
import io
import pathlib
import shutil
import subprocess
import tarfile
import tempfile

import tomllib


def run(*args: str, cwd: pathlib.Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def archive(repo: pathlib.Path, revision: str, destination: pathlib.Path) -> str:
    full = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", revision], text=True
    ).strip()
    payload = subprocess.check_output(["git", "-C", str(repo), "archive", full])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        source.extractall(destination)
    return full


def one_wheel(root: pathlib.Path, prefix: str) -> pathlib.Path:
    wheels = list(root.glob(prefix + "-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one {prefix} wheel, found {len(wheels)}")
    return wheels[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", required=True, type=pathlib.Path)
    parser.add_argument("--runtime-sha", required=True)
    parser.add_argument("--tensorfs-repo", required=True, type=pathlib.Path)
    parser.add_argument("--tensorfs-sha", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--source-out", required=True, type=pathlib.Path)
    parser.add_argument("--store", required=True, type=pathlib.Path)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modeled-editable-", dir=out) as scratch:
        work = pathlib.Path(scratch)
        runtime, tensorfs, tree = work / "runtime", work / "tensorfs", work / "tree"
        vendor = tree / "vendor"
        runtime_sha = archive(args.runtime_repo.resolve(), args.runtime_sha, runtime)
        tensorfs_sha = archive(args.tensorfs_repo.resolve(), args.tensorfs_sha, tensorfs)
        vendor.mkdir(parents=True)
        run("uv", "build", "--wheel", "--project", str(runtime), "--out-dir", str(vendor))
        run("uv", "build", "--wheel", "--project", str(tensorfs), "--out-dir", str(vendor))
        runtime_wheel = one_wheel(vendor, "cozy_runtime")
        tensorfs_wheel = one_wheel(vendor, "tensorfs")
        runtime_version = tomllib.loads((runtime / "pyproject.toml").read_text())["project"][
            "version"
        ]
        tensorfs_version = tomllib.loads((tensorfs / "pyproject.toml").read_text())["project"][
            "version"
        ]

        fixture = runtime / "proofs" / "fixtures" / "modeled-development-package"
        shutil.copytree(fixture / "src", tree / "src")
        shutil.copy2(fixture / "package.toml", tree / "package.toml")
        (tree / "pyproject.toml").write_text(
            "[project]\n"
            'name = "modeled-development-package"\n'
            'version = "1.0.0"\n'
            'requires-python = ">=3.11"\n'
            f'dependencies = ["cozy-runtime[derive]=={runtime_version}", '
            f'"tensorfs=={tensorfs_version}"]\n\n'
            "[tool.uv.sources]\n"
            f'cozy-runtime = {{ path = "vendor/{runtime_wheel.name}" }}\n'
            f'tensorfs = {{ path = "vendor/{tensorfs_wheel.name}" }}\n\n'
            "[tool.cozy]\n"
            'organization = "cozy"\n\n'
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            'packages = ["src/modeled_development_package"]\n'
        )
        run("uv", "lock", "--quiet", cwd=tree)
        run("uv", "sync", "--locked", "--quiet", cwd=tree)
        helper = runtime / "proofs" / "internal" / "modeled_development_store.py"
        run(str(tree / ".venv" / "bin" / "python"), str(helper), str(args.store.resolve()))
        shutil.copytree(
            tree,
            args.source_out.resolve(),
            ignore=shutil.ignore_patterns(".venv", "__pycache__", "dist"),
        )
        print(f"runtime:  {runtime_sha}")
        print(f"tensorfs: {tensorfs_sha}")
        print(f"source:   {args.source_out.resolve()}")
        print(f"store:    {args.store.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
