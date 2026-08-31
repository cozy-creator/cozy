"""Build the editable weightless source tree used by the product test."""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile

import tomllib

FIXTURE = pathlib.Path(__file__).with_name("weightless")


def run(
    *args: str, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-repo", default=os.getenv("RUNTIME_REPO", "~/cozy_v2/cozy-runtime")
    )
    parser.add_argument("--runtime-sha", default=os.getenv("RUNTIME_SHA", "HEAD"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-out")
    parser.add_argument("--version", default="1.0.0")
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
            'requires-python = ">=3.14,<3.15"\n'
            f'dependencies = ["cozy-runtime[media]=={runtime_version}"]\n\n'
            "[tool.uv.sources]\n"
            f'cozy-runtime = {{ path = "vendor/{wheels[0].name}" }}\n'
            '\n[tool.cozy]\norganization = "cozy"\n'
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

        if args.source_out:
            source_out = pathlib.Path(args.source_out).resolve()
            shutil.copytree(
                tree,
                source_out,
                ignore=shutil.ignore_patterns(".venv", "__pycache__", "dist"),
            )
        print(f"source:   {source_out if args.source_out else tree}")
        print(f"runtime: {full_sha} (git archive, read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
