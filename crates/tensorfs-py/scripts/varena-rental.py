# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena-standalone.py on a rental's GPU:
`cozy run varena-rental.py --on <rental> --gpus 1 --asset wheel=<tensorfs wheel> --asset script=varena-standalone.py`.

The script runs in a fresh process of the Runtime's interpreter (its torch), with the given
TensorFS wheel, and diffusers when the image lacks it, installed into a directory ahead of the
Runtime's own packages. Its output holds the script's log; the log's last line counts the checks.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

#: pip accepts a wheel only under its own name.
WHEEL = "tensorfs-0.13.0-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"


def install(target: Path, *specs: str) -> str:
    uv = shutil.which("uv")
    command = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
    done = subprocess.run([*command, "--target", str(target), *specs], capture_output=True, text=True)
    return f"$ {' '.join(command)} --target {target} {' '.join(specs)}: exit {done.returncode}\n{done.stdout}{done.stderr}"


def main(*, wheel: Annotated[FileAsset, AssetBound(max_bytes=1 << 27)],
         script: Annotated[FileAsset, AssetBound(max_bytes=1 << 20)], out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="varena-standalone-"))
    logs, site = root / "out", root / "site"
    logs.mkdir()
    (root / WHEEL).write_bytes(wheel.read_bytes())
    test = root / "varena-standalone.py"
    test.write_bytes(script.read_bytes())
    text = install(site, "--no-deps", str(root / WHEEL))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    if subprocess.run([sys.executable, "-c", "import diffusers"], env=env, capture_output=True).returncode:
        text += install(site, "diffusers")
    run = subprocess.run([sys.executable, str(test)], env=env, capture_output=True, text=True, cwd=root)
    text += f"$ {sys.executable} varena-standalone.py: exit {run.returncode}\n{run.stdout}{run.stderr}"
    (logs / "log.txt").write_text(text)
    print(text, flush=True)
    return out.save_tree(logs)
