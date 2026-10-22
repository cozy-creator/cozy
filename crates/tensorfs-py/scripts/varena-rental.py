# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4", "torch", "diffusers"]
# ///
"""A varena test script (varena-standalone.py, varena-sdxl-cap.py) on a rental's GPU:
`cozy run varena-rental.py --on <rental> --gpus 1 --asset wheel=<tensorfs wheel> --asset script=<script>`.

The script runs in a fresh process of this script's interpreter (its environment has torch and
diffusers), with the given TensorFS wheel installed into a directory ahead of the environment's
own TensorFS. Its output streams to the run's log as it comes, and the output holds it too.
"""
import os
import io
import hashlib
import zipfile
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
    text = f"$ {' '.join(command)} --target {target} {' '.join(specs)}: exit {done.returncode}\n{done.stdout}{done.stderr}"
    if done.returncode:
        raise RuntimeError(text)
    return text


def main(*, wheel: Annotated[FileAsset, AssetBound(max_bytes=1 << 27)],
         script: Annotated[FileAsset, AssetBound(max_bytes=1 << 20)], out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="varena-standalone-"))
    logs, site = root / "out", root / "site"
    logs.mkdir()
    data = wheel.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))
        distribution = metadata.split(".dist-info/", 1)[0]
        tag = next(line[5:] for line in archive.read(metadata).decode().splitlines() if line.startswith("Tag: "))
    wheel_path = root / f"{distribution}-{tag}.whl"
    wheel_path.write_bytes(data)
    test = root / "script.py"
    test.write_bytes(script.read_bytes())
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    with (logs / "log.txt").open("w") as log:
        def say(line: str) -> None:  # streamed: the run's log shows where a stall sits
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        say(f"wheel {wheel_path.name} sha256={hashlib.sha256(data).hexdigest()}\n")
        say(install(site, "--no-deps", str(wheel_path)))
        command = [sys.executable, "-u", str(test)]
        with subprocess.Popen(command, env=env, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
            for line in child.stdout or ():
                say(line)
        say(f"$ {' '.join(command)}: exit {child.returncode}\n")
    result = out.save_tree(logs)
    if child.returncode:
        raise RuntimeError(f"Varena child failed with exit {child.returncode}; see streamed log")
    return result
