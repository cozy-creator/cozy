#!/usr/bin/env python3
"""The installed wheel in a fresh venv: write a checkpoint, open it, read it back, and run the
bundled `tfs`. CI runs it on every platform and CPython version (publish-tensorfs.yaml)."""

import json
import pathlib
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from array import array

import tensorfs
import tensorfs.varena


def main() -> None:
    work = pathlib.Path(tempfile.mkdtemp(prefix="tensorfs-install-"))
    tensorfs.configure(cache_dir=work / "store")
    weight = array("f", [1.5, -2.0])
    with tensorfs.Writer(work / "model.cozytensors") as writer:
        component = writer.component("model", cls=("example", "1.0", "Model"), config={"width": 2})
        component.put("weight", weight, dtype="F32", shape=[2])
        component.asset("vocab.txt", b"one\ntwo\n")
    saved = writer.commit()

    checkpoint = tensorfs.open(work / "model.cozytensors")
    model = checkpoint.components["model"]
    value = model.tensors.read("weight")
    assert checkpoint.id == saved.id, (checkpoint.id, saved.id)
    assert (tuple(model.cls), model.config) == (("example", "1.0", "Model"), {"width": 2}), (model.cls, model.config)
    assert model.cls == tensorfs.ClassRef("example", "1.0", "Model") and model.cls.name == "Model"
    assert (value.dtype, value.shape, value.tobytes()) == ("F32", (2,), weight.tobytes()), value
    assert model.asset("vocab.txt") == b"one\ntwo\n"

    tfs = shutil.which("tfs", path=sysconfig.get_path("scripts"))
    assert tfs, f"no tfs in {sysconfig.get_path('scripts')}"
    for args in (["--help"], ["version"], ["--json", "--store", str(work / "store"), "inspect", str(work / "model.cozytensors")]):
        ran = subprocess.run([tfs, *args], capture_output=True, text=True, check=False)
        assert ran.returncode == 0, (args, ran.returncode, ran.stdout, ran.stderr)
    inspected = json.loads(ran.stdout)
    assert [c["name"] for c in inspected["components"]] == ["model"], inspected
    print(f"tensorfs {tensorfs.__version__} on CPython {platform.python_version()} {sys.platform} "
          f"{platform.machine()}: wrote, opened and read a checkpoint; tfs --help, version and inspect ran")


if __name__ == "__main__":
    main()
