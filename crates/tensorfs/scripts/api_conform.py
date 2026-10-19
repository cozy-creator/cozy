"""The public component API against real local stores and folders."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import tensorfs


with TemporaryDirectory() as temp:
    root = Path(temp)
    tensorfs.configure(cache_dir=root / "cache")
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    with tensorfs.Writer(root / "folder", variant="f32") as writer:
        writer.pipeline("diffusers.TestPipeline", {"flag": None, "scale": 0.5})
        component = writer.component("model", cls="transformers.TestModel", config={"width": 6})
        component.put("weight", values)
        component.asset("tokenizer.json", b"{}")
    ck = writer.commit()
    assert tensorfs.open(root / "folder").id == ck.id
    assert tensorfs.open(root / "folder/model.f32.cozytensors").id == ck.id
    assert np.array_equal(np.from_dlpack(ck.read("model", "weight")), values)
    component = ck.components["model"]
    assert json.loads((Path(component.path) / "config.json").read_text()) == {"width": 6}
    assert (Path(component.path) / "tokenizer.json").read_bytes() == b"{}"
    tree = tensorfs.upload_folder(repo_id="local/example", revision="draft", folder_path=ck.path)
    named = tensorfs.open("local/example", revision="draft", local_files_only=True)
    assert named.id == ck.id and named.tree == tree
    assert named.repo_id == "local/example" and named.revision == "draft"
    with tensorfs.Writer(base=named) as derived:
        derived.component("copy", cls=component.cls, config=component.config).copy("weight", component)
    again = derived.commit()
    assert np.array_equal(np.from_dlpack(again.read("copy", "weight")), values)
    downloaded = tensorfs.snapshot_download("local/example", revision="draft", local_dir=root / "download")
    assert tensorfs.open(downloaded).id == ck.id
    try:
        with tensorfs.Writer() as bad:
            bad.component("model", cls="", config={}).put("weight", values[:, ::2])
    except tensorfs.InvalidError:
        pass
    else:
        raise AssertionError("a non-contiguous DLPack input was accepted")
print("public component API: folder, shared store, local revision, DLPack write/copy/read passed")
