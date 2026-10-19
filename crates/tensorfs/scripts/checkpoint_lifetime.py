#!/usr/bin/env python3
"""Real component handles retain native access, then release their request provider."""

import gc
import pathlib
import tempfile
import weakref

import tensorfs


class Provider:
    def headers(self, method, url, headers, payload):
        return {}


with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    with tensorfs.Writer(root / "source", cache_dir=root / "writer") as writer:
        component = writer.component("model", cls="test.Model", config={"size": 4})
        component.put("weight", b"data", dtype="U8", shape=[4])
        component.asset("vocab.txt", b"a\nb\n")
    endpoint = (root / "origin").as_uri()
    tensorfs.upload_folder(repo_id="acme/lifetime", revision="one", folder_path=root / "source",
                           endpoint=endpoint, cache_dir=root / "upload")
    provider = Provider()
    alive = weakref.ref(provider)
    checkpoint = tensorfs.open("acme/lifetime", revision="one", endpoint=endpoint,
                               token=provider, cache_dir=root / "reader")
    components = checkpoint.components
    component = components["model"]
    tensors = component.tensors
    del provider, checkpoint, components
    gc.collect()
    assert alive() is not None, "a live component lost its request provider"
    assert component.asset("vocab.txt") == b"a\nb\n"
    assert (pathlib.Path(component.path) / "vocab.txt").read_bytes() == b"a\nb\n"
    assert tensors.read("weight").tobytes() == b"data"
    with tensorfs.Writer(root / "copy", cache_dir=root / "copy-cache") as copy:
        copy.component("copy", cls="test.Copy", config={}).copy("weight", component)
    assert tensorfs.open(root / "copy", cache_dir=root / "copy-cache").components["copy"].tensors.read("weight").tobytes() == b"data"
    del component
    gc.collect()
    assert alive() is not None, "a live Tensors handle lost its request provider"
    array = tensors.read("weight")
    del tensors
    gc.collect()
    assert alive() is None, "the dropped checkpoint/components/Tensors retained their request provider"
    assert array.tobytes() == b"data", "Array bytes depend on the checkpoint/provider lifetime"

print("checkpoint lifetime: surviving component/tensor reads and copy; provider released after last handle")
