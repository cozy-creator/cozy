"""The public component API against real local stores and folders."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import tensorfs
from tensorfs import _ext, raw, varena

assert not hasattr(tensorfs, "Store"), "the cache implementation leaked into the public API"
for name in ("materialize", "download_file", "convert_to_cozytensors"):
    assert name in tensorfs.__all__ and callable(getattr(tensorfs, name)), name
for obsolete in ("snapshot_download", "hub_download", "download", "convert", "resolve_source"):
    assert not hasattr(tensorfs, obsolete) and not hasattr(tensorfs.api, obsolete), obsolete
for obsolete in ("Closure", "ReadPlan", "parse_header", "plan", "regions", "Info", "Part"):
    assert not hasattr(tensorfs, obsolete) and not hasattr(_ext, obsolete), obsolete
assert importlib.util.find_spec("tensorfs.records") is None, "the legacy projection remains importable"
assert varena.regions is _ext.varena.regions

with TemporaryDirectory() as temp:
    root = Path(temp)
    tensorfs.configure(cache_dir=root / "cache")
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    with tensorfs.Writer(variant="f32") as writer:
        writer.pipeline("diffusers.TestPipeline", {"flag": None, "scale": 0.5})
        component = writer.component("model", cls="transformers.TestModel", config={"width": 6})
        component.put("weight", values)
        component.asset("tokenizer.json", b"{}")
        writer.component("optimizer", cls="", config={}).put("step", np.array([7], dtype=np.int64))
    with tensorfs.Writer(variant="f16") as half:
        half.component("model", cls="transformers.TestModel", config={"width": 6}).put("weight", values.astype(np.float16))
        half.component("optimizer", cls="", config={}).put("step", np.array([8], dtype=np.int64))
    tensorfs.export(writer.commit(), root / "folder", format="cozytensors")
    tensorfs.export(half.commit(), root / "folder", format="cozytensors")
    (root / "folder/README.md").write_text("a repo file\n")
    ck = writer.commit()
    assert not hasattr(ck, "read") and not hasattr(ck, "tensors") and not hasattr(ck, "path"), "checkpoint-wide tensor APIs remain public"
    assert callable(ck.download) and not hasattr(ck, "fetch")
    assert tensorfs.open(root / "folder", variant="f32").id == ck.id
    assert tensorfs.open(root / "folder/model.f32.cozytensors").id == ck.id
    assert np.array_equal(np.from_dlpack(ck.components["model"].tensors.read("weight")), values)
    component = ck.components["model"]
    assert component.config == {"width": 6} and component.asset("tokenizer.json") == b"{}"
    assert not hasattr(component, "path") and not hasattr(component, "config_file")
    assert isinstance(component.tensors["weight"], tensorfs.TensorMetadata)
    assert not hasattr(component.tensors["weight"], "parts") and not hasattr(component.tensors, "read_all")
    assert raw.parts(ck, "model", "weight")[""].length == values.nbytes
    support = Path(component.export_files(root / "support"))
    assert sorted(p.name for p in support.iterdir()) == ["config.json", "tokenizer.json"]
    assert json.loads((support / "config.json").read_text()) == component.config
    assert (support / "tokenizer.json").read_bytes() == b"{}"
    handle = root / "saved.cozytensors"
    with tensorfs.Writer(handle) as saved:
        saved.component("model", cls="test.File", config={}).put("data", b"handle" * 4096, dtype="U8", shape=[6 * 4096])
    assert handle.is_file() and handle.stat().st_size < 4096
    assert not (root / "chunks").exists(), "normal native Writer made adjacent payload files"
    moved = handle.rename(root / "renamed.cozytensors")
    with tensorfs.open(moved) as reopened:
        assert reopened.id == saved.commit().id
        assert reopened.components["model"].tensors.read("data").tobytes() == b"handle" * 4096
    tree = tensorfs.upload_folder(repo_id="local/example", revision="draft", folder_path=root / "folder")
    named = tensorfs.open("local/example", revision="draft", variant="f32", local_files_only=True)
    assert named.id == ck.id and named.tree == tree
    assert named.repo_id == "local/example" and named.revision == "draft"
    with tensorfs.Writer(base=named) as derived:
        derived.component("copy", cls=component.cls, config=component.config).copy("weight", component)
    again = derived.commit()
    assert np.array_equal(np.from_dlpack(again.components["copy"].tensors.read("weight")), values)
    downloaded = tensorfs.materialize("local/example", revision="draft", local_dir=root / "download")
    assert tensorfs.open(downloaded, variant="f32").id == ck.id
    assert (Path(downloaded) / "model.f16.cozytensors").is_file()
    assert Path(tensorfs.download_file("local/example", "README.md", revision="draft")).read_text() == "a repo file\n"
    default_snapshot = tensorfs.materialize("local/example", revision="draft")
    assert Path(default_snapshot).name == tree.removeprefix("sha256:")
    origin = (root / "origin").as_uri()
    remote_tree = tensorfs.upload_folder(repo_id="acme/example", revision="1", folder_path=root / "folder", endpoint=origin)
    remote_cache = root / "remote-cache"
    metadata = tensorfs.open("acme/example", revision="1", variant="f32", endpoint=origin, cache_dir=remote_cache)
    assert metadata.id == ck.id and metadata.tree == remote_tree
    assert metadata.header_ref[1] == (root / "folder/model.f32.cozytensors").stat().st_size
    assert metadata.missing(), "opening metadata fetched the closure"
    assert varena.regions(metadata, only=["model/weight"]) == [["model/weight"]]
    assert metadata.missing(), "region planning fetched payload bytes"
    exported = Path(metadata.components["model"].export_files(root / "remote-support"))
    assert (exported / "tokenizer.json").read_bytes() == b"{}"
    assert metadata.missing(["model"]), "exporting support files fetched its tensors"
    only = tensorfs.materialize("acme/example", revision="1", variant="f32", components=["model"], endpoint=origin, cache_dir=remote_cache)
    assert not (Path(only) / "model.f16.cozytensors").exists()
    selected = tensorfs.open(only, cache_dir=remote_cache)
    assert selected.missing(["model"]) == []
    assert selected.missing(["optimizer"]), "a model-only snapshot fetched the optimizer"
    assert np.array_equal(np.from_dlpack(selected.components["model"].tensors.read("weight")), values)
    assert metadata.missing(["optimizer"]), "reading one component fetched another"
    eager_cache = root / "eager-cache"
    eager = tensorfs.open("acme/example", revision="1", variant="f32", endpoint=origin,
                          cache_dir=eager_cache, lazy=False)
    assert eager.missing() == [] and eager.id == ck.id
    assert eager.components["model"].asset("tokenizer.json") == b"{}"
    assert not (eager_cache / "snapshots" / remote_tree.removeprefix("sha256:")).exists()
    thin = root / "thin.cozytensors"
    thin.write_bytes((root / "folder/model.f32.cozytensors").read_bytes())
    offline = dict(endpoint="http://127.0.0.1:1", cache_dir=root / "offline", local_files_only=True)
    assert tensorfs.open(thin, **offline).id == ck.id
    try:
        tensorfs.open(thin, lazy=False, **offline)
    except tensorfs.NotFoundError:
        pass
    else:
        raise AssertionError("eager local-only open returned with missing checkpoint data")
    progress = []
    metadata.download(["model"], progress=lambda stage, held, total: progress.append((stage, held, total)))
    assert progress[0][0] == "fetch" and progress[-1][1] == progress[-1][2]
    callback_failure = tensorfs.NetworkError("consumer progress failed")

    def stop(stage, held, total):
        raise callback_failure

    try:
        metadata.download(progress=stop)
    except tensorfs.NetworkError as error:
        assert error is callback_failure
    else:
        raise AssertionError("the progress callback failure disappeared")
    endpoint = tensorfs.Endpoint("local", cache_dir=root / "empty-endpoint")
    try:
        endpoint.update_repo("proof/invalid", lambda old: {"oops": True})
    except tensorfs.InvalidError:
        pass
    else:
        raise AssertionError("an invalid repo change was not an invalid argument")
    callback_conflict = tensorfs.ConflictError("consumer revision conflict")

    def conflict(old):
        raise callback_conflict

    try:
        endpoint.update_repo("proof/invalid", conflict)
    except tensorfs.ConflictError as error:
        assert error is callback_conflict
    else:
        raise AssertionError("the repository callback lost its error family")
    blocked = root / "not-a-directory"
    blocked.write_bytes(b"blocked")
    failed = tensorfs.Writer(repo_id="acme/blocked", endpoint=blocked.as_uri())
    failed.component("model", cls="", config={}).put("weight", values)
    try:
        failed.commit()
    except tensorfs.Error as error:
        assert failed.local_checkpoint is not None
        assert np.array_equal(np.from_dlpack(failed.local_checkpoint.components["model"].tensors.read("weight")), values)
        try:
            failed.commit()
        except tensorfs.Error as repeated:
            assert repeated is error
        else:
            raise AssertionError("failed publication became a successful commit")
    else:
        raise AssertionError("an unavailable endpoint accepted publication")
    try:
        with tensorfs.Writer() as bad:
            bad.component("model", cls="", config={}).put("weight", values[:, ::2])
    except tensorfs.InvalidError:
        pass
    else:
        raise AssertionError("a non-contiguous DLPack input was accepted")
print("public component API: folder, shared store, local revision, DLPack write/copy/read passed")
