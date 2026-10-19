"""The public API with ordinary owner credentials against the real CI Hub."""

import json
import ssl
import sys
import threading
import time
from pathlib import Path

import tensorfs


request = json.load(sys.stdin)
cache = Path(request["cache_dir"])
ca = ssl.PEM_cert_to_DER_cert(Path(request["ca"]).read_text())
calls, retries = [], []


class Owner:
    def headers(self, method, url, headers, payload):
        calls.append((method, url, headers, payload))
        assert isinstance(headers, dict)
        return {"authorization": "Bearer " + request["token"]}

    def retry(self, method, url, headers, payload, status, response_headers):
        retries.append((method, status))
        return False


owner = Owner()
options = dict(endpoint=request["endpoint"], token=owner, ca_der=ca, cache_dir=cache)
data = bytes(range(256)) * 4096
writer = tensorfs.Writer(repo_id=request["model"], variant="f32", **options)
component = writer.component("model", cls="conformance.Fixture", config={"scale": 0.5, "optional": None})
component.put("weight", data, dtype="U8", shape=[len(data)])
component.asset("tokenizer.json", b"{}")
published = writer.commit()
assert published.tree is not None and published.tree_ref is not None
repo = tensorfs.Endpoint(request["endpoint"], token=owner, ca_der=ca).repo(request["model"])[0]
assert any(tree["sha256"] == published.tree_ref[0] for tree in repo["checkpoints"])
assert retries
assert any(method == "PUT" and isinstance(payload, str) and len(payload) == 64 for method, _, _, payload in calls)

reader = tensorfs.open(request["model"], revision=published.tree, variant="f32",
                       endpoint=request["endpoint"], token=request["token"], ca_der=ca, cache_dir=cache / "reader")
assert reader.missing(), "opening metadata fetched data"
assert (Path(reader.components["model"].path) / "tokenizer.json").read_bytes() == b"{}"
assert reader.missing(), "a component path fetched its weights"
progress = []
reader.fetch(progress=lambda stage, held, total: progress.append((stage, held, total)))
assert progress and progress[-1][0] == "fetch" and progress[-1][1] == progress[-1][2]
assert reader.components["model"].tensors.read("weight").tobytes() == data
folder = tensorfs.snapshot_download(published.tree, local_files_only=True, cache_dir=cache)
(Path(folder) / "README.md").write_text("normal owner upload\n")
tree = tensorfs.upload_folder(repo_id=request["model"], revision="1", folder_path=folder,
                              endpoint=request["endpoint"], token=owner.headers, ca_der=ca, cache_dir=cache)
assert tree.startswith("sha256:")
readme = tensorfs.hub_download(request["model"], "README.md", revision="1", endpoint=request["endpoint"],
                               token=request["token"], ca_der=ca, cache_dir=cache / "reader")
assert Path(readme).read_text() == "normal owner upload\n"


def unavailable(method, url, headers, payload):
    raise ValueError("the test credential provider is unavailable")


failed = tensorfs.Writer(repo_id=request["model"], variant="f32", **{**options, "token": unavailable})
failed.component("model", cls="", config={}).put("weight", b"retained", dtype="U8", shape=[8])
try:
    failed.commit()
except tensorfs.DeniedError as error:
    assert failed.local_checkpoint is not None
    assert failed.local_checkpoint.components["model"].tensors.read("weight").tobytes() == b"retained"
    try:
        failed.commit()
    except tensorfs.DeniedError as repeated:
        assert repeated is error
    else:
        raise AssertionError("a failed upload later reported success")
else:
    raise AssertionError("an unavailable credential provider accepted an upload")

entered, release, inspecting = threading.Event(), threading.Event(), threading.Event()
failures = []


class BlockingOwner(Owner):
    def headers(self, method, url, headers, payload):
        if method == "PUT":
            entered.set()
            assert release.wait(20)
        return super().headers(method, url, headers, payload)


streaming = tensorfs.Writer(repo_id=request["model"], variant="f32", **{**options, "token": BlockingOwner()})
component = streaming.component("model", cls="", config={})
component.put("threaded", b"two threads", dtype="U8", shape=[11])


def commit():
    try:
        streaming.commit()
    except BaseException as error:
        failures.append(error)


def inspect_and_abandon():
    inspecting.set()
    try:
        streaming.has("model", "threaded")
        component.has("threaded")
        streaming.abandon()
    except BaseException as error:
        failures.append(error)


committer = threading.Thread(target=commit)
committer.start()
assert entered.wait(20), "the upload never called its provider"
inspector = threading.Thread(target=inspect_and_abandon)
inspector.start()
assert inspecting.wait(20)
time.sleep(0.05)
release.set()
committer.join(20)
inspector.join(20)
assert not committer.is_alive() and not inspector.is_alive(), "writer and token callback deadlocked"
assert not failures, failures
print("Owner Endpoint API: read, write, tree receipt, bearer/callback credentials, failure families and writer threads passed")
