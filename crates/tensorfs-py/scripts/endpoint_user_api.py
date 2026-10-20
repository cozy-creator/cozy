"""Ordinary credentials, retained output and Runtime library replay on the real CI Hub."""

import hashlib
import importlib.metadata
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import tensorfs


request = json.load(sys.stdin)
cache = Path(request["cache_dir"])
ca = ssl.PEM_cert_to_DER_cert(Path(request["ca"]).read_text())


def source_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def installed_wheel(distribution, package, wheel_path, expected_sha256=None):
    """Check the installed Python and native bytes, without a source-path overlay."""
    wheel_path = Path(wheel_path)
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        assert digest == expected_sha256, "the native wheel differs from the gate's pinned artifact"
    installed = importlib.metadata.distribution(distribution)
    assert Path(package.__file__).resolve() == Path(installed.locate_file(package.__name__ + "/__init__.py")).resolve()
    checked = 0
    with zipfile.ZipFile(wheel_path) as wheel:
        for name in wheel.namelist():
            if name.startswith(package.__name__ + "/") and not name.endswith("/"):
                assert Path(installed.locate_file(name)).read_bytes() == wheel.read(name), f"installed wheel mismatch: {name}"
                checked += 1
    assert checked > 0, "the supplied wheel contains no package files"
    return {"version": installed.version, "wheel_sha256": digest, "files_checked": checked}


native_provenance = installed_wheel("tensorfs", tensorfs, request["native_wheel"], request["native_sha256"])
# TensorFS has no embedded source SHA: the gate binds this artifact hash to its producer.
native_provenance["source"] = request["native_source"]
creator = Path(request["creator_source"])
native_provenance["test_source"] = source_sha(creator)
for source in (creator / "crates/tensorfs-py/python/tensorfs").rglob("*.py"):
    relative = source.relative_to(creator / "crates/tensorfs-py/python")
    assert source.read_bytes() == Path(importlib.metadata.distribution("tensorfs").locate_file(str(relative))).read_bytes(), f"Python source differs from the installed wheel: {relative}"

import cozy_runtime
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.weights_store import Connection, Derivation, Outputs

runtime_provenance = installed_wheel("cozy-runtime", cozy_runtime, request["runtime_wheel"])
runtime_provenance["source"] = source_sha(request["runtime_source"])
assert len(COMMIT) == 40 and COMMIT == runtime_provenance["source"], "Runtime wheel and source checkout differ"
assert Path(cozy_runtime.__file__).with_name("weights_store.py").read_bytes() == (Path(request["runtime_source"]) / "src/cozy_runtime/weights_store.py").read_bytes()


class Credential:
    def __init__(self, token):
        self.token = token
        self.calls, self.retries = [], []

    def headers(self, method, url, headers, payload):
        self.calls.append((method, payload))
        assert isinstance(headers, dict)
        return {"authorization": "Bearer " + self.token}

    def retry(self, method, url, headers, payload, status, response_headers):
        self.retries.append((method, status))
        return False


owner, rival = Credential(request["token"]), Credential(request["rival_token"])
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
assert owner.retries
assert any(method == "PUT" and isinstance(payload, str) and len(payload) == 64 for method, payload in owner.calls)

# The rival is a valid ordinary account: its same credential can publish its own repo.
rival_writer = tensorfs.Writer(repo_id=request["rival_model"], variant="f32", **{**options, "token": rival})
rival_writer.component("model", cls="", config={}).put("weight", b"rival-ok", dtype="U8", shape=[8])
assert rival_writer.commit().tree is not None

reader = tensorfs.open(request["model"], revision=published.tree, variant="f32",
                       endpoint=request["endpoint"], token=request["token"], ca_der=ca, cache_dir=cache / "reader")
assert reader.missing(), "opening metadata fetched data"
assert reader.components["model"].asset("tokenizer.json") == b"{}"
assert reader.missing(), "reading an asset fetched its weights"
progress = []
reader.download(progress=lambda stage, held, total: progress.append((stage, held, total)))
assert progress and progress[-1][0] == "fetch" and progress[-1][1] == progress[-1][2]
assert reader.components["model"].tensors.read("weight").tobytes() == data
folder = cache / "owner-export"
tensorfs.materialize(published, local_dir=folder, format="cozytensors")
(folder / "README.md").write_text("normal owner upload\n")
tree = tensorfs.upload_folder(repo_id=request["model"], revision="1", folder_path=folder,
                              endpoint=request["endpoint"], token=owner.headers, ca_der=ca, cache_dir=cache)
assert tree.startswith("sha256:")
readme = tensorfs.download_file(request["model"], "README.md", revision="1", endpoint=request["endpoint"],
                               token=request["token"], ca_der=ca, cache_dir=cache / "reader")
assert Path(readme).read_text() == "normal owner upload\n"

# A separate process has no configured provider or inherited in-memory checkpoint.
# Its empty store and deliberately unusable default endpoint prevent a warm/default
# path from concealing failure to forward the ordinary anonymous endpoint and CA.
anonymous_program = r'''
import hashlib
import json
import os
import ssl
import sys
from pathlib import Path

import tensorfs
import cozy_runtime
from cozy_runtime.weights_store import Connection, open_checkpoint

request = json.load(sys.stdin)
cache = Path(request["cache_dir"])
assert not cache.exists(), "the anonymous reader's store must start empty"
assert os.environ["TENSORFS_ENDPOINT"] == "http://127.0.0.1:1"
assert "TENSORFS_TOKEN" not in os.environ
assert "TENSORFS_MIRROR" not in os.environ
assert "SSL_CERT_FILE" not in os.environ
assert "SSL_CERT_DIR" not in os.environ
assert request["endpoint"].startswith("https://"), "the CA forwarding case requires TLS"
connection = Connection(
    endpoint=request["endpoint"],
    token=None,
    ca_der=ssl.PEM_cert_to_DER_cert(Path(request["ca"]).read_text()),
)
assert connection.token is None
with open_checkpoint(cache, request["reference"], connection) as checkpoint:
    assert checkpoint.id == request["header"]
    assert checkpoint.missing(), "the cold metadata open fetched all payloads"
    value = checkpoint.components["model"].tensors.read("weight").tobytes()
    assert len(value) == request["length"]
    assert hashlib.sha256(value).hexdigest() == request["sha256"]
    assert checkpoint.components["model"].asset("tokenizer.json") == b"{}"
    assert not checkpoint.missing()
print(json.dumps({
    "cold_store": True,
    "anonymous": connection.token is None,
    "explicit_endpoint_and_ca": True,
    "default_endpoint_disabled": True,
    "metadata_only_then_payload": True,
    "header": request["header"],
    "bytes": len(value),
    "tensorfs_path": str(Path(tensorfs.__file__).resolve()),
    "runtime_path": str(Path(cozy_runtime.__file__).resolve()),
}))
'''
anonymous_env = os.environ.copy()
anonymous_env["TENSORFS_ENDPOINT"] = "http://127.0.0.1:1"
for name in (
    "TENSORFS_TOKEN", "TENSORFS_MIRROR", "HF_TOKEN", "CIVITAI_TOKEN",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
):
    anonymous_env.pop(name, None)
anonymous_result = subprocess.run(
    [sys.executable, "-I", "-c", anonymous_program],
    input=json.dumps({
        "endpoint": request["anonymous_endpoint"], "ca": request["ca"],
        "cache_dir": str(cache / "anonymous-cold"),
        "reference": request["model"] + "@1/f32", "header": published.id,
        "length": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }),
    env=anonymous_env, capture_output=True, text=True, timeout=60,
)
assert anonymous_result.returncode == 0, anonymous_result.stderr
cold_anonymous_read = json.loads(anonymous_result.stdout)
assert cold_anonymous_read["tensorfs_path"] == str(Path(tensorfs.__file__).resolve())
assert cold_anonymous_read["runtime_path"] == str(Path(cozy_runtime.__file__).resolve())


class RetryFailure(Credential):
    def retry(self, method, url, headers, payload, status, response_headers):
        raise tensorfs.NetworkError("retry callback network failure")


try:
    tensorfs.Endpoint(request["endpoint"], token=RetryFailure(request["token"]), ca_der=ca).repo(request["model"])
except tensorfs.NetworkError as error:
    assert error.code == "network" and "retry callback network failure" in error.detail
else:
    raise AssertionError("the provider retry failure disappeared")


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

# A real server refusal must keep all local bytes, and retry only uploads that result.
productions = 0
repo_before_denial = tensorfs.Endpoint(request["endpoint"], token=owner, ca_der=ca).repo(request["model"])[0]
denied = tensorfs.Writer(repo_id=request["model"], variant="f32", **{**options, "token": rival})
denied.component("model", cls="conformance.Retained", config={"attempt": 1}).put("weight", b"retained-after-denial", dtype="U8", shape=[21])
productions += 1
denials_before = len(rival.retries)
try:
    denied.commit()
except tensorfs.DeniedError as error:
    assert error.code == "denied"
    retained = denied.local_checkpoint
    assert retained is not None
    retained_id = retained.id
    assert retained.components["model"].tensors.read("weight").tobytes() == b"retained-after-denial"
    assert not retained.missing()
    try:
        denied.commit()
    except tensorfs.DeniedError as repeated:
        assert repeated is error
    else:
        raise AssertionError("a refused publication later reported success")
else:
    raise AssertionError("a rival published into the owner's repo")
assert any(status in (401, 403) for _, status in rival.retries[denials_before:]), "no HTTP authorization refusal was observed"
after_denial = tensorfs.Endpoint(request["endpoint"], token=owner, ca_der=ca).repo(request["model"])[0]
assert after_denial == repo_before_denial, "a denied publication changed the owner's repo"
retry_folder = cache / "retained-export"
tensorfs.materialize(retained, local_dir=retry_folder, format="cozytensors")
retry_tree = tensorfs.upload_folder(repo_id=request["model"], revision="2", folder_path=retry_folder, **options)
retried = tensorfs.open(request["model"], revision="2", variant="f32", **{**options, "cache_dir": cache / "retry-reader"})
assert retried.tree == retry_tree and retried.id == retained_id
assert retried.components["model"].tensors.read("weight").tobytes() == b"retained-after-denial"
assert productions == 1

# Exercise Runtime's real publication/replay library; this is not an executor/CLI test.
runtime_cache = cache / "runtime"
shown = []
runtime_options = dict(store=runtime_cache, request_id="endpoint-proof", actor="ordinary-owner",
                       code="endpoint-proof", bounds={"model": 4096}, show=lambda *item: shown.append(item),
                       destination=request["runtime_model"])
runtime_denials_before = len(rival.retries)
outputs = Outputs(**runtime_options, connection=Connection(endpoint=request["endpoint"], token=rival, ca_der=ca))
runtime_productions = 0
with outputs.open("model", Derivation(None)) as output:
    output.put("model", "weight", "U8", [16], b"runtime-retained")
    runtime_productions += 1
    try:
        outputs.adopt(output)
    except Exception as error:
        assert getattr(error, "code", None) == "upload_failed"
        assert isinstance(error.__cause__, tensorfs.DeniedError), "Runtime failed before the real authorization refusal"
    else:
        raise AssertionError("Runtime published with the rival credential")
assert any(status in (401, 403) for _, status in rival.retries[runtime_denials_before:])
assert len(shown) == 1 and shown[0][1] == "local/derived"
runtime_retained = tensorfs.open(shown[0][2], cache_dir=runtime_cache, local_files_only=True)
assert runtime_retained.components["model"].tensors.read("weight").tobytes() == b"runtime-retained"
assert not runtime_retained.missing()
retry_outputs = Outputs(**runtime_options, connection=Connection(endpoint=request["endpoint"], token=owner, ca_der=ca))
with retry_outputs.open("model", Derivation(None)) as replay:
    assert replay.done is not None and replay.written == 0 and replay.has("model", "weight")
    artifact = retry_outputs.adopt(replay)
    assert replay.written == 0
assert artifact.checkpoint == runtime_retained.id and runtime_productions == 1
assert len(shown) == 3 and shown[-1][1] == request["runtime_model"]
runtime_read = tensorfs.open(request["runtime_model"], revision=shown[-1][2], **{**options, "cache_dir": cache / "runtime-reader"})
assert runtime_read.id == runtime_retained.id
assert runtime_read.components["model"].tensors.read("weight").tobytes() == b"runtime-retained"

entered, release, inspecting = threading.Event(), threading.Event(), threading.Event()
failures = []


class BlockingOwner(Credential):
    def headers(self, method, url, headers, payload):
        if method == "PUT":
            entered.set()
            assert release.wait(20)
        return super().headers(method, url, headers, payload)


streaming = tensorfs.Writer(repo_id=request["model"], variant="f32", **{**options, "token": BlockingOwner(request["token"])})
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
print(json.dumps({
    "native": native_provenance,
    "runtime": runtime_provenance,
    "tensorfs": {"owner_read_write": True, "rival_denied": True, "retained_upload_without_rerun": productions == 1},
    "runtime_library": {"publication_replay_without_rewrite": runtime_productions == 1},
    "cold_anonymous_read": cold_anonymous_read,
    "failure_families_and_writer_threads": True,
}, sort_keys=True))
