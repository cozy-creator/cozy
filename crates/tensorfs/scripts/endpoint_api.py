"""Public Python bindings against the real Hub/AuthKit/MinIO CI fixture; secrets arrive on stdin."""

import json
import ssl
import sys
from pathlib import Path

import tensorfs


request = json.load(sys.stdin)
cache = Path(request["cache_dir"])
ca = ssl.PEM_cert_to_DER_cert(Path(request["ca"]).read_text())
grant = dict(token_endpoint=request["token_endpoint"], client_id="tensord", resource=request["origin"],
             claims={"capability": request["capability"]}, endpoint=request["endpoint"], ca_der=ca)
provider = tensorfs.DPoP.grant(request["key"], **grant)
assert provider.issued_token() is None
options = dict(endpoint=request["endpoint"], token=provider, ca_der=ca, cache_dir=cache)
ck = tensorfs.open(request["model"], revision=request["tree"], variant="int8", **options)
assert ck.header_ref is not None and ck.tree == request["tree"]
assert ck.missing(), "metadata-only open fetched model bytes"
issued = provider.issued_token()
assert issued is not None
weights = bytes((i * 2_654_435_761) & 255 for i in range(300_000))
assert ck.components["model"].tensors.read("w").tobytes() == weights
warm = tensorfs.DPoP.grant(request["key"], **grant)
cached = tensorfs.open(request["model"], revision=request["tree"], variant="int8", **{**options, "token": warm})
assert cached.components["model"].tensors.read("w").tobytes() == weights
assert warm.issued_token() is None, "a warm request traded a capability"
resumed = tensorfs.DPoP.token(request["key"], issued, request["origin"], endpoint=request["endpoint"])
assert resumed.issued_token() == issued

writer = tensorfs.Writer(base=ck, repo_id=request["model"], variant="int8", **options)
writer.component("model", cls="conformance.Fixture", config={}).put("delta", b"derived", dtype="U8", shape=[7])
published = writer.commit()
assert published.id != ck.id and published.tree is not None
repo = tensorfs.Endpoint(request["endpoint"], token=provider, ca_der=ca).repo(request["model"])[0]
assert any(tree["sha256"] == published.tree.removeprefix("sha256:") for tree in repo["checkpoints"])

try:
    tensorfs.Endpoint(request["endpoint"], token=provider, ca_der=ca).update_repo(request["model"], lambda old: {
        **old, "releases": [*old["releases"], {"version": "forbidden", "tree": {
            "sha256": published.tree_ref[0], "length": published.tree_ref[1]}}]})
except tensorfs.DeniedError as error:
    assert error.code == "denied", "capability_exceeded policy was classified as a revision conflict"
else:
    raise AssertionError("a run capability published a release")

denied = tensorfs.Writer(repo_id=request["other"], variant="int8", **options)
denied.component("model", cls="", config={}).put("w", b"denied", dtype="U8", shape=[6])
try:
    denied.commit()
except tensorfs.DeniedError as error:
    assert denied.local_checkpoint is not None and error.code == "denied"
    assert denied.local_checkpoint.components["model"].tensors.read("w").tobytes() == b"denied"
    try:
        denied.commit()
    except tensorfs.DeniedError as repeated:
        assert repeated is error
    else:
        raise AssertionError("a denied upload later reported success")
else:
    raise AssertionError("a capability wrote another destination")

calls = []
retries = []


class Owner:
    def headers(self, method, url, headers, payload):
        calls.append((method, url, headers, payload))
        assert isinstance(headers, dict)
        return {"authorization": "Bearer " + request["owner_token"]}

    def retry(self, method, url, headers, payload, status, response_headers):
        retries.append((method, status))
        return False


owner = Owner()
assert tensorfs.Endpoint(request["endpoint"], token=owner, ca_der=ca).repo(request["model"]) is not None
assert retries, "the provider retry callback never saw a response"
local = tensorfs.Writer(cache_dir=cache)
local.component("model", cls="", config={}).put("w", b"callable", dtype="U8", shape=[8])
local = local.commit()
tensorfs.upload_folder(repo_id=request["model"], folder_path=local.path, endpoint=request["endpoint"],
                       cache_dir=cache, ca_der=ca, token=owner.headers)
assert any(method == "PUT" and isinstance(payload, str) and len(payload) == 64 for method, _, _, payload in calls)
print("Python Endpoint/AuthKit: lazy DPoP, warm cache, exact tree, denied upload and provider callbacks passed")
