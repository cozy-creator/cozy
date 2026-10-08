#!/usr/bin/env python3
"""Exercise the compiled source facade against loopback HF/Civitai providers and real Stores."""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading

import tensorfs
from tensorfs import records
from record_contract import check
from tensorfs.errors import Refusal


def main() -> None:
    body = b"selected raw checkpoint bytes"
    digest = hashlib.sha256(body).hexdigest()
    ordinary = {"config.json": b'{"model_type":"test"}', "tokenizer.json": b'{"tokens":["a"]}', "empty.py": b""}
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            requests.append(self.path)
            if self.path.startswith("/api/models/"):
                data = json.dumps([
                    {"type": "file", "path": name, "size": len(body),
                     "lfs": {"oid": digest, "size": len(body)}}
                    for name in ["chosen.safetensors", "unselected.safetensors"]
                ] + [{"type": "file", "path": name, "size": len(data)}
                     for name, data in ordinary.items()]).encode()
            elif self.path == "/api/v1/model-versions/42":
                data = json.dumps({"files": [{
                    "id": 92696, "name": "chosen.safetensors", "primary": True,
                    "metadata": {"format": "SafeTensor"}, "sizeKB": 123456.789,
                    "hashes": {"SHA256": digest},
                    "downloadUrl": f"http://127.0.0.1:{self.server.server_port}/chosen.safetensors",
                }]}).encode()
            else:
                name = self.path.rsplit("/", 1)[-1]
                assert name == "chosen.safetensors" or name in ordinary, self.path
                data = body if name == "chosen.safetensors" else ordinary[name]
                if requested := self.headers.get("Range"):
                    first, last = (int(value) for value in requested.removeprefix("bytes=").split("-"))
                    full_length = len(data)
                    data = data[first:last + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first}-{last}/{full_length}")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        with tempfile.TemporaryDirectory(prefix="tfs-source-facade-") as temporary:
            store = tensorfs.Store.init(Path(temporary) / "store")
            owner = tensorfs.object_id(b"producer")
            recipient = tensorfs.object_id(b"recipient")
            civitai_owner = tensorfs.object_id(b"civitai-producer")
            hf = store.resolve_source("hf://org/repo@" + "a" * 40,
                                      ["chosen.safetensors"], huggingface=endpoint, allow_local=True)
            civitai = store.resolve_source("civitai://42", ["civitai/files/92696"],
                                           civitai=endpoint, allow_local=True)
            check(hf, records.SelectedSource)
            check(civitai, records.SelectedSource)
            assert hf["content_manifest_digest"] != civitai["content_manifest_digest"]
            assert civitai["members"][0][0] == "civitai/files/92696"
            assert hf["selection_sha256"] != civitai["selection_sha256"]
            assert civitai["members"][0][2] == len(body), "rounded Civitai size was used"
            assert all("unselected" not in path for path in requests)
            class Stop(Exception):
                pass

            def stop(present: int, total: int) -> None:
                raise Stop((present, total))

            try:
                store.materialize_source(owner, hf["members"], allowed_hosts=hf["allowed_hosts"],
                                         credential_hosts=hf["credential_hosts"], allow_local=True,
                                         progress=stop)
            except Stop as raised:
                assert raised.args[0] == (0, len(body)), raised.args
            else:
                raise AssertionError("a raising progress callback did not stop the download")
            reports: list[tuple[int, int]] = []
            result = store.materialize_source(owner, hf["members"], allowed_hosts=hf["allowed_hosts"],
                                              credential_hosts=hf["credential_hosts"], allow_local=True,
                                              progress=lambda present, total: reports.append((present, total)))
            check(result, records.MaterializedSource)
            assert result["transferred_bytes"] == len(body)
            assert reports[0] == (0, len(body)) and reports[-1] == (len(body), len(body)), reports
            assert result["artifact"]["manifest_digest"] == hf["content_manifest_digest"]
            before = len(requests)
            again = store.materialize_source(civitai_owner, civitai["members"], allowed_hosts=civitai["allowed_hosts"],
                                             allow_local=True)
            assert again["transferred_bytes"] == 0 and len(requests) == before
            assert again["artifact"]["manifest_digest"] == civitai["content_manifest_digest"]
            store.release_tree_root(civitai_owner)
            root = store.tree_root(owner)
            assert root is not None
            assert "http" not in root["receipt"].decode()
            assert tensorfs.object_id(root["receipt"]) == root["receipt_digest"]
            adopted = store.retain_tree_root(owner, recipient)
            check(adopted, records.TreeRoot)
            assert adopted["producer"] == owner
            assert adopted["receipt"] == root["receipt"]
            store.release_tree_root(owner)
            tensorfs.gc(store.root)
            assert store.contains(digest)
            store.release_tree_root(recipient)
            tensorfs.gc(store.root)
            assert not store.contains(digest)
            assert len(store.implementation_sha256) == 64
            source_uri = "hf://org/repo@" + "a" * 40
            before = len(requests)
            for selection in ({"carriers": ["chosen.safetensors"]}, {"profiles": ["some-profile"]}):
                try:
                    store.resolve_source(source_uri, files=["config.json"], **selection,
                                         huggingface=endpoint, allow_local=True)
                except Refusal as error:
                    assert error.code == "TRANSACTION_CONFLICT", error
                else:
                    raise AssertionError("mixed selection accepted")
            assert len(requests) == before, "mixed selection made a network request"
            files = store.resolve_source(source_uri, files=sorted(ordinary),
                                         huggingface=endpoint, allow_local=True)
            assert [row[0] for row in files["members"]] == sorted(ordinary)
            for name, object_id, length, _url in files["members"]:
                assert object_id == tensorfs.object_id(ordinary[name]) and length == len(ordinary[name])
            files_owner = tensorfs.object_id(b"ordinary-files")
            downloaded = store.materialize_source(files_owner, files["members"],
                                                  allowed_hosts=files["allowed_hosts"], allow_local=True)
            assert downloaded["transferred_bytes"] == sum(map(len, ordinary.values()))
            del store
            store = tensorfs.Store.open(Path(temporary) / "store")
            before = len(requests)
            reused = store.materialize_source(files_owner, files["members"],
                                             allowed_hosts=files["allowed_hosts"], allow_local=True)
            assert reused["transferred_bytes"] == 0 and len(requests) == before
            store.release_tree_root(files_owner)
            tensorfs.gc(store.root)
        print("source facade green: native HF/Civitai and exact ordinary-file selection, identity, zero-network reuse after restart, custody and GC")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
