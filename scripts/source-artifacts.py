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


def main() -> None:
    body = b"selected raw checkpoint bytes"
    digest = hashlib.sha256(body).hexdigest()
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
                ]).encode()
            elif self.path == "/api/v1/model-versions/42":
                data = json.dumps({"files": [{
                    "name": "chosen.safetensors", "primary": True,
                    "metadata": {"format": "SafeTensor"}, "sizeKB": 123456.789,
                    "hashes": {"SHA256": digest},
                    "downloadUrl": f"http://127.0.0.1:{self.server.server_port}/chosen.safetensors",
                }]}).encode()
            else:
                assert "chosen.safetensors" in self.path, self.path
                data = body
                if requested := self.headers.get("Range"):
                    first, last = (int(value) for value in requested.removeprefix("bytes=").split("-"))
                    data = body[first:last + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
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
            hf = store.resolve_source("hf://org/repo@" + "a" * 40,
                                      ["chosen.safetensors"], huggingface=endpoint, allow_local=True)
            civitai = store.resolve_source("civitai://42", ["chosen.safetensors"],
                                           civitai=endpoint, allow_local=True)
            assert hf["content_manifest_digest"] == civitai["content_manifest_digest"]
            assert hf["selection_sha256"] != civitai["selection_sha256"]
            assert civitai["members"][0][2] == len(body), "rounded Civitai size was used"
            assert all("unselected" not in path for path in requests)
            result = store.materialize_source(owner, hf["members"], allowed_hosts=hf["allowed_hosts"],
                                              credential_hosts=hf["credential_hosts"], allow_local=True)
            assert result["transferred_bytes"] == len(body)
            assert result["artifact"]["manifest_digest"] == hf["content_manifest_digest"]
            before = len(requests)
            again = store.materialize_source(owner, civitai["members"], allowed_hosts=civitai["allowed_hosts"],
                                             allow_local=True)
            assert again["transferred_bytes"] == 0 and len(requests) == before
            root = store.tree_root(owner)
            assert root is not None
            assert "http" not in root["receipt"].decode()
            assert tensorfs.object_id(root["receipt"]) == root["receipt_digest"]
            adopted = store.retain_tree_root(owner, recipient)
            assert adopted["producer"] == owner
            assert adopted["receipt"] == root["receipt"]
            store.release_tree_root(owner)
            tensorfs.gc(store.root)
            assert store.contains(digest)
            store.release_tree_root(recipient)
            tensorfs.gc(store.root)
            assert not store.contains(digest)
            assert len(store.implementation_sha256) == 64
        print("source facade green: native HF/Civitai selection, exact lengths, shared content identity, zero-network replay, custody and GC")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
