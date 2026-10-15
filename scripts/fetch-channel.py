#!/usr/bin/env python3
"""The executor's fetch client over a real channel: a native owner (`serve_fetch`, what TensorD
runs per attempt) fetching from a loopback hub into a real Store.

    python scripts/fetch-channel.py <tfs binary>
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import tensorfs
from tensorfs import _ext, errors
from tensorfs.fetch import Fetch
from tensorfs.records import ObjectRef


def prove(binary: str, root: Path) -> None:
    producer = root / "producer"
    producer.mkdir()

    def native(*args: str) -> str:
        return subprocess.check_output([binary, *args], text=True)

    written = native("checkpoint", "write", str(producer), "--blocks", "3", "--plain", "--no-configs")
    manifest = next(fields[1] for line in written.splitlines() if len(fields := line.split()) == 2 and fields[0] == "manifest_id")
    refs = root / "refs.jsonl"
    native("manifest", "walk", str(producer), manifest, "--refs", str(refs))
    objects = [cast(ObjectRef, json.loads(line)) for line in refs.read_text().splitlines()]
    bodies: dict[str, bytes] = {}
    for obj in objects:
        target = root / obj["sha256"]
        native("get", str(producer), obj["sha256"], "--out", str(target))
        bodies[obj["sha256"]] = target.read_bytes()
    manifest_hex = manifest.removeprefix("sha256:")
    native("manifest", "get", str(producer), manifest, "--out", str(root / manifest_hex))
    bodies[manifest_hex] = (root / manifest_hex).read_bytes()
    gets: Counter[str] = Counter()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def answer(self, status: int, data: object) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = cast(dict[str, object], json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if self.path.endswith("/closure"):
                gets["closure"] += 1
                self.answer(200, {"model": "proof/model", "release": "1.0.0", "lane": "plain", "scope": "runtime",
                                  "manifest": {"sha256": manifest_hex, "length": len(bodies[manifest_hex])},
                                  "objects": sorted(objects, key=lambda obj: obj["sha256"]),
                                  "complete": True, "presign_max_digests": 4096})
            elif self.path.endswith("/presign"):
                now = int(time.time())
                digests = cast(list[str], request["digests"])
                self.answer(200, {"expires_at_unix": now + 3600, "server_time_unix": now,
                                  "urls": {digest: origin + "/object/" + digest for digest in digests}})
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            digest = self.path.rsplit("/", 1)[-1]
            gets[digest] += 1
            body = bodies[digest]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = f"http://127.0.0.1:{server.server_port}"
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    try:
        store = tensorfs.Store.init(root / "destination")
        owner, executor = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        granted = [{"hub": origin, "ref": "proof/model@1.0.0", "lane": "plain", "manifest": manifest,
                    "manifest_length": len(bodies[manifest_hex]), "allow_local": True}]
        served = threading.Thread(target=_ext.serve_fetch, args=(store, owner.detach()),
                                  kwargs={"origins": granted, "run": 1})
        served.start()
        parts = [obj for obj in objects if len(bodies[obj["sha256"]]) > 0]
        with Fetch.from_fd(executor.detach()) as fetch:
            assert fetch.present(manifest, parts) == [False] * len(parts)
            fetch.schedule(manifest, parts[:1], rank=0)
            # Waits from several threads at once; each answers when its object is here.
            failures: list[BaseException] = []

            def wait(obj: ObjectRef) -> None:
                try:
                    fetch.wait(manifest, [obj])
                except BaseException as exc:
                    failures.append(exc)

            waiters = [threading.Thread(target=wait, args=(obj,)) for obj in parts]
            for waiter in waiters:
                waiter.start()
            for waiter in waiters:
                waiter.join()
            assert not failures, failures
            assert fetch.present(manifest, parts) == [True] * len(parts)
            assert all(gets[obj["sha256"]] == 1 for obj in objects), gets
            assert gets["closure"] == 1, gets
            try:
                fetch.wait("sha256:" + "ab" * 32, parts)
            except errors.Refusal as error:
                assert "SOURCE_NOT_ALLOWED" in str(error), error
            else:
                raise AssertionError("a checkpoint outside the grant was fetched")
        served.join()
        print("fetch channel proof passed: present, schedule, concurrent waits, one GET each, grant refusal")
    finally:
        server.shutdown()
        serving.join()
        server.server_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tensorfs-fetch-") as directory:
        prove(sys.argv[1], Path(directory))
