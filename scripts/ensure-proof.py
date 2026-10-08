#!/usr/bin/env python3
"""Exercise the built Python `ensure` API against a native checkpoint and a loopback hub.

    python scripts/ensure-proof.py <tfs binary>
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict, cast

import tensorfs
from tensorfs import errors, records
from record_contract import check


class ObjectRef(TypedDict):
    sha256: str
    length: int


def prove(binary: str, root: Path) -> None:
    producer = root / "producer"
    producer.mkdir()

    def native(*args: str) -> str:
        return subprocess.check_output([binary, *args], text=True)

    written = native("checkpoint", "write", str(producer), "--blocks", "0", "--plain", "--no-configs")
    manifest = next(fields[1] for line in written.splitlines() if len(fields := line.split()) == 2 and fields[0] == "manifest_id")
    refs_path = root / "refs.jsonl"
    native("manifest", "walk", str(producer), manifest, "--refs", str(refs_path))
    objects = [cast(ObjectRef, json.loads(line)) for line in refs_path.read_text().splitlines()]
    bodies: dict[str, bytes] = {}
    for obj in objects:
        target = root / obj["sha256"]
        native("get", str(producer), obj["sha256"], "--out", str(target))
        bodies[obj["sha256"]] = target.read_bytes()
    manifest_hex = manifest.removeprefix("sha256:")
    target = root / manifest_hex
    native("manifest", "get", str(producer), manifest, "--out", str(target))
    bodies[manifest_hex] = target.read_bytes()
    manifest_ref: ObjectRef = {"sha256": manifest_hex, "length": len(bodies[manifest_hex])}
    total = sum(len(body) for body in bodies.values())

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
            if self.path.endswith("/closure") and request["ref"] != "proof/model@1.0.0":
                self.answer(404, {"error": {"code": "model_not_found", "message": "no such model"}})
            elif self.path.endswith("/closure"):
                self.answer(200, {"model": "proof/model", "release": "1.0.0", "lane": "plain", "scope": "runtime",
                                  "manifest": manifest_ref, "objects": sorted(objects, key=lambda obj: obj["sha256"]),
                                  "complete": True, "presign_max_digests": 4096})
            elif self.path.endswith("/presign"):
                now = int(time.time())
                digests = cast(list[str], request["digests"])
                self.answer(200, {"expires_at_unix": now + 3600, "server_time_unix": now,
                                  "urls": {digest: origin + "/object/" + digest for digest in digests}})
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            body = bodies[self.path.rsplit("/", 1)[-1]]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = f"http://127.0.0.1:{server.server_port}"
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    try:
        assert {"ensure/1", "get-range/1", "deliver/1", "pressure/1"} <= tensorfs.CAPABILITIES
        store = tensorfs.Store.init(root / "destination")
        events: list[tensorfs.EnsureEvent] = []
        result = tensorfs.ensure(store, "proof/model@1.0.0", hub=origin, lane="plain", step="load",
                                 allow_local=True, progress=events.append)
        check(result, records.EnsureResult)
        for event in events:
            check(event, records.EnsureEvent)
        assert result["manifest"] == manifest and result["bytes_total"] == total, result
        assert result["bytes_fetched"] == total and result["bytes_held"] == 0 and not result["joined"]
        assert events[0]["phase"] == "resolving" and all(e["step"] == "load" and e["model"] == "proof/model" for e in events)
        assert events[-1]["phase"] == "fetching" and events[-1]["bytes_done"] == total == events[-1]["bytes_total"]
        assert isinstance(events[-1]["rate"], float)

        warm = tensorfs.ensure(store, f"proof/model@1.0.0@{manifest}", hub=origin, lane="plain", allow_local=True)
        assert warm["bytes_fetched"] == 0 and warm["bytes_held"] == total, warm

        try:
            tensorfs.ensure(store, "proof/missing@1.0.0", hub=origin, allow_local=True)
        except errors.RefNotFound as error:
            assert "model_not_found" in str(error)
        else:
            raise AssertionError("an unknown model was not refused")

        def callback_failure(event: tensorfs.EnsureEvent) -> None:
            raise ValueError("callback sentinel")

        try:
            tensorfs.ensure(tensorfs.Store.init(root / "callback"), "proof/model@1.0.0", hub=origin,
                            allow_local=True, progress=callback_failure)
        except ValueError as error:
            assert str(error) == "callback sentinel"
        else:
            raise AssertionError("callback failure was swallowed")

        token = tensorfs.PullCancellation()
        token.cancel()
        try:
            tensorfs.ensure(tensorfs.Store.init(root / "cancelled"), "proof/model@1.0.0", hub=origin,
                            allow_local=True, cancellation=token)
        except errors.TransferFailed as error:
            assert "cancelled" in str(error)
        else:
            raise AssertionError("cancellation was ignored")

        relief = tensorfs.relieve(store, keep=[manifest])
        assert relief["capacity_bytes"] >= relief["available_bytes"] > 0, relief
        try:
            store.deliver_tree_root("sha256:" + "0" * 64)
        except errors.RootAbsent:
            pass
        else:
            raise AssertionError("an absent tree was delivered")
        print("ensure proof passed: events, result, warm, typed refusal, callback, cancellation, relieve")
    finally:
        server.shutdown()
        serving.join()
        server.server_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tensorfs-ensure-") as directory:
        prove(sys.argv[1], Path(directory))
