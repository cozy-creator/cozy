#!/usr/bin/env python3
"""Exercise the built Python pull API against a native checkpoint and loopback hub."""

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


class PullResult(TypedDict):
    bytes_held: int
    bytes_total: int
    bytes_moved: int


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
    hits: dict[str, int] = {}
    lock = threading.Lock()
    interrupt_next: list[tensorfs.PullCancellation | None] = [None]
    release_body = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def answer(self, data: object) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = cast(dict[str, object], json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if self.path.endswith("/closure"):
                self.answer({"model": "proof/model", "release": "1.0.0", "lane": "plain", "scope": "runtime",
                             "manifest": manifest_ref, "objects": sorted(objects, key=lambda obj: obj["sha256"]),
                             "complete": True, "presign_max_digests": 4096})
            elif self.path.endswith("/presign"):
                now = int(time.time())
                digests = cast(list[str], request["digests"])
                self.answer({"expires_at_unix": now + 3600, "server_time_unix": now,
                             "urls": {digest: origin + "/object/" + digest for digest in digests}})
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            digest = self.path.rsplit("/", 1)[-1]
            body = bodies[digest]
            with lock:
                hits[digest] = hits.get(digest, 0) + 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with lock:
                stop = interrupt_next[0]
                interrupt_next[0] = None
            if stop is not None:
                self.wfile.write(body[:1])
                self.wfile.flush()
                stop.cancel()
                release_body.wait()
                try:
                    self.wfile.write(body[1:])
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = f"http://127.0.0.1:{server.server_port}"
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    try:
        store = tensorfs.Store.init(root / "destination")
        token = tensorfs.PullCancellation()
        samples: list[tuple[int, int]] = []

        def stop_after_admission(available: int, total: int) -> None:
            samples.append((available, total))
            if available > 0:
                token.cancel()

        try:
            tensorfs.pull(store, origin, "proof/model@1.0.0", lane="plain", allow_local=True,
                          streams=1, cancellation=token, progress=stop_after_admission)
        except errors.Refusal as error:
            assert token.cancelled and "cancelled by its caller" in str(error), str(error)
        else:
            raise AssertionError("native pull ignored cancellation")
        assert len(samples) >= 2 and samples[0][0] == 0 and samples[-1][0] > 0
        completed = dict(hits)
        resumed: list[tuple[int, int]] = []
        result = cast(PullResult, tensorfs.pull(
            store, origin, "proof/model@1.0.0", lane="plain", allow_local=True, streams=1,
            cancellation=tensorfs.PullCancellation(), progress=lambda available, total: resumed.append((available, total))))
        check(result, records.PullResult)
        assert result["bytes_held"] > 0 and resumed[0][0] == result["bytes_held"]
        assert resumed[-1][0] == result["bytes_total"]
        assert all(hits[digest] == count for digest, count in completed.items()), "resume re-downloaded admitted objects"
        assert all(left[0] <= right[0] for left, right in zip(resumed, resumed[1:]))

        # A Python callback failure remains the original exception, even though
        # native peer streams must stop cooperatively before the call returns.
        def callback_failure(available: int, total: int) -> None:
            raise ValueError("callback sentinel")

        try:
            tensorfs.pull(store, origin, "proof/model@1.0.0", lane="plain", allow_local=True,
                          progress=callback_failure)
        except ValueError as error:
            assert str(error) == "callback sentinel"
        else:
            raise AssertionError("callback failure was swallowed")
        # Stop a real HTTP response while its body is still incomplete. The
        # origin does not finish that body until the cancelled native call returns.
        interrupted = tensorfs.Store.init(root / "in-flight")
        active_token = tensorfs.PullCancellation()
        interrupt_next[0] = active_token
        try:
            tensorfs.pull(interrupted, origin, "proof/model@1.0.0", lane="plain", allow_local=True,
                          streams=1, sample_seconds=0.01, cancellation=active_token)
        except errors.Refusal as error:
            assert active_token.cancelled and interrupt_next[0] is None
            assert "cancelled by its caller" in str(error), str(error)
        else:
            raise AssertionError("in-flight body ignored cancellation")
        finally:
            release_body.set()
        # Model pull has per-object resume: the unfinished object is retried;
        # cancellation does not poison subsequent admission or completion.
        tensorfs.pull(interrupted, origin, "proof/model@1.0.0", lane="plain", allow_local=True, streams=1)
        print("pull control proof passed: in-flight cancellation, verified-byte resume, progress, callback exception")
    finally:
        release_body.set()
        server.shutdown()
        serving.join()
        server.server_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tensorfs-pull-control-") as directory:
        prove(sys.argv[1], Path(directory))
