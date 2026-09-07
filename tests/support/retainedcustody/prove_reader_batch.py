"""Four public native CLI children must overlap real immutable HTTP body uploads."""

import base64
import hashlib
import http.server
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import tensorfs


with tempfile.TemporaryDirectory(prefix="cozy-batch-recovery-proof-") as directory:
    root = Path(directory)
    store = tensorfs.Store.init(root / "store")
    plain = next(digest for name, digest in tensorfs.seed_digests() if name == "plain/1")
    length = 1 << 20
    bodies = {f"weight_{i}": bytes([i + 1]) * length for i in range(4)}
    transaction_id = "sha256:" + "2" * 64
    writer = store.begin_derived(
        transaction_id, 1, {},
        {"model": {"drop": [], "add": {
            name: {"logical_dtype": "f32", "shape": [length // 4], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": [length // 4]}}}
            for name in bodies
        }}},
        {"model": {"kind": "add"}}, [("model", name) for name in bodies],
        4 * length + 4096,
    )
    for name, body in bodies.items():
        writer.add_part("model", name, "value", io.BytesIO(body))
    writer.add_config("model", io.BytesIO(b"{}"))
    receipt = writer.commit()
    manifest = {"digest": "sha256:" + receipt["manifest"]["sha256"],
                "length": receipt["manifest"]["length"]}
    store.derived_adopt(transaction_id, "operator-batch-proof")
    expected = {hashlib.sha256(body).hexdigest(): body for body in bodies.values()}
    objects = [{"digest": str(row["id"]), "length": row["length"]}
               for row in store.walk(manifest["digest"])
               if str(row["id"])[7:] in expected]
    assert len(objects) == 4
    uploaded = {}
    attempts = []
    active = peak = 0
    lock = threading.Lock()
    all_bodies = threading.Barrier(4)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            global active, peak
            digest = self.path.split("?")[0].split("/")[-1]
            with lock:
                active += 1
                peak = max(peak, active)
                fresh = digest not in uploaded
                attempts.append(digest)
            try:
                if fresh:
                    # A serial adapter cannot pass: all four real HTTP handlers
                    # must be receiving bodies before any one completes.
                    all_bodies.wait(timeout=8)
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                assert raw == expected[digest]
                assert self.headers["If-None-Match"] == "*"
                assert self.headers["X-Amz-Checksum-Sha256"] == base64.b64encode(
                    bytes.fromhex(digest)
                ).decode()
                with lock:
                    status = 412 if digest in uploaded else 200
                    uploaded[digest] = raw
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
            finally:
                with lock:
                    active -= 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("reader.py")), "--store",
         str(root / "store"), "--tfs", str(Path(sys.executable).parent / "tfs"),
         "--allow-local"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        def request(obj):
            return {"manifest": manifest, "object": obj, "grant": {
                "url": f"http://127.0.0.1:{server.server_port}/objects/{obj['digest'][7:]}?secret=never-print",
                "required_headers": {
                    "if-none-match": "*", "x-amz-checksum-sha256":
                        base64.b64encode(bytes.fromhex(obj["digest"][7:])).decode(),
                },
            }}

        def ask(value):
            child.stdin.write(json.dumps(value) + "\n")
            child.stdin.flush()
            raw = child.stdout.readline()
            assert raw and "never-print" not in raw
            return json.loads(raw)

        batch = [request(obj) for obj in objects]
        invalid = json.loads(json.dumps(batch))
        invalid[-1]["grant"]["required_headers"]["if-none-match"] = "wrong"
        result = ask(invalid)
        assert len(result) == 4 and all(not row["ok"] for row in result) and not attempts
        invalid = json.loads(json.dumps(batch))
        invalid[-1]["manifest"] = {"digest": "sha256:" + "f" * 64, "length": 164}
        result = ask(invalid)
        assert len(result) == 4 and all(not row["ok"] for row in result) and not attempts
        invalid = json.loads(json.dumps(batch))
        invalid[-1]["object"] = {"digest": "sha256:" + "f" * 64, "length": length}
        invalid[-1]["grant"]["required_headers"]["x-amz-checksum-sha256"] = base64.b64encode(b"\xff" * 32).decode()
        result = ask(invalid)
        assert len(result) == 4 and all(not row["ok"] for row in result) and not attempts
        assert not ask(batch + [batch[0]])["ok"] and not attempts

        result = ask(batch)
        assert len(result) == 4 and peak == 4
        for obj, row in zip(objects, result, strict=True):
            assert row["ok"] and row["object_id"] == obj["digest"] and row["http_status"] == 200
            assert row["manifest_digest"] == manifest["digest"] and row["transferred_bytes"] == length
        assert uploaded == expected
        replay = ask(batch)
        assert all(row["ok"] and row["http_status"] == 412 for row in replay)
        single = ask(batch[0])
        assert single["ok"] and single["http_status"] == 412
        child.stdin.close()
        assert child.wait(timeout=5) == 0
        assert "never-print" not in child.stderr.read()
        assert store.derived_lookup(transaction_id)["disposition"]["kind"] == "adopted"
        print(json.dumps({"tensorfs": tensorfs.__version__, "uploads": 4, "bytes": 4 * length,
                          "peak_simultaneous_bodies": peak, "ordered_results": True,
                          "all_preflight_before_any_socket": True, "single_api_preserved": True,
                          "replay_412": True, "original_disposition_retained": True,
                          "secrets_not_printed": True}))
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        server.shutdown()
        server.server_close()
