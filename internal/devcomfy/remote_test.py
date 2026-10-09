"""Real CPU HTTP/child-process proof for the owned service protocol; no CUDA."""
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("comfy_dev_remote", HERE / "remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.children = []
        original_popen = subprocess.Popen

        def spawned(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            self.children.append(child)
            return child

        self.popen_patch = patch.object(subprocess, "Popen", side_effect=spawned)
        self.popen_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        (self.outputs / "sample.mp4").write_bytes(b"original retained bytes" * 1024)
        self.posts = []
        self.history_ready = threading.Event()
        self.drop_ack = False
        self.trace_steps = None
        self.history_status = "success"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                if self.path != "/prompt":
                    self.send_error(404)
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts.append(body)
                if outer.trace_steps is not None:
                    directory = outer.root / "traces" / body["prompt_id"]
                    directory.mkdir(parents=True)
                    ranges = [{"label": f"h3.step.{i}.{'dense' if i < 4 else 'sparse_eligible'}",
                               "start_unix_ns": 100 + 2*i, "end_unix_ns": 101 + 2*i}
                              for i in range(outer.trace_steps)]
                    if outer.trace_steps >= 0:
                        (directory / "summary.json").write_text(json.dumps({"expected_steps": 8,
                            "steps_observed": outer.trace_steps, "ranges": ranges,
                            "segments": [{"file": "trace.json.gz"}]}))
                    (directory / "trace.json.gz").write_bytes(b"retained diagnostic fixture")
                if outer.drop_ack:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.send_json({"prompt_id": body["prompt_id"]})

            def do_GET(self):
                if self.path == "/queue":
                    self.send_json({"queue_running": [], "queue_pending": []})
                    return
                if not outer.posts or not outer.history_ready.is_set():
                    self.send_json({})
                    return
                prompt = outer.posts[0]["prompt_id"]
                self.send_json({prompt: {"status": {"status_str": outer.history_status, "messages": [
                    ["execution_start", {"timestamp": 1000}],
                    ["execution_success", {"timestamp": 2500}]]},
                    "outputs": {"1": {"videos": [{"type": "output", "filename": "sample.mp4"}]}}}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.request = {"action": "start", "operation": "a" * 64, "state_root": str(self.root / "state"),
                        "observer_source": (HERE / "observer.py").read_text(),
                        "driver_source": (HERE / "remote.py").read_text(),
                        "payload": {"graph_json": '{"1":{"class_type":"Fixture","inputs":{}}}',
                                    "port": self.server.server_port, "timeout_s": 30, "trace": False,
                                    "trace_root": "", "output_root": str(self.outputs), "expected_steps": 8}}

    def tearDown(self):
        root = self.root / "state" / ("a" * 64)
        process = root / "process.json"
        if process.exists():
            owner = json.loads(process.read_text())
            if remote.identity(owner["pid"]) == owner["start_ticks"]:
                os.kill(owner["pid"], signal.SIGTERM)
            try:
                os.waitpid(owner["pid"], 0)
            except ChildProcessError:
                pass
        self.server.shutdown()
        self.server.server_close()
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        self.popen_patch.stop()
        self.temp.cleanup()

    def wait(self, condition):
        end = time.monotonic() + 15
        while time.monotonic() < end:
            value = condition()
            if value:
                return value
            time.sleep(.05)
        self.fail("condition did not become true")

    def query(self):
        return remote.dispatch({**self.request, "action": "status"})

    def finished(self):
        return self.wait(lambda: (r if (r := self.query())["status"] == "success" else None))

    def test_lost_ack_replay_keeps_one_prompt_and_exact_original_artifacts(self):
        self.drop_ack = True
        self.history_ready.set()
        remote.dispatch(self.request)
        self.wait(lambda: self.posts)
        remote.dispatch(self.request)  # an interrupted client retries the same identity
        result = self.finished()
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(result["prompt_id"], self.posts[0]["prompt_id"])
        self.assertEqual(result["server_execution_seconds"], 1.5)
        video = next(f for f in result["files"] if f["name"].endswith("sample.mp4"))
        root = self.root / "state" / ("a" * 64)
        self.assertEqual((root / "checkpoint/artifacts" / video["name"]).read_bytes(),
                         (self.outputs / "sample.mp4").read_bytes())
        # Exercise the same framed stdin/bootstrap used by SSH, including binary resume.
        request = {**self.request, "action": "fetch", "name": video["name"], "offset": 17}
        bootstrap = "import json,sys; _cozy_request=json.loads(sys.stdin.readline()); exec(compile(sys.stdin.read(), '<cozy-dev-comfy>', 'exec'))"
        response = subprocess.run([sys.executable, "-c", bootstrap],
                                  input=(json.dumps(request) + "\n" + self.request["driver_source"]).encode(),
                                  capture_output=True, check=True)
        self.assertEqual(response.stdout, (self.outputs / "sample.mp4").read_bytes()[17:])
        with self.assertRaisesRegex(ValueError, "idempotency conflict"):
            remote.dispatch({**self.request, "payload": {**self.request["payload"], "graph_json": '{"other":{}}'}})

    def test_dead_observer_resumes_read_only_after_post(self):
        remote.dispatch(self.request)
        self.wait(lambda: self.posts)
        owner = self.query()["process"]
        os.kill(owner["pid"], signal.SIGTERM)
        self.wait(lambda: remote.identity(owner["pid"]) is None)
        self.history_ready.set()
        remote.dispatch(self.request)
        result = self.finished()
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(result["prompt_id"], self.posts[0]["prompt_id"])

    def test_missing_submission_checkpoint_after_acceptance_never_posts_again(self):
        remote.dispatch(self.request)
        self.wait(lambda: self.posts)
        owner = self.query()["process"]
        os.kill(owner["pid"], signal.SIGTERM)
        self.wait(lambda: remote.identity(owner["pid"]) is None)
        root = self.root / "state" / ("a" * 64)
        (root / "checkpoint/state.json").unlink()
        (root / "process.json").unlink()  # Only the pre-spawn durable attempt marker remains.
        with self.assertRaisesRegex(ValueError, "checkpoint is missing"):
            remote.dispatch(self.request)
        self.assertEqual(len(self.posts), 1)

    def test_completed_replay_never_resubmits_and_fetch_refuses_unknown_path(self):
        self.history_ready.set()
        remote.dispatch(self.request)
        first = self.finished()
        self.assertEqual(remote.dispatch(self.request), first)
        self.assertEqual(len(self.posts), 1)
        with self.assertRaisesRegex(ValueError, "not in the retained manifest"):
            remote.dispatch({**self.request, "action": "fetch", "name": "../../secret"})

    def test_lost_acknowledged_remote_ledger_is_not_recreated(self):
        prepared = remote.dispatch({**self.request, "action": "prepare"})
        self.assertEqual(prepared["status"], "prepared")
        self.assertFalse(self.posts)
        path = self.root / "state" / ("a" * 64) / "request.json"
        path.unlink()  # Simulate loss of an acknowledged durable remote ledger.
        with self.assertRaisesRegex(ValueError, "receipt is missing"):
            remote.dispatch({**self.request, "require_existing": True})
        self.assertFalse(self.posts)

    def test_partial_trace_is_unsuccessful_but_original_evidence_is_retained(self):
        self.trace_steps = 7
        self.history_ready.set()
        self.request["payload"].update(trace=True, trace_root=str(self.root / "traces"))
        remote.dispatch(self.request)
        result = self.wait(lambda: (r if (r := self.query())["status"] == "failed" else None))
        self.assertIn("expected 8 complete step ranges", result["detail"])
        self.assertIn("trace/trace.json.gz", [item["name"] for item in result["files"]])
        self.assertEqual(len(self.posts), 1)

    def test_server_failure_still_retains_exported_trace(self):
        self.trace_steps = 8
        self.history_status = "error"
        self.history_ready.set()
        self.request["payload"].update(trace=True, trace_root=str(self.root / "traces"))
        remote.dispatch(self.request)
        result = self.wait(lambda: (r if (r := self.query())["status"] == "failed" else None))
        self.assertIn("trace/trace.json.gz", [item["name"] for item in result["files"]])
        self.assertIsNone(result["server_execution_seconds"])

    def test_failed_export_without_summary_retains_available_partial_files(self):
        self.trace_steps = -1
        self.history_status = "error"
        self.history_ready.set()
        self.request["payload"].update(trace=True, trace_root=str(self.root / "traces"))
        remote.dispatch(self.request)
        result = self.wait(lambda: (r if (r := self.query())["status"] == "failed" else None))
        self.assertIn("trace/trace.json.gz", [item["name"] for item in result["files"]])
        receipt = self.root / "state" / ("a" * 64) / "checkpoint/artifacts/trace-retention.json"
        self.assertFalse(json.loads(receipt.read_text())["completed_summary_present"])

    def test_timed_out_export_without_summary_retains_available_partial_files(self):
        self.trace_steps = -1
        self.request["payload"].update(trace=True, trace_root=str(self.root / "traces"), timeout_s=3)
        remote.dispatch(self.request)
        result = self.wait(lambda: (r if (r := self.query())["status"] == "timed_out" else None))
        self.assertIn("trace/trace.json.gz", [item["name"] for item in result["files"]])
        self.assertEqual(len(self.posts), 1)


if __name__ == "__main__":
    unittest.main()
