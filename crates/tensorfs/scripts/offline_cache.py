#!/usr/bin/env python3
"""Real HTTP/cache proof: only network failure may use the last named resolution."""

import functools
import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorfs


class OfflineCache(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="tensorfs-offline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "reader"
        with tensorfs.Writer(
            self.root / "source", cache_dir=self.root / "author"
        ) as writer:
            writer.component("model", cls="test.Model", config={}).put(
                "weight", b"data", dtype="U8", shape=[4]
            )
        tensorfs.upload_folder(
            repo_id="acme/offline",
            revision="one",
            folder_path=self.root / "source",
            endpoint=(self.root / "origin").as_uri(),
            cache_dir=self.root / "upload",
        )
        self.mode = "normal"
        owner = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/refs/") and owner.mode != "normal":
                    if owner.mode in ("denied", "missing"):
                        self.send_error(403 if owner.mode == "denied" else 404)
                    else:
                        body = b"not JSON"
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                super().do_GET()

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(Handler, directory=str(self.root / "origin")),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.stop_server)
        with self.open() as checkpoint:
            self.header = checkpoint.id
            self.assertEqual(
                checkpoint.components["model"].tensors.read("weight").tobytes(), b"data"
            )

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(5)
            self.assertFalse(
                self.thread.is_alive(), "the owned HTTP server did not stop"
            )
            self.server = None

    def open(self, reference="acme/offline", **kwargs):
        return tensorfs.open(
            reference, endpoint=self.endpoint, cache_dir=self.cache, **kwargs
        )

    def check_bytes(self, checkpoint):
        with checkpoint:
            self.assertEqual(checkpoint.id, self.header)
            self.assertEqual(
                checkpoint.components["model"].tensors.read("weight").tobytes(), b"data"
            )

    def test_stopped_server_keeps_explicit_cached_modes(self):
        self.stop_server()
        self.check_bytes(self.open(self.header))
        self.check_bytes(self.open(revision="one"))
        self.check_bytes(self.open(local_files_only=True))

    def test_stopped_server_uses_last_unversioned_resolution(self):
        self.stop_server()
        self.check_bytes(self.open())

    def test_online_default_refreshes_then_offline_uses_last_resolution(self):
        with tensorfs.Writer(
            self.root / "second", cache_dir=self.root / "author"
        ) as writer:
            writer.component("model", cls="test.Model", config={}).put(
                "weight", b"next", dtype="U8", shape=[4]
            )
        tensorfs.upload_folder(
            repo_id="acme/offline",
            revision="two",
            folder_path=self.root / "second",
            endpoint=(self.root / "origin").as_uri(),
            cache_dir=self.root / "upload",
        )
        with self.open() as checkpoint:
            self.assertNotEqual(checkpoint.id, self.header)
            self.header = checkpoint.id
            self.assertEqual(
                checkpoint.components["model"].tensors.read("weight").tobytes(), b"next"
            )
        self.stop_server()
        with self.open() as checkpoint:
            self.assertEqual(checkpoint.id, self.header)
            self.assertEqual(
                checkpoint.components["model"].tensors.read("weight").tobytes(), b"next"
            )

    def test_remote_refusals_do_not_become_stale_success(self):
        for mode, expected in (
            ("denied", tensorfs.DeniedError),
            ("missing", tensorfs.NotFoundError),
            ("invalid", tensorfs.InvalidError),
        ):
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaises(expected):
                    self.open()


if __name__ == "__main__":
    unittest.main(verbosity=2)
