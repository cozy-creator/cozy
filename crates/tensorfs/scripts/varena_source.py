#!/usr/bin/env python3
"""Cold remote placement retains its source and owned bytes through parent close."""
import ctypes
import functools
import gc
import http.server
import pathlib
import tempfile
import threading
import weakref

import tensorfs
from tensorfs import varena


class Provider:
    def __init__(self):
        self.calls = 0

    def headers(self, method, url, headers, payload):
        self.calls += 1
        return {}


with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    with tensorfs.Writer(root / "source", cache_dir=root / "writer") as writer:
        writer.component("model", cls="test.Model", config={}).put("weight", b"data", dtype="U8", shape=[4])
    tensorfs.upload_folder(repo_id="acme/placement", revision="one", folder_path=root / "source",
                           endpoint=(root / "origin").as_uri(), cache_dir=root / "upload")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root / "origin"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = Provider()
        alive = weakref.ref(provider)
        checkpoint = tensorfs.open("acme/placement", revision="one", endpoint=f"http://127.0.0.1:{server.server_port}",
                                   token=provider, cache_dir=root / "reader")
        assert checkpoint.missing(), "fixture must begin with cold chunk bytes"
        before = provider.calls
        weights = varena.Host().register("cold", checkpoint, [["model/weight"]])
        assert provider.calls > before, "placement did not use TensorFS's real HTTP provider"
        assert not checkpoint.missing(), "selected chunk was not fetched"
        checkpoint.close()
        del provider, checkpoint
        gc.collect()
        assert alive() is not None, "placement lost its retained source context"
        pin = weights.bind(0)
        part = weights.parts[0]
        assert ctypes.string_at(pin.ptr + part.offset, part.nbytes) == b"data"
        weights.close()
        del weights
        gc.collect()
        assert alive() is not None, "a bound region lost its source context"
        assert ctypes.string_at(pin.ptr + part.offset, part.nbytes) == b"data"
        del pin, part
        gc.collect()
        assert alive() is None, "last released placement retained its provider"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
print("varena source: cold HTTP fetch, parent close, bound bytes and provider release passed")
