#!/usr/bin/env python3
"""Real component handles retain native access, then release their request provider."""

import gc
import contextlib
import functools
import http.server
import os
import threading
import pathlib
import tempfile
import weakref

import tensorfs
from tensorfs import raw


class Provider:
    def __init__(self):
        self.calls = 0
        self.block = False
        self.entered = threading.Event()
        self.resume = threading.Event()

    def headers(self, method, url, headers, payload):
        self.calls += 1
        if self.block:
            self.entered.set()
            assert self.resume.wait(10), "the provider was never released"
        return {}


@contextlib.contextmanager
def serve(folder):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    with tensorfs.Writer(cache_dir=root / "writer") as writer:
        component = writer.component("model", cls="test.Model", config={"size": 4})
        component.put("weight", b"data", dtype="U8", shape=[4])
        component.asset("vocab.txt", b"a\nb\n")
    tensorfs.export(writer.commit(), root / "source", format="cozytensors")
    endpoint = (root / "origin").as_uri()
    tensorfs.upload_folder(repo_id="acme/lifetime", revision="one", folder_path=root / "source",
                           endpoint=endpoint, cache_dir=root / "upload")
    with serve(root / "origin") as endpoint:
        provider = Provider()
        alive = weakref.ref(provider)
        released = threading.Event()
        weakref.finalize(provider, released.set)
        checkpoint = tensorfs.open("acme/lifetime", revision="one", endpoint=endpoint,
                                   token=provider, cache_dir=root / "reader")
        assert provider.calls > 0, "the real HTTP transport did not use the provider"
        components = checkpoint.components
        component = components["model"]
        tensors = component.tensors
        del provider, checkpoint, components
        gc.collect()
        assert alive() is not None, "a live component lost its request provider"
        assert component.asset("vocab.txt") == b"a\nb\n"
        assert tensors.read("weight").tobytes() == b"data"
        with tensorfs.Writer(root / "copy", cache_dir=root / "copy-cache") as copy:
            copy.component("copy", cls="test.Copy", config={}).copy("weight", component)
        assert tensorfs.open(root / "copy", cache_dir=root / "copy-cache").components["copy"].tensors.read("weight").tobytes() == b"data"
        del component
        gc.collect()
        assert alive() is not None, "a live Tensors handle lost its request provider"
        array = tensors.read("weight")
        del tensors
        gc.collect()
        assert released.wait(10), "deferred final Tensors release did not finish within 10 seconds"
        assert alive() is None, "the dropped checkpoint/components/Tensors retained their request provider"
        assert array.tobytes() == b"data", "Array bytes depend on the checkpoint/provider lifetime"

        provider = Provider()
        alive = weakref.ref(provider)
        released = threading.Event()
        weakref.finalize(provider, released.set)
        checkpoint = tensorfs.open("acme/lifetime", revision="one", endpoint=endpoint,
                                   token=provider, cache_dir=root / "close-reader")
        assert "model" in checkpoint.components
        del provider
        checkpoint.close()
        gc.collect()
        assert released.wait(10), "deferred closed-parent release did not finish within 10 seconds"
        assert alive() is None, "a closed parent retained its cached components/provider"

        provider = Provider()
        alive = weakref.ref(provider)
        released = threading.Event()
        weakref.finalize(provider, released.set)
        checkpoint = tensorfs.open("acme/lifetime", revision="one", endpoint=endpoint,
                                   token=provider, cache_dir=root / "race-reader")
        provider.block = True
        entered, resume = provider.entered, provider.resume
        download = checkpoint.download
        errors = []

        def fetching():
            try:
                download()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=fetching)
        thread.start()
        assert entered.wait(10), "the fetch never reached the real HTTP provider"
        checkpoint.close()
        checkpoint.close()
        del provider
        gc.collect()
        assert alive() is not None, "close released an in-flight call's provider"
        resume.set()
        thread.join(10)
        assert not thread.is_alive() and not errors, errors
        del download, checkpoint
        gc.collect()
        assert released.wait(10), "deferred completed-call release did not finish within 10 seconds"
        assert alive() is None, "a completed call retained its closed parent's provider"

    with tensorfs.open(root / "source", cache_dir=root / "isolated") as checkpoint:
        assert checkpoint.__enter__() is checkpoint
        assert checkpoint.verify() == []
        component = checkpoint.components["model"]
        part = raw.parts(checkpoint, "model", "weight")[""]
        chunk = checkpoint.chunks[part.chunk]
        descriptor = checkpoint.open_chunk(part.chunk)
        os.lseek(descriptor, part.offset, os.SEEK_SET)
        assert os.read(descriptor, 4) == b"data"
        try:
            os.write(descriptor, b"x")
        except OSError:
            pass
        else:
            raise AssertionError("open_chunk returned a writable descriptor")
        os.close(descriptor)
        blob = root / "source/chunks" / chunk[0]
        saved = blob.read_bytes()
        corrupted = bytes([saved[0] ^ 1]) + saved[1:]
        blob.write_bytes(corrupted)
        assert chunk in checkpoint.verify()
        assert blob.read_bytes() == corrupted, "verify changed the user's file"
        blob.unlink()
        assert chunk in checkpoint.verify()
        blob.write_bytes(saved)
        assert checkpoint.verify() == []
        header = root / "source/model.cozytensors"
        saved_header = header.read_bytes()
        header.write_bytes(saved_header[:-1] + bytes([saved_header[-1] ^ 1]))
        assert checkpoint.header_ref in checkpoint.verify()
        header.write_bytes(saved_header)
        assert checkpoint.verify() == []
        descriptor = checkpoint.open_chunk(part.chunk)
    for call in [lambda: checkpoint.components, lambda: checkpoint.missing(), lambda: checkpoint.download(),
                 lambda: checkpoint.open_chunk(0), lambda: checkpoint.verify(), lambda: checkpoint.__enter__()]:
        try:
            call()
        except tensorfs.InvalidError as error:
            assert "closed" in error.detail
        else:
            raise AssertionError("a closed checkpoint still accepted byte access")
    assert component.tensors.read("weight").tobytes() == b"data", "a with-block invalidated an external child"
    os.lseek(descriptor, part.offset, os.SEEK_SET)
    assert os.read(descriptor, 4) == b"data", "closing the parent closed the caller's descriptor"
    os.close(descriptor)

    try:
        with tensorfs.open(root / "source", cache_dir=root / "exception") as checkpoint:
            raise RuntimeError("with-block body failed")
    except RuntimeError:
        pass
    try:
        checkpoint.missing()
    except tensorfs.InvalidError:
        pass
    else:
        raise AssertionError("an exception skipped checkpoint.close")

print("checkpoint lifetime: child reads/copy, close/fetch race, provider release, readonly descriptor and verify passed")
