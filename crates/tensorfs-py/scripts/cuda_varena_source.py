#!/usr/bin/env python3
"""Real CUDA lazy-provider/final-release regression; requires a coherent TensorFS wheel.

Run serially on an authorized CUDA device 0 with PyTorch installed:
    python cuda_varena_source.py --timeout 45

The outer process bounds each child, including a child completely deadlocked on the GIL.
Public Writer generates the fixture; HTTP reads use only that loopback data, with no model download.
"""

import argparse
import contextlib
import faulthandler
import functools
import gc
import hashlib
import http.server
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from pathlib import Path
from urllib.parse import urlsplit

CASES = ("last-weightset", "callback-close", "last-pin", "last-consumer", "pending-consumer-b", "pending-ptds-consumer", "pending-legacy-consumer")
WAIT = 12
AUTH = "r2-source-proof"
DATA = bytes(range(256)) * 256


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def wait_for(predicate, message):
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        gc.collect()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


class Provider:
    def __init__(self, mode):
        self.mode = mode
        self.chunks = set()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.weights = None
        self.observed = {"calls": 0, "chunks": 0, "thread": None}

    def headers(self, method, url, headers, payload):
        assert isinstance(headers, dict)
        self.observed["calls"] += 1
        if method == "GET" and self.chunks.intersection(urlsplit(url).path.split("/")):
            self.observed["chunks"] += 1
            self.observed["thread"] = threading.get_ident()
            self.entered.set()
            if self.mode != "read":
                assert self.release.wait(WAIT), (
                    "the main thread never released the provider"
                )
            if self.mode == "close":
                # Remove the Python back-reference before closing the actual native handle.
                weights, self.weights = self.weights, None
                assert weights is not None
                weights.close()
                self.closed.set()
        return {"x-tensorfs-proof": AUTH}


@contextlib.contextmanager
def fixture(tensorfs):
    with tempfile.TemporaryDirectory(prefix="tensorfs-cuda-source-") as temp:
        root = Path(temp)
        with tensorfs.Writer(cache_dir=root / "writer") as writer:
            writer.component("model", cls="test.Model", config={}).put(
                "weight", DATA, dtype="U8", shape=[len(DATA)]
            )
        tensorfs.materialize(writer.commit(), local_dir=root / "source")
        tensorfs.upload_folder(
            repo_id="acme/placement",
            revision="one",
            folder_path=root / "source",
            endpoint=(root / "origin").as_uri(),
            cache_dir=root / "upload",
        )
        requests = []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def authorized(self):
                if self.headers.get("x-tensorfs-proof") != AUTH:
                    self.send_error(403, "supplied provider header required")
                    return False
                requests.append((self.command, self.path))
                return True

            def do_GET(self):
                if self.authorized():
                    super().do_GET()

            def do_HEAD(self):
                if self.authorized():
                    super().do_HEAD()

            def log_message(self, *_args):
                pass

        handler = functools.partial(Handler, directory=str(root / "origin"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield root, f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)
            assert not thread.is_alive(), "the owned HTTP server did not stop"


def values(torch, pin):
    assert pin.__dlpack_device__() == (2, 0), "a placement fell back to host memory"
    tensor = torch.from_dlpack(pin)
    assert tensor.is_cuda and tensor.device.index == 0
    assert tensor.dtype == torch.uint8 and tensor.numel() == len(DATA)
    assert bytes(tensor.cpu().tolist()) == DATA
    return tensor


def idle(gpu):
    stats = gpu.stats()
    return not stats.sets and stats.mapped == 0 and stats.slots == 0


def warm(tensorfs, torch, varena, gpu, root, stream):
    # Initialize the same native read engine/driver before measuring tiny VMM reclamation.
    checkpoint = tensorfs.open(root / "source", cache_dir=root / "warm")
    weights = gpu.register("warm", checkpoint, [["model/weight"]])
    pin = weights.bind(0, stream.cuda_stream)
    tensor = values(torch, pin)
    weights.close()
    checkpoint.close()
    del tensor, pin, weights, checkpoint
    wait_for(lambda: idle(gpu), "warmup retained its native placement")
    gpu.trim()
    assert gpu.stats().idle == 0
    varena.copies()


def drop_last(holder, started, done):
    victim = holder.pop()
    started.set()
    del victim
    done.set()


def cleanup(gpu, alive, held_free, mapping_bytes):
    # Deliberately wait in pure Python first: native cleanup must flush its deferred decrefs.
    wait_for(lambda: alive() is None, "native source/provider survived its final owner")
    wait_for(
        lambda: idle(gpu),
        "native placement mappings/VA bookkeeping survived final release",
    )
    gpu.trim()
    wait_for(
        lambda: (
            gpu.stats().idle == 0
            and gpu.stats().driver_free + (64 << 10) >= held_free + mapping_bytes
        ),
        "native pages were not returned to the driver after trim",
    )
    stats = gpu.stats()
    return {
        "sets": len(stats.sets),
        "mapped": stats.mapped,
        "idle": stats.idle,
        "slots": stats.slots,
        "driver_recovered": stats.driver_free - held_free,
    }


def run_case(case):
    import tensorfs
    from tensorfs import _ext, varena

    assert "torch" not in sys.modules, "plain TensorFS imported a framework"
    import torch

    assert torch.cuda.is_available(), "this gate requires a real CUDA GPU"
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1)
    stream.synchronize()
    properties = torch.cuda.get_device_properties(0)
    with Path(_ext.__file__).open("rb") as binary:
        extension_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    emit(
        case=case,
        tensorfs=tensorfs.__version__,
        distribution=importlib.metadata.version("tensorfs"),
        extension=str(_ext.__file__),
        extension_sha256=extension_sha256,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=0,
        gpu=properties.name,
        uuid=str(getattr(properties, "uuid", "unavailable")),
        total_memory=properties.total_memory,
    )

    with fixture(tensorfs) as (root, endpoint, requests):
        gpu = varena.Gpu(0, chunk=2 << 20)
        warm(tensorfs, torch, varena, gpu, root, stream)
        mode = (
            "block"
            if case == "last-weightset"
            else "close"
            if case == "callback-close"
            else "read"
        )
        provider = Provider(mode)
        checkpoint = tensorfs.open(
            "acme/placement",
            revision="one",
            endpoint=endpoint,
            token=provider,
            cache_dir=root / "reader",
        )
        header = checkpoint.id
        assert header is not None and checkpoint.missing(), (
            "the fixture must be chunk-cold"
        )
        chunk_digests = {digest for digest, _length in checkpoint.chunks}
        provider.chunks = chunk_digests
        before = (provider.observed["calls"], len(requests))
        weights = gpu.register(case, checkpoint, [["model/weight"]])
        assert before == (provider.observed["calls"], len(requests)), (
            "registration performed HTTP I/O"
        )
        assert checkpoint.missing(), "registration fetched chunks"
        assert weights.device == 0 and len(weights.spans) == 1
        mapping_bytes = weights.spans[0]
        alive = weakref.ref(provider)
        entered, release, closed, observed = (
            provider.entered,
            provider.release,
            provider.closed,
            provider.observed,
        )
        if case == "callback-close":
            provider.weights = weights
        weights.prefetch(0)
        assert entered.wait(WAIT), "prefetch never called the real chunk provider"
        assert observed["thread"] != threading.get_ident(), (
            "payload provider did not run from native fetch"
        )
        held_free = gpu.stats().driver_free
        assert gpu.stats().mapped == mapping_bytes
        checkpoint.close()
        del checkpoint

        consumer_proof = {}
        if case == "last-weightset":
            holder = [weights]
            del weights, provider
            started, done = threading.Event(), threading.Event()
            deleter = threading.Thread(
                target=drop_last, args=(holder, started, done), daemon=True
            )
            deleter.start()
            assert started.wait(WAIT)
            assert done.wait(2), (
                "last WeightSet destruction waited for its blocked provider"
            )
            assert alive() is not None, "an in-flight fetch lost its source context"
            release.set()
            deleter.join(2)
            assert not deleter.is_alive()
        elif case == "callback-close":
            # prefetch has returned: no bind call retains a native Arc and masks self-close.
            release.set()
            assert closed.wait(WAIT), (
                "provider closing its own WeightSet waited on itself"
            )
            try:
                weights.bind(0)
            except varena.Closed:
                pass
            else:
                raise AssertionError("the provider did not close its actual handle")
            del weights, provider
        elif case in ("pending-consumer-b", "pending-legacy-consumer"):
            legacy = case == "pending-legacy-consumer"
            pin = weights.bind(0, stream.cuda_stream)
            # Make the fill already ready: readiness must not erase later consumers.
            stream.synchronize()
            consumer = torch.cuda.default_stream(0) if legacy else torch.cuda.Stream(device=0)
            assert stream.cuda_stream > 2
            assert consumer.cuda_stream == 0 if legacy else consumer.cuda_stream > 2
            assert consumer.cuda_stream != stream.cuda_stream
            with torch.cuda.stream(consumer):
                # An explicit capsule preserves stream=None: Torch must not replace it
                # with its own producer call. Native None means legacy1; Torch's
                # default stream uses raw0, which its CUDA runtime treats as legacy.
                tensor = torch.from_dlpack(pin.__dlpack__(stream=None)) if legacy else torch.from_dlpack(pin)
                assert tensor.is_cuda and tensor.device.index == 0
                assert tensor.dtype == torch.uint8 and tensor.numel() == len(DATA)
                assert tensor.data_ptr() == pin.ptr
                copied = torch.empty_like(tensor)
                assert copied.data_ptr() != tensor.data_ptr()
            # Allocate the independent result before the driver-free baseline. Neither
            # this output nor its byte comparison may keep the foreign source alive.
            held_free = gpu.stats().driver_free
            foreign = weakref.ref(tensor)
            heartbeat, stop = [0], threading.Event()

            def beat_consumer():
                while not stop.wait(0.001):
                    heartbeat[0] += 1

            ticker = threading.Thread(target=beat_consumer, daemon=True)
            ticker.start()
            end = torch.cuda.Event()
            try:
                with torch.cuda.stream(consumer):
                    torch.cuda._sleep(2_000_000_000)
                    copied.copy_(tensor, non_blocking=True)
                    end.record()
                assert not end.query(), "consumer B needs actual pending GPU work"
                weights.close()
                del weights, provider, pin
                ticks, started = heartbeat[0], time.monotonic()
                del tensor
                elapsed = time.monotonic() - started
                gc.collect()
                assert foreign() is None, "a Python tensor still masks final consumer release"
                # No .cpu(), stats/recovery, trim or synchronization here: those could
                # complete B and hide a release incorrectly fenced only on bind stream A.
                time.sleep(0.1)
                assert not end.query(), "consumer B completed before the retention observation"
                assert alive() is not None, "pending consumer B lost its native source/provider"
                assert elapsed < 0.25, "last consumer B release waited for CUDA work"
                assert heartbeat[0] - ticks >= 3, "consumer B release stopped Python's heartbeat"
                consumer_proof = {
                    "bind_stream": stream.cuda_stream,
                    "consumer_stream": 1 if legacy else consumer.cuda_stream,
                    "distinct_nondefault": not legacy,
                    "fill_ready_before_consume": True,
                    "pending_after_drop": True,
                    "source_retained_while_pending": True,
                    "foreign_python_owner_collected": True,
                    "drop_seconds": elapsed,
                    "heartbeat_ticks": heartbeat[0] - ticks,
                }
                if legacy:
                    consumer_proof.update(
                        scope="final-release-custody",
                        dlpack_stream=None,
                        torch_consumer_stream=consumer.cuda_stream,
                        default_legacy=True,
                        pending_readiness_device_proof=False,
                    )
            finally:
                # Rescue our own queued read even when an assertion fails. The existing
                # isolated-child timeout still bounds a broken driver/native teardown.
                try:
                    consumer.synchronize()
                finally:
                    stop.set()
                    ticker.join(2)
            assert not ticker.is_alive(), "the owned heartbeat did not stop"
            assert bytes(copied.cpu().tolist()) == DATA, "consumer B read released or reused pages"
            consumer_proof.update(copied_bytes=len(DATA), copied_sha256=hashlib.sha256(DATA).hexdigest())
            del copied
        elif case == "pending-ptds-consumer":
            # Equal raw PTDS handles on distinct live OS threads denote different streams.
            # Transfer ownership through a holder, never Thread args/a captured WeightSet.
            handoff = [weights]
            del weights
            bound, release_bind = threading.Event(), threading.Event()
            bind_proof, bind_errors = {}, []

            def bind_ptds():
                source = item = bind_stream = None
                try:
                    bind_stream = torch.cuda.ExternalStream(2, device=0)
                    assert bind_stream.cuda_stream == 2, "Torch did not preserve PTDS handle 2"
                    with torch.cuda.stream(bind_stream):
                        assert torch.cuda.current_stream(0).cuda_stream == 2
                        source = handoff.pop()
                        item = source.bind(0, 2)
                        # This fixture proves final release, not pending fill readiness.
                        bind_stream.synchronize()
                        source.close()
                        del source
                        handoff.append(item)
                        del item
                    bind_proof.update(thread=threading.get_ident(), stream=2,
                                      source_handles_released=True)
                    bound.set()
                    assert release_bind.wait(WAIT), "the PTDS consumer never released bind thread A"
                except BaseException:
                    bind_errors.append(traceback.format_exc())
                finally:
                    try:
                        if bind_stream is not None:
                            bind_stream.synchronize()
                    except BaseException:
                        bind_errors.append(traceback.format_exc())
                    source = item = None
                    bound.set()

            binder = threading.Thread(target=bind_ptds, daemon=True)
            binder.start()
            consumer = ticker = None
            stop = threading.Event()
            try:
                assert bound.wait(WAIT), "PTDS bind thread A never handed over its Pin"
                assert not bind_errors, bind_errors
                assert binder.is_alive() and bind_proof["source_handles_released"]
                assert bind_proof["thread"] != threading.get_ident()
                pin = handoff.pop()
                assert not handoff, "the handoff still retains a source owner"
                consumer = torch.cuda.ExternalStream(2, device=0)
                assert consumer.cuda_stream == 2, "Torch substituted another stream for PTDS B"
                with torch.cuda.stream(consumer):
                    assert torch.cuda.current_stream(0).cuda_stream == 2
                    tensor = torch.from_dlpack(pin)
                    assert tensor.is_cuda and tensor.device.index == 0
                    assert tensor.dtype == torch.uint8 and tensor.numel() == len(DATA)
                    assert tensor.data_ptr() == pin.ptr
                    copied = torch.empty_like(tensor)
                    assert copied.data_ptr() != tensor.data_ptr()
                held_free = gpu.stats().driver_free
                foreign = weakref.ref(tensor)
                heartbeat = [0]

                def beat_ptds():
                    while not stop.wait(0.001):
                        heartbeat[0] += 1

                ticker = threading.Thread(target=beat_ptds, daemon=True)
                ticker.start()
                end = torch.cuda.Event()
                with torch.cuda.stream(consumer):
                    torch.cuda._sleep(2_000_000_000)
                    copied.copy_(tensor, non_blocking=True)
                    end.record()
                assert not end.query(), "PTDS consumer B needs actual pending work"
                del provider, pin
                ticks, started = heartbeat[0], time.monotonic()
                del tensor
                elapsed = time.monotonic() - started
                gc.collect()
                assert foreign() is None, "a Python tensor masks PTDS final release"
                # Observe only Python owners and the event until B has been rescued.
                # Stats/trim/recovery or host reads here could synchronize the context.
                time.sleep(0.1)
                assert not end.query(), "PTDS B finished before the retention observation"
                assert binder.is_alive(), "bind thread A exited before the PTDS observation"
                assert alive() is not None, "pending PTDS B lost its source/provider"
                assert elapsed < 0.25, "PTDS final consumer release waited for CUDA"
                assert heartbeat[0] - ticks >= 3, "PTDS release stopped Python's heartbeat"
                consumer_proof = {
                    "scope": "final-release-custody",
                    "bind_stream": bind_proof["stream"],
                    "consumer_stream": consumer.cuda_stream,
                    "bind_thread": bind_proof["thread"],
                    "consumer_thread": threading.get_ident(),
                    "distinct_threads": True,
                    "bind_thread_alive_while_pending": True,
                    "bind_thread_source_handles_released": True,
                    "ptds": True,
                    "fill_ready_before_consume": True,
                    "pending_readiness_device_proof": False,
                    "pending_after_drop": True,
                    "source_retained_while_pending": True,
                    "foreign_python_owner_collected": True,
                    "drop_seconds": elapsed,
                    "heartbeat_ticks": heartbeat[0] - ticks,
                }
            finally:
                try:
                    # PTDS must be rescued on this owning OS thread, not the deleter's.
                    if consumer is not None:
                        consumer.synchronize()
                finally:
                    release_bind.set()
                    binder.join(WAIT)
                    stop.set()
                    if ticker is not None:
                        ticker.join(2)
            assert not binder.is_alive(), "the owned PTDS bind thread did not stop"
            assert not bind_errors, bind_errors
            assert ticker is not None and not ticker.is_alive()
            assert bytes(copied.cpu().tolist()) == DATA, "PTDS B read released or reused pages"
            consumer_proof.update(copied_bytes=len(DATA), copied_sha256=hashlib.sha256(DATA).hexdigest())
            del copied
        else:
            pin = weights.bind(0, stream.cuda_stream)
            tensor = values(torch, pin)
            assert observed["chunks"] > 0
            held_free = gpu.stats().driver_free
            weights.close()
            del weights, provider
            if case == "last-pin":
                del tensor
                holder = [pin]
                del pin
            else:
                del pin
                holder = [tensor]
                del tensor
            gc.collect()
            assert alive() is not None, (
                "a retained pin/consumer lost its source context"
            )
            heartbeat, stop = [0], threading.Event()

            def beat():
                while not stop.wait(0.001):
                    heartbeat[0] += 1

            ticker = threading.Thread(target=beat, daemon=True)
            ticker.start()
            end = torch.cuda.Event()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(2_000_000_000)
                end.record()
            assert not end.query(), "the final release gate needs pending GPU work"
            ticks, started = heartbeat[0], time.monotonic()
            holder.clear()
            elapsed = time.monotonic() - started
            time.sleep(0.03)
            assert elapsed < 0.25, "last pin/consumer release waited for CUDA work"
            assert heartbeat[0] - ticks >= 3, (
                "native release stopped Python's heartbeat"
            )
            stop.set()
            ticker.join(2)
            assert not ticker.is_alive()
            end.synchronize()

        released = cleanup(gpu, alive, held_free, mapping_bytes)
        copies = varena.copies()
        h2d = [copy for copy in copies if copy.kind == "h2d" and copy.gpu == 0]
        assert any(copy.bytes == len(DATA) and copy.ns > 0 for copy in h2d), (
            "prefetch never completed its native H2D transfer"
        )
        assert observed["chunks"] > 0
        assert any(
            method == "GET"
            and chunk_digests.intersection(urlsplit(path).path.split("/"))
            for method, path in requests
        ), "no real chunk HTTP request reached the endpoint"
        with tensorfs.open(
            header, local_files_only=True, cache_dir=root / "reader"
        ) as cached:
            assert not cached.missing()
            assert cached.components["model"].tensors.read("weight").tobytes() == DATA
        emit(
            case=case,
            result="passed",
            callback=observed,
            cleanup=released,
            **({"consumer": consumer_proof} if consumer_proof else {}),
            h2d=[
                {
                    "bytes": copy.bytes,
                    "ns": copy.ns,
                    "staged": copy.staged,
                    "busy_ns": copy.busy_ns,
                    "ceiling": copy.ceiling,
                    "link": copy.link,
                }
                for copy in h2d
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="strict seconds per isolated child (20..120)",
    )
    parser.add_argument("--child-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not __debug__:
        parser.error("this proof requires Python assertions; do not use -O")
    if not 20 <= args.timeout <= 120:
        parser.error("--timeout must be between 20 and 120 seconds")
    if args.child_case:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.timeout - 5)
        try:
            run_case(args.child_case)
        finally:
            faulthandler.cancel_dump_traceback_later()
        return
    for case in CASES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-case",
            case,
            "--timeout",
            str(args.timeout),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            # subprocess.run kills and reaps only this owned child, even for a total GIL deadlock.
            raise AssertionError(
                f"{case}: exceeded the outer {args.timeout}s timeout; output={error.stdout!r}; stderr={error.stderr!r}"
            ) from error
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        assert result.returncode == 0, f"{case}: child exited {result.returncode}"
        report = json.loads(result.stdout.rstrip().rsplit("\n", 1)[-1])
        assert report.get("case") == case and report.get("result") == "passed", (
            f"{case}: child exited without its completed proof receipt"
        )
    emit(result="passed", cases=list(CASES), timeout_per_child=args.timeout)


if __name__ == "__main__":
    main()
