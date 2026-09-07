#!/usr/bin/env python3
"""Real source-backed writer reads must not deadlock readers or terminal operations."""

import concurrent.futures
import io
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import tensorfs

ARMS = ("readers", "commit", "fence")


def exercise(arm):
    with tempfile.TemporaryDirectory(prefix="tensorfs-writer-concurrency-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
        body = b"x" * (8 << 20)
        shape = [len(body)]
        target = {"model": {"drop": [], "add": {
            "weight": {"logical_dtype": "u8", "shape": shape, "encoding": plain,
                       "parts": {"value": {"dtype": "u8", "shape": shape}}},
        }}}
        source = store.begin_derived("sha256:"+"01"*32, 1, {}, target, {},
                                     [("model", "weight")], len(body), work_fingerprint="sha256:"+"11"*32)
        source.add_part("model", "weight", "value", io.BytesIO(body))
        source_ref = source.commit()["manifest"]
        writer = store.begin_derived("sha256:"+"02"*32, 1,
                                    {"source": ("sha256:"+source_ref["sha256"], source_ref["length"])},
                                    {"model": {"source": "source", "source_component": "model",
                                               "drop": [], "add": {}}},
                                    {}, [("model", "weight")], len(body), work_fingerprint="sha256:"+"12"*32)
        barrier = threading.Barrier(4)
        first_read, closing = threading.Event(), threading.Event()

        def read(_):
            destination = bytearray(len(body))
            completed = 0
            barrier.wait()
            for _ in range(20):
                try:
                    writer.source_read_into("source", "model", "weight", "value", 0, destination)
                except tensorfs.errors.Refusal as error:
                    assert closing.is_set(), error
                    assert error.code in ("LEASE_REVOKED", "WRITER_FENCED", "TRANSACTION_CLOSED"), error
                    return completed
                assert destination == body
                completed += 1
                first_read.set()
            return completed

        def close():
            assert first_read.wait(3), "source readers made no progress"
            closing.set()
            if arm == "commit":
                assert writer.commit()["manifest"] == source_ref
            else:
                assert writer.fence()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            readers = [pool.submit(read, i) for i in range(4)]
            terminal = pool.submit(close) if arm != "readers" else None
            counts = [future.result() for future in readers]
            if terminal is not None:
                terminal.result()
        assert sum(counts) > 0
        if arm == "readers":
            assert counts == [20]*4
            writer.fence()
        try:
            writer.source_read_into("source", "model", "weight", "value", 0, bytearray(len(body)))
        except tensorfs.errors.LeaseRevoked:
            pass
        else:
            raise AssertionError("terminal writer served source bytes")
        print(f"PASS {arm}: exact source reads {counts}; terminal source access refused", flush=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--arm":
        exercise(sys.argv[2])
        return
    for arm in ARMS:
        try:
            result = subprocess.run([sys.executable, __file__, "--arm", arm],
                                    capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f"DerivedWriter {arm} froze the interpreter") from error
        assert result.returncode == 0, (arm, result.stdout, result.stderr)
        print(result.stdout.strip(), flush=True)
    print("DerivedWriter concurrency green: shared source reads, commit/fence contention, exact bytes")


if __name__ == "__main__":
    main()
