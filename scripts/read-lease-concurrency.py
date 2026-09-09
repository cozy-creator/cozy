#!/usr/bin/env python3
"""Exercise real lease contention in children so a GIL deadlock cannot hang CI."""

import concurrent.futures
import hashlib
import io
import math
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import tensorfs

ARMS = ("readers", "hold_id", "manifest", "bytes", "objects", "live", "recheck",
        "read_into", "read_asset", "read_part_into", "stream", "release", "exit", "callback_error")


def readers():
    with tempfile.TemporaryDirectory(prefix="tensorfs-shared-lease-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        body = b"x" * (8 << 20)
        digest = tensorfs.object_id(body)
        store.put_reader(io.BytesIO(body), digest, len(body))
        with store.acquire("sha256:" + "a" * 64, [(digest, len(body))]) as lease:
            barrier = threading.Barrier(4)

            def read(_):
                target = bytearray(len(body))
                barrier.wait()
                for _ in range(20):
                    lease.read_into(digest, len(body), 0, len(body), target)
                assert target == body

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(read, range(4)))


def contention(arm, root, manifest):
    store = tensorfs.Store.open(root)
    raw = store.manifest(manifest)["header"]
    header = tensorfs.parse_header(raw)
    order = [(component, key) for component, rows in header["components"].items() for key in rows]
    plan = tensorfs.plan(raw, order, window=1 << 16)
    entered, competing = threading.Event(), threading.Event()
    lease = store.acquire_cozytensors(manifest)
    getters = {name: getattr(lease, name) for name in ("hold_id", "manifest", "bytes", "objects", "live")}
    object_id, length = max(lease.objects, key=lambda item: item[1])
    expected = bytearray(length)
    lease.read_into(object_id, length, 0, length, expected)
    asset_name = next(iter(header["assets"]))
    asset = lease.read_asset(raw, asset_name)
    assert hashlib.sha256(asset).hexdigest() == header["assets"][asset_name]["logical_sha256"]
    component, key = order[0]
    role, part = next(iter(header["components"][component][key]["parts"].items()))
    part_bytes = min(17, math.prod(part["shape"]) * tensorfs.dtypes()[part["dtype"]])
    expected_part = bytearray(part_bytes)
    lease.read_part_into(raw, component, key, role, 0, expected_part)

    def stream(block=False):
        slots = [bytearray(1 << 18) for _ in range(2)]
        digest = hashlib.sha256()
        first = True

        def receive(batch):
            nonlocal first
            if block and first:
                first = False
                entered.set()
                assert competing.wait(2), "contending call never started"
                # The stream owns the native lease mutex. A contending attached
                # caller must let this callback reacquire the GIL after the sleep.
                time.sleep(0.05)
            digest.update(slots[batch.slot][:batch.nbytes])
            batch.release()
            if block and arm == "callback_error":
                raise ValueError("consumer failed")

        stats = lease.stream(plan, slots, receive, readers=2)
        return digest.digest(), stats["bytes"], stats["items"]

    baseline = stream()

    def compete():
        assert entered.wait(2), "stream callback never started"
        competing.set()
        if arm in getters:
            assert getattr(lease, arm) == getters[arm]
        elif arm == "recheck":
            lease.recheck()
        elif arm in ("read_into", "callback_error"):
            target = bytearray(length)
            lease.read_into(object_id, length, 0, length, target)
            assert target == expected
        elif arm == "read_asset":
            assert lease.read_asset(raw, asset_name) == asset
        elif arm == "read_part_into":
            target = bytearray(part_bytes)
            lease.read_part_into(raw, component, key, role, 0, target)
            assert target == expected_part
        elif arm == "stream":
            assert stream() == baseline
        elif arm == "release":
            lease.release()
        elif arm == "exit":
            assert lease.__exit__(None, None, None) is False
        else:
            raise AssertionError(arm)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(stream, True)
        second = pool.submit(compete)
        if arm == "callback_error":
            try:
                first.result()
            except ValueError as error:
                assert str(error) == "consumer failed"
            else:
                raise AssertionError("callback exception was lost")
        else:
            assert first.result() == baseline
        second.result()
    if arm not in ("release", "exit"):
        assert lease.live
        lease.release()
    assert not lease.live
    try:
        lease.read_into(object_id, length, 0, length, bytearray(length))
    except tensorfs.errors.LeaseRevoked:
        pass
    else:
        raise AssertionError("released lease served bytes")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--arm":
        arm, root, manifest = sys.argv[2:]
        if arm == "readers":
            readers()
        else:
            contention(arm, root, manifest)
        print("PASS", arm, flush=True)
        return
    root, manifest = sys.argv[1:]
    for arm in ARMS:
        try:
            result = subprocess.run([sys.executable, __file__, "--arm", arm, root, manifest],
                                    capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f"ReadLease {arm} froze the interpreter") from error
        assert result.returncode == 0, (arm, result.stdout, result.stderr)
        print(result.stdout.strip(), flush=True)
    print("ReadLease concurrency green: shared reads, every lock entry point, stream bytes/order, "
          "callback exception, and release fencing")


if __name__ == "__main__":
    main()
