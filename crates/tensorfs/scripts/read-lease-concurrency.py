#!/usr/bin/env python3
"""Real lease contention in a child, so a GIL deadlock cannot hang CI."""

import concurrent.futures
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import tensorfs


def readers():
    with tempfile.TemporaryDirectory(prefix="tensorfs-shared-lease-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        body = b"x" * (8 << 20)
        digest = tensorfs.object_id(body)
        source = Path(root) / "body"
        source.write_bytes(body)
        store.put_file(source, digest, len(body))
        lease = store.acquire("sha256:" + "a" * 64, [(digest, len(body))])
        barrier = threading.Barrier(4)

        def read(_):
            target = bytearray(len(body))
            barrier.wait()
            for _ in range(20):
                lease.read_into(digest, len(body), 0, len(body), target)
            assert target == body

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(read, range(4)))
        assert lease.live
        lease.release()
        assert not lease.live
        try:
            lease.read_into(digest, len(body), 0, len(body), bytearray(len(body)))
        except tensorfs.errors.LeaseRevoked:
            pass
        else:
            raise AssertionError("released lease served bytes")


def main():
    if sys.argv[1:] == ["--child"]:
        readers()
        return
    try:
        result = subprocess.run([sys.executable, __file__, "--child"],
                                capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise AssertionError("concurrent ReadLease reads froze the interpreter") from error
    assert result.returncode == 0, (result.stdout, result.stderr)
    print("ReadLease concurrency green: shared reads from four threads, release fencing")


if __name__ == "__main__":
    main()
