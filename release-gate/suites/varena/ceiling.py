# /// script
# requires-python = ">=3.12"
# [tool.cozy]
# accelerator = true
# ///
"""This machine's load ceilings, measured on it (hardware-ceiling-bars): pinned host-to-device copy bandwidth over four
streams, and a cold sequential read of a fresh 2 GiB file on the disk the machine's software lives on. The varena
suite judges every load and refill as a share of them."""
import json
import os
import sys
import tempfile
import time

import torch

GIB = 1 << 30


def h2d(total: int = 2 * GIB, streams: int = 4) -> float:
    n = total // streams
    src = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for _ in range(streams)]
    dst = [torch.empty(n, dtype=torch.uint8, device="cuda") for _ in range(streams)]
    lanes = [torch.cuda.Stream() for _ in range(streams)]
    best = 0.0
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for lane, a, b in zip(lanes, src, dst):
            with torch.cuda.stream(lane):
                b.copy_(a, non_blocking=True)
        torch.cuda.synchronize()
        best = max(best, total / (time.perf_counter() - start) / 1e9)
    return best


def disk(total: int = 2 * GIB, chunk: int = 16 << 20) -> float:
    block = os.urandom(chunk)
    fd, path = tempfile.mkstemp(dir=sys.prefix)
    try:
        for _ in range(total // chunk):
            os.write(fd, block)
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.lseek(fd, 0, os.SEEK_SET)
        start = time.perf_counter()
        while os.read(fd, chunk):
            pass
        return total / (time.perf_counter() - start) / 1e9
    finally:
        os.close(fd)
        os.unlink(path)


def main(ctx) -> str:
    return json.dumps({"h2d_gbps": round(h2d(), 2), "disk_gbps": round(disk(), 2), "gpu": torch.cuda.get_device_name()})
