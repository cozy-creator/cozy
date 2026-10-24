"""The tensorfs suite's probe, run on the rented machine: the disk the machine's software and store live on, written and
then read cold (the varena suite's ceiling method), the ceiling the suite's downloads are reported beside
(hardware-ceiling-bars). It never dials: an executor is handed files, never an address (cr-012), so the network side
of the suite runs on the gate's host (straggler.py)."""

import os
import sys
import tempfile
import time

import msgspec

from cozy_runtime.author import App, Telemetry

app = App()
GIB = 1 << 30


class Probe(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Measured(msgspec.Struct):
    disk_write_bytes_per_s: int
    disk_read_bytes_per_s: int


@app.entrypoint(accelerator=False)
def measure(payload: Probe, tel: Telemetry) -> Measured:
    total, chunk = 2 * GIB, 16 << 20
    block = os.urandom(chunk)
    fd, path = tempfile.mkstemp(dir=sys.prefix)
    try:
        with tel.stage("writing", overall_range=(0.0, 0.5)):
            start = time.perf_counter()
            for i in range(total // chunk):
                os.write(fd, block)
                tel.progress((i + 1) * chunk / total)
            os.fsync(fd)
            write = total / (time.perf_counter() - start)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.lseek(fd, 0, os.SEEK_SET)
        with tel.stage("reading cold", overall_range=(0.5, 1.0)):
            start, done = time.perf_counter(), 0
            while n := len(os.read(fd, chunk)):
                done += n
                tel.progress(done / total)
            read = total / (time.perf_counter() - start)
        return Measured(disk_write_bytes_per_s=round(write), disk_read_bytes_per_s=round(read))
    finally:
        os.close(fd)
        os.unlink(path)
