"""`Checkpoint.open_file()` on a real folder and store (driven by tests/checkpoint_file.rs).

stdin: one JSON line. The store half then prints `held` with only the file open and `closed`
after closing it, each time waiting for a line back while the parent collects the store.
"""

import io
import json
import random
import sys
from pathlib import Path

import tensorfs

request = json.loads(sys.stdin.readline())
expected = Path(request["expected"]).read_bytes()
starts = request["starts"]


def check(f: tensorfs.CheckpointFile) -> None:
    assert isinstance(f, io.RawIOBase) and f.readable() and f.seekable() and not f.writable()
    assert f.read() == expected
    assert f.tell() == len(expected) and f.read(9) == b""
    for start in starts:
        assert f.seek(start - 3) == start - 3
        assert f.read(7) == expected[start - 3 : start + 4]
    assert f.seek(-5, io.SEEK_END) == len(expected) - 5 and f.read(100) == expected[-5:]
    assert f.seek(10, io.SEEK_END) == len(expected) + 10 and f.read(5) == b""
    rng = random.Random(7)
    pos = f.seek(0)
    for _ in range(400):
        whence = rng.choice([io.SEEK_SET, io.SEEK_CUR, io.SEEK_END])
        offset = rng.randrange(-len(expected) // 2, len(expected) + 64)
        base = {io.SEEK_SET: 0, io.SEEK_CUR: pos, io.SEEK_END: len(expected)}[whence]
        if base + offset < 0:
            try:
                f.seek(offset, whence)
            except (ValueError, tensorfs.InvalidError):
                assert f.tell() == pos
                continue
            raise AssertionError("seek before the start")
        pos = f.seek(offset, whence)
        assert pos == base + offset
        size = rng.randrange(0, 3 * 4096)
        buffer = bytearray(size)
        n = f.readinto(buffer)
        assert bytes(buffer[:n]) == expected[pos : pos + size], (pos, size)
        pos += n
    buffered = io.BufferedReader(f, buffer_size=1000)
    buffered.seek(starts[1] - 10)
    assert buffered.read(5000) == expected[starts[1] - 10 : starts[1] + 4990]


with tensorfs.open(request["folder"], cache_dir=request["cache"], local_files_only=True) as ck:
    check(ck.open_file())

ck = tensorfs.open(request["header"], cache_dir=request["store"], local_files_only=True)
f = ck.open_file()
ck.close()
print("held", flush=True)
sys.stdin.readline()
check(f)
f.close()
assert f.closed
try:
    f.read(1)
except ValueError:
    pass
else:
    raise AssertionError("a closed file read")
print("closed", flush=True)
sys.stdin.readline()
