# /// script
# requires-python = ">=3.12"
# [tool.cozy]
# accelerator = false
# ///
"""Disk-cold weights for a cold-load cell: every object of this machine's TensorFS store leaves the page cache
(POSIX_FADV_DONTNEED drops clean pages, no privilege needed; each file's fdatasync and a second pass take what
writeback held), then each file's residency is counted with mincore. Returns the store, its files and bytes, and the
bytes resident before and after: a cell asserts `resident_after` is 0 before its run."""
import ctypes
import json
import mmap
import os
from pathlib import Path

LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PAGE = mmap.PAGESIZE


def store() -> Path:
    for root in (os.environ.get("COZY_TENSORFS_ROOT"), "/var/lib/tensorfs", str(Path.home() / ".cozy/var/lib/tensorfs")):
        if root and (Path(root) / "blobs").is_dir():
            return Path(root)
    raise RuntimeError("no TensorFS store with blobs/ found")


def resident(path: Path, length: int) -> int:
    """The bytes of `path` the page cache holds (mincore over a read-only mapping)."""
    if length == 0:
        return 0
    with open(path, "rb") as f:
        address = LIBC.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, f.fileno(), 0)
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            pages = (length + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(pages)
            if LIBC.mincore(address, length, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {path}")
            return sum(b & 1 for b in vec.raw) * PAGE
        finally:
            LIBC.munmap(address, length)


def drop(path: Path, sync: bool) -> None:
    with open(path, "rb") as f:
        if sync:
            os.fdatasync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main(ctx) -> str:
    root = store()
    files = [(p, p.stat().st_size) for p in (root / "blobs").rglob("*") if p.is_file()]
    before = sum(resident(p, n) for p, n in files)
    for sync in (False, True):
        for p, _ in files:
            drop(p, sync)
    after = sum(resident(p, n) for p, n in files)
    total = sum(n for _, n in files)
    return json.dumps({"store": str(root), "files": len(files), "bytes": total, "resident_before": before, "resident_after": after,
                       "resident_after_pct": round(100 * after / total, 3) if total else 0.0})
