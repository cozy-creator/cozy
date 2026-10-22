#!/usr/bin/env python3
"""The driver's rule behind varena's Keep: cuMemSetAccess on a mapped VMM page while kernels
read it. Two 64 MiB pages, mapped read-write; each round queues reads of page 0 (sums over it),
then protects a page read-only before the reads finish, then synchronizes. Each case runs in its
own process (a sticky error ends its context):
  other:  page 1 is protected while page 0 is read (control: expected clean);
  same:   page 0 is protected while it is read (varena's Keep racing a bound block);
  export: as same, then exported as a POSIX handle, as Keep does.

    python scripts/varena-protect-probe.py
GPU 0 only.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

import torch

ROUNDS = 200
READS = 40  # queued per round: the protect lands while they run
PROT_READ, READWRITE = 1, 3


class Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class Flags(ctypes.Structure):
    _fields_ = [("compression", ctypes.c_ubyte), ("rdma", ctypes.c_ubyte), ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handles", ctypes.c_int), ("location", Location), ("win32", ctypes.c_void_p), ("flags", Flags)]


class Access(ctypes.Structure):
    _fields_ = [("location", Location), ("flags", ctypes.c_int)]


class Buffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes // 4,), "typestr": "<f4", "data": (ptr, False), "version": 3}


def check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with CUDA {rc}")


def case(which: str) -> int:
    d = ctypes.CDLL("libcuda.so.1")
    torch.zeros(1, device="cuda")
    prop = Prop(type=1, handles=1, location=Location(type=1, id=0))  # POSIX fd handles, as varena's pages
    gran = ctypes.c_size_t()
    check(d.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0), "granularity")
    size = max(gran.value, 64 << 20) // gran.value * gran.value
    va = ctypes.c_uint64()
    check(d.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(2 * size), ctypes.c_size_t(0), ctypes.c_uint64(0), ctypes.c_ulonglong(0)), "reserve")
    pages = [ctypes.c_uint64(va.value + n * size) for n in range(2)]
    handles = []
    for p in pages:
        handle = ctypes.c_ulonglong()
        check(d.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0)), "create")
        check(d.cuMemMap(p, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0)), "map")
        handles.append(handle)

    def access(page: ctypes.c_uint64, flags: int) -> None:
        desc = Access(location=Location(type=1, id=0), flags=flags)
        check(d.cuMemSetAccess(page, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1)), "set access")

    for p in pages:
        access(p, READWRITE)
    read = torch.as_tensor(Buffer(pages[0].value, size), device="cuda")
    read.fill_(1.0)
    torch.cuda.synchronize()
    target = pages[1] if which == "other" else pages[0]
    began = time.perf_counter()
    for n in range(ROUNDS):
        access(target, READWRITE)
        torch.cuda.synchronize()
        for _ in range(READS):
            read.sum()
        access(target, PROT_READ)  # while the reads queued above run
        if which == "export":
            fd = ctypes.c_int()
            check(d.cuMemExportToShareableHandle(ctypes.byref(fd), handles[0], ctypes.c_int(1), ctypes.c_ulonglong(0)), "export")
            os.close(fd.value)
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - the sticky error is the result
            print(f"PROBE {which}: round {n}: {str(e).splitlines()[0]}", flush=True)
            return 1
    print(f"PROBE {which}: {ROUNDS} rounds clean in {time.perf_counter() - began:.1f} s", flush=True)
    return 0


def main() -> int:
    results = {}
    for which in ("other", "same", "export"):
        done = subprocess.run([sys.executable, "-u", __file__, which], capture_output=True, text=True)
        print(done.stdout + done.stderr[-2000:], end="", flush=True)
        results[which] = done.returncode
    print(f"PROBE verdict: protecting another page {'faults' if results['other'] else 'is clean'}; protecting the page in use "
          f"{'faults' if results['same'] else 'is clean'}; with its export {'faults' if results['export'] else 'is clean'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(case(sys.argv[1]) if len(sys.argv) > 1 else main())
