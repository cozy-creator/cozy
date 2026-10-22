#!/usr/bin/env python3
"""Owned GPU diagnostic: native address reuse and freeing on a contextless CPU thread."""
from __future__ import annotations

import ctypes
import gc
import threading
import time

import torch
from tensorfs import _ext
from tensorfs.varena.torch import Memory

native = ctypes.CDLL(_ext.__file__)
driver = ctypes.CDLL("libcuda.so.1")
native.varena_torch_alloc.restype = ctypes.c_void_p
native.varena_torch_alloc.argtypes = (ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p)
native.varena_torch_free.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
driver.cuCtxSetCurrent.argtypes = (ctypes.c_void_p,)
driver.cuCtxGetCurrent.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
driver.cuMemsetD8Async.argtypes = (ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p)
driver.cuMemcpyDtoH_v2.argtypes = (ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t)


def settle(expected: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        _ext.varena.allocator_check(0)
        if _ext.varena.allocator_stats(0)[0] == expected and _ext.varena.allocator_pending(0) == 0:
            return
        time.sleep(0.001)
    raise AssertionError(("native physical retirement did not finish", _ext.varena.allocator_stats(0), _ext.varena.allocator_pending(0)))


def contextless_pool_drop(memory: Memory) -> None:
    baseline = _ext.varena.allocator_stats(0)[0]
    stream = torch.cuda.Stream()
    pool = memory.new_pool()
    with torch.cuda.stream(stream), memory.request("contextless-free"), memory.allocating(pool):
        value = torch.ones(1 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda._sleep(1 << 30)
        value.add_(1)
    holders = [value, pool]
    del value, pool
    errors: list[BaseException] = []
    restored: list[bool] = []

    def release() -> None:
        try:
            assert driver.cuCtxSetCurrent(None) == 0
            holders.clear()  # last Tensor/independent MemPool references on this CPU thread
            gc.collect()
            context = ctypes.c_void_p()
            assert driver.cuCtxGetCurrent(ctypes.byref(context)) == 0
            restored.append(context.value is None)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=release)
    worker.start()
    worker.join(15)
    assert not worker.is_alive(), "pool destructor waited on GPU completion"
    assert not errors, errors
    assert restored == [True], restored
    _ext.varena.allocator_check(0)
    stream.synchronize()
    settle(baseline)
    print("PASS last Tensor/Pool drop on a fresh contextless thread; allocation context restored and bytes retired", flush=True)


def reused_addresses(memory: Memory) -> None:
    baseline = _ext.varena.allocator_stats(0)[0]
    stream = torch.cuda.Stream()
    pools = [memory.new_pool(), memory.new_pool()]
    alive: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    reused = 0
    sample = (ctypes.c_ubyte * 1)()
    with torch.cuda.stream(stream), memory.request("reused-addresses"):
        for index in range(2048):
            size = (2 + 2 * (index % 3)) << 20
            with memory.allocating(pools[index % 2], route=False):
                ptr = native.varena_torch_alloc(size, 0, stream.cuda_stream)
            assert ptr, (index, _ext.varena.allocator_refusals(0))
            reused += ptr in seen
            seen.add(ptr)
            byte = (index % 251) + 1
            assert driver.cuMemsetD8Async(ptr, byte, size, stream.cuda_stream) == 0
            alive.append((ptr, size, byte))
            if len(alive) > 16:
                old, length, _ = alive.pop(0)
                native.varena_torch_free(old, length, 0, stream.cuda_stream)
            if index % 32 == 31:
                stream.synchronize()
                for current, length, expected in alive:
                    assert driver.cuMemcpyDtoH_v2(sample, current + length - 1, 1) == 0
                    assert sample[0] == expected, (index, current, sample[0], expected)
                _ext.varena.allocator_check(0)
        for ptr, size, _ in alive:
            native.varena_torch_free(ptr, size, 0, stream.cuda_stream)
    stream.synchronize()
    settle(baseline)
    assert reused > 0, "driver did not reuse an address; ownership stress is unqualified"
    print(f"PASS 2048 interleaved allocations/frees across two pool owners; {reused} reused addresses; live/pending returned to baseline", flush=True)


def main() -> None:
    torch.zeros(1, device="cuda")
    memory = Memory.open(0, limit=1 << 30, pool=True)
    contextless_pool_drop(memory)
    reused_addresses(memory)


if __name__ == "__main__":
    main()
