#!/usr/bin/env python3
"""Real-driver regression: a native free returns while its consumer still runs, and its
physical bytes remain in the adapter's reserved count until completion. No mocked Torch."""
import ctypes
import json
import time

import tensorfs
import torch
from tensorfs import _ext
from tensorfs.varena import allocator_stats
from tensorfs.varena.torch import Memory

mem = Memory.open(0, limit=2 << 30)
torch.ones(1, device=mem.device).cpu()
print(json.dumps({"torch": torch.__version__, "tensorfs": tensorfs.__version__, "context_measurable": mem.native.context_measurable()}), flush=True)
lib = ctypes.CDLL(_ext.__file__)
allocate, release = lib.varena_torch_alloc, lib.varena_torch_free
allocate.argtypes, allocate.restype = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p], ctypes.c_void_p
release.argtypes, release.restype = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p], None
cuda = ctypes.CDLL("libcuda.so.1")
memset = cuda.cuMemsetD8Async
memset.argtypes, memset.restype = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p], ctypes.c_int
stream, completed = torch.cuda.Stream(), torch.cuda.Event()
size = 16 << 20
with mem.request("native-retirement"), torch.cuda.stream(stream):
    ptr = allocate(size, 0, stream.cuda_stream)
    assert ptr, "native allocation must succeed"
    torch.cuda._sleep(500_000_000)
    assert memset(ptr, 19, size, stream.cuda_stream) == 0
    completed.record(stream)
    assert not completed.query(), "consumer must still be running at the release"
    started = time.perf_counter()
    release(ptr, size, 0, stream.cuda_stream)
    elapsed = time.perf_counter() - started
    physical = allocator_stats(0)[0]
    actual, recorded = torch.cuda.memory_reserved(0), mem.torch.now()[1]
    outstanding = not completed.query()
    print(json.dumps({"free_callback_ms": elapsed * 1000, "consumer_pending": outstanding, "native_physical": physical, "torch_reserved": actual, "adapter_reserved": recorded, "retiring_bytes": size}), flush=True)
    try:
        assert outstanding, "free callback must not wait for the consumer"
        assert physical >= size, "the driver still holds the retiring segment"
        assert recorded >= actual + size, "logical Torch free must not remove physically retiring bytes from accounting"
    finally:
        stream.synchronize()
        deadline = time.monotonic() + 10
        while allocator_stats(0)[0] >= size and time.monotonic() < deadline:
            time.sleep(0.01)
assert allocator_stats(0)[0] < size, "native worker must retire completed storage"
print("PASS native free is asynchronous and physical accounting follows completion", flush=True)
