#!/usr/bin/env python3
"""Real CUDA Array/DLPack lifetime gate against an installed AOT wheel.

Run explicitly on an allocated GPU with NumPy and a CUDA PyTorch consumer installed.
TensorFS itself never imports either framework.
"""

import gc
import sys
import threading
import time

import tensorfs

assert "torch" not in sys.modules
import torch

from read_engine import HostReads

assert torch.cuda.is_available(), "this gate requires an actual CUDA GPU"
HostReads.setUpClass()
try:
    tensors = HostReads.tensors
    arrays = tensors.read_all(["empty", "plain", "big", "quantized"], device="cuda:0")
    for key, array in arrays.items():
        assert isinstance(array, tensorfs.Array)
        assert array.__dlpack_device__() == (2, 0)
        assert array.tobytes() == tensors.read(key).tobytes(), key
    encoded = tensors.read_all(["quantized"], device="cuda", decode=False)["quantized"]
    host_parts = tensors.read("quantized", decode=False).parts
    for role, array in encoded.parts.items():
        assert array.__dlpack_device__() == (2, 0)
        assert array.tobytes() == host_parts[role].tobytes()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        destination = torch.full((3, 4), -1.0, device="cuda:0")
        tensors.read_into("plain", destination)
        odd = torch.empty(9, dtype=torch.uint8, device="cuda:0")
        tensors.read_into("quantized", odd, offset=3)
    assert torch.equal(destination, torch.tensor(HostReads.plain, device="cuda:0"))
    assert bytes(odd.cpu().tolist()) == tensors.read("quantized").tobytes()[3:12]
    # An unconsumed capsule is destroyed with Python's GIL held. Its final allocation
    # release may synchronize CUDA work, so an independent Python heartbeat must progress.
    victim = tensors.read_all(["plain"], device="cuda:0")["plain"]
    capsule = victim.__dlpack__()
    del victim
    heartbeat = [0]
    stopped = threading.Event()

    def beat():
        while not stopped.wait(0.001):
            heartbeat[0] += 1

    thread = threading.Thread(target=beat)
    thread.start()
    end = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000)
        end.record()
    assert not end.query(), "the deletion gate needs pending CUDA work"
    before = heartbeat[0]
    started = time.monotonic()
    del capsule
    elapsed = time.monotonic() - started
    stopped.set()
    thread.join()
    assert end.query() and elapsed > 0.02, "cuMemFree did not exercise a synchronizing release"
    assert heartbeat[0] - before >= 3, "the synchronizing DLPack deleter retained the GIL"
    first = torch.from_dlpack(arrays["plain"])
    second = torch.from_dlpack(arrays["plain"])
    copied = torch.from_dlpack(arrays["plain"].__dlpack__(copy=True))
    assert first.is_cuda and first.data_ptr() == second.data_ptr()
    assert first.data_ptr() != copied.data_ptr()
    expected = torch.tensor(HostReads.plain, device="cuda:0")
    # Consumers on a separate stream see a completed read and retain native allocations
    # after the parent checkpoint, cached component map and Array wrappers are released.
    with torch.cuda.stream(stream):
        independent = torch.from_dlpack(arrays["quantized"]).clone()
    HostReads.ck.close()
    HostReads.ck = HostReads.tensors = tensors = None
    del arrays, encoded, host_parts, array
    gc.collect()
    stream.synchronize()
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert torch.equal(copied, expected)
    assert independent.shape == (2, 16)
    torch.cuda.synchronize()
finally:
    HostReads.tearDownClass()

print("CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts, DLPack sharing/copy and retained consumers passed")
