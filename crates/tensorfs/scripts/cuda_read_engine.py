#!/usr/bin/env python3
"""Real CUDA Array/DLPack lifetime gate against an installed AOT wheel.

Run explicitly on an allocated GPU with NumPy and a CUDA PyTorch consumer installed.
TensorFS itself never imports either framework.
"""

import gc
import sys

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
    first = torch.from_dlpack(arrays["plain"])
    second = torch.from_dlpack(arrays["plain"])
    copied = torch.from_dlpack(arrays["plain"].__dlpack__(copy=True))
    assert first.is_cuda and first.data_ptr() == second.data_ptr()
    assert first.data_ptr() != copied.data_ptr()
    expected = torch.tensor(HostReads.plain, device="cuda:0")
    # Consumers on a separate stream see a completed read and retain native allocations
    # after the parent checkpoint, cached component map and Array wrappers are released.
    stream = torch.cuda.Stream()
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
