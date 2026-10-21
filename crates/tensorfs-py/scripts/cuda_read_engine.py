#!/usr/bin/env python3
"""Real CUDA Array/DLPack lifetime gate against an installed AOT wheel.

Run explicitly on an allocated GPU with NumPy and a CUDA PyTorch consumer installed.
TensorFS itself never imports either framework.
"""

import gc
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import tensorfs

assert "torch" not in sys.modules
import torch

assert torch.cuda.is_available(), "this gate requires an actual CUDA GPU"
tmp = tempfile.TemporaryDirectory()
try:
    writer = tensorfs.Writer(cache_dir=Path(tmp.name) / "store")
    component = writer.component("model", cls=("test", "1.0", "Model"), config={})
    plain = np.arange(12, dtype=np.float32).reshape(3, 4)
    component.put("plain", plain)
    component.put("empty", b"", dtype="U8", shape=[0])
    component.put("big", np.arange((64 << 20) + 257, dtype=np.uint8))
    component.put_encoded(
        "quantized", dtype="F32", shape=[2, 16], scheme="fp8.rowwise",
        parts={"weight": ("F8_E4M3", [2, 16], bytes([0x38]) * 32),
               "scale": ("F32", [2], np.array([0.5, 2.0], dtype=np.float32))},
        params={},
    )
    ck = writer.commit()
    tensors = ck.components["model"].tensors
    arrays = tensors.read(["empty", "plain", "big", "quantized"], device="cuda:0", decode=True)
    for key, array in arrays.items():
        assert isinstance(array, tensorfs.Array)
        assert array.__dlpack_device__() == (2, 0)
        assert array.tobytes() == tensors.read(key, decode=True).tobytes(), key
    encoded = tensors.read(["quantized"], device="cuda", decode=False)["quantized"]
    host_parts = tensors.read("quantized", decode=False).parts
    for role, array in encoded.parts.items():
        assert array.__dlpack_device__() == (2, 0)
        assert array.tobytes() == host_parts[role].tobytes()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        destination = torch.full((3, 4), -1.0, device="cuda:0")
        tensors.read_into("plain", destination)
        odd = torch.empty(9, dtype=torch.uint8, device="cuda:0")
        tensors.read_into("quantized", odd, decode=True, offset=3)
    assert torch.equal(destination, torch.tensor(plain, device="cuda:0"))
    assert bytes(odd.cpu().tolist()) == tensors.read("quantized", decode=True).tobytes()[3:12]
    for consumer_stream in [-1, 1, 2, stream.cuda_stream]:
        try:
            arrays["plain"].__dlpack__(copy=True, stream=consumer_stream)
        except BufferError:
            pass
        else:
            raise AssertionError("a CUDA copy ignored the explicit consumer stream")
    # An unconsumed capsule is destroyed with Python's GIL held. Its final allocation
    # release may synchronize CUDA work, so an independent Python heartbeat must progress.
    free_before, _ = torch.cuda.mem_get_info()
    victim = tensors.read(["big"], device="cuda:0")["big"]
    allocation_size = victim.nbytes
    capsule = victim.__dlpack__()
    del victim
    free_allocated, _ = torch.cuda.mem_get_info()
    assert free_before - free_allocated >= allocation_size
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
    time.sleep(0.03)
    stopped.set()
    thread.join()
    assert elapsed < 0.1, "the capsule deleter waited for CUDA under the GIL"
    assert heartbeat[0] - before >= 3, "Python stopped while native cleanup waited for CUDA"
    end.synchronize()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        recovered, _ = torch.cuda.mem_get_info()
        if recovered >= free_allocated + allocation_size:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("native cleanup never released the capsule allocation")
    first = torch.from_dlpack(arrays["plain"])
    second = torch.from_dlpack(arrays["plain"])
    copied = torch.from_dlpack(arrays["plain"].__dlpack__(copy=True))
    assert first.is_cuda and first.data_ptr() == second.data_ptr()
    assert first.data_ptr() != copied.data_ptr()
    expected = torch.tensor(plain, device="cuda:0")
    expected_quantized = torch.frombuffer(bytearray(tensors.read("quantized", decode=True).tobytes()), dtype=torch.float32).reshape(2, 16)
    # Consumers on a separate stream see a completed read and retain native allocations
    # after the parent checkpoint, cached component map and Array wrappers are released.
    with torch.cuda.stream(stream):
        independent = torch.from_dlpack(arrays["quantized"]).clone()
    ck.close()
    ck = tensors = None
    del arrays, encoded, host_parts, array
    gc.collect()
    stream.synchronize()
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert torch.equal(copied, expected)
    assert torch.equal(independent.cpu(), expected_quantized), "a retained consumer lost its source bytes"
    torch.cuda.synchronize()
finally:
    tmp.cleanup()

print("CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts, DLPack sharing/copy and retained consumers passed")
