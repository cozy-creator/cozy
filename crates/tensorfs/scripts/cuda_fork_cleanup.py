#!/usr/bin/env python3
"""A CPU-only TensorFS import, then a real fork: child CUDA allocations must be released.

Run on a Linux GPU with only the installed TensorFS wheel. No CUDA/Torch initialization
happens in the parent, and no private native test API is exposed for the memory oracle.
"""

import ctypes
import gc
import json
import os
import select
import signal
import sys
import tempfile
import time
import traceback

import tensorfs

assert "torch" not in sys.modules and "triton" not in sys.modules
assert hasattr(os, "fork"), "this gate requires a real CPU-parent fork"


def child():
    # Independent test-only driver observation, after the fork. TensorFS's production
    # allocations and copies still use its sole native driver and primary context.
    driver = ctypes.CDLL("libcuda.so.1")

    def checked(name, *args):
        status = getattr(driver, name)(*args)
        assert status == 0, f"{name}: CUDA {status}"

    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    driver.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    release = "cuDevicePrimaryCtxRelease_v2" if hasattr(driver, "cuDevicePrimaryCtxRelease_v2") else "cuDevicePrimaryCtxRelease"
    getattr(driver, release).argtypes = [ctypes.c_int]
    checked("cuInit", 0)
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    checked("cuDeviceGet", ctypes.byref(device), 0)
    checked("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
    checked("cuCtxSetCurrent", context)

    def free_bytes():
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        checked("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        return free.value

    try:
        with tempfile.TemporaryDirectory() as temporary:
            size = (32 << 20) + 17
            pattern = (b"cozy-fork-bytes!" * (size // 16 + 1))[:size]
            assert len(pattern) == size
            writer = tensorfs.Writer(cache_dir=temporary)
            component = writer.component("model", cls="test.Fork", config={})
            component.put("warm", b"w", dtype="U8", shape=[1])
            component.put("value", pattern, dtype="U8", shape=[size])
            checkpoint = writer.commit()
            tensors = checkpoint.components["model"].tensors
            warm = tensors.read_all(["warm"], device="cuda:0")["warm"]
            assert warm.tobytes() == b"w"
            before = free_bytes()
            array = tensors.read_all(["value"], device="cuda:0")["value"]
            assert array.tobytes() == pattern
            allocated = free_bytes()
            assert before - allocated >= size, "the oracle did not observe the native allocation"
            capsule = array.__dlpack__()
            del array
            gc.collect()
            assert free_bytes() <= before - size, "the capsule did not retain its allocation"
            del capsule
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                gc.collect()
                recovered = free_bytes()
                if recovered >= allocated + size:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("the child inherited an orphaned cleanup queue: CUDA bytes never released")
            checkpoint.close()
            return {"allocated_bytes": size, "recovered_bytes": recovered - allocated}
    finally:
        checked("cuCtxSetCurrent", None)
        checked(release, device)


read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    try:
        result = {"ok": True, **child()}
    except BaseException:
        result = {"ok": False, "error": traceback.format_exc()}
    os.write(write_fd, json.dumps(result).encode())
    os.close(write_fd)
    os._exit(0 if result["ok"] else 1)
os.close(write_fd)
chunks = []
status = None
deadline = time.monotonic() + 60
try:
    os.set_blocking(read_fd, False)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
            raise TimeoutError("the owned CUDA child did not finish its cleanup proof within 60 seconds")
        data = os.read(read_fd, 8192)
        if not data:
            break
        chunks.append(data)
    while status is None:
        ended, result = os.waitpid(pid, os.WNOHANG)
        if ended == pid:
            status = result
        elif time.monotonic() >= deadline:
            raise TimeoutError("the owned CUDA child wrote its proof but did not exit")
        else:
            time.sleep(0.01)
finally:
    os.close(read_fd)
    if status is None:
        # This exact PID is the child created above; never signal another process. Reap
        # it before this parent exits. The paid-worker wrapper also bounds the process
        # group in case a broken kernel/driver cannot complete SIGKILL teardown.
        try:
            ended, _ = os.waitpid(pid, os.WNOHANG)
            if ended == 0:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
        except ChildProcessError:
            pass
result = json.loads(b"".join(chunks))
assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and result["ok"], result
print(f"CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered {result['recovered_bytes']} bytes")
