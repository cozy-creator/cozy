# /// script
# requires-python = ">=3.12"
# [tool.cozy]
# accelerator = true
# ///
"""A game on the machine's GPU: raw driver allocations (cuMemAlloc, never torch's or varena's), so
TensorD sees them only as device memory leaving its books, as it sees a game the user starts.

hold: allocate 256 MiB blocks until the driver's free bytes are at most `leave_mib`, then hold
them `hold_s` seconds, reading free every 100 ms; then take `then_mib` more, block by block.
ramp: allocate free plus `extra_mib`, block by block, retrying a refused block every 50 ms; give
up after `give_up_s` with no block taken.

Returns JSON: every block's wait (s), the ramp's total, and the lowest free seen while holding."""
import ctypes
import json
import time

BLOCK = 256 << 20


def check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"{what}: CUDA error {result}")


def driver():
    cu = ctypes.CDLL("libcuda.so.1")
    check(cu.cuInit(0), "cuInit")
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    check(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return cu


def free(cu) -> int:
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    cu.cuMemGetInfo_v2(ctypes.byref(f), ctypes.byref(t))
    return f.value


def take(cu, held: list, give_up_s: float) -> float | None:
    """One block, retried until the driver gives it; its wait, or None after give_up_s."""
    start = time.monotonic()
    while True:
        p = ctypes.c_uint64()
        if cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(BLOCK)) == 0:
            held.append(p.value)
            return time.monotonic() - start
        if time.monotonic() - start > give_up_s:
            return None
        time.sleep(0.05)


def main(ctx, mode: str = "ramp", extra_mib: int = 4096, leave_mib: int = 2560, hold_s: float = 60, then_mib: int = 1024, give_up_s: float = 120) -> str:
    cu = driver()
    held: list[int] = []
    out: dict = {"mode": mode, "free_start": free(cu)}
    if mode == "hold":
        while free(cu) - BLOCK > leave_mib << 20:
            take(cu, held, give_up_s)
        out["held"] = len(held) * BLOCK
        lowest, until = free(cu), time.monotonic() + hold_s
        while time.monotonic() < until:
            lowest = min(lowest, free(cu))
            time.sleep(0.1)
        out["free_lowest_while_holding"] = lowest
        waits = [take(cu, held, give_up_s) for _ in range((then_mib << 20) // BLOCK)]
    else:
        start = time.monotonic()
        waits = []
        for _ in range((out["free_start"] + (extra_mib << 20)) // BLOCK):
            waits.append(take(cu, held, give_up_s))
            if waits[-1] is None:
                break
        out["ramp_s"] = time.monotonic() - start
    out["waits"] = waits
    out["blocks"] = sum(w is not None for w in waits)
    out["gave_up"] = any(w is None for w in waits)
    out["free_end"] = free(cu)
    for p in held:
        cu.cuMemFree_v2(ctypes.c_uint64(p))
    return json.dumps(out)
