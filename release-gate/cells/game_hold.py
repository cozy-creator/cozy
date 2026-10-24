# /// script
# requires-python = ">=3.12"
# ///
"""A game holding the machine's GPU (game.py's raw driver allocations): it takes 256 MiB blocks
until the driver's free bytes are at most LEAVE (high + 1 GiB), holds them HOLD seconds reading free
every 100 ms, then takes THEN more, block by block. Returns JSON: the lowest free seen while holding
and each later block's wait (s)."""
import ctypes
import json
import time

BLOCK = 256 << 20
LEAVE, HOLD, THEN, EXTRA, GIVE_UP = 2560 << 20, 300.0, 1 << 30, 4 << 30, 120.0


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


def main(ctx) -> str:
    cu = driver()
    held: list[int] = []
    out: dict = {"mode": "hold", "free_start": free(cu)}
    while free(cu) - BLOCK > LEAVE:
        take(cu, held, GIVE_UP)
    out["held"] = len(held) * BLOCK
    lowest, until = free(cu), time.monotonic() + HOLD
    while time.monotonic() < until:
        lowest = min(lowest, free(cu))
        time.sleep(0.1)
    out["free_lowest_while_holding"] = lowest
    waits = [take(cu, held, GIVE_UP) for _ in range(THEN // BLOCK)]
    return report(cu, held, waits, out)


def report(cu, held: list, waits: list, out: dict) -> str:
    out["waits"] = waits
    out["blocks"] = sum(w is not None for w in waits)
    out["gave_up"] = any(w is None for w in waits)
    out["free_end"] = free(cu)
    for p in held:
        cu.cuMemFree_v2(ctypes.c_uint64(p))
    return json.dumps(out)
