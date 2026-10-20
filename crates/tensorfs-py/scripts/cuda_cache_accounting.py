"""Actual CUDA allocations must distinguish managed caches from retained package data."""

import torch
from tensorfs.varena.torch import Memory

assert torch.cuda.device_count() == 1
device = torch.device("cuda", 0)
torch.cuda.synchronize(device)
baseline = torch.cuda.memory_allocated(device)
memory = Memory.open(device)
owned: list[torch.Tensor] = []


def shrink(_need: int) -> int:
    before = torch.cuda.memory_allocated(device)
    owned.clear()
    torch.cuda.synchronize(device)
    return max(before - torch.cuda.memory_allocated(device), 0)


cache = memory.cache("accounted-speed-cache", shrink)
try:
    with memory.request("owned-cache-growth") as first:
        before = torch.cuda.memory_allocated(device)
        owned.append(torch.empty(1 << 20, dtype=torch.uint8, device=device))
        cache.nbytes = torch.cuda.memory_allocated(device) - before
        assert cache.nbytes >= 1 << 20
    assert first.stats is not None and first.stats.leftover == ((0, 0),)

    with memory.request("unowned-tensor-growth") as second:
        before = torch.cuda.memory_allocated(device)
        retained = torch.empty(1 << 20, dtype=torch.uint8, device=device)
        leaked = torch.cuda.memory_allocated(device) - before
    assert leaked >= 1 << 20
    assert second.stats is not None and second.stats.leftover == ((0, leaked),)

    with memory.request("cache-shrink-does-not-hide-new-leak") as third:
        shrink(cache.nbytes)
        cache.nbytes = 0
        before = torch.cuda.memory_allocated(device)
        another = torch.empty(1 << 20, dtype=torch.uint8, device=device)
        leaked = torch.cuda.memory_allocated(device) - before
    assert leaked >= 1 << 20
    assert third.stats is not None and third.stats.leftover == ((0, leaked),)

    # A nested scope's released activations must become allocator slack before
    # its still-running caller needs room, without starting another request.
    with memory.request("outer-after-inner-cleanup"):
        torch.cuda.empty_cache()
        memory._measure()
        reusable = memory.available()
        with memory.request("temporary-allocation") as fourth:
            temporary = torch.empty(4 << 20, dtype=torch.uint8, device=device)
            memory._measure()  # A real model measures its allocator while live.
            del temporary
        assert fourth.stats is not None and fourth.stats.leftover == ((0, 0),)
        assert memory.free(device, reusable), (
            "completed scope left released activations charged as pinned scratch"
        )
    del retained, another

    class MeasurementFailed(RuntimeError):
        pass

    def refuse_measurement() -> None:
        raise MeasurementFailed("cleanup measurement refused")

    previous = memory.measure
    try:
        with memory.request("measurement-failure") as failed:
            memory.measure = refuse_measurement
    except MeasurementFailed:
        pass
    else:
        raise AssertionError("measurement failure was hidden")
    finally:
        memory.measure = previous
    assert memory.native is not None
    try:
        memory.native.stats(failed.native_id)
    except Exception as error:
        assert "unknown request" in str(error)
    else:
        raise AssertionError("failed cleanup retained an active native request")
    with memory.request("healthy-after-measurement-failure") as healthy:
        pass
    assert healthy.stats is not None and healthy.stats.leftover == ((0, 0),)
finally:
    owned.clear()
    cache.close()
    memory.close()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
assert torch.cuda.memory_allocated(device) == baseline
print("CUDA cache accounting: owned growth, real leaks, symmetric shrink")
