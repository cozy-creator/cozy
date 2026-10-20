"""Actual CUDA allocations must distinguish managed caches from retained package data."""

import torch
from tensorfs.varena import VarenaError
from tensorfs.varena.torch import Memory
from torch.utils._python_dispatch import _disable_current_modes

assert torch.cuda.device_count() == 1
device = torch.device("cuda", 0)
torch.cuda.synchronize(device)
baseline = torch.cuda.memory_allocated(device)
memory = Memory.open(device)
owned: list[torch.Tensor] = []


class ObservedPlane:
    def __init__(self, plane, available: int, growth: int) -> None:
        self.plane, self.available, self.growth = plane, available, growth
        self.observed: list[int] = []

    def __getattr__(self, name: str):
        return getattr(self.plane, name)

    def relief(self, *args):
        available = self.plane.available()
        self.observed.append(available)
        assert self.available - available == self.growth, (
            "relief saw stale live CUDA allocations",
            self.available, available, self.growth,
        )
        return self.plane.relief(*args)


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
    except VarenaError as error:
        assert "unknown request" in str(error)
    else:
        raise AssertionError("failed cleanup retained an active native request")
    with memory.request("healthy-after-measurement-failure") as healthy:
        pass
    assert healthy.stats is not None and healthy.stats.leftover == ((0, 0),)

    # A changed available budget must reach the real caching allocator before
    # retrying the same allocation. Disable dispatch recovery only for the two
    # allocations so it cannot hide whether free() refreshed the actual limit.
    original_limit = memory.limit_bytes
    try:
        with memory.request("allocator-budget-after-relief") as sixth:
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(device)
            memory.limit_bytes = allocated + (64 << 20)
            memory._measure()
            with _disable_current_modes():
                try:
                    torch.empty(80 << 20, dtype=torch.uint8, device=device)
                except torch.OutOfMemoryError:
                    pass
                else:
                    raise AssertionError("the actual 64 MiB allocator limit was not enforced")
            memory.limit_bytes = allocated + (128 << 20)
            assert memory.free(device, 80 << 20)
            with _disable_current_modes():
                recovered = torch.empty(80 << 20, dtype=torch.uint8, device=device)
                recovered.fill_(7)
                assert recovered[-1].item() == 7
                del recovered
        assert sixth.stats is not None and sixth.stats.leftover == ((0, 0),)
    finally:
        memory.limit_bytes = original_limit
        memory._measure()

    # Real allocations between model boundaries must reach the placement policy
    # before it decides whether relief is necessary. Observe the real plane at
    # that boundary; no allocator, policy decision or allocation is simulated.
    for lowering in (False, True):
        with memory.request(f"current-allocations-before-relief-{lowering}") as current:
            torch.cuda.empty_cache()
            memory._measure()
            available = memory.available()
            reserved = torch.cuda.memory_reserved(device)
            temporary = torch.empty(8 << 20, dtype=torch.uint8, device=device)
            growth = torch.cuda.memory_reserved(device) - reserved
            assert growth >= 8 << 20
            plane = memory.native
            assert plane is not None
            observer = ObservedPlane(plane, available, growth)
            memory.native = observer
            try:
                (memory.lower if lowering else memory.free)(device, 1)
            finally:
                memory.native = plane
                del temporary
            assert len(observer.observed) == 1
        assert current.stats is not None and current.stats.leftover == ((0, 0),)
finally:
    owned.clear()
    cache.close()
    memory.close()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
assert torch.cuda.memory_allocated(device) == baseline
print("CUDA cache accounting: owned growth, real leaks, symmetric shrink")
