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
    del retained, another
finally:
    owned.clear()
    cache.close()
    memory.close()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
assert torch.cuda.memory_allocated(device) == baseline
print("CUDA cache accounting: owned growth, real leaks, symmetric shrink")
