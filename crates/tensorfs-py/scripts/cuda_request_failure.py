"""Actual failed-request slack release, preserving warm weights and owned cache data."""

import json
import tempfile
from pathlib import Path

import tensorfs
import torch
from tensorfs.varena.torch import Memory

assert torch.cuda.device_count() == 1
device = torch.device("cuda", 0)
# Framework context belongs to the serving thread, before the request baseline.
with torch.cuda.device(device):
    torch.cuda.current_blas_handle()


class PayloadFailed(Exception):
    pass


with tempfile.TemporaryDirectory(prefix="request-failure-") as folder:
    writer = tensorfs.Writer(cache_dir=Path(folder))
    writer.component("net", cls=None, config={}).put("weight", torch.ones(4, 4))
    checkpoint = writer.commit()
    with torch.device("meta"):
        module = torch.nn.Linear(4, 4, bias=False)
    memory = Memory.open(device)
    cache = None
    owned: list[torch.Tensor] = []

    def shrink(_need: int) -> int:
        before = torch.cuda.memory_allocated(device)
        owned.clear()
        return max(before - torch.cuda.memory_allocated(device), 0)

    try:
        model = memory.load(module, ck=checkpoint, component="net", level="gpu")
        value = torch.ones(1, 4, device=device)
        with torch.no_grad(), memory.request("warm", plan=[model]):
            assert torch.equal(module(value), torch.full((1, 4), 4.0, device=device))
        resident = memory.resident()
        assert sum(resident.values()) > 0
        cache = memory.cache("surviving-owned-data", shrink)
        before = torch.cuda.memory_allocated(device)
        owned.append(torch.full((1 << 20,), 7, dtype=torch.uint8, device=device))
        cache.nbytes = torch.cuda.memory_allocated(device) - before
        assert cache.nbytes >= 1 << 20
        results = []
        for kind in ("returned", "escaping", "success"):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_reserved(device)
            peak = baseline
            try:
                with memory.request(f"{kind}-slack", plan=[model]) as request:
                    temporary = torch.empty(64 << 20, dtype=torch.uint8, device=device)
                    temporary.fill_(3)
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.memory_reserved(device)
                    del temporary
                    if kind == "returned":
                        # On the old source this remains an actual allocator RED,
                        # rather than failing only because the new API is absent.
                        mark = getattr(request, "mark_failed", None)
                        if mark is not None:
                            mark()
                    elif kind == "escaping":
                        raise PayloadFailed("intentional package failure")
            except PayloadFailed:
                assert kind == "escaping"
            reserved = torch.cuda.memory_reserved(device)
            assert peak - baseline >= 64 << 20
            if kind == "success":
                assert reserved > baseline, "successful requests must retain warm allocator slack"
            else:
                assert reserved <= baseline, ("failed scope kept allocator slack", kind, baseline, reserved)
            assert not request.cleanup_errors
            assert request.stats is not None and request.stats.leftover == ((0, 0),)
            assert not memory.active and not memory.local.requests
            assert memory.resident() == resident, "slack release evicted warm weights"
            assert cache.nbytes >= 1 << 20 and owned[0][-1].item() == 7
            with torch.no_grad(), memory.request(f"healthy-after-{kind}", plan=[model]) as healthy:
                assert torch.equal(module(value), torch.full((1, 4), 4.0, device=device))
            assert healthy.stats is not None and healthy.stats.copied == 0
            results.append({"kind": kind, "before": baseline, "peak": peak, "after": reserved})
        print("PASS actual failed-request slack release with warm weights and cache preserved")
        print(json.dumps({"cases": results, "gpu": str(torch.cuda.get_device_properties(device).uuid)}))
    finally:
        if cache is not None:
            shrink(cache.nbytes)
            cache.close()
        memory.close()
        checkpoint.close()
