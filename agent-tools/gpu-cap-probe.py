# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4,<0.31", "torch==2.14.0"]
# ///
"""Public weightless GPU allocator proof; run only on the real capped machine."""

from __future__ import annotations

import json
import math
import os

from cozy_runtime.author import Outputs, Tree


def main(*, limit_gib: int, out: Outputs) -> Tree:
    import torch

    if limit_gib < 0:
        raise ValueError("limit_gib is positive, or zero for no cap")
    pid = os.getpid()
    device = torch.cuda.current_device()
    cuda_device = torch.device("cuda", device)
    properties = torch.cuda.get_device_properties(device)
    physical = int(properties.total_memory)
    uuid = getattr(properties, "uuid", None)
    if physical <= 0 or uuid is None:
        raise RuntimeError("Torch must report actual physical GPU memory and UUID")
    requested = limit_gib << 30
    effective = min(requested, physical) if requested else physical
    expected_fraction = effective / physical
    fraction = float(torch.cuda.get_per_process_memory_fraction(device))
    # Native budgeting may reserve context/other live allocations below machine authority.
    if not (math.isfinite(fraction) and 0.0 < fraction <= expected_fraction + 1e-8):
        raise RuntimeError(
            f"allocator fraction {fraction} exceeds ceiling {expected_fraction} or is invalid; "
            f"requested={requested}, physical={physical}, pid={pid}"
        )

    oom = None
    attempted = None
    if requested and requested < physical:
        attempted = effective + (64 << 20)
        allocation = None
        try:
            allocation = torch.empty(attempted, dtype=torch.uint8, device=cuda_device)
        except torch.OutOfMemoryError:
            oom = True
        finally:
            del allocation
            torch.cuda.empty_cache()
        if not oom:
            raise RuntimeError(
                f"allocator accepted {attempted} bytes above its {effective}-byte cap"
            )

    tiny = torch.ones(1024, dtype=torch.float32, device=cuda_device).mul_(3)
    result = float(tiny.sum().item())
    torch.cuda.synchronize(device)
    if not tiny.is_cuda or result != 3072.0 or os.getpid() != pid:
        raise RuntimeError(
            "tiny CUDA computation failed to recover in the same process"
        )
    report = {
        "pid": pid,
        "device": device,
        "gpu_uuid": str(uuid),
        "physical_bytes": physical,
        "requested_bytes": requested,
        "effective_limit_bytes": effective,
        "allocator_fraction": fraction,
        "fraction_ceiling": expected_fraction,
        "allocator_budget_bytes": int(fraction * physical),
        "attempted_bytes": attempted,
        "allocator_cap_oom": oom,
        "same_pid_recovery": True,
        "tiny_cuda_sum": result,
    }
    folder = out.temporary_file().parent / "gpu-cap-proof"
    folder.mkdir()
    (folder / "proof.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    return out.save_tree(folder)
