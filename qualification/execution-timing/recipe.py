# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.18.89", "cozy-timing-proof==0.1.0"]
# [tool.uv.sources]
# cozy-timing-proof = {path = "./child"}
# ///

import asyncio
import importlib.metadata
import os
import time

from cozy_runtime.author import ThreadPoolExecutor
from timing_proof import busy, pause_then_work


async def main(ctx) -> dict[str, float | int | str]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as scanning:
        parallel = scanning.submit(busy, 150)
        ctx.progress(0.1, stage="Waiting for nested child")
        child = await pause_then_work(wait_seconds=4.0)
        parallel_ms, _ = parallel.result()
        ctx.progress(0.9, stage="Finishing CPU work")
        final_ms, _ = await asyncio.to_thread(busy, 150)
    return {
        "runtime": importlib.metadata.version("cozy-runtime"),
        "parent_pid": os.getpid(),
        "child_pid": child.pid,
        "authored_wall_ms": (time.monotonic() - started) * 1000,
        "child_wait_ms": child.waited_ms,
        "cpu_work_ms": parallel_ms + child.worked_ms + final_ms,
        "checksum": child.checksum,
    }
