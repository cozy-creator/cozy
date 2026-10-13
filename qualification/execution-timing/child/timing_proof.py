import asyncio
import os
import time

import msgspec

from cozy_runtime.author import App, Context, Telemetry, invocable

app = App()


class Result(msgspec.Struct):
    pid: int
    waited_ms: float
    worked_ms: float
    checksum: int


def busy(milliseconds: int) -> tuple[float, int]:
    started = time.monotonic()
    checksum = 3
    while (time.monotonic() - started) * 1000 < milliseconds:
        checksum = (checksum * 17 + 11) % 1000003
    return (time.monotonic() - started) * 1000, checksum


@invocable
async def pause_then_work(ctx: Context, tel: Telemetry, *, wait_seconds: float = 4.0) -> Result:
    tel.progress(0.0, stage="Waiting")
    started = time.monotonic()
    await asyncio.sleep(wait_seconds)
    waited = (time.monotonic() - started) * 1000
    ctx.raise_if_cancelled()
    tel.progress(0.9, stage="CPU work")
    worked, checksum = busy(150)
    return Result(os.getpid(), waited, worked, checksum)


app.job(pause_then_work)
