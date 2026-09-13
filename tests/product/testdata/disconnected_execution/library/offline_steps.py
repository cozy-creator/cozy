import asyncio
from pathlib import Path

import msgspec
from cozy_runtime.author import App, Context, invocable


class Result(msgspec.Struct, frozen=True):
    value: int


ROOT = Path("/tmp/cozy-offline-proto051")
app = App()


@invocable(memoize=True)
async def prepare(ctx: Context, *, value: int) -> Result:
    ROOT.mkdir(exist_ok=True)
    # Qualification-only execution count, independent of the deterministic value.
    with (ROOT / "prepare-count").open("a") as handle:
        handle.write("executed\n")
    ctx.log("expensive preparation completed")
    return Result(value * value)


@invocable(memoize=False)
async def after_disconnect(ctx: Context, *, value: int) -> Result:
    ROOT.mkdir(exist_ok=True)
    (ROOT / "waiting").write_text(str(value))
    while not (ROOT / "continue").exists():
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    (ROOT / "continued").write_text(str(value + 1))
    return Result(value + 1)


@invocable(memoize=False)
async def finish(ctx: Context, *, value: int) -> Result:
    (ROOT / "finished").write_text(str(value + 1))
    return Result(value + 1)


app.job(prepare)
app.job(after_disconnect)
app.job(finish)
