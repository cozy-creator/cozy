"""A quantizer's shape without torch: Runtime writes one declared output directly through
public TensorFS in the configured store and returns its model reference."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()


class TableInput(msgspec.Struct):
    size: int = 64
    hold: str = ""
    count: bool = False
    #: A file under TENSORFS_HOME: written with `.reached` once committed, then awaited before
    #: the adopt (and its upload), so a test acts in the middle of the call.
    gate: str = ""


async def table(ctx: Context, payload: TableInput, tel: Telemetry) -> ModelArtifact:
    async def hold(phase: str) -> None:
        if payload.hold == phase:
            tel.step_callback(1, stage=f"native {phase}")(0)
            while True:
                ctx.raise_if_cancelled()
                await asyncio.sleep(0.01)

    output = ctx.output("model", Derivation(None, order=(("model", "table"),)))
    if not output.has("model", "table"):
        output.put("model", "table", "F32", [payload.size], bytes(range(256)) * (payload.size // 64))
        await hold("checkpoint")  # TensorFS retains the keyed Writer's resumable state
    await hold("commit")
    if payload.gate:
        gate = Path(os.environ["TENSORFS_HOME"]) / payload.gate
        gate.with_suffix(".reached").touch()
        while not gate.exists():
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.01)
    result = ctx.adopt_model(output)  # Runtime commits, then reports a checkpoint reference
    await hold("adopt")
    return result


app.entrypoint(table, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)


async def convert(
    ctx: Context, payload: TableInput, tel: Telemetry, *, source: Weights[Any]
) -> ModelArtifact:
    """The same CPU Writer behind the CLI's existing source/destination conversion syntax."""
    values = ctx.tensorfs_source(source).components["model"].tensors.read()
    assert values["seed"].tobytes() == bytes(4)
    if payload.count:
        # A bounded test observation: a code-less upload must not execute this function again.
        with (Path(os.environ["TENSORFS_HOME"]) / "test-convert-invocations").open("ab") as stream:
            stream.write(b"invoked\n")
    return await table(ctx, payload, tel)


app.entrypoint(convert, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)
