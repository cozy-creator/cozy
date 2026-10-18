"""A quantizer's shape without torch: a CPU function writes one weights output through its
owner's derive channel, adopts it and returns the model."""
from __future__ import annotations

import asyncio

import msgspec

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, WeightsOutput
from cozy_runtime.tensord_channels import Derivation

app = App()


class TableInput(msgspec.Struct):
    size: int = 64
    hold: str = ""


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
        await hold("checkpoint")  # TensorD journals each put: the write is resumable from here
    await hold("commit")
    result = ctx.adopt_model(output)  # commits, then TensorD checks and keeps the manifest
    await hold("adopt")
    return result


app.entrypoint(table, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)
