"""A torch-free quantizer for cozy's model tests: any checkpoint's attention and MLP projections
to fp8, every other tensor at its source bytes. Installs in seconds on a CPU machine."""

from typing import Any

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, Weights, WeightsOutput, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

app = App()


@invocable(memo_version="tiny-quantizer-fp8/1")
async def fp8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The fp8 lane (`fp8.rowwise`)."""
    plan = QuantizationPlan(
        components=source.components(),
        include=("*.c_attn.weight", "*.c_fc.weight"),
        output_precision="preserve",
    )
    return quantize_into(ctx, tel, source=source, plan=plan, encoding="fp8.rowwise", output="fp8")


app.entrypoint(fp8, weights=(WeightsOutput("fp8", max_new_bytes=1 << 30),), accelerator=False)
