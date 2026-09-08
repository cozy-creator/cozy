"""Qualification-only fault boundary over the production generic quantizer."""
import json
import os
import time

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, WeightsOutput, WeightsSink, WeightsTarget, WeightsTransaction, invocable
from cozy_runtime.derive import plan, quantize_artifact
from cozy_runtime.derive.quantization import QuantizationSource

# The harness supplies this private fault-injection path once when copying the
# fixture. It is not an operation argument or a production source registry option.
INTERRUPT_MARKER = ""


@invocable(memoize=True)
async def quantize(ctx: Context, *, source: QuantizationSource, encoding: str,
                   weights: WeightsSink, tel: Telemetry) -> ModelArtifact:
    checkpoint = WeightsTransaction.checkpoint

    def pause_after_checkpoint(writer):
        checkpoint(writer)
        try:
            fd = os.open(INTERRUPT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "request": ctx.request_id,
                       "parts": sorted(writer.completed_parts)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        while True:
            time.sleep(0.05)

    WeightsTransaction.checkpoint = pause_after_checkpoint
    try:
        return quantize_artifact(source, plan(("encoder",), encoding), sink=weights, ctx=ctx, tel=tel)
    finally:
        WeightsTransaction.checkpoint = checkpoint


app = App()
app.job(quantize, weights=(WeightsOutput("model", max_new_bytes=1 << 20),))


@invocable(memoize=True)
async def graft(ctx: Context, *, source: QuantizationSource, weights: WeightsSink) -> ModelArtifact:
    """Retain the raw fixture's tensors without permitting new payload bytes."""
    structure = weights.structure(source)
    assert not structure.configs
    components = sorted({tensor.component for tensor in structure.tensors})
    with weights.open("model", sources={"source": source},
                      targets={name: WeightsTarget(source="source", source_component=name)
                               for name in components},
                      order=tuple((tensor.component, tensor.key) for tensor in structure.tensors)) as writer:
        return writer.commit().artifact


app.job(graft, weights=(WeightsOutput("model", max_new_bytes=0),))
