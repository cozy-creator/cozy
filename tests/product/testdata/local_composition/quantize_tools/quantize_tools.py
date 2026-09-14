"""Qualification-only fault boundary over the production generic quantizer."""
import json
import os
import time

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, WeightsOutput, invocable
from tensorfs.derived import Derivation, DerivedTransaction, Target
from cozy_runtime.derive import plan, quantize_artifact
from cozy_runtime.derive.quantization import QuantizationSource

# The harness supplies this private fault-injection path once when copying the
# fixture. It is not an operation argument or a production source registry option.
INTERRUPT_MARKER = ""


@invocable(memoize=True)
async def quantize(ctx: Context, *, source: QuantizationSource, encoding: str,
                   tel: Telemetry) -> ModelArtifact:
    checkpoint = DerivedTransaction.checkpoint

    def pause_after_checkpoint(writer):
        checkpoint(writer)
        try:
            fd = os.open(INTERRUPT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "request": ctx.request_id,
                       "parts": sorted(writer.completed_parts())}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        while True:
            time.sleep(0.05)

    DerivedTransaction.checkpoint = pause_after_checkpoint
    try:
        return quantize_artifact(source, plan(("encoder",), encoding), ctx=ctx, tel=tel)
    finally:
        DerivedTransaction.checkpoint = checkpoint


app = App()
app.job(quantize, weights=(WeightsOutput("model", max_new_bytes=1 << 20),))


@invocable(memoize=True)
async def graft(ctx: Context, *, source: QuantizationSource) -> ModelArtifact:
    """Retain the raw fixture's tensors without permitting new payload bytes."""
    capability = ctx.tensorfs_source(source)
    structure = capability.inspect()
    assert not structure.configs
    components = structure.components
    with ctx.output("model").open(Derivation(sources={"source": capability},
                      targets={name: Target(source="source", source_component=name)
                               for name in components},
                      configs={}, order=tuple((component, key) for component, tensors in components.items() for key in tensors))) as writer:
        return ctx.adopt_model(writer.receipt if writer.receipt is not None else writer.commit())


app.job(graft, weights=(WeightsOutput("model", max_new_bytes=0),))
