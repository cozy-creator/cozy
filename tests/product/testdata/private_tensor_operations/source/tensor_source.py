import io
import struct

import tensorfs
from cozy_runtime.author import (
    App, Context, ModelArtifact, WeightsOutput, invocable,
)
from tensorfs.derived import Derivation, Part, Target, Tensor

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


@invocable(memoize=True)
async def compute(ctx: Context, *, value: int) -> ModelArtifact:
    tensor = Tensor(
        logical_dtype="f32", shape=(512,), encoding=PLAIN,
        parts={"value": Part("f32", (512,))},
    )
    with ctx.output("weights").open(Derivation(
        sources={}, targets={"model": Target(add={"weight": tensor})},
        configs={}, order=(("model", "weight"),),
    )) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        writer.add_part("model", "weight", "value", io.BytesIO(struct.pack("<512f", *([float(value)] * 512))))
        return ctx.adopt_model(writer.commit())


app = App()
app.job(compute, weights=(WeightsOutput("weights", max_new_bytes=4096),))
