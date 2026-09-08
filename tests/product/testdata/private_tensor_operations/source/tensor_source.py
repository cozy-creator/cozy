import struct

import tensorfs
from cozy_runtime.author import (
    App, Context, ModelArtifact, WeightsOutput, WeightsPart, WeightsSink,
    WeightsTarget, WeightsTensor, invocable,
)

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


@invocable(memoize=True)
async def compute(ctx: Context, *, artifacts: WeightsSink, value: int) -> ModelArtifact:
    tensor = WeightsTensor(
        logical_dtype="f32", shape=(512,), encoding=PLAIN,
        parts={"value": WeightsPart("f32", (512,))},
    )
    with artifacts.open(
        "weights", sources={}, targets={"model": WeightsTarget(add={"weight": tensor})},
        configs={}, order=(("model", "weight"),),
    ) as writer:
        writer.add_part("model", "weight", "value", struct.pack("<512f", *([float(value)] * 512)))
        return writer.commit().artifact


app = App()
app.job(compute, weights=(WeightsOutput("weights", max_new_bytes=4096),))
