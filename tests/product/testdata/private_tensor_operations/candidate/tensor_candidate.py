import io
import struct

import tensorfs
from cozy_runtime.author import (
    App, Context, Loader, Model, ModelArtifact, WeightsOutput, invocable,
)
from tensorfs.derived import Derivation, Part, Target, Tensor

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("derive-only input must not load")


@invocable(memoize=True)
async def compute(ctx: Context, *, source: Source, factor: int) -> ModelArtifact:
    if factor == 0:
        raise ValueError("candidate quality gate failed")
    tensor = Tensor(
        logical_dtype="f32", shape=(512,), encoding=PLAIN,
        parts={"value": Part("f32", (512,))},
    )
    with ctx.output("weights").open(Derivation(
        sources={"original": ctx.tensorfs_source(source)},
        targets={"model": Target(
            source="original", source_component="model", drop=("weight",), add={"weight": tensor},
        )}, configs={}, order=(("model", "weight"),),
    )) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        buf = bytearray(2048)
        writer.source_read_into("original", "model", "weight", "value", 0, buf)
        values = struct.unpack("<512f", buf)
        writer.add_part("model", "weight", "value", io.BytesIO(struct.pack("<512f", *(value * factor for value in values))))
        return ctx.adopt_model(writer.commit())


app = App()
app.job(compute, weights=(WeightsOutput("weights", max_new_bytes=4096),))
