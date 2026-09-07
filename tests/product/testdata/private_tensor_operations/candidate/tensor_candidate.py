import struct
from importlib.metadata import distributions

import tensorfs
from cozy_runtime.author import (
    App, Context, Loader, Model, ModelArtifact, WeightsOutput, WeightsPart,
    WeightsSink, WeightsTarget, WeightsTensor, invocable,
)

if any((item.metadata.get("Name") or "").startswith("cozy-script-") for item in distributions()):
    raise RuntimeError("tensor implementation imported into parent")

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("derive-only input must not load")


@invocable(memoize=True)
async def compute(ctx: Context, *, artifacts: WeightsSink, source: Source, factor: int) -> ModelArtifact:
    if factor == 0:
        raise ValueError("candidate quality gate failed")
    tensor = WeightsTensor(
        logical_dtype="f32", shape=(512,), encoding=PLAIN,
        parts={"value": WeightsPart("f32", (512,))},
    )
    with artifacts.open(
        "weights", sources={"original": source},
        targets={"model": WeightsTarget(
            source="original", source_component="model", drop=("weight",), add={"weight": tensor},
        )}, configs={}, order=(("model", "weight"),),
    ) as writer:
        buf = bytearray(2048)
        writer.source_read_into("original", "model", "weight", "value", 0, buf)
        values = struct.unpack("<512f", buf)
        writer.add_part("model", "weight", "value", struct.pack("<512f", *(value * factor for value in values)))
        return writer.commit().artifact


app = App()
app.job(compute, weights=(WeightsOutput("weights", max_new_bytes=4096),))
