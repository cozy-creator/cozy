"""Small deterministic native preparation, with no downloader or Torch."""
import struct
import tensorfs
from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, WeightsPart, WeightsSink, WeightsTarget, WeightsTensor, invocable

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")

@invocable(memoize=True)
async def produce(ctx: Context, *, weights: WeightsSink, value: int) -> ModelArtifact:
    assert 1 <= value <= 32
    names = ("linear_a.weight", "linear_b.weight")
    tensors = {name: WeightsTensor(logical_dtype="bf16", shape=(16, 64), encoding=PLAIN,
        parts={"value": WeightsPart("bf16", (16, 64))}) for name in names}
    with weights.open("model", sources={}, targets={"encoder": WeightsTarget(add=tensors)},
                      configs={}, order=tuple(("encoder", name) for name in names)) as writer:
        for index, name in enumerate(names):
            ctx.raise_if_cancelled()
            raw = struct.pack("<1024H", *([0x3F80 + value if index == 0 else 0x3F00 + value] * 1024))
            writer.add_part("encoder", name, "value", raw)
        return writer.commit().artifact

app = App()
app.job(produce, weights=(WeightsOutput("model", max_new_bytes=1 << 16),))
