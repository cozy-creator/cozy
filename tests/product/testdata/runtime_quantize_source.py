"""Native bytes and a fresh reader for the shared Runtime invocation contract."""
import struct
import msgspec
import tensorfs
from cozy_runtime.author import (
    App, Context, ModelArtifact, WeightsOutput, WeightsPart, WeightsReader,
    WeightsSink, WeightsTarget, WeightsTensor, invocable,
)
from cozy_runtime.derive.operations import QuantizationSource

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]

@invocable(memoize=True)
async def produce(ctx: Context, *, weights: WeightsSink) -> ModelArtifact:
    matrix = WeightsTensor("f16", (16, 32), PLAIN,
                           {"value": WeightsPart("f16", (16, 32))})
    bias = WeightsTensor("f16", (16,), PLAIN,
                         {"value": WeightsPart("f16", (16,))})
    with weights.open("model", sources={}, targets={"body": WeightsTarget(
        add={"layer.weight": matrix, "layer.bias": bias})},
        order=(("body", "layer.weight"), ("body", "layer.bias"))) as output:
        if output.replayed:
            assert output.receipt is not None
            return output.receipt.artifact
        output.add_part("body", "layer.weight", "value",
                        struct.pack("<512e", *(i / 257 - 1 for i in range(512))))
        output.add_part("body", "layer.bias", "value", struct.pack("<16e", *range(16)))
        return output.commit().artifact

class Checked(msgspec.Struct):
    nonzero: bool

@invocable(memoize=False)
async def inspect(ctx: Context, *, original: QuantizationSource,
                  candidate: QuantizationSource, reader: WeightsReader) -> Checked:
    with reader.open(original) as source, reader.open(candidate) as result:
        tensor = result.tensor("body", "layer.weight")
        assert tensor.logical_dtype == "f16"
        assert {part.name for part in tensor.parts} == {"data", "scale"}
        assert source.identity("body", "layer.bias") == result.identity("body", "layer.bias")
        payload = bytearray(512)
        result.read_part_into("body", "layer.weight", "data", 0, payload)
        assert any(payload)
        return Checked(True)

app.job(produce, weights=(WeightsOutput("model", max_new_bytes=65536),))
app.job(inspect)
