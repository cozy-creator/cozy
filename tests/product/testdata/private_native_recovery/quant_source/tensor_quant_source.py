import io
import struct
import tensorfs
from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, invocable
from tensorfs.derived import Derivation, Part, Target, Tensor


@invocable(memoize=True)
async def compute(ctx: Context) -> ModelArtifact:
    plain = dict(tensorfs.seed_digests())["plain/1"]
    tensors = {key: Tensor("f16", (512, 1024), plain,
                          {"value": Part("f16", (512, 1024))})
               for key in ("a.weight", "b.weight")}
    tensors["bias"] = Tensor("f32", (1,), plain, {"value": Part("f32", (1,))})
    with ctx.output("model").open(Derivation(
        sources={}, targets={"text_encoder": Target(add=tensors)}, configs={},
        order=tuple(("text_encoder", key) for key in tensors),
    )) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        for index, key in enumerate(("a.weight", "b.weight")):
            writer.add_part("text_encoder", key, "value",
                            io.BytesIO(struct.pack("<e", 1.25 + index) * (512 * 1024)))
        writer.add_part("text_encoder", "bias", "value", io.BytesIO(struct.pack("<f", 1.25)))
        return ctx.adopt_model(writer.commit())

app = App()
app.job(compute, weights=(WeightsOutput("model", max_new_bytes=3145728),))
