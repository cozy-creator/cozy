"""Small deterministic native preparation, with no downloader or Torch."""
import io
import struct
import tensorfs
from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, invocable
from tensorfs.derived import Derivation, Part, Target, Tensor

PLAIN = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")

@invocable(memoize=True)
async def produce(ctx: Context, *, value: int, private_note: str = "") -> ModelArtifact:
    assert isinstance(private_note, str)
    assert 1 <= value <= 32
    names = ("linear_a.weight", "linear_b.weight")
    tensors = {name: Tensor(logical_dtype="bf16", shape=(16, 64), encoding=PLAIN,
        parts={"value": Part("bf16", (16, 64))}) for name in names}
    with ctx.output("model").open(Derivation(sources={}, targets={"encoder": Target(add=tensors)},
                      configs={}, order=tuple(("encoder", name) for name in names))) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        for index, name in enumerate(names):
            ctx.raise_if_cancelled()
            raw = struct.pack("<1024H", *([0x3F80 + value if index == 0 else 0x3F00 + value] * 1024))
            writer.add_part("encoder", name, "value", io.BytesIO(raw))
        return ctx.adopt_model(writer.commit())

app = App()
app.job(produce, weights=(WeightsOutput("model", max_new_bytes=1 << 16),))
