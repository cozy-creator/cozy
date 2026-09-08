"""A tiny real model whose construction order cannot come from the checkpoint header."""

from __future__ import annotations

import os

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, uses_components

app = App()
REFUSE_INITIALIZATION = False


class Request(msgspec.Struct, forbid_unknown_fields=True):
    value: int = 3


class Result(msgspec.Struct):
    value: float
    components: list[str]
    device: str


class OrderedPipeline:
    def __init__(self) -> None:
        import torch

        self.components = {
            "zeta": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32),
            "alpha": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32),
        }
        if REFUSE_INITIALIZATION:
            raise ValueError(
                f"local-serving-init-canary pid={os.getpid()} ppid={os.getppid()} "
                f"device={self.components['zeta'].weight.device.type}"
            )


def build_pipeline(config: Config) -> OrderedPipeline:
    return OrderedPipeline()


class OrderedModel(Model[OrderedPipeline]):
    pipe: OrderedPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OrderedPipeline, factory=build_pipeline)

    @uses_components("alpha", "zeta")
    def measure(self, value: int) -> Result:
        import torch

        zeta = self.pipe.components["zeta"]
        alpha = self.pipe.components["alpha"]
        with torch.inference_mode():
            vector = torch.full((1, 2), float(value), device=zeta.weight.device)
            output = alpha(zeta(vector))
            return Result(
                float(output.sum().item()), list(self.pipe.components), output.device.type
            )


@app.entrypoint
def generate(ctx: Context, payload: Request, model: OrderedModel) -> Result:
    ctx.raise_if_cancelled()
    return model.measure(payload.value)
