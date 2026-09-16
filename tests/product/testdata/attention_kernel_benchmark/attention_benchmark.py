"""Small real CUDA attention workload for ordinary CLI kernel A/B/A qualification."""

from __future__ import annotations

import hashlib
import time
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    App,
    AttentionContext,
    Config,
    Context,
    Loader,
    Model,
    uses_components,
)

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 91
    tokens: Annotated[int, msgspec.Meta(ge=16, le=8192)] = 512
    iterations: Annotated[int, msgspec.Meta(ge=1, le=100)] = 10


class Result(msgspec.Struct):
    device: str
    dtype: str
    shape: list[int]
    backend: str
    output_sha256: str
    checksum: float
    max_abs: float
    mean_ms: float
    finite: bool
    sdpa_max_abs_error: float
    sdpa_relative_l2_error: float
    distributions: dict[str, str]


class AttentionPipeline:
    def __init__(self, config: Config) -> None:
        del config
        import torch
        from diffusers.models.transformers.transformer_wan import (
            WanAttention,
            WanAttnProcessor,
        )

        self.components = {
            "dit": WanAttention(
                dim=128, heads=1, dim_head=128, processor=WanAttnProcessor()
            )
            .to(dtype=torch.bfloat16)
            .eval()
        }


class AttentionModel(Model[AttentionPipeline]):
    pipe: AttentionPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(AttentionPipeline, factory=AttentionPipeline)

    def choose_attention(self, context: AttentionContext) -> str:
        del context
        return "sdpa"

    @uses_components("dit")
    def benchmark(self, payload: Request) -> Result:
        import torch

        site = self.pipe.components["dit"]
        device = next(site.parameters()).device
        if device.type != "cuda":
            raise ValueError("attention kernel benchmark requires a CUDA rental")
        generator = torch.Generator(device="cpu").manual_seed(payload.seed)
        hidden = torch.randn(1, payload.tokens, 128, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            site(hidden)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(payload.iterations):
                output = site(hidden)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        # Compare against the same real projection weights with Diffusers' SDPA
        # processor. This copy never changes the prepared model or request pin.
        import copy
        from importlib.metadata import PackageNotFoundError, version

        from diffusers.models.attention_dispatch import AttentionBackendName
        from diffusers.models.transformers.transformer_wan import WanAttnProcessor

        selected = getattr(site.processor, "_attention_backend", None)
        reference = copy.deepcopy(site)
        reference.set_processor(WanAttnProcessor())
        reference.processor._attention_backend = AttentionBackendName.NATIVE
        with torch.inference_mode():
            expected = reference(hidden).float()
        difference = output.float() - expected
        distributions = {}
        for name in (
            "cozy-runtime",
            "torch",
            "diffusers",
            "kernels",
            "cozy-kernel-flash-attn3",
            "cozy-kernel-flash-attn4",
        ):
            try:
                distributions[name] = version(name)
            except PackageNotFoundError:
                pass
        data = output.detach().cpu().contiguous()
        return Result(
            device=str(device),
            dtype=str(data.dtype),
            shape=list(data.shape),
            backend=getattr(selected, "value", str(selected)),
            output_sha256=hashlib.sha256(
                data.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            checksum=float(data.float().sum()),
            max_abs=float(data.float().abs().max()),
            mean_ms=elapsed * 1000 / payload.iterations,
            finite=bool(data.isfinite().all()),
            sdpa_max_abs_error=float(difference.abs().max()),
            sdpa_relative_l2_error=float(
                difference.norm() / expected.norm().clamp_min(1e-12)
            ),
            distributions=distributions,
        )


@app.entrypoint
def generate(ctx: Context, payload: Request, model: AttentionModel) -> Result:
    ctx.raise_if_cancelled()
    return model.benchmark(payload)
