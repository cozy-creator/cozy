"""cl-003's proof entrypoint, added to cr-008b's four-component pipeline RELEASE.

The M4 arm "an endpoint that ignores cancel" needs an endpoint that ignores cancel, and
cr-008b's pipeline is cooperative by construction: `generate` calls
`ctx.raise_if_cancelled()` on every denoising step, which is what a real endpoint should
do. So the proof release adds ONE entrypoint whose body is `generate`'s loop minus the
cancellation check — the non-cooperative half of the honest-cancellation claim, in the
endpoint's own source rather than as a plant inside the runtime.

It touches ONLY the UNet (the `denoise` component scope), with zero-valued conditioning of
the right geometry: what the arm is about is a handler that will not stop, not the picture
it would have made. The release's `[application]` object points here; importing this module
imports `sdxl_txt2img`, which is what registers `generate` and `condition` beside it.
"""

from __future__ import annotations

from typing import Any

import msgspec
from sdxl_txt2img import Txt2ImgInput, app

from corpus.sdxl_pipeline import SdxlPipelineModel
from cozy_runtime.author import Context, Telemetry


class StubbornOutput(msgspec.Struct):
    passes: int


@app.entrypoint
def stubborn(
    ctx: Context,
    payload: Txt2ImgInput,
    model: SdxlPipelineModel,
    tel: Telemetry,
) -> StubbornOutput:
    """UNet passes in a loop, never asking whether the request was cancelled.

    A cooperative handler converges on the next `raise_if_cancelled`. This one has no such
    line, so the only thing that can end it is the runtime: cancel, grace expiry, then a
    kill of the device process — and a typed CANCELED terminal with the card released.
    """
    import torch

    device = torch.device("cuda", 0)
    model.for_request(ctx, seed=payload.seed)
    side = 512 if ctx.boot_warmup else payload.size
    latent = side // 8
    passes = 1 if ctx.boot_warmup else max(payload.steps, 1)
    latents = torch.zeros(2, 4, latent, latent, device=device, dtype=torch.float16)
    prompt = torch.zeros(2, 77, 2048, device=device, dtype=torch.float16)
    pooled = torch.zeros(2, 1280, device=device, dtype=torch.float16)
    time_ids = torch.zeros(2, 6, device=device, dtype=torch.float16)
    on_step = tel.step_callback(passes, stage="denoise")
    with tel.stage("denoise"):
        for index in range(passes):
            timestep = torch.tensor(999, device=device)
            model.denoise(latents, timestep, prompt, pooled, time_ids)
            on_step(index)
    return StubbornOutput(passes=passes)
