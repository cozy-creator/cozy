"""Qualification-only scoring of supplied immutable PNGs, not model inference."""
from pathlib import Path

import msgspec
import numpy as np
from PIL import Image
from cozy_eval.metrics.reference import psnr, ssim
from cozy_runtime.author import App, Context, Model, Telemetry, invocable


class Candidate(Model[object]):
    def load(self, loader):
        raise AssertionError("the composition proof does not run inference")


class Result(msgspec.Struct, frozen=True):
    checkpoint: str
    psnr: float
    ssim: float


@invocable(memoize=True)
async def score(ctx: Context, *, model: Candidate, tel: Telemetry) -> Result:
    ctx.raise_if_cancelled()
    root = Path(__file__).parent
    with Image.open(root / "baseline.png") as base, Image.open(root / "candidate.png") as candidate:
        reference = np.asarray(base.convert("RGB"))
        compared = np.asarray(candidate.convert("RGB"))
        values = psnr(reference, compared), ssim(reference, compared)
    tel.log("scored supplied qualification images", checkpoint=model.checkpoint_ref)
    return Result(model.checkpoint_ref, *values)


app = App()
app.job(score)
