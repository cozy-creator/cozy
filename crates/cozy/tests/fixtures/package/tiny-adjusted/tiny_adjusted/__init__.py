"""A torch-free package that serves its request other than as sent, and says so."""

import msgspec
from cozy_runtime.author import Adjustments, App

app = App()


class StepsInput(msgspec.Struct, forbid_unknown_fields=True):
    steps: int = 50


class StepsOutput(msgspec.Struct):
    steps: int


@app.entrypoint(accelerator=False)
def steps(payload: StepsInput, adjustments: Adjustments) -> StepsOutput:
    """Runs at most 30 steps."""
    applied = min(payload.steps, 30)
    if applied != payload.steps:
        adjustments.clamp("steps", payload.steps, applied, "at most 30 steps")
    adjustments.warn("seed", "none given; the result is not reproducible")
    return StepsOutput(applied)
