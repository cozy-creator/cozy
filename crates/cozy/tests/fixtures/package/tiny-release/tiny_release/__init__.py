"""A torch-free package whose releases differ in one literal: the release tests' fixture."""

from typing import Literal

import msgspec
from cozy_runtime.author import App

app = App()


class EchoInput(msgspec.Struct, forbid_unknown_fields=True):
    kind: Literal["audio", "scene"] = "audio"


class EchoOutput(msgspec.Struct):
    kind: str


@app.entrypoint(accelerator=False)
def echo(payload: EchoInput) -> EchoOutput:
    """Answers the kind it was asked for."""
    return EchoOutput(payload.kind)
