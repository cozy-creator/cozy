"""A torch-free package for cozy's package tests: it installs in seconds on any machine."""

import msgspec
from cozy_runtime.author import App

app = App()


class EchoInput(msgspec.Struct, forbid_unknown_fields=True):
    text: str = "hello"


class EchoOutput(msgspec.Struct):
    text: str


@app.entrypoint(accelerator=False)
def echo(payload: EchoInput) -> EchoOutput:
    """Answers its input."""
    return EchoOutput(payload.text)
