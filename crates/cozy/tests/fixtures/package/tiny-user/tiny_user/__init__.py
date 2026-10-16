"""Uses tiny, resolved from its author's account index."""

import msgspec
from cozy_runtime.author import App

app = App()


class ShoutInput(msgspec.Struct, forbid_unknown_fields=True):
    text: str = "hello"


class ShoutOutput(msgspec.Struct):
    text: str


@app.entrypoint(accelerator=False)
def shout(payload: ShoutInput) -> ShoutOutput:
    """Answers its input, louder."""
    return ShoutOutput(payload.text.upper())
