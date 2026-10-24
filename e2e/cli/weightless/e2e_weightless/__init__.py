"""The cli suite's weightless request: from cozy through TensorD, the Runtime and TensorFS to the executor's
reply and back, with no model, so it runs on any computer."""

import msgspec

from cozy_runtime.author import App

app = App()


class Text(msgspec.Struct, forbid_unknown_fields=True):
    text: str


class Reversed(msgspec.Struct):
    text: str


@app.entrypoint(accelerator=False)
def reverse(payload: Text) -> Reversed:
    return Reversed(text=payload.text[::-1])
