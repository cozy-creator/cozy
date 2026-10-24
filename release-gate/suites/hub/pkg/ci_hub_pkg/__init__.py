"""The hub suite's throwaway package: one weightless function that names its release, so a run proves which
release an install brought (cells.sh writes RELEASE and the pyproject version together)."""

import msgspec

from cozy_runtime.author import App

RELEASE = "0.1.0"
app = App()


class Text(msgspec.Struct, forbid_unknown_fields=True):
    text: str


class Echoed(msgspec.Struct):
    text: str
    release: str


@app.entrypoint(accelerator=False)
def echo(payload: Text) -> Echoed:
    return Echoed(text=payload.text, release=RELEASE)
