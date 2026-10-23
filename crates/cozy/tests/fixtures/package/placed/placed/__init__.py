"""A torch-free package whose one function takes a GPU and loads no model: placement's subject
(run_placement.rs). Nothing here is ever run."""

import msgspec
from cozy_runtime.author import App

app = App()


class Prompt(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a lighthouse at dusk"


class Drawn(msgspec.Struct):
    prompt: str


@app.entrypoint
def draw(payload: Prompt) -> Drawn:
    """Answers its prompt, on a GPU."""
    return Drawn(payload.prompt)
