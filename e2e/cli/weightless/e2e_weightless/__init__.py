"""The cli suite's weightless requests: from cozy through TensorD, the Runtime and TensorFS to the executor's
reply and back, with no model, so they run on any computer."""

import time

import msgspec

from cozy_runtime.author import App, Assets, FileAsset

app = App()


class Text(msgspec.Struct, forbid_unknown_fields=True):
    text: str


class Reversed(msgspec.Struct):
    text: str


@app.entrypoint(accelerator=False)
def reverse(payload: Text) -> Reversed:
    return Reversed(text=payload.text[::-1])


class Wait(msgspec.Struct, forbid_unknown_fields=True):
    seconds: float


class Waited(msgspec.Struct):
    seconds: float


@app.entrypoint(accelerator=False)
def wait(payload: Wait) -> Waited:
    """A run that stays in progress: followed, snapshotted and canceled by the suite."""
    time.sleep(payload.seconds)
    return Waited(seconds=payload.seconds)


class Nothing(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Seen(msgspec.Struct):
    assets: list[str]


@app.entrypoint(accelerator=False)
def fidelities(payload: Nothing, assets: Assets[FileAsset]) -> Seen:
    """Each attached asset as the executor was granted it: `label=fidelity`."""
    return Seen(assets=[f"{assets.info(i).label}={assets.info(i).fidelity}" for i in range(len(assets))])
