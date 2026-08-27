"""cl-013's acceptance endpoint: a REAL invoke with no model, no weights, no GPU.

`gpu` DERIVES from the signature (cozy-runtime `_describe.py`): a handler with no `Model`
parameter is a CPU handler, so the orchestrator grants it no device and this endpoint runs
on any machine — including a Windows runner, which is the whole point of the launch tier's
Windows slice. Its release depends on the BASE `cozy-runtime` wheel only (msgspec,
protobuf, grpcio — no torch, no tensorfs), so a clean machine can install it over a
residential line in seconds.

What an invoke of `tile` exercises is everything except the model: the install generation's
own venv, `describe --check`'s derived surface, the supervisor, the executor, the worker
protocol, the record owner's terminal transaction, the output publication, the media id, and
`cozy run --out`. `refuse` is the failure terminal on the same path.

`video_transport` exists only so the remote live driver can exercise a text/image request
and MP4 result contract through an adversarial worker. It refuses local execution and is
explicitly not a model or video-generation fixture.
"""

from __future__ import annotations

import hashlib
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    App,
    Context,
    ImageAsset,
    ImageFrame,
    InvalidRequest,
    Outputs,
    Telemetry,
    VideoAsset,
)

app = App()


class TileInput(msgspec.Struct, forbid_unknown_fields=True):
    size: Annotated[int, msgspec.Meta(ge=8, le=256)] = 64
    seed: int = 13


class TileOutput(msgspec.Struct):
    image: ImageAsset
    size: int
    pixels: int
    digest: str
    warm: bool


class RefuseInput(msgspec.Struct, forbid_unknown_fields=True):
    why: str = "cl-013 asked for it"


class VideoTransportInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    first_frame: ImageAsset


class VideoTransportOutput(msgspec.Struct):
    video: VideoAsset


@app.entrypoint
def tile(ctx: Context, payload: TileInput, out: Outputs, tel: Telemetry) -> TileOutput:
    """A deterministic RGB tile from a linear congruential sequence — real computation
    whose output is a real PNG the runtime encodes, with nothing to load first."""
    side = 8 if ctx.boot_warmup else payload.size
    tel.log("filling the tile", side=side, seed=payload.seed)
    state = payload.seed & 0xFFFFFFFF
    pixels = bytearray(side * side * 3)
    for i in range(0, len(pixels), 3):
        ctx.raise_if_cancelled()
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        pixels[i] = (state >> 24) & 0xFF
        pixels[i + 1] = (state >> 16) & 0xFF
        pixels[i + 2] = (state >> 8) & 0xFF
    tel.metric("tile_bytes", len(pixels))
    return TileOutput(
        image=out.save_image(ImageFrame(side, side, bytes(pixels)), format="png"),
        size=side,
        pixels=side * side,
        digest=hashlib.sha256(pixels).hexdigest(),
        warm=ctx.boot_warmup,
    )


@app.entrypoint
def refuse(payload: RefuseInput) -> TileOutput:
    """The FAILED terminal on the same weightless path, so the fixture observes both
    verdicts of the terminal transaction rather than only the happy one."""
    raise InvalidRequest(payload.why)


@app.entrypoint
def video_transport(payload: VideoTransportInput) -> VideoTransportOutput:
    """A descriptor-only seam for the remote byte-plane live proof.

    The fake remote worker consumes this request under an exact DeliveryGrant and emits a
    tiny MP4-shaped transport fixture. This handler deliberately refuses if somebody runs
    it locally: it is not video generation and must never be presented as H3 inference.
    """
    del payload
    raise InvalidRequest("video_transport is a transport-contract fixture, not inference")
