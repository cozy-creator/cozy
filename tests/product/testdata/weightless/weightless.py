"""cl-013's acceptance package: a REAL invoke with no model, no weights, no GPU.

`gpu` DERIVES from the signature (cozy-runtime `_describe.py`): a handler with no `Model`
parameter is a CPU handler, so the orchestrator grants it no device and this package runs
on any machine — including a Windows runner, which is the whole point of the launch tier's
Windows slice. Its release depends on `cozy-runtime[media]` only (msgspec, protobuf,
grpcio and the exact PyAV wheel — no torch, CUDA or tensorfs), so a clean machine can
install it over a residential line in seconds. PyAV exists solely for the `relay` fixture.

What an invoke of `tile` exercises is everything except the model: the install generation's
own venv, `describe --check`'s derived surface, the Runtime worker, the executor, the worker
protocol, the record owner's terminal transaction, the output publication, the media id, and
`cozy run --out`. `refuse` is the failure terminal on the same path.

`video_transport` exists only so the remote product proof can exercise a text/image request
and MP4 result contract through an adversarial worker. It refuses local execution and is
explicitly not a model or video-generation fixture.
"""

from __future__ import annotations

import hashlib
import time
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    InvalidRequest,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
)

app = App()
REVISION = "first"


class TileInput(msgspec.Struct, forbid_unknown_fields=True):
    size: Annotated[int, msgspec.Meta(ge=8, le=256)] = 64
    seed: int = 13
    delay_ms: Annotated[int, msgspec.Meta(ge=0, le=5_000)] = 0


class TileOutput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(media_types=("image/webp",))]
    size: int
    pixels: int
    digest: str
    warm: bool
    revision: str


class TileJobOutput(msgspec.Struct):
    size: int
    seed: int
    revision: str


class RefuseInput(msgspec.Struct, forbid_unknown_fields=True):
    why: str = "cl-013 asked for it"


class VideoTransportInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    first_frame: Annotated[ImageAsset, AssetBound(max_bytes=8 << 20, max_decoded_bytes=16 << 20)]


class VideoTransportOutput(msgspec.Struct):
    video: VideoAsset


class RelayInput(msgspec.Struct, forbid_unknown_fields=True):
    image: Annotated[ImageAsset, AssetBound(max_bytes=8 << 20, max_decoded_bytes=16 << 20)]
    delay_ms: Annotated[int, msgspec.Meta(ge=0, le=5_000)] = 0


class RelayOutput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(media_types=("image/webp",))]


@app.entrypoint
def tile(ctx: Context, payload: TileInput, out: Outputs, tel: Telemetry) -> TileOutput:
    """A deterministic RGB tile from a linear congruential sequence — real computation
    whose output is a real WebP the runtime encodes, with nothing to load first."""
    side = 8 if ctx.boot_warmup else payload.size
    tel.log("filling the tile", side=side, seed=payload.seed)
    tel.progress(0.25, stage="tile")
    # Leave the real CLI enough time to observe this deliberately lossy frame. The fixture
    # proves presentation of a live Runtime stream; it does not pretend a completed run can
    # replay every transient tick.
    time.sleep(0.05)
    remaining = payload.delay_ms
    if remaining > 0:
        # cl-104: the delay is a MEASURED step loop — the fixture's stand-in for a
        # denoising loop, one `on_step` per iteration (cancellation included).
        on_step = tel.step_callback((remaining + 24) // 25, stage="tile_steps")
        index = 0
        while remaining > 0:
            step = min(remaining, 25)
            time.sleep(step / 1_000)
            remaining -= step
            on_step(index)
            index += 1
    state = payload.seed & 0xFFFFFFFF
    with tel.stage("fill_pixels"):
        pixels = bytearray(side * side * 3)
        for i in range(0, len(pixels), 3):
            ctx.raise_if_cancelled()
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            pixels[i] = (state >> 24) & 0xFF
            pixels[i + 1] = (state >> 16) & 0xFF
            pixels[i + 2] = (state >> 8) & 0xFF
    tel.progress(1.0, stage="tile")
    tel.metric("tile_bytes", len(pixels))
    return TileOutput(
        image=out.save_image(ImageFrame(side, side, bytes(pixels)), format="webp"),
        size=side,
        pixels=side * side,
        digest=hashlib.sha256(pixels).hexdigest(),
        warm=ctx.boot_warmup,
        revision=REVISION,
    )


@app.job
def tile_job(payload: TileInput) -> TileJobOutput:
    """Weightless remote-job fixture; its value proves frozen local revision replay."""
    return TileJobOutput(size=payload.size, seed=payload.seed, revision=REVISION)


@app.entrypoint
def refuse(payload: RefuseInput) -> TileOutput:
    """The REFUSED terminal on the same weightless path, so the fixture observes both
    verdicts of the terminal transaction rather than only the happy one."""
    raise InvalidRequest(payload.why)


@app.entrypoint
def fail(payload: RefuseInput) -> TileOutput:
    """The FAILED terminal: an exception the handler never typed. Its traceback is what a
    triage bundle exists to carry off a dead pod (cl-101)."""
    raise RuntimeError(payload.why)


@app.entrypoint
def relay(
    payload: RelayInput, ctx: Context, decoder: MediaDecoder, out: Outputs
) -> RelayOutput:
    """CPU-only exact workflow handoff: decode the prior accepted image and save it again."""
    remaining = payload.delay_ms
    if remaining > 0:
        # cl-104: the delay is a MEASURED step loop — the fixture's stand-in for a
        # denoising loop, one `on_step` per iteration (cancellation included).
        on_step = tel.step_callback((remaining + 24) // 25, stage="tile_steps")
        index = 0
        while remaining > 0:
            step = min(remaining, 25)
            time.sleep(step / 1_000)
            remaining -= step
            on_step(index)
            index += 1
    image = decoder.decode_image(payload.image)
    return RelayOutput(
        image=out.save_image(ImageFrame(image.width, image.height, image.rgb), format="webp")
    )


@app.entrypoint
def video_transport(payload: VideoTransportInput) -> VideoTransportOutput:
    """A package-descriptor-only seam for the remote byte-plane product proof.

    The fake remote worker consumes this request under an exact DeliveryGrant and emits a
    tiny MP4-shaped transport fixture. This handler deliberately refuses if somebody runs
    it locally: it is not video generation and must never be presented as H3 inference.
    """
    del payload
    raise InvalidRequest("video_transport is a transport-contract fixture, not inference")
