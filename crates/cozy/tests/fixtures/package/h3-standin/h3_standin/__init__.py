"""minimax-h3's request shape without torch: `long_form` and `ref2va` take named references
(character and background images, a video, an audio) and answer an MP4 of one MiB per second
whose bytes follow from the prompts and every reference's file. The result names each file's
sha256 as the machine read it, so a test sees which file reached it."""

import asyncio
import hashlib
import random
import struct
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    AudioAsset,
    Context,
    FileAsset,
    ImageAsset,
    Outputs,
    VideoAsset,
)

app = App()

Video = Annotated[FileAsset, AssetBound(max_bytes=1 << 30, media_types=("video/mp4",))]
Seconds = Annotated[
    int,
    msgspec.Meta(ge=5, le=15, description="Required. Clip length in whole seconds, 5 to 15."),
]


class StoryReference(msgspec.Struct):
    """A supplied file, or a character or background described in words."""

    name: str
    kind: Literal["character", "background", "audio", "video"]
    description: str = ""
    image: Annotated[
        ImageAsset | None,
        AssetBound(max_bytes=64 << 20, media_types=("image/png", "image/jpeg", "image/webp")),
    ] = None
    audio: Annotated[AudioAsset | None, AssetBound(max_bytes=256 << 20)] = None
    video: Annotated[VideoAsset | None, AssetBound(max_bytes=256 << 20)] = None
    voice: Annotated[AudioAsset | None, AssetBound(max_bytes=256 << 20)] = None


class Seen(msgspec.Struct):
    """A reference as the machine read it: its file's sha256, or its description."""

    name: str
    kind: str
    sha256: str = ""
    description: str = ""


class LongFormSegment(msgspec.Struct, kw_only=True):
    prompt: str
    seed: int | None = None
    duration_s: Seconds


class LongFormInput(msgspec.Struct, kw_only=True):
    segments: Annotated[list[LongFormSegment], msgspec.Meta(min_length=1)]
    references: Annotated[list[StoryReference], msgspec.Meta(min_length=1)]
    style: str = ""
    #: Seconds each segment takes to render (H3's take minutes).
    render_s: float = 0.0


class Ref2VAInput(msgspec.Struct, kw_only=True):
    prompt: str
    seed: int | None = None
    duration_s: Seconds
    resolution: Literal["768p", "1080p", "1440p", "2160p"] = "768p"
    references: list[StoryReference] | None = None
    style: str = ""
    render_s: float = 0.0


class VideoOutput(msgspec.Struct):
    video: Video
    #: The video's sha256 as the machine wrote it.
    video_sha256: str
    references: list[Seen]
    warnings: list[str]


def _seen(references: list[StoryReference]) -> list[Seen]:
    out = []
    for ref in references:
        held = ref.image or ref.audio or ref.video or ref.voice
        digest = hashlib.sha256(held.read_bytes()).hexdigest() if held else ""
        out.append(Seen(ref.name, ref.kind, digest, ref.description))
    return out


def _clip(prompt: str, seed: int | None, seconds: int, style: str, seen: list[Seen]) -> bytes:
    key = hashlib.sha256(msgspec.json.encode([prompt, seed, style, seen])).digest()
    return random.Random(key).randbytes(seconds << 20)


def _mp4(body: bytes) -> bytes:
    """An `ftyp` box, then the clips in one `free` box."""
    ftyp = struct.pack(">I", 24) + b"ftypmp42" + b"\0\0\0\0" + b"mp42isom"
    return ftyp + struct.pack(">I", 8 + len(body)) + b"free" + body


async def _render(ctx: Context, seconds: float) -> None:
    for _ in range(int(seconds * 20)):
        await asyncio.sleep(0.05)
        ctx.raise_if_cancelled()


def _result(out: Outputs, film: bytes, seen: list[Seen]) -> VideoOutput:
    video = _mp4(film)
    return VideoOutput(
        out.save_bytes(video, media_type="video/mp4"),
        hashlib.sha256(video).hexdigest(),
        seen,
        [],
    )


async def long_form(ctx: Context, payload: LongFormInput, out: Outputs) -> VideoOutput:
    """Renders each segment in turn and publishes the film so far after each."""
    seen = _seen(payload.references)
    film = b""
    for index, segment in enumerate(payload.segments):
        await _render(ctx, payload.render_s)
        film += _clip(segment.prompt, segment.seed, segment.duration_s, payload.style, seen)
        so_far = out.save_bytes(_mp4(film), media_type="video/mp4")
        out.publish("video", so_far, label=f"Video (segments 1-{index + 1})")
    return _result(out, film, seen)


async def ref2va(ctx: Context, payload: Ref2VAInput, out: Outputs) -> VideoOutput:
    """One clip from its prompt and references."""
    seen = _seen(payload.references or [])
    await _render(ctx, payload.render_s)
    clip = _clip(payload.prompt, payload.seed, payload.duration_s, payload.style, seen)
    return _result(out, clip, seen)


app.entrypoint(long_form, emits_media=True, accelerator=False)
app.entrypoint(ref2va, emits_media=True, accelerator=False)
