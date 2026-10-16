"""Outputs for Calcifer's export tests: one rewritten in place, a list, a big file and a tree."""
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, Tree

app = App()

Video = Annotated[FileAsset, AssetBound(max_bytes=64 << 20, media_types=("video/mp4",))]


class Ask(msgspec.Struct):
    revisions: int = 3
    #: The big output's length in bytes.
    big: int = 20 << 20
    #: After its first revision the run waits until this file exists.
    gate: str = ""


class Made(msgspec.Struct):
    video: Video
    frames: list[Video]
    big: Video
    tree: Tree


def film(segments: int) -> bytes:
    return b"".join(f"segment {i}\n".encode() for i in range(1, segments + 1))


async def make(ctx: Context, payload: Ask, out: Outputs) -> Made:
    frames = []
    for revision in range(1, payload.revisions + 1):
        frames.append(out.save_bytes(f"frame {revision}\n".encode(), media_type="video/mp4"))
        out.publish("frames", frames[-1], label=f"Frame {revision}")
        video = out.save_bytes(film(revision), media_type="video/mp4")
        out.publish("video", video, label=f"Segments 1-{revision}")
        while revision == 1 and payload.gate and not Path(payload.gate).exists():
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.05)
    big = out.save_bytes(random.Random(7).randbytes(payload.big), media_type="video/mp4")
    root = out.temporary_file()
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "a.txt").write_text("same")
    (root / "b.txt").write_text("same")
    (root / "c.txt").write_text("other")
    return Made(video=video, frames=frames, big=big, tree=out.save_tree(root))


app.entrypoint(make, emits_media=True, accelerator=False)
