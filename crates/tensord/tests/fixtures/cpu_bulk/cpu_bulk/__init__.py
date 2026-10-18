"""One output of `size` bytes, for reads that take their time; a call that takes its time; the
GPUs a call's process was given."""
from __future__ import annotations

import asyncio
import os

import msgspec

from cozy_runtime.author import App, Context, FileAsset, Outputs

app = App()


class Size(msgspec.Struct):
    size: int


class Written(msgspec.Struct):
    file: FileAsset


async def bulk(ctx: Context, *, payload: Size, out: Outputs) -> Written:
    block = bytes(range(256)) * 4096
    data = (block * (payload.size // len(block) + 1))[: payload.size]
    return Written(out.save_bytes(data, media_type="application/octet-stream"))


app.entrypoint(bulk, accelerator=False)


class Nap(msgspec.Struct):
    seconds: float


class Napped(msgspec.Struct):
    seconds: float


async def nap(ctx: Context, *, payload: Nap) -> Napped:
    await asyncio.sleep(payload.seconds)
    return Napped(payload.seconds)


app.entrypoint(nap, accelerator=False)


class Which(msgspec.Struct):
    label: str = ""


class Seen(msgspec.Struct):
    devices: str


async def devices(ctx: Context, *, payload: Which) -> Seen:
    return Seen(os.environ.get("CUDA_VISIBLE_DEVICES", ""))


app.entrypoint(devices)
