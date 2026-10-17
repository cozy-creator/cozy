"""One output of `size` bytes, for reads that take their time."""
from __future__ import annotations

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
