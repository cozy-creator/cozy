"""Ordinary library operations used by the native artifact product proof."""
import hashlib
from typing import Annotated

import msgspec
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, Tree, invocable

ReportFile = Annotated[FileAsset, AssetBound(max_bytes=200000, media_types=("application/json",))]


class Produced(msgspec.Struct):
    report: ReportFile
    tree: Tree


class Verified(msgspec.Struct):
    digest: str
    length: int


@invocable(memoize=True)
async def produce(ctx: Context, *, variant: str, out: Outputs) -> Produced:
    data = msgspec.json.encode({"variant": variant, "observations": "verified" * 12000})
    root = out.temporary_file()
    root.mkdir()
    (root / "report.json").write_bytes(data)
    (root / "extra.txt").write_text(variant)
    return Produced(out.save_bytes(data, media_type="application/json"), out.save_tree(root))


@invocable()
async def verify(ctx: Context, *, report: ReportFile, tree: Tree) -> Verified:
    raw = report.read_bytes()
    assert len(raw) > 48 * 1024
    assert raw == (tree.path / "report.json").read_bytes()
    assert (tree.path / "extra.txt").read_text() == msgspec.json.decode(raw)["variant"]
    return Verified(hashlib.sha256(raw).hexdigest(), len(raw))


app = App()
app.job(produce)
app.job(verify)
