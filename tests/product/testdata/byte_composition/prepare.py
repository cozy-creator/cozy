# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime==__VERSION__", "byte-tools==0.0.1"]
# [tool.uv.sources]
# cozy-runtime={path=__WHEEL__}
# byte-tools={path="./byte_tools",editable=true}
# ///
from byte_tools import produce, verify


async def main(ctx):
    artifact = await produce(variant="reviewed")
    assert artifact.report.read_bytes() == (artifact.tree.path / "report.json").read_bytes()
    checked = await verify(report=artifact.report, tree=artifact.tree)
    ctx.log(f"read {checked.length} bytes with SHA256 {checked.digest}")
