# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.5.2", "tensor-source==0.1.0", "tensor-candidate==0.1.0"]
# [tool.uv.sources]
# tensor-source = {path = "./source"}
# tensor-candidate = {path = "./candidate"}
# ///
from tensor_source import compute as source
from tensor_candidate import compute as candidate


async def main():
    original = await source(value=7)
    await candidate(source=original, factor=0)
