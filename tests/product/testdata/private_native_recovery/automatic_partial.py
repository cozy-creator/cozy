# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.24,<1", "tensorfs>=0.3.51,<0.4", "tensor-source>=0.1.0,<0.2", "tensor-candidate>=0.1.0,<0.2"]
# [tool.uv.sources]
# tensor-source = {path = "./source", editable = true}
# tensor-candidate = {path = "./candidate", editable = true}
# ///
from cozy_runtime.author import ModelArtifact
from tensor_source import compute as source
from tensor_candidate import compute as candidate

async def main() -> ModelArtifact:
    original = await source(value=11)
    return await candidate(source=original, factor=2)
