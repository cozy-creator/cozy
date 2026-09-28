# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.24,<1", "tensorfs>=0.3.51,<0.4", "numpy>=1.26,<3", "tensor-quant-source>=0.1.0,<0.2"]
# [tool.uv.sources]
# tensor-quant-source = {path = "./quant_source", editable = true}
# ///
from cozy_runtime.author import ModelArtifact
from cozy_runtime.derive.operations import QuantizationPlan, quantize
from tensor_quant_source import compute

PLAN = QuantizationPlan(components=("text_encoder",), keys=("a.weight", "b.weight"), output_precision="preserve")

async def main() -> ModelArtifact:
    source = await compute()
    return await quantize(source=source, plan=PLAN, encoding="fp8-rowwise/1")
