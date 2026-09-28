# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=__RUNTIME_VERSION__","quantize-tools>=0.0.1","score-tools>=0.0.1"]
# [tool.uv.sources]
# cozy-runtime={path=__RUNTIME_WHEEL__}
# quantize-tools={path="./quantize_tools"}
# score-tools={path="./score_tools"}
# ///
from cozy_runtime.author.sources import download_huggingface,convert_cozytensors
from quantize_tools import graft,quantize
from score_tools import score

async def main(ctx):
    source=await download_huggingface("example/model",revision="__REVISION__")
    original=await convert_cozytensors(source,profile="fixture/quantize/1")
    original=await graft(source=original)
    candidate=await quantize(source=original,encoding="fp8-rowwise/1")
    report=await score(model=candidate)
    ctx.log(f"Supplied fixture media SSIM={report.ssim}; checkpoint={report.checkpoint}")
    if report.ssim < 1.01:
        raise ValueError("qualification media gate rejected")
