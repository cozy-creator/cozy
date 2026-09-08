# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==__VERSION__", "model-tools==0.0.1"]
# [tool.uv.sources]
# cozy-runtime = {path=__WHEEL__}
# model-tools = {path="./model_tools"}
# ///
from cozy_runtime.author import ActivationCapture
from model_tools import generate, helper, produce

async def main(ctx):
    model = await produce()
    first = generate(seed=helper(), steps=2, model=model,
        capture=ActivationCapture(components=("alpha", "zeta"), steps=(0, 1)))
    baseline = await first
    repeated = generate(seed=helper(), steps=2, model=model)
    candidate = await repeated
    assert baseline.value == candidate.value
    assert baseline.cpu_rng == candidate.cpu_rng
    assert baseline.cuda_rng == candidate.cuda_rng
    assert baseline.next_noise == candidate.next_noise
    assert first.request_id != repeated.request_id
    assert first.observation is not None and first.observation.capture is not None
    assert first.observation.capture.tree is not None
    assert first.observation.capture.tree.path.is_dir()
    ctx.log(f"Serving with held native model: {baseline.value}")
