#!/usr/bin/env python3
"""Exercise the public TensorFS API with a real, tiny CPU Diffusers UNet.

Run against an installed native TensorFS wheel, torch 2.14.0 (CPU), and diffusers
0.40.0, matching the Runtime lock. No pretrained model or Runtime is loaded.

The consumer follows Diffusers' UNet2DConditionModel.from_config and PyTorch's
meta-device + load_state_dict(assign=True) recipes:
https://huggingface.co/docs/diffusers/v0.40.0/en/api/models/unet2d-cond
https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
"""

import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Keep this local-artifact proof offline even when the caller has Hub credentials.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import diffusers
import torch
from diffusers import UNet2DConditionModel

import tensorfs


def prove(root):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261019)
    model = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=8,
    ).eval()
    state = model.state_dict()
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in state.values())
    sample = torch.randn(1, 4, 8, 8)
    conditioning = torch.randn(1, 4, 16)
    # Keep an independent reference from before TensorFS sees any source tensor.
    with torch.inference_mode():
        expected = model(sample, 5, encoder_hidden_states=conditioning).sample.detach().clone()
    config = json.loads(json.dumps(dict(model.config)))
    pipeline_config = {"unet": ["diffusers", "UNet2DConditionModel"]}
    asset = b"Locally generated CPU Diffusers UNet; no pretrained weights.\n"
    folder, cache = root / "unet", root / "cache"
    with tensorfs.Writer(folder, cache_dir=cache) as writer:
        writer.pipeline("diffusers.DiffusionPipeline", pipeline_config)
        component = writer.component("unet", cls="diffusers.UNet2DConditionModel", config=config)
        component.asset("proof.txt", asset)
        for key, tensor in state.items():
            component.put(key, tensor)
    committed = writer.commit()
    header_id = committed.id
    committed.close()
    checkpoint = tensorfs.open(folder, cache_dir=cache)
    assert checkpoint.id == header_id
    assert checkpoint.pipeline.cls == "diffusers.DiffusionPipeline"
    assert checkpoint.pipeline.config == pipeline_config
    assert list(checkpoint.components) == ["unet"]
    component = checkpoint.components["unet"]
    assert component.cls == "diffusers.UNet2DConditionModel"
    assert component.config == config
    assert component.asset("proof.txt") == asset
    assert (Path(component.path) / "proof.txt").read_bytes() == asset
    assert json.loads((Path(component.path) / "config.json").read_text()) == config
    arrays = component.tensors.read_all(device="cpu")
    assert set(arrays) == set(state)
    loaded = {key: torch.from_dlpack(array) for key, array in arrays.items()}
    for key, tensor in loaded.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0, msg=key)
    with torch.device("meta"):
        restored = UNet2DConditionModel.from_config(component.config)
    assert all(p.device.type == "meta" for p in restored.parameters())
    restored.load_state_dict(loaded, assign=True, strict=True)
    restored.eval()
    for key, tensor in restored.state_dict().items():
        assert tensor.data_ptr() == loaded[key].data_ptr(), f"assign copied {key}"
    # The assigned Torch tensors own their DLPack storage after every TensorFS handle closes.
    checkpoint.close()
    del checkpoint, component, arrays, loaded, committed, writer
    gc.collect()
    with torch.inference_mode():
        actual = restored(sample, 5, encoder_hidden_states=conditioning).sample
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    return {
        "proof": "public-tensorfs-diffusers-unet-cpu",
        "device": "cpu",
        "tensorfs": tensorfs.__version__,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "header": header_id,
        "tensors": len(state),
        "bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "output_shape": list(actual.shape),
        "max_abs_error": (actual - expected).abs().max().item(),
        "assigned_without_copy": True,
        "survives_checkpoint_close": True,
        "reference_computed_before_write": True,
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tensorfs-diffusers-cpu-") as temp:
        print(json.dumps(prove(Path(temp)), sort_keys=True))
