#!/usr/bin/env python3
"""Exercise the public TensorFS API with a real, tiny Diffusers UNet.

CPU is the default; --device cuda:N requires a CUDA-enabled wheel and actual GPU.
Use torch 2.14.0 and diffusers 0.40.0, matching the Runtime lock. No pretrained
model or Runtime is loaded. Supply --native-source and --wheel for a pinned receipt.

The consumer follows Diffusers' UNet2DConditionModel.from_config and PyTorch's
meta-device + load_state_dict(assign=True) recipes:
https://huggingface.co/docs/diffusers/v0.40.0/en/api/models/unet2d-cond
https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
import weakref

# Keep this local-artifact proof offline even when the caller has Hub credentials.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import diffusers
import torch
from diffusers import UNet2DConditionModel

import tensorfs


def prove(root, device_name="cpu"):
    device = torch.device(device_name)
    device_info = {"type": device.type}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device}: this proof requires an available CUDA device")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        device_info.update(name=props.name, total_memory=props.total_memory,
                           compute_capability=[props.major, props.minor],
                           uuid=str(getattr(props, "uuid", "unavailable")))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    timings = {}

    @contextmanager
    def phase(name):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[name] = (perf_counter() - start) * 1000

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261019)
    with phase("producer_init"), torch.device(device):
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
    assert all(t.device == device and t.is_contiguous() for t in state.values())
    tensor_count = len(state)
    nbytes = sum(t.numel() * t.element_size() for t in state.values())
    producer = weakref.ref(model)
    producer_tensors = [weakref.ref(t) for t in state.values()]
    sample = torch.randn(1, 4, 8, 8, device=device)
    conditioning = torch.randn(1, 4, 16, device=device)
    # Keep an independent reference from before TensorFS sees any source tensor.
    with phase("reference_forward"), torch.inference_mode():
        expected = model(sample, 5, encoder_hidden_states=conditioning).sample.detach().clone()
    config = json.loads(json.dumps(dict(model.config)))
    pipeline_config = {"unet": ["diffusers", "UNet2DConditionModel"]}
    asset = f"Locally generated {device.type.upper()} Diffusers UNet; no pretrained weights.\n".encode()
    folder, cache = root / "unet", root / "cache"
    with phase("writer"), tensorfs.Writer(folder, cache_dir=cache) as writer:
        writer.pipeline("diffusers.DiffusionPipeline", pipeline_config)
        component = writer.component("unet", cls="diffusers.UNet2DConditionModel", config=config)
        component.asset("proof.txt", asset)
        for key, tensor in state.items():
            component.put(key, tensor)
    committed = writer.commit()
    header_id = committed.id
    committed.close()
    with phase("open"):
        checkpoint = tensorfs.open(folder, cache_dir=cache)
    assert checkpoint.id == header_id
    assert checkpoint.pipeline.cls == "diffusers.DiffusionPipeline"
    assert checkpoint.pipeline.config == pipeline_config
    assert list(checkpoint.components) == ["unet"]
    component = checkpoint.components["unet"]
    assert component.cls == "diffusers.UNet2DConditionModel"
    assert component.config == config
    assert component.asset("proof.txt") == asset
    assert component.asset("proof.txt") == asset
    assert component.config == config
    with phase("read_all"):
        arrays = component.tensors.read(device=device_name)
    assert set(arrays) == set(state)
    with phase("dlpack_assign_verify"):
        loaded = {key: torch.from_dlpack(array) for key, array in arrays.items()}
        for key, tensor in loaded.items():
            assert tensor.device == device
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
    del checkpoint, component, arrays, loaded, committed, writer, tensor, state, model
    gc.collect()
    assert producer() is None and all(t() is None for t in producer_tensors)
    with phase("restored_forward"), torch.inference_mode():
        actual = restored(sample, 5, encoder_hidden_states=conditioning).sample
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    return {
        "proof": f"public-tensorfs-diffusers-unet-{device.type}",
        "device": device_name,
        "device_info": device_info,
        "torch_cuda": torch.version.cuda,
        "tensorfs": tensorfs.__version__,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "header": header_id,
        "tensors": tensor_count,
        "bytes": nbytes,
        "output_shape": list(actual.shape),
        "max_abs_error": (actual - expected).abs().max().item(),
        "assigned_without_copy": True,
        "survives_checkpoint_close": True,
        "reference_computed_before_write": True,
        "producer_released_before_forward": True,
        "gpu_origin_writer": device.type == "cuda",
        "timings_ms": timings,
        "timing_scope": "one acceptance run; wall clock synchronized on the selected device",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu (default) or an explicit cuda:N")
    parser.add_argument("--native-source", help="exact Creator commit used to build the installed wheel")
    parser.add_argument("--wheel", type=Path, default=os.environ.get("COZY_TEST_TENSORFS_WHEEL"))
    args = parser.parse_args()
    if not re.fullmatch(r"cpu|cuda:[0-9]+", args.device):
        parser.error("--device must be cpu or cuda:N with an explicit nonnegative device index")
    with TemporaryDirectory(prefix="tensorfs-diffusers-") as temp:
        receipt = prove(Path(temp), args.device)
    receipt.update(native_source=args.native_source,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.wheel is not None:
        receipt.update(wheel_name=args.wheel.name,
                       wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest())
    print(json.dumps(receipt, sort_keys=True))
