#!/usr/bin/env python3
"""Exercise the public TensorFS API with a real, tiny Diffusers UNet.

CPU is the default; --device cuda:N requires a CUDA-enabled wheel and actual GPU.
Use torch 2.14.0 and diffusers 0.40.0, matching the Runtime lock. No pretrained
model or Runtime is loaded. Requires a coherent literal-file Writer wheel. Supply
--native-source, --wheel and --wheel-sha256 from its build receipt; installed package
bytes (including the native extension) must match that exact wheel, without overlays.

The consumer follows Diffusers' UNet2DConditionModel.from_config and PyTorch's
meta-device + load_state_dict(assign=True) recipes:
https://huggingface.co/docs/diffusers/v0.40.0/en/api/models/unet2d-cond
https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib import machinery, metadata
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
import weakref
import zipfile
from email.parser import BytesParser

# Keep this local-artifact proof offline even when the caller has Hub credentials.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import diffusers
import torch
from diffusers import UNet2DConditionModel

import tensorfs
from tensorfs import _ext


def installed_provenance(wheel, expected_sha256, native_source):
    with wheel.open("rb") as source:
        wheel_hash = hashlib.file_digest(source, "sha256").hexdigest()
    assert wheel_hash == expected_sha256, (wheel_hash, expected_sha256)
    package = Path(tensorfs.__file__).resolve().parent
    distribution = metadata.distribution("tensorfs")
    assert package == Path(distribution.locate_file("tensorfs")).resolve(), package
    native = Path(_ext.__file__).resolve()
    assert isinstance(_ext.__loader__, machinery.ExtensionFileLoader), native
    assert native.parent == package, native
    installed = {}
    with zipfile.ZipFile(wheel) as archive:
        manifests = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(manifests) == 1, manifests
        info = BytesParser().parsebytes(archive.read(manifests[0]))
        assert info["Name"] == "tensorfs", info["Name"]
        assert info["Version"] == distribution.version == tensorfs.__version__
        for name in archive.namelist():
            if not name.startswith("tensorfs/") or name.endswith("/"):
                continue
            relative = Path(name).relative_to("tensorfs")
            assert ".." not in relative.parts, name
            path = (package / relative).resolve()
            assert path.is_relative_to(package), path
            with path.open("rb") as local, archive.open(name) as shipped:
                actual = hashlib.file_digest(local, "sha256").hexdigest()
                expected = hashlib.file_digest(shipped, "sha256").hexdigest()
            assert actual == expected, f"installed file differs from wheel: {name}"
            installed[name] = actual
    assert f"tensorfs/{native.name}" in installed, native
    stamp = re.search(r"\+dev\.g([0-9a-f]{8,40})(?:\.|$)", distribution.version)
    if stamp:
        assert native_source.startswith(stamp[1]), (native_source, distribution.version)
    return {
        "native_source": native_source,
        "source_evidence": "caller build receipt; dev version checked when stamped",
        "wheel_name": wheel.name,
        "wheel_sha256": wheel_hash,
        "installed_tensorfs_package": str(package),
        "native_extension": str(native),
        "installed_file_sha256": installed,
        "installed_package_matches_wheel": True,
    }


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
    assert torch.isfinite(expected).all() and expected.abs().max().item() > 0
    config = json.loads(json.dumps(dict(model.config)))
    pipeline_config = {"unet": ["diffusers", "UNet2DConditionModel"]}
    asset = f"Locally generated {device.type.upper()} Diffusers UNet; no pretrained weights.\n".encode()
    header, store = root / "model.cozytensors", root / "store"
    os.environ["TENSORFS_HOME"] = str(store)
    with phase("writer"), tensorfs.Writer(header) as writer:
        writer.pipeline("diffusers.DiffusionPipeline", pipeline_config)
        component = writer.component("unet", cls="diffusers.UNet2DConditionModel", config=config)
        component.asset("proof.txt", asset)
        for key, tensor in state.items():
            component.put(key, tensor)
    committed = writer.commit()
    header_id = committed.id
    committed.close()
    assert header.is_file(), "Writer(dest) did not create a literal header file"
    assert set(root.iterdir()) == {header, store}, "Writer created adjacent payload files"
    with phase("open"):
        checkpoint = tensorfs.open(header, local_files_only=True)
    assert checkpoint.id == header_id
    assert checkpoint.pipeline.cls == "diffusers.DiffusionPipeline"
    assert checkpoint.pipeline.config == pipeline_config
    assert list(checkpoint.components) == ["unet"]
    component = checkpoint.components["unet"]
    assert component.cls == "diffusers.UNet2DConditionModel"
    assert component.config == config
    assert component.asset("proof.txt") == asset
    with phase("read_all"):
        arrays = component.tensors.read(device=device_name, decode=True)
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
    tolerance = 0 if device.type == "cpu" else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    return {
        "proof": f"public-tensorfs-diffusers-unet-{device.type}",
        "device": device_name,
        "device_info": device_info,
        "torch_cuda": torch.version.cuda,
        "tensorfs": tensorfs.__version__,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "header": header_id,
        "literal_header_file": True,
        "payload_location": "TENSORFS_HOME; no adjacent payload files",
        "header_bytes": header.stat().st_size,
        "decode": True,
        "tensors": tensor_count,
        "bytes": nbytes,
        "output_shape": list(actual.shape),
        "max_abs_error": (actual - expected).abs().max().item(),
        "output_sha256": hashlib.sha256(actual.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "tolerance": {"rtol": tolerance, "atol": tolerance},
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
    parser.add_argument("--native-source", required=True, help="full Creator SHA from the wheel build receipt")
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True, help="expected wheel hash from the build receipt")
    args = parser.parse_args()
    if not re.fullmatch(r"cpu|cuda:[0-9]+", args.device):
        parser.error("--device must be cpu or cuda:N with an explicit nonnegative device index")
    if not re.fullmatch(r"[0-9a-f]{40}", args.native_source):
        parser.error("--native-source must be a full lowercase 40-character commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.wheel_sha256):
        parser.error("--wheel-sha256 must be a lowercase SHA-256 digest")
    provenance = installed_provenance(args.wheel, args.wheel_sha256, args.native_source)
    with TemporaryDirectory(prefix="tensorfs-diffusers-") as temp:
        receipt = prove(Path(temp), args.device)
    receipt.update(provenance,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(receipt, sort_keys=True))
