#!/usr/bin/env python3
"""Tiny upstream Comfy UNet/library forward through the public TensorFS API.

This does not run a ComfyUI graph, server, GUI, sampler or visual workflow. Supply
an unmodified upstream checkout at COMFY_COMMIT and an installed TensorFS wheel.
CPU is the default; --device cuda:N requires the coherent CUDA native wheel.
Requires a literal-file Writer wheel. Supply --native-source, --wheel and
--wheel-sha256 from its build receipt; the installed package and native extension
must match that exact wheel without a source overlay. CUDA is not qualified here.
Hosted dependencies: torch 2.14.0, numpy, einops, safetensors, Pillow, scipy, packaging,
tqdm, psutil, pyyaml, comfy-kitchen==0.2.39 and comfy-aimdo==0.5.6. The last two
versions match the pinned upstream requirements. No pretrained model is loaded.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
from email.parser import BytesParser
from importlib import machinery, metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import weakref
import zipfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import tensorfs
from tensorfs import _ext

COMFY_COMMIT = "35f6d4f8bdefe6eea17bfc4631bba8c7f5a66360"
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI"


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


def upstream(root, device):
    root = root.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMFY_COMMIT:
        raise RuntimeError(f"expected upstream Comfy {COMFY_COMMIT}, found {commit}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True
    )
    if changed:
        raise RuntimeError("the pinned upstream Comfy checkout is not clean")
    sys.path.insert(0, str(root))
    # Configure upstream's actual CLI options before its model imports initialize devices.
    # Keep its processed defaults (including args.fast) intact.
    import comfy.cli_args
    comfy.cli_args.args.cpu = device.type == "cpu"
    comfy.cli_args.args.disable_dynamic_vram = True
    comfy.cli_args.args.disable_xformers = True
    comfy.cli_args.args.use_pytorch_cross_attention = True
    from comfy.ldm.modules.diffusionmodules import openaimodel
    import comfy.ops
    Path(openaimodel.__file__).resolve().relative_to(root)
    Path(comfy.ops.__file__).resolve().relative_to(root)
    return openaimodel.UNetModel, comfy.ops.manual_cast


def prove(root, comfy_root, device_name):
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
    model_class, operations = upstream(comfy_root, device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261019)
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

    config = {
        "image_size": 8, "in_channels": 4, "model_channels": 32, "out_channels": 4,
        "num_res_blocks": 1, "channel_mult": [1, 2], "num_heads": 4,
        "use_spatial_transformer": True, "context_dim": 32,
        "transformer_depth": [0, 0], "transformer_depth_output": [0, 0, 0, 0],
        "transformer_depth_middle": 1,
    }
    with phase("producer_init"):
        model = model_class(**config, dtype=torch.float32, device=device, operations=operations).eval()
        # Real Comfy operations disable initialization because production loads checkpoints.
        # Fill every parameter so this generated model has a finite, nontrivial reference.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.05, 0.05)
        del parameter
    state = model.state_dict()
    assert all(t.device == device and t.is_contiguous() for t in state.values())
    tensor_count = len(state)
    nbytes = sum(t.numel() * t.element_size() for t in state.values())
    producer = weakref.ref(model)
    producer_tensors = [weakref.ref(t) for t in state.values()]
    sample = torch.randn(1, 4, 8, 8, device=device)
    timestep = torch.tensor([5], device=device)
    context = torch.randn(1, 4, 32, device=device)
    with phase("reference_forward"), torch.inference_mode():
        expected = model(sample, timestep, context=context, transformer_options={}).detach().clone()
    assert torch.isfinite(expected).all().item() and expected.abs().max().item() > 0
    cls = f"{model_class.__module__}.{model_class.__name__}"
    origin = f"{COMFY_REPO}/tree/{COMFY_COMMIT}\n".encode()
    header, cache = root / "model.cozytensors", root / "cache"
    with phase("writer"), tensorfs.Writer(header, cache_dir=cache) as writer:
        component = writer.component("unet", cls=cls, config=config)
        component.asset("upstream.txt", origin)
        for key, tensor in state.items():
            component.put(key, tensor)
    committed = writer.commit()
    header_id = committed.id
    committed.close()
    assert header.is_file(), "Writer(dest) did not write the literal header file"
    assert set(root.iterdir()) == {header, cache}, "Writer created adjacent payload files"
    with phase("open"):
        checkpoint = tensorfs.open(header, cache_dir=cache, local_files_only=True)
    assert checkpoint.id == header_id and list(checkpoint.components) == ["unet"]
    component = checkpoint.components["unet"]
    assert component.cls == cls and component.config == config
    assert component.asset("upstream.txt") == origin
    with phase("read_all"):
        arrays = component.tensors.read(device=device_name, decode=True)
    assert set(arrays) == set(state)
    with phase("dlpack_assign_verify"):
        loaded = {key: torch.from_dlpack(array) for key, array in arrays.items()}
        for key, tensor in loaded.items():
            original = state[key]
            assert tensor.device == device and tensor.dtype == original.dtype and tensor.shape == original.shape
            assert torch.equal(tensor.detach().reshape(-1).view(torch.uint8),
                               original.detach().reshape(-1).view(torch.uint8)), f"weight bytes changed: {key}"
        del original
        restored = model_class(**component.config, dtype=torch.float32, device="meta", operations=operations)
        assert all(p.device.type == "meta" for p in restored.parameters())
        restored.load_state_dict(loaded, assign=True, strict=True)
        restored.eval()
        for key, tensor in restored.state_dict().items():
            assert tensor.data_ptr() == loaded[key].data_ptr(), f"assign copied {key}"
    checkpoint.close()
    del checkpoint, component, arrays, loaded, committed, writer, tensor, state, model
    # Native producer cleanup is deferred; require it to finish before the next forward.
    deadline = perf_counter() + 10
    while True:
        gc.collect()
        if producer() is None and all(t() is None for t in producer_tensors):
            break
        assert perf_counter() < deadline, "the producer remained alive after every TensorFS handle was released"
        sleep(0.01)
    with phase("restored_forward"), torch.inference_mode():
        actual = restored(sample, timestep, context=context, transformer_options={})
    tolerance = 0 if device.type == "cpu" else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    output_hash = hashlib.sha256(actual.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    expected_hash = hashlib.sha256(expected.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if device.type == "cpu":
        assert output_hash == expected_hash, "CPU forward output bytes changed"
    return {
        "proof": f"public-tensorfs-comfy-unet-{device.type}",
        "scope": "tiny real Comfy model/library forward; no graph/GUI/visual qualification",
        "comfy_source": COMFY_COMMIT, "comfy_repo": COMFY_REPO, "model_class": cls,
        "operations": "comfy.ops.manual_cast", "device": device_name, "device_info": device_info,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "tensorfs": tensorfs.__version__,
        "comfy_kitchen": metadata.version("comfy-kitchen"), "comfy_aimdo": metadata.version("comfy-aimdo"),
        "header": header_id, "tensors": tensor_count, "bytes": nbytes,
        "literal_header_file": True, "header_bytes": header.stat().st_size,
        "payload_location": "configured shared store; no adjacent payload files",
        "decode": True, "weights_bit_exact": True,
        "output_shape": list(actual.shape), "max_abs_error": (actual - expected).abs().max().item(),
        "output_sha256": output_hash, "expected_output_sha256": expected_hash,
        "tolerance": {"rtol": tolerance, "atol": tolerance},
        "assigned_without_copy": True, "survives_checkpoint_close": True,
        "reference_computed_before_write": True, "producer_released_before_forward": True,
        "tensorfs_handles_released_before_forward": True, "gpu_qualified": False,
        "gpu_origin_writer": device.type == "cuda", "timings_ms": timings,
        "timing_scope": "one acceptance run; wall clock synchronized on the selected device",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True, help="unmodified pinned upstream Git checkout")
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
    with TemporaryDirectory(prefix="tensorfs-comfy-unet-") as temp:
        receipt = prove(Path(temp), args.comfy_root, args.device)
    receipt.update(provenance,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(receipt, sort_keys=True))
