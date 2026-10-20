"""CPU oracle for bounded RMS normalization using unchanged Diffusers forwards."""

from __future__ import annotations

import ast
import copy
import gc
import importlib.metadata
import importlib.util
import sys
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode


def source_module(path: Path):
    spec = importlib.util.spec_from_file_location("rmsnorm_bands_cuts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def upstream_class(filename: str, name: str):
    """Compile the installed class verbatim, without unrelated pipeline imports."""
    distribution = importlib.metadata.distribution("diffusers")
    path = Path(distribution.locate_file(f"diffusers/models/autoencoders/{filename}.py"))
    parsed = ast.parse(path.read_text())
    definition = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = {"__name__": f"diffusers.models.autoencoders.{filename}", "torch": torch, "nn": nn, "F": F}
    # The pinned installed ClassDef has the same trust boundary as importing its module.
    exec(compile(ast.Module([definition], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


class Room:
    def available(self, device):
        return 0  # Unknown room: the existing cut ladder halves the band.

    def free(self, device, need):
        return False

    def lower(self, device, need):
        return False

    def exhausted(self, device, need, name):
        return torch.OutOfMemoryError(f"RMS oracle exhausted: {name} needs {need} B")


class BoundedMultiply(TorchDispatchMode):
    """The actual failed aten.mul seam, with a deterministic CPU allocation bound."""

    def __init__(self, limit):
        super().__init__()
        self.limit, self.failures, self.largest = limit, 0, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mul.Tensor:
            count = max((value.numel() for value in args if isinstance(value, torch.Tensor)), default=0)
            if count > self.limit:
                self.failures += 1
                raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
            self.largest = max(self.largest, count)
        return func(*args, **(kwargs or {}))


def prove(cuts, cls, shape, dtype, channel_first, images, bias):
    channels = shape[1] if channel_first else shape[-1]
    reference = cls(channels, channel_first=channel_first, images=images, bias=bias).to(dtype=dtype)
    with torch.no_grad():
        reference.gamma.uniform_(0.5, 1.5)
        if isinstance(reference.bias, torch.Tensor):
            reference.bias.uniform_(-0.1, 0.1)
    module = copy.deepcopy(reference)
    installed = cuts.install(module)
    assert cuts.install(module) == 0, "normalization was wrapped twice"
    memo = cuts.Memo()
    cuts.serve(memo, Room())
    measured = memo.shape("rms", str((shape, dtype, channel_first)))
    site = cuts.Site(("rms", 1), torch.device("cpu"), cuts.Track("rms", measured))
    previous, cuts._local.site = cuts._local.site, site
    try:
        value = torch.randn(shape, dtype=dtype)
        pointer = weakref.ref(value)
        with torch.no_grad():
            expected = reference(value)
            with BoundedMultiply(value.numel() // 3) as bound, cuts.caught():
                actual = module(value)
        assert installed == 1 and bound.failures > 0
        assert 0 < bound.largest <= bound.limit
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(measured.scopes) == 1
        del value
        gc.collect()
        assert pointer() is None, "RMS shrink memo retained its activation"
        assert cuts.finer(measured), "RMS cut could not become finer after its input was collected"
        return bound.failures
    finally:
        cuts._local.site = previous


def unsupported(cuts, cls):
    spatial_affine = cls(7, channel_first=True, images=True)
    spatial_affine.gamma = nn.Parameter(torch.ones(7, 2, 1))
    assert cuts.install(spatial_affine) == 0, "spatial affine escaped the channel-only contract"
    module = cls(7, channel_first=True, images=True)
    reference = copy.deepcopy(module)
    cuts.install(module)
    value = torch.randn(2, 7, 5, 11).transpose(-1, -2)
    assert not value.is_contiguous()
    with torch.no_grad():
        torch.testing.assert_close(module(value), reference(value), rtol=0, atol=0)
        try:
            with BoundedMultiply(1), cuts.caught():
                module(value)
        except torch.OutOfMemoryError:
            pass
        else:
            raise AssertionError("unsupported layout silently entered a banded RMS path")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "python/tensorfs/varena/cuts.py"
    cuts = source_module(path)
    torch.manual_seed(73)
    cases, failures = 0, 0
    classes = [upstream_class("autoencoder_kl_qwenimage21", "QwenImage21RMS_norm"),
               upstream_class("autoencoder_kl_wan", "WanRMS_norm")]
    for cls in classes:
        for shape, channel_first, images in [((2, 7, 5, 11), True, True),
                                           ((2, 7, 2, 5, 11), True, False),
                                           ((2, 5, 11, 7), False, True)]:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for bias in (False, True):
                    failures += prove(cuts, cls, shape, dtype, channel_first, images, bias)
                    cases += 1
        unsupported(cuts, cls)
    print(f"PASS RMS normalization: {cases} exact upstream CPU cases, {failures} forced mul OOMs, "
          "both channel layouts, 4D/5D, bias, unsupported strides and tensor-free memo")


if __name__ == "__main__":
    main()
