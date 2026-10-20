"""CPU seam oracle: exact bands when a complete device result cannot be allocated.

The admitted destination is a CPU stand-in. Actual CUDA mapping, stream ownership and
native scatter are qualified separately; this checks the production cut and its bridge.
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from rmsnorm_bands import source_module, upstream_class


class Exhausted(Exception):
    pass


class Destination:
    def __init__(self, like):
        with _disable_current_modes():
            self.tensor = torch.empty_like(like, memory_format=torch.contiguous_format)
        self.writes = []

    def write_band(self, source, axis, offset):
        self.writes.append((axis, offset, tuple(source.shape)))
        with _disable_current_modes():
            self.tensor.narrow(axis, offset, source.shape[axis]).copy_(source)


class Room:
    def __init__(self, admit=True):
        self.admit = admit
        self.events, self.outputs = [], []

    def available(self, device):
        return 0

    def free(self, device, need):
        self.events.append("free")
        return False

    def lower(self, device, need):
        self.events.append("lower")
        return False

    def exhausted(self, device, need, name):
        return Exhausted(f"{name} needs {need} B")

    def mapped_output(self, like, device):
        assert self.events and self.events[-1] == "lower", "host bypassed device relief"
        self.events.append("mapped")
        if not self.admit:
            return None
        output = Destination(like)
        self.outputs.append(output)
        return output


class Bound(TorchDispatchMode):
    def __init__(self, shape, *, deny_output=True):
        super().__init__()
        self.shape, self.deny_output = tuple(shape), deny_output
        self.full, self.arithmetic = 0, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided.default):
            if self.deny_output and tuple(args[0]) == self.shape:
                self.full += 1
                raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
        if func is torch.ops.aten.mul.Tensor:
            count = max((x.numel() for x in args if isinstance(x, torch.Tensor)), default=0)
            if count > torch.Size(self.shape).numel() // 3:
                self.arithmetic += 1
                raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
        return func(*args, **(kwargs or {}))


def rms(cuts, cls, shape, first, images, dtype, bias, *, admit=True, deny_output=True):
    channels = shape[1] if first else shape[-1]
    module = cls(channels, channel_first=first, images=images, bias=bias).to(dtype=dtype)
    value = torch.randn(shape, dtype=dtype)
    pointer = weakref.ref(value)
    with torch.no_grad():
        expected = module(value)
    assert cuts.install(module) == 1
    room, memo = Room(admit), cuts.Memo()
    cuts.serve(memo, room)
    with torch.no_grad(), Bound(shape, deny_output=deny_output) as bound, cuts.caught():
        try:
            actual = module(value)
        except Exhausted:
            assert deny_output and not admit, "admitted mapped output still exhausted"
            assert room.events.count("mapped") == 1
            return
    assert admit or not deny_output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == value.shape and actual.dtype == dtype
    assert bound.arithmetic > 0
    if deny_output:
        assert bound.full > 0 and len(room.outputs) == 1
        output = room.outputs[0]
        assert actual is output.tensor and len(output.writes) > 1
        assert sum(shape_[axis] for axis, _, shape_ in output.writes) == shape[output.writes[0][0]]
    else:
        assert not room.outputs and "mapped" not in room.events
    del value
    gc.collect()
    assert pointer() is None, "mapped result memo retained its input"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "python/tensorfs/varena/cuts.py"
    cuts = source_module(path)
    torch.manual_seed(119)
    cases = 0
    for filename, name in [("autoencoder_kl_qwenimage21", "QwenImage21RMS_norm"),
                           ("autoencoder_kl_wan", "WanRMS_norm")]:
        cls = upstream_class(filename, name)
        for shape, first, images in [((2, 7, 5, 11), True, True),
                                     ((2, 7, 2, 5, 11), True, False),
                                     ((2, 5, 11, 7), False, True)]:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for bias in (False, True):
                    rms(cuts, cls, shape, first, images, dtype, bias)
                    cases += 1
        rms(cuts, cls, (2, 7, 5, 11), True, True, torch.float32, False, admit=False)
        rms(cuts, cls, (2, 7, 5, 11), True, True, torch.float32, False, deny_output=False)
    print(f"PASS mapped-output CPU seam: {cases} exact upstream RMS cases, denied admission, "
          "device-first allocation and no retained input; native CUDA feasibility is separate")


if __name__ == "__main__":
    main()
