"""CPU seam oracle: exact bands when a complete device result cannot be allocated.

The admitted destination is a CPU stand-in. Actual CUDA mapping, stream ownership and
native scatter are qualified separately; this checks the production cut and its bridge.
"""

from __future__ import annotations

import ast
import gc
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol
from unittest.mock import patch

import torch
from rmsnorm_bands import source_module, upstream_class
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


class Exhausted(Exception):
    pass


class Destination:
    def __init__(self, like):
        with _disable_current_modes():
            self.tensor = torch.empty(tuple(like.shape), dtype=like.dtype, device="cpu")
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
        shape = args[1] if func is torch.ops.aten.new_empty.default else args[0] if args else ()
        allocating = func in (torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided.default,
                              torch.ops.aten.new_empty.default)
        if (allocating and self.deny_output and tuple(shape) == self.shape
                and str((kwargs or {}).get("device")) != "meta"):
            self.full += 1
            raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
        if func is torch.ops.aten.mul.Tensor:
            count = max((x.numel() for x in args if isinstance(x, torch.Tensor)), default=0)
            if count > torch.Size(self.shape).numel() // 3:
                self.arithmetic += 1
                raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
        return func(*args, **(kwargs or {}))


def generic(cuts):
    value = torch.arange(2 * 3 * 7 * 9, dtype=torch.float64).reshape(2, 3, 7, 9)
    weight = torch.arange(4 * 3 * 3 * 3, dtype=torch.float64).reshape(4, 3, 3, 3) % 5
    cases = [
        (cuts._conv, torch.ops.aten.convolution.default,
         (value, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)),
        (cuts._pad, torch.ops.aten.constant_pad_nd.default, (value, [1, 2, 2, 1], 3.0)),
        (cuts._upsample, torch.ops.aten.upsample_nearest2d.default, (value, [14, 18], None, None)),
    ]
    for rule, func, args in cases:
        expected = func(*args)
        plan = rule(func, args, {})
        assert plan is not None
        room = Room()
        cuts.serve(cuts.Memo(), room)
        with torch.no_grad(), Bound(expected.shape) as bound:
            actual = cuts.Layer()._banded(func, plan, (str(func),), plan.align)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bound.full and len(room.outputs) == 1 and len(room.outputs[0].writes) > 1


def bridge(cuts):
    """Run the actual Memory adapter with a CPU DLPack owner and observed CUDA calls."""
    path = Path(__file__).resolve().parents[1] / "python/tensorfs/varena/torch.py"
    parsed = ast.parse(path.read_text())
    definitions = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                   and node.name in {"_BandWriter", "_MappedOutput"}]
    memory = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "Memory")
    method = next(node for node in memory.body if isinstance(node, ast.FunctionDef) and node.name == "mapped_output")
    definitions.append(ast.ClassDef(name="Memory", bases=[], keywords=[], body=[method], decorator_list=[]))
    module = ModuleType("mapped_output_bridge")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({"torch": torch, "Tensor": torch.Tensor, "cuts": cuts, "dataclass": dataclass,
                      "Protocol": Protocol, "DTYPES": {"F16": torch.float16, "BF16": torch.bfloat16,
                                                       "F32": torch.float32, "F64": torch.float64}})
    # Compile only this checkout's own adapter definitions, not an alternate implementation.
    compiled = compile(ast.fix_missing_locations(ast.Module(
        [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
         *definitions], type_ignores=[])), str(path), "exec")
    exec(compiled, namespace)  # noqa: S102
    factory_calls, writes = [], []
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    class Array:
        def __dlpack_device__(self):
            return source.__dlpack_device__()

        def __dlpack__(self):
            return source.__dlpack__()

        def _write_band(self, band, axis, offset):
            writes.append((band, axis, offset))

    array = Array()

    def factory(shape, dtype, stream):
        factory_calls.append((shape, dtype, stream))
        return array

    memory = SimpleNamespace(native=SimpleNamespace(mapped_output=factory), device=source.device)
    like = torch.empty(source.shape, dtype=source.dtype, device="meta")
    call = namespace["Memory"].mapped_output
    with torch.no_grad(), patch.object(torch.cuda, "is_initialized", return_value=False), \
            patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=321)):
        result = call(memory, like, source.device)
        assert result is not None and factory_calls == [(tuple(source.shape), "F32", 321)]
        torch.testing.assert_close(result.tensor, source, rtol=0, atol=0)
        result.write_band(source, 1, 0)
        assert writes == [(source, 1, 0)]
        factory_calls.clear()
        for unsupported in (like.transpose(1, 2), like.to(dtype=torch.int64)):
            assert call(memory, unsupported, source.device) is None
        with patch.object(torch.cuda, "is_initialized", return_value=True), \
                patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            assert call(memory, like, source.device) is None
        assert not factory_calls
        memory.native.mapped_output = lambda *_: None
        assert call(memory, like, source.device) is None

        def failed(*_):
            raise RuntimeError("native allocation failed")

        memory.native.mapped_output = failed
        try:
            call(memory, like, source.device)
        except RuntimeError as error:
            assert str(error) == "native allocation failed"
        else:
            raise AssertionError("native allocation failure was swallowed")


def rms(cuts, cls, shape, first, images, dtype, bias, *, admit=True, deny_output=True, promote=False):
    channels = shape[1] if first else shape[-1]
    module = cls(channels, channel_first=first, images=images, bias=bias).to(dtype=torch.float32 if promote else dtype)
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
    assert actual.shape == value.shape and actual.dtype == expected.dtype
    if promote:
        assert actual.dtype == torch.float32 and value.dtype != actual.dtype
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
        for dtype in (torch.float16, torch.bfloat16):
            rms(cuts, cls, (2, 7, 5, 11), True, True, dtype, True, promote=True)
            cases += 1
    generic(cuts)
    bridge(cuts)
    print(f"PASS mapped-output CPU seam: {cases} exact upstream RMS cases, denied admission, "
          "device-first allocation, convolution/pad/upsample, actual Memory bridge and no retained input; "
          "native CUDA feasibility is separate")


if __name__ == "__main__":
    main()
