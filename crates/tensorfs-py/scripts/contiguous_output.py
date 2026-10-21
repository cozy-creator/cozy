"""Exact contiguous cuts: hosted CPU oracle and isolated real-CUDA output proof.

CUDA uses a real Torch allocator quota and admitted Memory output. Its small quota
adapter does not qualify native relief policy; the full Qwen request does that.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import weakref
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("contiguous_output_cuts", path)
    assert spec is not None and spec.loader is not None
    cuts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cuts
    spec.loader.exec_module(cuts)
    return cuts


class ObservedOutput:
    def __init__(self, output):
        self.output = output
        self.writes = []

    @property
    def tensor(self):
        return self.output.tensor

    def write_band(self, source, axis, offset):
        assert source.is_contiguous(), "permuted source was not coalesced before native scatter"
        self.writes.append((axis, offset, tuple(source.shape), source.numel() * source.element_size()))
        self.output.write_band(source, axis, offset)


class HostOutput:
    def __init__(self, like):
        with _disable_current_modes():
            self.tensor = torch.empty(tuple(like.shape), dtype=like.dtype)

    def write_band(self, source, axis, offset):
        with _disable_current_modes():
            self.tensor.narrow(axis, offset, source.shape[axis]).copy_(source)


class Room:
    def __init__(self, memory=None, room=0):
        self.memory, self.room = memory, room
        self.outputs, self.events = [], []

    def available(self, device):
        return self.room

    def free(self, device, need):
        self.events.append("free")
        return False

    def lower(self, device, need):
        self.events.append("lower")
        return False

    def mapped_output(self, like, device):
        assert self.events[-1] == "lower", "mapped storage bypassed device relief"
        self.events.append("mapped")
        output = self.memory.mapped_output(like, device) if self.memory is not None else HostOutput(like)
        assert output is not None, "small mapped output must be admitted"
        observed = ObservedOutput(output)
        self.outputs.append(observed)
        return observed

    def exhausted(self, device, need, name):
        return torch.OutOfMemoryError(f"{name} needs {need} B")


class CpuBound(TorchDispatchMode):
    """Refuse the exact full contiguous result; smaller real CPU bands still run."""

    def __init__(self, shape):
        super().__init__()
        self.shape, self.refused = tuple(shape), 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        full_copy = (func is torch.ops.aten.clone.default and tuple(args[0].shape) == self.shape)
        full_empty = (func in (torch.ops.aten.empty_strided.default, torch.ops.aten.empty.memory_format)
                      and tuple(args[0]) == self.shape and str(kwargs.get("device")) != "meta")
        full_silu = func is torch.ops.aten.silu.default and tuple(args[0].shape) == self.shape
        full_add = func is torch.ops.aten.add.Tensor and tuple(args[0].shape) == self.shape
        if full_copy or full_empty or full_silu or full_add:
            self.refused += 1
            raise torch.OutOfMemoryError("Tried to allocate 1 MiB")
        return func(*args, **kwargs)


def check(cuts, source, expected, *, room=0):
    memo, authority = cuts.Memo(), Room(room=room)
    cuts.serve(memo, authority)
    with torch.no_grad(), CpuBound(source.shape) as bound, cuts.caught():
        actual = torch.ops.aten.contiguous.default(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous() and actual.dtype == source.dtype
    assert actual.shape == source.shape and actual.data_ptr() != source.data_ptr()
    assert bound.refused and len(authority.outputs) == 1
    writes = authority.outputs[0].writes
    assert len(writes) > 1 and max(row[3] for row in writes) < actual.numel() * actual.element_size()
    assert torch.equal(source, expected), "contiguous copy mutated its source"
    return actual, authority


def default_formats(cuts, source):
    """Exercise the rule itself, before CPU can lower omitted contiguous to clone."""
    contiguous, clone = torch.ops.aten.contiguous.default, torch.ops.aten.clone.default
    defaults = {name: repr(cuts._arguments(op, (source,), {}).get("memory_format"))
                for name, op in (("contiguous", contiguous), ("clone", clone))}
    assert cuts._contiguous(contiguous, (source,), {}) is not None, defaults
    for op in (contiguous, clone):
        assert cuts._contiguous(op, (source,), {"memory_format": torch.contiguous_format}) is not None
        assert cuts._contiguous(op, (source,), {"memory_format": 0}) is not None
        assert cuts._contiguous(op, (source,), {"memory_format": torch.preserve_format}) is None
    assert cuts._contiguous(clone, (source,), {}) is None, "omitted clone must preserve its format"
    return defaults


def silu_cpu(cuts):
    op = torch.ops.aten.silu.default
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raw = torch.linspace(-7, 7, 2 * 4 * 8 * 32, dtype=dtype).reshape(2, 4, 8, 32)
        for source in (raw[..., :16].contiguous(), raw[..., 1::2]):
            expected, before = op(source), source.clone()
            version = source._version
            room = Room()
            cuts.serve(cuts.Memo(), room)
            with torch.no_grad(), CpuBound(source.shape) as bound, cuts.caught():
                actual = op(source)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.dtype == source.dtype and actual.shape == source.shape
            assert actual.stride() == expected.stride() and actual.data_ptr() != source.data_ptr()
            assert torch.equal(source, before) and source._version == version
            assert bound.refused and len(room.outputs) == 1 and len(room.outputs[0].writes) > 1
        with torch.no_grad():
            assert cuts._silu(op, (raw.transpose(1, 2),), {}) is None
            destination = torch.empty_like(raw)
            assert cuts._silu(torch.ops.aten.silu.out, (raw,), {"out": destination}) is None
            with cuts.caught():
                assert torch.ops.aten.silu.out(raw, out=destination) is destination
            torch.testing.assert_close(destination, op(raw), rtol=0, atol=0)
            inplace = raw.clone()
            pointer = inplace.data_ptr()
            assert cuts._silu(torch.ops.aten.silu_.default, (inplace,), {}) is None
            with cuts.caught():
                assert torch.ops.aten.silu_.default(inplace) is inplace
            assert inplace.data_ptr() == pointer
            torch.testing.assert_close(inplace, op(raw), rtol=0, atol=0)
        assert cuts._silu(op, (raw,), {}) is None, "autograd mode must keep the original op"
        train = raw.clone().requires_grad_()
        reference = train.detach().clone().requires_grad_()
        with cuts.caught():
            op(train).sum().backward()
        op(reference).sum().backward()
        torch.testing.assert_close(train.grad, reference.grad, rtol=0, atol=0)
    assert torch.ops.aten.silu_ not in cuts._RULES, "an in-place activation cannot get a fresh result"
    print("PASS SiLU CPU: exact native bands, four dtypes, strided offsets, fresh output and unchanged autograd")


def add_cpu(cuts):
    op = torch.ops.aten.add.Tensor
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raw = torch.linspace(-7, 7, 2 * 4 * 8 * 32, dtype=dtype).reshape(2, 4, 8, 32)
        other = torch.cos(raw)
        pairs = [(raw[..., :16].contiguous(), other[..., :16].contiguous()),
                 (raw[..., 1::2], other[..., ::2]), (raw[..., 1::2], raw[..., 1::2])]
        for left, right in pairs:
            for alpha in (-2, -0.5, 0, 1, 2.25):
                expected = op(left, right, alpha=alpha)
                before, versions = (left.clone(), right.clone()), (left._version, right._version)
                room = Room()
                cuts.serve(cuts.Memo(), room)
                with torch.no_grad(), CpuBound(left.shape) as bound, cuts.caught():
                    actual = op(left, right, alpha=alpha)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.stride() == expected.stride()
                assert actual.data_ptr() not in (left.data_ptr(), right.data_ptr())
                assert torch.equal(left, before[0]) and torch.equal(right, before[1])
                assert (left._version, right._version) == versions
                assert bound.refused and len(room.outputs) == 1 and len(room.outputs[0].writes) > 1
        with torch.no_grad():
            assert cuts._add(op, (raw, raw[..., :1]), {}) is None, "broadcast is unsupported"
            assert cuts._add(op, (raw, 1), {}) is None, "scalar add is unsupported"
            assert cuts._add(op, (raw, raw.to("meta")), {}) is None, "devices must match"
            transposed = raw.transpose(1, 2)
            assert cuts._add(op, (transposed, transposed), {}) is None, "output must be contiguous"
            destination = torch.empty_like(raw)
            assert cuts._add(torch.ops.aten.add.out, (raw, raw), {"out": destination}) is None
            with cuts.caught():
                assert torch.ops.aten.add.out(raw, raw, alpha=-0.5, out=destination) is destination
            torch.testing.assert_close(destination, op(raw, raw, alpha=-0.5), rtol=0, atol=0)
            inplace = raw.clone()
            pointer = inplace.data_ptr()
            assert cuts._add(torch.ops.aten.add_.Tensor, (inplace, raw), {}) is None
            with cuts.caught():
                assert torch.ops.aten.add_.Tensor(inplace, raw, alpha=-0.5) is inplace
            assert inplace.data_ptr() == pointer
            torch.testing.assert_close(inplace, op(raw, raw, alpha=-0.5), rtol=0, atol=0)
        assert cuts._add(op, (raw, raw), {}) is None, "autograd keeps the native op"
        train, peer = raw.clone().requires_grad_(), other.clone().requires_grad_()
        reference, reference_peer = train.detach().clone().requires_grad_(), peer.detach().clone().requires_grad_()
        with cuts.caught():
            op(train, peer, alpha=-0.5).sum().backward()
        op(reference, reference_peer, alpha=-0.5).sum().backward()
        torch.testing.assert_close(train.grad, reference.grad, rtol=0, atol=0)
        torch.testing.assert_close(peer.grad, reference_peer.grad, rtol=0, atol=0)
    # Promotion and its larger result use the native meta dtype and both input byte sizes.
    left, right = torch.arange(512, dtype=torch.float16).reshape(2, 4, 64), torch.ones(2, 4, 64)
    cuts.serve(cuts.Memo(), Room())
    with torch.no_grad(), CpuBound(left.shape), cuts.caught():
        actual = op(left, right, alpha=-0.5)
    torch.testing.assert_close(actual, op(left, right, alpha=-0.5), rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert torch.ops.aten.add_ not in cuts._RULES
    print("PASS add CPU: four native dtypes and alpha, offsets, input aliases, promotion and unchanged autograd")


def cpu(cuts):
    from rmsnorm_bands import upstream_class

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        # Exact decoder DupUp3D view/permute, including first_chunk's output alias.
        cls = upstream_class("autoencoder_kl_qwenimage21", "QwenImage21DupUp3D")
        module = cls(12, 3, factor_t=2, factor_s=2)
        x = torch.arange(12 * 3 * 5, dtype=dtype).reshape(1, 12, 1, 3, 5)
        repeated = x.repeat_interleave(module.repeats, dim=1)
        source = repeated.view(1, 3, 2, 2, 2, 1, 3, 5).permute(0, 1, 5, 2, 6, 3, 7, 4)
        expected = source.contiguous()
        with torch.no_grad():
            defaults = default_formats(cuts, source)
        actual, authority = check(cuts, source, expected)
        expected_decode = module(x, first_chunk=True)
        cuts.serve(cuts.Memo(), Room())
        with torch.no_grad(), CpuBound(source.shape), cuts.caught():
            decoded = module(x, first_chunk=True)
        torch.testing.assert_close(decoded, expected_decode, rtol=0, atol=0)
        assert decoded.stride() == expected_decode.stride()
        # A real sliced/offset source, and later-axis bands when one leading row is too large.
        raw = torch.arange(4 * 17 * 9, dtype=dtype).reshape(4, 17, 9)
        sliced = raw[1:, 1::2, ::2].transpose(0, 2)
        _, alternate = check(cuts, sliced, sliced.contiguous(), room=2 * sliced.element_size() * 20)
        assert alternate.outputs[0].writes[0][0] != 0
        same = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
        with torch.no_grad(), cuts.caught():
            assert same.contiguous() is same, "contiguous alias fast path changed"
        with torch.no_grad():
            assert cuts._contiguous(torch.ops.aten.contiguous.default, (same,), {}) is None
            channels = torch.empty(2, 3, 4, 5).transpose(2, 3)
            assert cuts._contiguous(torch.ops.aten.contiguous.default, (channels,),
                                    {"memory_format": torch.channels_last}) is None
        assert cuts._contiguous(torch.ops.aten.contiguous.default, (source,), {}) is None
        pointer = weakref.ref(source)
        del source, actual, authority
        gc.collect()
        assert pointer() is None, "contiguous memo retained source storage"
    print("PASS contiguous CPU: exact locked DupUp3D, strided offsets, later-axis bands, formats and aliases")
    print(json.dumps({"omitted_memory_format": defaults}))
    silu_cpu(cuts)
    add_cpu(cuts)


def measured_quota(device, total, name):
    """Twelve MiB beyond existing allocator residency, not live tensor bytes.

    A live cuBLAS workspace can split a larger segment: its inactive remainder
    cannot be returned by empty_cache or reused on our other stream. That
    hardware-dependent baseline must not consume the staging allowance.
    """
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    allowance = 12 << 20
    allowed = reserved + allowance
    torch.cuda.set_per_process_memory_fraction(allowed / total, device)
    assert round(torch.cuda.get_per_process_memory_fraction(device) * total) == allowed
    torch.cuda.reset_peak_memory_stats(device)
    return {"phase": name, "allocated_baseline": allocated, "reserved_baseline": reserved,
            "allowance": allowance, "allowed": allowed}


def quota_peaks(device, quota):
    torch.cuda.synchronize(device)
    total = torch.cuda.get_device_properties(device).total_memory
    assert round(torch.cuda.get_per_process_memory_fraction(device) * total) == quota["allowed"], quota
    allocated = torch.cuda.max_memory_allocated(device) - quota["allocated_baseline"]
    reserved = torch.cuda.max_memory_reserved(device) - quota["reserved_baseline"]
    observed = {**quota, "peak_allocated_delta": allocated, "peak_reserved_delta": reserved}
    assert 0 <= allocated <= quota["allowance"], observed
    assert 0 <= reserved <= quota["allowance"], observed
    return observed


def add_cuda(memory, device, side, total):
    from tensorfs.varena import cuts

    op = torch.ops.aten.add.Tensor
    quota = measured_quota(device, total, "add")
    like = torch.empty((1, 32, 128, 2048), dtype=torch.float32, device="meta")
    left_host, right_host = memory.mapped_output(like, device), memory.mapped_output(like, device)
    assert left_host is not None and right_host is not None
    left, right = left_host.tensor, right_host.tensor
    pointers = weakref.ref(left), weakref.ref(right)
    authority = Room(memory, room=8 << 20)
    cuts.serve(cuts.Memo(), authority)
    with torch.cuda.stream(side):
        for at in range(32):
            piece = torch.linspace(-7 + at / 32, 7 - at / 32, 128 * 2048,
                                   device=device).reshape(1, 1, 128, 2048)
            peer = torch.cos(piece)
            left_host.write_band(piece, 1, at)
            right_host.write_band(peer, 1, at)
            del piece, peer
        with cuts.caught():
            result = op(left, right, alpha=-0.5)
    side.synchronize()
    assert result.shape == left.shape and result.dtype == left.dtype and result.is_contiguous()
    assert result.data_ptr() not in (left.data_ptr(), right.data_ptr())
    assert len(authority.outputs) == 1
    writes = authority.outputs[0].writes
    assert len(writes) > 1 and max(row[3] for row in writes) <= 4 << 20
    for at in range(32):
        expected_left = torch.linspace(-7 + at / 32, 7 - at / 32, 128 * 2048,
                                       device=device).reshape(1, 1, 128, 2048)
        expected_right = torch.cos(expected_left)
        assert torch.equal(left.narrow(1, at, 1), expected_left), "add mutated its left input"
        assert torch.equal(right.narrow(1, at, 1), expected_right), "add mutated its right input"
        expected = op(expected_left, expected_right, alpha=-0.5)
        assert torch.equal(result.narrow(1, at, 1), expected)
        del expected, expected_left, expected_right
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        assert cuts._add(op, (left, right), {}) is None, "capture cannot allocate a mapped result"
    del graph
    destination = torch.empty(0, device=device)
    assert cuts._add(torch.ops.aten.add.out, (left, right), {"out": destination}) is None
    assert cuts._add(torch.ops.aten.add_.Tensor, (left, right), {}) is None
    before_outputs = len(authority.outputs)
    try:
        with cuts.caught():
            torch.ops.aten.add.out(left, right, alpha=-0.5, out=destination)
    except torch.OutOfMemoryError:
        pass
    else:
        raise AssertionError("add out= bypassed its real allocator quota")
    assert len(authority.outputs) == before_outputs, "out= acquired fresh replacement storage"
    del destination
    alias = result.view(-1)[7:39]
    expected_alias = alias.clone()
    del result, left, right, left_host, right_host
    authority.outputs.clear()
    gc.collect()
    assert all(pointer() is None for pointer in pointers), "add memo retained mapped input storage"
    with torch.cuda.stream(side):
        observed = op(alias, alias, alpha=-0.5)
    side.synchronize()
    torch.testing.assert_close(observed, op(expected_alias, expected_alias, alpha=-0.5), rtol=0, atol=0)
    del alias, observed, expected_alias
    return max(row[3] for row in writes), quota_peaks(device, quota)


def cuda():
    from tensorfs.varena import cuts
    from tensorfs.varena.torch import Memory

    assert torch.cuda.device_count() == 1
    device = torch.device("cuda", 0)
    with torch.cuda.device(device):
        torch.cuda.current_blas_handle()
    memory = Memory.open(device)
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction(device)
    side = torch.cuda.Stream(device=device)
    try:
        with memory.request("contiguous-mapped-output") as request, torch.inference_mode():
            # 32MiB source; a real quota above existing reserved memory fits bounded staging,
            # but cannot hold another complete result. No physical hog/ledger cap mutation.
            raw = torch.arange(1 << 23, dtype=torch.float32, device=device).reshape(1, 32, 2, 2, 2, 1, 128, 256)
            source = raw.permute(0, 1, 5, 2, 6, 3, 7, 4)
            del raw
            pointer = weakref.ref(source)
            torch.cuda.synchronize(device)
            quota = measured_quota(device, total, "contiguous")
            authority = Room(memory, room=8 << 20)
            cuts.serve(cuts.Memo(), authority)
            defaults = default_formats(cuts, source)
            with torch.cuda.stream(side):
                side.wait_stream(torch.cuda.current_stream(device))
                with cuts.caught():
                    result = torch.ops.aten.contiguous.default(source)
            side.synchronize()
            assert result.dtype == source.dtype and result.shape == source.shape and result.is_contiguous()
            assert result.data_ptr() != source.data_ptr() and len(authority.outputs) == 1
            writes = authority.outputs[0].writes
            assert len(writes) > 1 and max(row[3] for row in writes) <= 4 << 20
            axis = writes[0][0]
            for at in range(source.shape[axis]):
                expected = source.narrow(axis, at, 1).contiguous()
                assert torch.equal(result.narrow(axis, at, 1), expected)
                del expected
            alias = result.view(-1)[7:39]
            del source, result
            authority.outputs.clear()
            gc.collect()
            assert pointer() is None, "contiguous memo retained its source"
            with torch.cuda.stream(side):
                observed = alias + 1
            side.synchronize()
            assert torch.equal(observed - 1, alias)
            del alias, observed
            quota_observations = [quota_peaks(device, quota)]
            torch.cuda.set_per_process_memory_fraction(fraction, device)

            # The real decode now feeds native SiLU a mapped contiguous result.
            # Build that input through the admitted writer in bounded GPU bands too.
            quota = measured_quota(device, total, "silu")
            like = torch.empty((1, 32, 128, 2048), dtype=torch.float32, device="meta")
            host_input = memory.mapped_output(like, device)
            assert host_input is not None
            activation_source = host_input.tensor
            activation_pointer = weakref.ref(activation_source)
            with torch.cuda.stream(side):
                for at in range(32):
                    piece = torch.linspace(-7 + at / 32, 7 - at / 32, 128 * 2048,
                                           device=device).reshape(1, 1, 128, 2048)
                    host_input.write_band(piece, 1, at)
                    del piece
                activation_room = Room(memory, room=8 << 20)
                cuts.serve(cuts.Memo(), activation_room)
                with cuts.caught():
                    activated = torch.ops.aten.silu.default(activation_source)
            side.synchronize()
            assert activated.shape == activation_source.shape and activated.dtype == activation_source.dtype
            assert activated.is_contiguous() and activated.data_ptr() != activation_source.data_ptr()
            assert len(activation_room.outputs) == 1
            activation_writes = activation_room.outputs[0].writes
            assert len(activation_writes) > 1 and max(row[3] for row in activation_writes) <= 4 << 20
            for at in range(32):
                expected = torch.ops.aten.silu.default(activation_source.narrow(1, at, 1))
                assert torch.equal(activated.narrow(1, at, 1), expected)
                del expected
            destination = torch.empty(0, device=device)
            assert cuts._silu(torch.ops.aten.silu.out, (activation_source,), {"out": destination}) is None
            before_outputs = len(activation_room.outputs)
            try:
                with cuts.caught():
                    torch.ops.aten.silu.out(activation_source, out=destination)
            except torch.OutOfMemoryError:
                pass
            else:
                raise AssertionError("the mutating out overload bypassed its real allocator quota")
            assert len(activation_room.outputs) == before_outputs, "out= acquired replacement storage"
            del destination
            del activated, activation_source, host_input
            activation_room.outputs.clear()
            gc.collect()
            assert activation_pointer() is None, "activation memo retained its mapped input"
            quota_observations.append(quota_peaks(device, quota))
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            add_largest, add_quota = add_cuda(memory, device, side, total)
            quota_observations.append(add_quota)
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        assert request.stats is not None and not request.cleanup_errors
        assert request.stats.leftover == ((0, 0),)
        print("PASS contiguous mapped CUDA: real quota, exact strided bands, actual stream and alias lifetime")
        print("PASS SiLU mapped CUDA: real quota, mapped input, exact native activation bands and no retained input")
        print("PASS add mapped CUDA: real quota, two mapped inputs, exact native alpha bands and alias lifetime")
        print(json.dumps({"gpu": str(torch.cuda.get_device_properties(device).uuid),
                          "input_bytes": 32 << 20, "largest_band_bytes": max(row[3] for row in writes),
                          "omitted_memory_format": defaults,
                          "silu_largest_band_bytes": max(row[3] for row in activation_writes),
                          "add_largest_band_bytes": add_largest,
                          "quotas": quota_observations,
                          "leftover": request.stats.leftover, "scope": "isolated cut/output; not native relief policy"}))
    finally:
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        memory.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cpu":
        path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "python/tensorfs/varena/cuts.py"
        cpu(load_source(path))
    else:
        cuda()
