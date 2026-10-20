"""Actual CPU resident-part marshaling and transferred-owner proof.

Run against the coherent installed native wheel, beside prove-dlpack-owner.py.
The latter's existing real ctypes managed allocation fixture is reused unchanged.
No Torch, CUDA, model download, network or mocked decoder is involved.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes as C
import gc
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import tensorfs
from tensorfs import _ext, raw


def fixture():
    path = Path(__file__).with_name("prove-dlpack-owner.py")
    spec = importlib.util.spec_from_file_location("resident_decode_owner_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def refuses(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def cpu_main():
    assert "torch" not in sys.modules, "the CPU proof imported Torch"
    helper, helper_path = fixture()
    passed = []
    custom = []
    weight = np.array([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=np.int8)
    scale = np.array([0.5, 2.0], dtype=np.float32)
    expected = weight.astype(np.float32) * scale[:, None]

    def parts(w=None, s=None):
        return {
            "weight": ("I8", [2, 4], memoryview(weight) if w is None else w),
            "scale": ("F32", [2], scale.tobytes() if s is None else s),
        }

    def decode(values, out):
        raw.decode_into(
            "int8.rowwise",
            values,
            out,
            dtype="F32",
            shape=[2, 4],
            params={"axis": 0, "scale_dtype": "F32"},
        )

    def owned(data, *, logical=None, shape=None, offset=0, on_delete=None):
        value = helper.OwnedTensor(data, on_delete)
        custom.append(value)
        if shape is not None:
            value.shape = (C.c_int64 * len(shape))(*shape)
            value.managed.dl_tensor.ndim = len(shape)
            value.managed.dl_tensor.shape = value.shape
        if logical is not None:
            value.managed.dl_tensor.dtype = helper.Dtype(*logical)
        value.managed.dl_tensor.byte_offset = offset
        return value

    out = np.full((2, 4), 99, dtype=np.float32)
    decode(parts(), memoryview(out))
    assert out.tobytes() == expected.tobytes()
    passed.append("primitive_buffer_leases_and_raw_part_views_decode_exactly")

    arrays = [weight.copy(), scale.copy(), np.empty_like(expected)]
    capsules = [value.__dlpack__() for value in arrays]
    aliases = [
        helper.CapsuleAlias(cap, value.__dlpack_device__())
        for cap, value in zip(capsules, arrays, strict=True)
    ]
    decode(parts(aliases[0], aliases[1]), aliases[2])
    assert arrays[2].tobytes() == expected.tobytes()
    assert all(helper._name(id(cap)) == helper._used for cap in capsules)
    sentinel = np.full_like(expected, 99)
    refuses(
        tensorfs.InvalidError, lambda: decode(parts(aliases[0]), memoryview(sentinel))
    )
    assert np.all(sentinel == 99)
    passed.append("real_numpy_capsules_transfer_once_and_reuse_refuses")

    class CountingSource:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __dlpack_device__(self):
            return self.value.__dlpack_device__()

        def __dlpack__(self, *, stream=None):
            assert stream is None
            self.calls += 1
            return self.value.__dlpack__()

    untouched = CountingSource(weight)
    refuses(
        tensorfs.UnknownSchemeError,
        lambda: raw.decode_into(
            "not.a.registered.scheme",
            parts(untouched),
            memoryview(sentinel),
            dtype="F32",
            shape=[2, 4],
        ),
    )
    assert untouched.calls == 0 and np.all(sentinel == 99)
    passed.append("core_metadata_refusal_precedes_capsule_requests")

    short = owned(bytes(7))
    not_requested = CountingSource(np.empty_like(expected))
    refuses(tensorfs.InvalidError, lambda: decode(parts(short), not_requested))
    helper.released(short)
    assert short.released_bytes == bytes(7) and not_requested.calls == 0
    passed.append("declared_byte_span_refuses_and_releases_claim_once")

    first = owned(weight.tobytes())
    late = owned(bytes(7))
    refuses(tensorfs.InvalidError, lambda: decode(parts(first, late), not_requested))
    helper.released(first)
    helper.released(late)
    assert first.released_bytes == weight.tobytes() and not_requested.calls == 0
    passed.append("later_invalid_input_releases_all_earlier_claims_once")

    wrong_out = owned(bytes(16), logical=(2, 16, 1), shape=[2, 4])
    refuses(tensorfs.InvalidError, lambda: decode(parts(), wrong_out))
    helper.released(wrong_out)
    assert wrong_out.released_bytes == bytes(16)
    passed.append("actual_output_dtype_mismatch_refuses_before_write")

    readonly = memoryview(expected.tobytes()).cast("f", shape=[2, 4])
    before = readonly.tobytes()
    refuses(tensorfs.InvalidError, lambda: decode(parts(), readonly))
    assert readonly.tobytes() == before
    passed.append("readonly_output_refuses_before_write")

    overlapping = np.full_like(expected, 99)
    before = overlapping.tobytes()
    overlap_bytes = memoryview(overlapping).cast("B")[:8]
    refuses(
        tensorfs.InvalidError,
        lambda: decode(parts(overlap_bytes), memoryview(overlapping)),
    )
    assert overlapping.tobytes() == before
    passed.append("input_output_alias_refuses_without_mutation")

    shared = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    same = memoryview(shared)
    duplicate_expected = (
        np.frombuffer(shared, dtype=np.int8).reshape(2, 4).astype(np.float32)
        * np.frombuffer(shared, dtype=np.float32)[:, None]
    )
    decode(parts(same, same), memoryview(out))
    assert out.tobytes() == duplicate_expected.tobytes()
    passed.append("duplicate_readonly_input_spans_are_allowed")

    strided = np.arange(16, dtype=np.int8).reshape(2, 8)[:, ::2]
    capsule = strided.__dlpack__()
    alias = helper.CapsuleAlias(capsule, strided.__dlpack_device__())
    refuses(
        tensorfs.InvalidError,
        lambda alias=alias: decode(parts(alias), memoryview(sentinel)),
    )
    assert helper._name(id(capsule)) == helper._dltensor
    assert np.all(sentinel == 99)
    del alias, capsule
    passed.append("actual_noncontiguous_descriptor_keeps_capsule_unconsumed")

    reentered = []

    def reenter():
        nested = np.empty_like(expected)
        decode(parts(), memoryview(nested))
        assert nested.tobytes() == expected.tobytes()
        reentered.append(True)

    prefix = b"prefix!!"
    source = owned(prefix + weight.tobytes(), shape=[8], offset=8, on_delete=reenter)
    scales = owned(scale.tobytes(), on_delete=reenter)
    destination = owned(
        prefix + bytes(expected.nbytes),
        logical=(2, 32, 1),
        shape=[2, 4],
        offset=8,
        on_delete=reenter,
    )
    decode(parts(source, scales), destination)
    for value in (source, scales, destination):
        helper.released(value)
    assert source.released_bytes == prefix + weight.tobytes()
    assert destination.released_bytes == prefix + expected.tobytes()
    assert len(reentered) == 3
    passed.append("byte_offsets_and_reentrant_managed_deleters_complete_once")

    # These capsules contain genuinely freed managed pointers. Only their used names
    # may be inspected on retry, never their stale shape/data fields.
    refuses(tensorfs.InvalidError, lambda: decode(parts(source), memoryview(sentinel)))
    refuses(tensorfs.InvalidError, lambda: decode(parts(), destination))
    assert source.deletions == destination.deletions == 1 and np.all(sentinel == 99)
    passed.append("used_aliases_refuse_before_freed_pointer_access")

    bad_weight = owned(bytes(2))
    bad_scale = owned(np.array([np.nan], np.float32).tobytes())
    channel = owned(np.array([1.0], np.float32).tobytes())
    failed_out = owned(bytes(16), logical=(2, 32, 1), shape=[1, 4])
    refuses(
        tensorfs.DecodeError,
        lambda: raw.decode_into(
            "comfy.asym_w4a8_int8",
            {
                "weight": ("I8", [1, 2], bad_weight),
                "weight_s_rel": ("F32", [1, 1], bad_scale),
                "weight_s_channel": ("F32", [1], channel),
            },
            failed_out,
            dtype="F32",
            shape=[1, 4],
            params={"group_size": 4, "rotation_size": 4},
        ),
    )
    for value in (bad_weight, bad_scale, channel, failed_out):
        helper.released(value)
    passed.append("native_value_error_releases_every_claimed_owner_once")

    for value in custom:
        value.capsule = None
    aliases.clear()
    capsules.clear()
    gc.collect()
    assert not helper._live and not helper._callback_errors
    assert len(custom) == 11 and all(value.deletions == 1 for value in custom)
    passed.append("used_capsule_destruction_never_repeats_managed_release")
    assert len(passed) == 14 and "torch" not in sys.modules
    print(
        json.dumps(
            {
                "passed": len(passed),
                "failed": 0,
                "skipped": 0,
                "cases": passed,
                "managed_deletions": len(custom),
                "payload_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "tensorfs_version": tensorfs.__version__,
                "numpy_version": np.__version__,
                "native_path": _ext.__file__,
                "native_sha256": hashlib.sha256(
                    Path(_ext.__file__).read_bytes()
                ).hexdigest(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "owner_fixture_sha256": hashlib.sha256(
                    helper_path.read_bytes()
                ).hexdigest(),
                "scope": "CPU resident-part decode and actual managed ownership; no GPU completion qualification",
            },
            sort_keys=True,
        )
    )


def gpu_main(vectors):
    """Opt-in device proof; the default CPU path never imports this framework."""
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    helper, _helper_path = fixture()
    corpus = vectors.read_bytes()
    assert hashlib.sha256(corpus).hexdigest() == (
        "56758d0c82b7b0d173d8db25cd5362ad9db348839e5f4e22ce664784c0e56043"
    )
    cases = json.loads(corpus)["cases"]
    positive = [row for row in cases if not row.get("refuse")]
    negative = [row for row in cases if row.get("refuse")]
    assert len(positive) == 18 and len(negative) == 14
    dtype = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}

    # The extra F16 operand has exactly the same real values as its frozen BF16
    # correction. Its independent expected logical bytes remain unchanged.
    variant = copy.deepcopy(
        next(
            row
            for row in positive
            if any(
                p["role"] == "weight_correction" and p["dtype"] == "BF16"
                for p in row["parts"]
            )
        )
    )
    correction = next(p for p in variant["parts"] if p["role"] == "weight_correction")
    values = [
        struct.unpack("<f", struct.pack("<I", word[0] << 16))[0]
        for word in struct.iter_unpack("<H", base64.b64decode(correction["bits"]))
    ]
    narrowed = [struct.pack("<e", value) for value in values]
    assert all(
        math.isfinite(value) and struct.unpack("<e", bits)[0] == value
        for value, bits in zip(values, narrowed, strict=True)
    )
    correction.update(dtype="F16", bits=base64.b64encode(b"".join(narrowed)).decode())
    variant["key"] += "-exact-f16-correction"
    positive.append(variant)

    class Export:
        """A real Torch producer that honors its supplied consumer stream."""

        def __init__(self, value, stream=None):
            self.value = value
            self.stream = stream
            self.capsules = []

        def __dlpack_device__(self):
            return self.value.__dlpack_device__()

        def __dlpack__(self, *, stream=None):
            if self.stream is None:
                capsule = self.value.__dlpack__(stream=stream)
            else:
                with torch.cuda.stream(self.stream):
                    capsule = self.value.__dlpack__(stream=stream)
            self.capsules.append(capsule)
            return capsule

    class UsedAlias:
        def __init__(self, capsule, device):
            self.capsule = capsule
            self.device = device

        def __dlpack_device__(self):
            return self.device

        def __dlpack__(self, *, stream=None):
            # A supplied stream cannot make this consumed real capsule's stale
            # managed pointer legal to inspect again.
            return self.capsule

    def operands(row):
        # POD byte carriers deliberately keep stored format metadata separate.
        held = {}
        for part in row["parts"]:
            value = torch.frombuffer(
                bytearray(base64.b64decode(part["bits"])), dtype=torch.uint8
            )
            held[part["role"]] = (
                part["dtype"],
                part["shape"],
                Export(value.to("cuda")),
            )
        return held

    def invoke(row, parts, out):
        raw.decode_into(
            row["encoding"]["scheme"],
            parts,
            out,
            dtype=row["dtype"],
            shape=row["shape"],
            params=row["encoding"]["params"],
        )

    def bytes_of(value):
        return value.view(torch.uint8).cpu().numpy().tobytes()

    def claimed(parts, out):
        exported = [value[2] for value in parts.values()] + [out]
        assert all(len(value.capsules) == 1 for value in exported)
        assert all(
            helper._name(id(value.capsules[0])) == helper._used for value in exported
        )

    def stream_control(row):
        source, previous = torch.cuda.Stream(), torch.cuda.Stream()
        assert source.cuda_stream != previous.cuda_stream
        ready = operands(row)
        parts = {
            role: (kind, shape, Export(torch.empty_like(value.value), source))
            for role, (kind, shape, value) in ready.items()
        }
        out = Export(
            torch.empty(row["shape"], dtype=dtype[row["dtype"]], device="cuda"),
            previous,
        )
        torch.cuda.synchronize()
        produced, prior = torch.cuda.Event(), torch.cuda.Event()
        try:
            with torch.cuda.stream(source):
                torch.cuda._sleep(1_000_000_000)
                for role, (_, _, value) in parts.items():
                    value.value.copy_(ready[role][2].value)
                produced.record()
            with torch.cuda.stream(previous):
                torch.cuda._sleep(1_000_000_000)
                out.value.fill_(99)
                prior.record()
            assert not produced.query() and not prior.query()
            invoke(row, parts, out)
            # These checks happen AT RETURN, before rescue synchronization or a
            # tensor read could mask a missing producer/output ordering contract.
            assert produced.query() and prior.query()
            claimed(parts, out)
            assert bytes_of(out.value) == base64.b64decode(row["expect"])
            return {
                "producer_pending_before": True,
                "output_pending_before": True,
                "producer_complete_at_return": True,
                "output_complete_at_return": True,
                "source_stream": source.cuda_stream,
                "output_stream": previous.cuda_stream,
            }
        finally:
            source.synchronize()
            previous.synchronize()

    records, metadata_casts = [], set()
    for row in positive:
        parts = operands(row)
        out = Export(
            torch.empty(row["shape"], dtype=dtype[row["dtype"]], device="cuda")
        )
        invoke(row, parts, out)
        claimed(parts, out)
        assert bytes_of(out.value) == base64.b64decode(row["expect"]), row["key"]
        metadata_casts.update(p["dtype"] for p in row["parts"] if p["role"] != "weight")
        records.append(
            {
                "key": row["key"],
                "result": "passed",
                "kind": "golden",
                "bytes": out.value.numel() * out.value.element_size(),
            }
        )
    assert len(records) == 19 and {"F8_E4M3", "BF16", "F16"} <= metadata_casts
    ordering = stream_control(positive[0])

    for row in negative:
        parts = operands(row)
        out = Export(
            torch.full(row["shape"], 99, dtype=dtype[row["dtype"]], device="cuda")
        )
        before = bytes_of(out.value)
        refuses(
            tensorfs.DecodeError,
            lambda row=row, parts=parts, out=out: invoke(row, parts, out),
        )
        value_error = row["key"] == "s-rel-nan"
        if value_error:
            claimed(parts, out)
            first = next(iter(parts))
            kind, shape, producer = parts[first]
            alias = UsedAlias(producer.capsules[0], producer.__dlpack_device__())
            retry = {**parts, first: (kind, shape, alias)}
            fresh = torch.empty(row["shape"], dtype=dtype[row["dtype"]], device="cuda")
            refuses(
                tensorfs.InvalidError,
                lambda row=row, retry=retry, fresh=fresh: invoke(row, retry, fresh),
            )
        else:
            assert not out.capsules and all(not p[2].capsules for p in parts.values())
            assert bytes_of(out.value) == before
        records.append(
            {
                "key": row["key"],
                "result": "passed",
                "kind": "expected-refusal",
                "family": "decode",
                "value_error_output_unspecified": value_error,
            }
        )

    # A zero logical destination may still have a real, pending optional codebook.
    # Bindings cleanup is asynchronous: checking the event right at RETURN proves
    # Core's success contract, not eventual cleanup of an early return.
    source, previous = torch.cuda.Stream(), torch.cuda.Stream()
    codebook = torch.empty(16, dtype=torch.float32, device="cuda")
    ready = torch.arange(16, dtype=torch.float32, device="cuda")
    marker = torch.zeros(1, dtype=torch.uint8, device="cuda")
    parts = {
        "weight": (
            "I8",
            [0, 128],
            Export(torch.empty((0, 128), dtype=torch.int8, device="cuda")),
        ),
        "weight_s_rel": (
            "F32",
            [0, 16],
            Export(torch.empty((0, 16), dtype=torch.float32, device="cuda")),
        ),
        "weight_s_channel": (
            "F32",
            [0],
            Export(torch.empty(0, dtype=torch.float32, device="cuda")),
        ),
        "weight_codebook": ("F32", [16], Export(codebook, source)),
    }
    out = Export(torch.empty((0, 256), dtype=torch.float32, device="cuda"), previous)
    produced, prior = torch.cuda.Event(), torch.cuda.Event()
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(source):
            torch.cuda._sleep(1_000_000_000)
            codebook.copy_(ready)
            produced.record()
        with torch.cuda.stream(previous):
            torch.cuda._sleep(1_000_000_000)
            marker.fill_(7)
            prior.record()
        assert not produced.query() and not prior.query()
        raw.decode_into(
            "comfy.asym_w4a8_int8",
            parts,
            out,
            dtype="F32",
            shape=[0, 256],
            params={"group_size": 16, "rotation_size": 256},
        )
        assert produced.query() and prior.query(), (
            "zero decode returned before its producers"
        )
        claimed(parts, out)
        assert bytes_of(codebook) == struct.pack("<16f", *range(16))
        assert marker.item() == 7 and bytes_of(out.value) == b""
        records.append(
            {
                "key": "zero-output-pending-codebook",
                "result": "passed",
                "kind": "completion-control",
                "producer_complete_at_return": True,
                "output_complete_at_return": True,
                "codebook_sha256": hashlib.sha256(bytes_of(codebook)).hexdigest(),
            }
        )
    finally:
        source.synchronize()
        previous.synchronize()
    assert len(records) == 34 and len({row["key"] for row in records}) == 34
    print(
        json.dumps(
            {
                "mode": "gpu",
                "passed": 34,
                "failed": 0,
                "skipped": 0,
                "original_positive": 18,
                "derived_f16_positive": 1,
                "expected_refusal": 14,
                "completion_control": 1,
                "cases": records,
                "stream_control": ordering,
                "metadata_cast_dtypes": sorted(metadata_casts),
                "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
                "tensorfs_version": tensorfs.__version__,
                "torch_version": torch.__version__,
                "device_count": torch.cuda.device_count(),
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "native_sha256": hashlib.sha256(
                    Path(_ext.__file__).read_bytes()
                ).hexdigest(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "scope": "resident GPU decode corpus/refusals and checked producer/output ordering; no full model or unknown CUDA-fault injection",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="explicit opt-in; requires one allocated CUDA device",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        help="exact unchanged Comfy conformance JSON, required with --gpu",
    )
    args = parser.parse_args()
    if args.gpu:
        if args.vectors is None:
            parser.error("--gpu requires --vectors")
        gpu_main(args.vectors)
    else:
        cpu_main()
