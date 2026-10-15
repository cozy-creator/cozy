#!/usr/bin/env python3
"""Real native Store/scoped-source tensor reads; CPU by default, optional owned CUDA."""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import tempfile
import threading
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import tensorfs
from tensorfs.derived import (
    Derivation,
    Part,
    Source,
    SourceCapability,
    Target,
    Tensor,
    serve_source,
)


def fixture(store):
    import torch

    plain = dict(tensorfs.seed_digests())["plain/1"]
    flat, keep = [
        digest for alias, digest in tensorfs.seed_digests() if alias == "fp8-rowwise/1"
    ]
    values = {
        "plain": torch.arange(128, dtype=torch.float32).reshape(8, 16),
        "scalar": torch.tensor(7, dtype=torch.int64),
    }
    tensors = {
        key: Tensor(
            {torch.float32: "f32", torch.int64: "i64", torch.float16: "f16"}[
                value.dtype
            ],
            tuple(value.shape),
            plain,
            {
                "value": Part(
                    {torch.float32: "f32", torch.int64: "i64", torch.float16: "f16"}[
                        value.dtype
                    ],
                    tuple(value.shape),
                )
            },
        )
        for key, value in values.items()
    }
    parts = {(key, "value"): value for key, value in values.items()}
    codes = (
        torch.arange(128, dtype=torch.float32).reshape(8, 16).to(torch.float8_e4m3fn)
    )
    scale = torch.linspace(0.01, 0.08, 8, dtype=torch.float32)
    for key, encoding, scales in (
        ("flat", flat, scale),
        ("keep", keep, scale[:, None]),
    ):
        tensors[key] = Tensor(
            "bf16",
            (8, 16),
            encoding,
            {
                "data": Part("f8_e4m3fn", (8, 16)),
                "scale": Part("f32", tuple(scales.shape)),
            },
        )
        parts[key, "data"], parts[key, "scale"] = codes, scales
    unsupported_encoding = next(
        digest for alias, digest in tensorfs.seed_digests() if alias == "mxfp8/1"
    )
    tensors["unsupported"] = Tensor(
        "bf16",
        (8, 16),
        unsupported_encoding,
        {"data": Part("f8_e4m3fn", (8, 16)), "scale": Part("u8", (8, 1))},
    )
    parts["unsupported", "data"] = codes
    parts["unsupported", "scale"] = torch.full((8, 1), 127, dtype=torch.uint8)
    declaration = Derivation(
        {}, {"dit": Target(add=tensors)}, {}, [("dit", key) for key in tensors]
    )
    identity = tensorfs.object_id(b"torch-interface-fixture")
    writer = store.begin_derived(
        identity,
        1,
        *declaration.native_arguments(sum(value.nbytes for value in parts.values())),
        work_fingerprint=identity,
    )
    for (key, role), value in parts.items():
        writer.add_part(
            "dit",
            key,
            role,
            io.BytesIO(value.reshape(-1).view(torch.uint8).numpy().tobytes()),
        )
    manifest = writer.commit()["manifest"]
    return (
        Source("sha256:" + manifest["sha256"], manifest["length"]),
        tensors,
        parts,
        values,
    )


@contextlib.contextmanager
def source_handle(store, source, current=lambda: None):
    local, remote = socket.socketpair()
    errors = []

    def serve():
        try:
            serve_source(store, remote.detach(), source.manifest, source.length, check_current=current)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    handle = SourceCapability.from_fd(source.manifest, source.length, local.detach())
    try:
        yield handle, errors
    finally:
        handle.close()
        thread.join(10)
        assert not thread.is_alive()


def refuses(function, phrase):
    try:
        function()
    except (ValueError, tensorfs.errors.Refusal) as exc:
        assert phrase in str(exc), (phrase, exc)
    else:
        raise AssertionError("invalid tensor access was accepted")


def main():
    import torch
    import torchao
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference

    assert metadata.version("torchao") == torchao.__version__
    assert torchao.__version__ in {"0.18.0", "0.18.0+cpu"}
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        or os.environ.get("TENSORFS_TEST_CUDA") == "1"
    )
    with tempfile.TemporaryDirectory(prefix="tensorfs-torch-") as area:
        store = tensorfs.Store.init(Path(area) / "store")
        source, specs, original_parts, values = fixture(store)
        with tensorfs.tensors(
            store, source.manifest, components=("dit",), buffer_bytes=31
        ) as reader:
            for key, original in values.items():
                got = reader.get("dit", key)
                assert type(got) is torch.Tensor and torch.equal(got, original)
                assert got.data_ptr() != original.data_ptr() or not got.numel()
            for key in ("flat", "keep"):
                got = reader.get("dit", key)
                assert type(got).__name__ == "Float8Tensor"
                assert torch.equal(
                    got.qdata.view(torch.uint8),
                    original_parts[key, "data"].view(torch.uint8),
                )
                assert torch.equal(
                    got.scale.flatten(), original_parts[key, "scale"].flatten()
                )
                expected = (
                    original_parts[key, "data"].float()
                    * original_parts[key, "scale"].reshape(8, 1)
                ).to(torch.bfloat16)
                assert torch.equal(got.dequantize(), expected)
                # Default upstream weight-only semantics work in a standard Linear.
                linear = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
                linear.weight = torch.nn.Parameter(got, requires_grad=False)
                x = torch.ones(2, 16, dtype=torch.bfloat16)
                assert torch.equal(linear(x), torch.nn.functional.linear(x, expected))
            out = torch.empty_like(values["plain"])
            previous_version = out._version
            assert reader.get("dit", "plain", out=out) is out
            assert out._version > previous_version
            parameter = torch.nn.Parameter(torch.empty_like(out))
            assert reader.get("dit", "plain", out=parameter) is parameter
            assert parameter.requires_grad and torch.equal(
                parameter.detach(), values["plain"]
            )
            allocated = []

            def allocate(ask):
                value = torch.empty(ask.shape, dtype=ask.dtype, device=ask.device)
                allocated.append(value)
                return value

            refuses(
                lambda: reader.get("dit", "unsupported", allocator=allocate),
                "unsupported tensor encoding",
            )
            assert not allocated
            shared = torch.empty(128, dtype=torch.uint8)

            def overlapping(ask):
                needed = 128 if ask.role == "data" else 32
                return shared[:needed].view(ask.dtype).reshape(ask.shape)

            refuses(lambda: reader.get("dit", "keep", allocator=overlapping), "overlap")
            try:
                reader.spec("dit", "plain").parts["value"] = Part("f16", (8, 16))
            except TypeError:
                pass
            else:
                raise AssertionError("caller mutated stored tensor metadata")

            weight = reader.get(
                "dit",
                "keep",
                allocator=allocate,
                mm_config=Float8MMConfig(use_fast_accum=True),
                kernel_preference=KernelPreference.TORCH,
            )
            assert (
                weight.qdata is allocated[0]
                and weight.scale.data_ptr() == allocated[1].data_ptr()
            )
            assert (
                weight.mm_config.use_fast_accum
                and weight.kernel_preference == KernelPreference.TORCH
            )
            assert set(id(t) for t in tensorfs.physical_tensors(weight)) == set(
                id(t) for t in (weight.qdata, weight.scale)
            )
            previous_versions = [
                part._version for part in tensorfs.physical_tensors(weight)
            ]
            refilled = reader.get("dit", "keep", out=weight)
            assert refilled is weight
            assert all(
                part._version > before
                for part, before in zip(
                    tensorfs.physical_tensors(weight), previous_versions
                )
            )
            refuses(lambda: reader.get("other", "plain"), "outside")
            refuses(
                lambda: reader.get("dit", "plain", out=torch.empty(2)), "shape or dtype"
            )
            refuses(
                lambda: reader.get("dit", "plain", mm_config=Float8MMConfig()),
                "do not accept",
            )
            refuses(
                lambda: reader.get("dit", "plain", out=out, allocator=allocate),
                "mutually exclusive",
            )
            direct = tensorfs.from_parts(
                specs["keep"],
                {
                    "data": original_parts["keep", "data"],
                    "scale": original_parts["keep", "scale"],
                },
            )
            assert direct.qdata is original_parts["keep", "data"]
            bad = replace(specs["keep"], encoding="sha256:" + "a" * 64)
            refuses(lambda: tensorfs.from_parts(bad, {}), "unsupported tensor encoding")
            refuses(
                lambda: tensorfs.from_parts(replace(specs["plain"], shape=(0, 16)), {}),
                "positive integers",
            )
            refuses(
                lambda: reader.get(
                    "dit", "keep", out=weight, mm_config=Float8MMConfig()
                ),
                "different FP8",
            )
            out.fill_(-1)
            assert torch.equal(reader.get("dit", "plain"), values["plain"])
        assert torch.isfinite(weight.dequantize()).all()
        refuses(lambda: reader.get("dit", "plain"), "closed")
        # Per-tensor local leases are closed even while returned allocations live.
        tensorfs.gc(store.root)
        with source_handle(store, source) as (handle, errors):
            with tensorfs.tensors(
                handle, components=("dit",), buffer_bytes=17
            ) as scoped:
                assert torch.equal(scoped.get("dit", "plain"), values["plain"])
                assert torch.equal(
                    scoped.get("dit", "flat").qdata.view(torch.uint8),
                    original_parts["flat", "data"].view(torch.uint8),
                )
            handle.check()  # Reader borrows, rather than closes, the caller's capability.
            assert not errors
        live = [True]

        def current():
            if not live[0]:
                raise tensorfs.errors.LeaseRevoked("source execution ended")

        with source_handle(store, source, current) as (handle, errors):
            scoped = tensorfs.tensors(handle, components=("dit",))
            live[0] = False
            revoked_allocations = []

            def revoked_allocator(ask):
                revoked_allocations.append(ask)
                return torch.empty(ask.shape, dtype=ask.dtype, device=ask.device)

            refuses(
                lambda: scoped.get("dit", "plain", allocator=revoked_allocator),
                "source",
            )
            assert not revoked_allocations, "revoked source allocated a destination"
        checks = [0]
        stop_after = [None]

        def revoke_mid_read():
            checks[0] += 1
            if stop_after[0] is not None and checks[0] >= stop_after[0]:
                raise tensorfs.errors.LeaseRevoked("source revoked during read")

        with source_handle(store, source, revoke_mid_read) as (handle, errors):
            with tensorfs.tensors(
                handle, components=("dit",), buffer_bytes=17
            ) as scoped:
                stop_after[0] = checks[0] + 8
                partial = torch.full_like(values["plain"], -100)
                previous_version = partial._version
                refuses(lambda: scoped.get("dit", "plain", out=partial), "source")
                assert partial._version > previous_version
                assert not torch.equal(partial, values["plain"])
                assert not torch.equal(partial, torch.full_like(partial, -100))
        if os.environ.get("TENSORFS_TEST_CUDA") == "1":
            with tensorfs.tensors(store, source.manifest, buffer_bytes=31) as gpu:
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    actual = gpu.get("dit", "keep", device="cuda")
                    plain = gpu.get("dit", "plain", device="cuda")
                    plain_out = torch.empty_like(plain)
                    assert gpu.get("dit", "plain", out=plain_out) is plain_out
                    assert gpu.get("dit", "keep", out=actual) is actual
                    refuses(
                        lambda: gpu.get("dit", "plain", out=plain_out, device="cpu"),
                        "different device",
                    )
                assert torch.equal(
                    actual.qdata.cpu().view(torch.uint8),
                    original_parts["keep", "data"].view(torch.uint8),
                )
                assert torch.equal(plain.cpu(), values["plain"])
                assert torch.equal(plain_out.cpu(), values["plain"])
            torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "passed",
                "native_version": tensorfs.__version__,
                "torch": torch.__version__,
                "torchao": torchao.__version__,
                "torchao_path": torchao.__file__,
                "cuda_exercised": os.environ.get("TENSORFS_TEST_CUDA") == "1",
            }
        )
    )


if __name__ == "__main__":
    main()
