"""Create the fixture's tiny native checkpoint in an explicitly supplied TensorFS store."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import tensorfs
import torch
from attention_benchmark import AttentionPipeline
from cozy_runtime.author import Config
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("--org", required=True)
    parser.add_argument("--name", default="attention-kernel-benchmark")
    args = parser.parse_args()
    torch.manual_seed(17)
    pipeline = AttentionPipeline(Config({}))
    module = pipeline.components["dit"]
    rows = {
        key: value.detach().cpu().contiguous()
        for key, value in module.state_dict().items()
    }
    encoding = dict(tensorfs.seed_digests())["plain/1"]
    types = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int64: "i64"}
    total = sum(value.numel() * value.element_size() for value in rows.values())
    identity = hashlib.sha256(b"attention-kernel-benchmark/1").hexdigest()
    store = tensorfs.Store.ensure(str(args.store.resolve()))
    writer = store.begin_derived(
        "sha256:" + identity,
        1,
        *Derivation(
            sources={},
            targets={
                "dit": Target(
                    add={
                        key: Tensor(
                            types[value.dtype],
                            tuple(value.shape),
                            encoding,
                            {"value": Part(types[value.dtype], tuple(value.shape))},
                        )
                        for key, value in rows.items()
                    }
                )
            },
            configs={"model": NativeConfig("add")},
            order=tuple(("dit", key) for key in rows),
        ).native_arguments(total + 1_000_000),
        work_fingerprint="sha256:" + identity,
    )
    for key, value in rows.items():
        writer.add_part(
            "dit", key, "value", io.BytesIO(value.view(torch.uint8).numpy().tobytes())
        )
    writer.add_config("model", io.BytesIO(b"{}"))
    receipt = writer.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = int(receipt["manifest"]["length"])
    operation = store.begin_operation("attention-benchmark-seed", args.org, args.name)
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "bf16", manifest, length)
    print(
        json.dumps(
            {
                "model": f"{args.org}/{args.name}",
                "release": "1",
                "lane": "bf16",
                "manifest": manifest,
                "length": length,
                "weight_bytes": total,
            }
        )
    )


if __name__ == "__main__":
    main()
