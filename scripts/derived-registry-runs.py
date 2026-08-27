#!/usr/bin/env python3
"""Focused real-store arm for compiled-registry adoption by a derived transaction."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import subprocess
from typing import Any

import numpy as np
import tensorfs
from cozy_jobs.microscale import encode_fp8_rowwise


def run(*argv: str) -> str:
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n{completed.stderr}"
        )
    return completed.stdout


def target(encoding: str) -> dict[str, Any]:
    return {
        "text_encoder": {
            "source": "bf16",
            "source_component": "text_encoder",
            "drop": ["embeddings.weight"],
            "add": {
                "embeddings.weight": {
                    "logical_dtype": "bf16",
                    "shape": [2, 2],
                    "encoding": encoding,
                    "parts": {
                        "data": {"dtype": "f8_e4m3fn", "shape": [2, 2]},
                        "scale": {"dtype": "f32", "shape": [2]},
                    },
                }
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", required=True, type=pathlib.Path)
    parser.add_argument("--store", required=True, type=pathlib.Path)
    args = parser.parse_args()

    output = run(
        str(args.tfs),
        "checkpoint",
        "write",
        str(args.store),
        "--blocks",
        "0",
        "--scale",
        "8192",
        "--no-configs",
    )
    match = re.search(r"checkpoint_id\s+(sha256:[0-9a-f]{64})", output)
    if match is None:
        raise RuntimeError("checkpoint write did not print checkpoint_id")
    source_id = match.group(1)
    run(str(args.tfs), "root", "register", str(args.store), "plain-source", source_id)

    store = tensorfs.Store.open(str(args.store))
    source_length = len(store.document(source_id))
    source = store.snapshot(source_id)
    source_header = tensorfs.parse_header(source["header"])
    if len(source_header["encodings"]) != 1:
        raise RuntimeError("the source is not a plain/1-only closure")

    variants = [
        digest
        for alias, digest in tensorfs.seed_digests()
        if alias == "fp8-rowwise/1"
    ]
    if len(variants) < 2:
        raise RuntimeError("the compiled registry lacks both fp8-rowwise variants")
    encoding = variants[0]
    changed = encoding[:-1] + ("0" if encoding[-1] != "0" else "1")
    sources = {"bf16": (source_id, source_length)}

    try:
        store.begin_derived(
            tensorfs.object_id(b"derived-registry-unknown"),
            1,
            sources,
            target(changed),
            {},
            12,
        )
    except tensorfs.errors.UnknownEncoding:
        pass
    else:
        raise RuntimeError("a changed/unregistered encoding digest was admitted")

    values = np.array([[1.0, -2.0], [3.0, 0.25]], dtype=np.float32)
    encoded = encode_fp8_rowwise("embeddings.weight", values)
    scale = encoded.companions[0]
    writer = store.begin_derived(
        tensorfs.object_id(b"derived-registry-positive"),
        1,
        sources,
        target(encoding),
        {},
        len(encoded.payload) + len(scale.raw),
    )
    writer.add_part("text_encoder", "embeddings.weight", "data", io.BytesIO(encoded.payload))
    writer.add_part("text_encoder", "embeddings.weight", "scale", io.BytesIO(scale.raw))
    receipt = writer.commit()
    replay = store.derived_lookup(tensorfs.object_id(b"derived-registry-positive"))
    if replay != {"state": "committed", "receipt": receipt}:
        raise RuntimeError("the committed receipt did not replay byte-for-byte")

    spec = json.loads(store.document(encoding))
    expected_registry = {
        encoding,
        f"sha256:{spec['vectors']['sha256']}",
    }
    snapshot_id = f"sha256:{receipt['snapshot']['sha256']}"
    reached = store.walk(snapshot_id)
    reached_registry = {
        row["id"] for row in reached if row["kind"] in {"spec", "vectors"}
    }
    if not expected_registry <= reached_registry:
        raise RuntimeError("target snapshot closure does not reach adopted spec+vectors")

    target_snapshot = store.snapshot(snapshot_id)
    closure = tensorfs.Closure()
    for row in reached:
        if row["kind"] == "spec":
            closure.insert(store.document(row["id"]))
    tensorfs.run_doc("header", target_snapshot["header"], closure)
    tensorfs.run_doc("header-serving", target_snapshot["header"], closure)

    print(
        "DERIVED REGISTRY PASS plain/1->fp8-rowwise/1 "
        f"encoding={encoding} closure_objects=2 receipt_replay=stable "
        "unknown=refused serving=admitted"
    )


if __name__ == "__main__":
    main()
