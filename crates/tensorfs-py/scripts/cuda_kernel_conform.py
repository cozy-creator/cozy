#!/usr/bin/env python3
"""Execute the committed encoded green vectors from an installed AOT wheel on a real GPU.

Only the standard library and TensorFS are needed: no GPU-side Rust build, Torch or Triton.
Run once per allocated GPU family; CUDA execution and bit-exact bytes are actual evidence.
"""

import base64
import importlib.util
import json
import pathlib
import sys
import tempfile

import tensorfs


def bits(encoded):
    return base64.b64decode(encoded, validate=True)


assert "torch" not in sys.modules and "triton" not in sys.modules
assert importlib.util.find_spec("triton") is None, "run this gate in a TensorFS-only venv without Triton"
assert importlib.util.find_spec("torch") is None, "the core kernel gate must also work without Torch installed"
vectors = pathlib.Path(__file__).resolve().parents[2] / "tensorfs/conformance/vectors/cases"
checked = 0
schemes = set()
with tempfile.TemporaryDirectory() as temporary:
    for family in sorted(vectors.glob("*.json")):
        for case in json.loads(family.read_text())["cases"]:
            encoding = case.get("encoding")
            if encoding is None or case.get("expect") is None:
                continue
            expected = bits(case["expect"])
            writer = tensorfs.Writer(cache_dir=temporary)
            component = writer.component("model", cls="test.Conformance", config={})
            parts = {part["role"]: (part["dtype"], part["shape"], bits(part["bits"]))
                     for part in case["parts"]}
            component.put_encoded("value", dtype=case["dtype"], shape=case["shape"],
                                  scheme=encoding["scheme"], params=encoding.get("params", {}), parts=parts)
            length = min(13, max(0, len(expected) - 1))
            component.put("window", bytes(length), dtype="U8", shape=[length])
            checkpoint = writer.commit()
            tensors = checkpoint.components["model"].tensors
            arrays = tensors.read_all(["value", "window"], device="cuda:0")
            label = f"{family.stem}/{case['component']}/{case['key']}"
            assert arrays["value"].__dlpack_device__() == (2, 0), label
            assert arrays["value"].tobytes() == expected, f"{label}: full decode differs"
            checked += 1
            if length:
                # Byte offset/length intentionally cross logical element boundaries.
                tensors.read_into("value", arrays["window"], offset=1)
                assert arrays["window"].tobytes() == expected[1:1 + length], f"{label}: window differs"
                checked += 1
            schemes.add(encoding["scheme"])
            checkpoint.close()
            del arrays, tensors, checkpoint, component, writer

assert checked > 50, f"only {checked} encoded full/window vectors were checked"
assert "torch" not in sys.modules and "triton" not in sys.modules
print(f"CUDA prebuilt conformance: {checked} full/window vectors across {len(schemes)} schemes bit-exact; no Torch/Triton runtime")
