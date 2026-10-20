#!/usr/bin/env python3
"""Execute the committed encoded green vectors from an installed AOT wheel on a real GPU.

Only the standard library and TensorFS are needed: no GPU-side Rust build, Torch or Triton.
Run once per allocated GPU family; CUDA execution and bit-exact bytes are actual evidence.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

import tensorfs

CORPUS_SHA256 = "92a0f4f01f4490c2f7f8c391e5eb600f5e5b33d647eed48d91ff7c21e7af57d4"
EXPECTED_CASES = 209
EXPECTED_SCHEMES = 39


def bits(encoded):
    return base64.b64decode(encoded, validate=True)


assert "torch" not in sys.modules and "triton" not in sys.modules
assert importlib.util.find_spec("triton") is None, "run this gate in a TensorFS-only venv without Triton"
assert importlib.util.find_spec("torch") is None, "the core kernel gate must also work without Torch installed"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=0, help="CUDA ordinal in the inherited visible-device scope")
args = parser.parse_args()
if args.device < 0:
    parser.error("--device must be a nonnegative visible CUDA ordinal")
vectors = pathlib.Path(__file__).resolve().parents[2] / "tensorfs/conformance/vectors/cases"
files = sorted(vectors.glob("*.json"))
corpus = hashlib.sha256()
for path in files:
    corpus.update(path.name.encode() + b"\0" + path.read_bytes())
assert corpus.hexdigest() == CORPUS_SHA256, "the frozen conformance corpus is missing or changed"
full = windows = 0
schemes = set()
cases = []
devices = set()
with tempfile.TemporaryDirectory() as temporary:
    for family in files:
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
            arrays = tensors.read(["value", "window"], device=f"cuda:{args.device}")
            label = f"{family.stem}/{case['component']}/{case['key']}"
            actual = arrays["value"].__dlpack_device__()
            assert actual == (2, args.device), label
            devices.add(actual)
            assert arrays["value"].tobytes() == expected, f"{label}: full decode differs"
            full += 1
            result = {"case": label, "scheme": encoding["scheme"], "dtype": case["dtype"],
                      "shape": case["shape"], "full_bytes": len(expected), "full_passed": True}
            if length:
                # Byte offset/length intentionally cross logical element boundaries.
                tensors.read_into("value", arrays["window"], offset=1)
                assert arrays["window"].tobytes() == expected[1:1 + length], f"{label}: window differs"
                windows += 1
                result["window"] = {"offset": 1, "length": length, "passed": True}
            cases.append(result)
            schemes.add(encoding["scheme"])
            checkpoint.close()
            del arrays, tensors, checkpoint, component, writer

assert full == windows == EXPECTED_CASES and len(schemes) == EXPECTED_SCHEMES, (full, windows, schemes)
assert len(devices) == 1
assert "torch" not in sys.modules and "triton" not in sys.modules
runtime = {"torch_installed": importlib.util.find_spec("torch") is not None,
           "triton_installed": importlib.util.find_spec("triton") is not None,
           "torch_imported": "torch" in sys.modules, "triton_imported": "triton" in sys.modules}
assert not any(runtime.values())
evidence = {"schema": "tensorfs.cuda-kernel-conformance/1",
            "device": {"type": "cuda", "ordinal": next(iter(devices))[1]},
            "counts": {"full": full, "windows": windows, "total": full + windows,
                       "cases": len(cases), "schemes": len(schemes)},
            "schemes": sorted(schemes), "cases": cases, "corpus_sha256": corpus.hexdigest(),
            "routing": {"prepared": "Kernel", "host_only_exclusions": []}, "runtime": runtime}
print(f"CUDA prebuilt conformance: {full + windows} full/window vectors across {len(schemes)} schemes bit-exact")
# The final stdout line is the complete structured receipt. The pinned native CPU routing
# gate proves all corpus plans are Prepared::Kernel; a Host fallback is not GPU proof.
print(json.dumps(evidence, sort_keys=True, allow_nan=False))
