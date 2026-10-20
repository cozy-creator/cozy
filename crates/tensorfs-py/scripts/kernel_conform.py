#!/usr/bin/env python3
"""The installed wheel carries every prebuilt image, with no Triton or GPU at runtime."""

import hashlib
import importlib.util
import json
import pathlib
import sys

from tensorfs import _ext


helper = pathlib.Path(__file__).parents[1] / "kernels/abi.py"
spec = importlib.util.spec_from_file_location("tensorfs_aot_abi", helper)
assert spec is not None and spec.loader is not None
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)

manifest = pathlib.Path(__file__).parents[2] / "tensorfs/kernels/aot/manifest.json"
records = json.loads(manifest.read_text())["binaries"]
assert len(records) == 378
for expected in records:
    image, symbol, shared, warps = _ext._kernel_binary(expected["family"], expected["kind"], expected["dtype"])
    assert (expected["parameters"], expected["global_scratch_size"], expected["profile_scratch_size"]) == (13, 0, 0)
    assert abi.check_image(image, symbol) == 13
    assert image.startswith(b"\x7fELF")
    assert hashlib.sha256(image).hexdigest() == expected["sha256"], expected
    assert (symbol, shared, warps) == (expected["symbol"], expected["shared"], expected["warps"])
assert "triton" not in sys.modules and "torch" not in sys.modules
print(f"installed wheel: {len(records)} prebuilt CUDA images read back exactly, 13-parameter zero-scratch ABI; no Triton, framework or GPU")
