#!/usr/bin/env python3
"""The installed wheel carries every prebuilt image, with no Triton or GPU at runtime."""

import hashlib
import json
import pathlib
import sys

from tensorfs import _ext


manifest = pathlib.Path(__file__).parents[2] / "tensorfs/kernels/aot/manifest.json"
records = json.loads(manifest.read_text())["binaries"]
assert len(records) == 324
for expected in records:
    image, symbol, shared, warps = _ext._kernel_binary(expected["family"], expected["kind"], expected["dtype"])
    assert image.startswith(b"\x7fELF")
    assert hashlib.sha256(image).hexdigest() == expected["sha256"], expected
    assert (symbol, shared, warps) == (expected["symbol"], expected["shared"], expected["warps"])
assert "triton" not in sys.modules and "torch" not in sys.modules
print(f"installed wheel: {len(records)} prebuilt CUDA images read back exactly; no Triton, framework or GPU")
