#!/usr/bin/env python3
"""Exercise the request-time in-memory fit seam through the installed wheel."""

from pathlib import Path

import tensorfs


header_bytes = Path("vectors/cases/header/plain-fp8.cbor").read_bytes()
header = tensorfs.parse_header(header_bytes)
requirements = [
    tensorfs.TensorRequirement(
        component,
        key,
        tensor["logical"]["shape"],
        tensor["logical"]["logical_dtype"],
    )
    for component, tensors in header["components"].items()
    for key, tensor in tensors.items()
]

answer = tensorfs.fit(
    requirements,
    header_bytes,
    custody="canonical",
    encoded_leaves=True,
)
assert answer["ok"] and answer["code"] == "ok", answer

planted = list(requirements)
first = planted[0]
planted[0] = tensorfs.TensorRequirement(
    first.component,
    first.key,
    [7],
    first.logical_dtype,
)
answer = tensorfs.fit(
    planted,
    header_bytes,
    custody="local",
    encoded_leaves=True,
)
assert not answer["ok"] and answer["code"] == "shape_mismatch", answer

print("request-time in-memory fit green")
