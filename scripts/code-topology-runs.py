#!/usr/bin/env python3
"""Live Python author-seam proof and red arms for canonical CodeTopology bytes."""

from __future__ import annotations

import json
from collections.abc import Callable

import tensorfs


contract = "sha256:" + "11" * 32
encoding_a = "sha256:" + "22" * 32
encoding_b = "sha256:" + "33" * 32
tensors = [
    tensorfs.CodeTensor("dit", "block.0.weight", "bf16", [2, 4], "any"),
    tensorfs.CodeTensor(
        "dit", "block.1.weight", "f32", [4], [encoding_a, encoding_b]
    ),
]
raw = tensorfs.code_topology(contract, tensors)
document: dict[str, object] = json.loads(raw)
assert document["contract"] == contract
assert document["constructed_components"] == ["dit"]
components = document["components"]
assert isinstance(components, dict)
assert list(components) == ["dit"]
assert "pinned" not in components["dit"]["block.0.weight"]
assert tensorfs.object_id(raw).startswith("sha256:")
pinned = tensorfs.code_topology(
    contract, [tensorfs.CodeTensor("dit", "ids", "i64", [77], "any", pinned=True)]
)
assert json.loads(pinned)["components"]["dit"]["ids"]["pinned"] is True


def refuses(code: str, call: Callable[[], bytes]) -> None:
    try:
        call()
    except tensorfs.errors.Refusal as error:
        assert error.code == code, (error.code, error.detail)
    else:
        raise AssertionError(f"CodeTopology author accepted {code} red arm")


refuses("EMPTY_COMPONENTS", lambda: tensorfs.code_topology(contract, []))
refuses(
    "MISSING_COMPONENT",
    lambda: tensorfs.code_topology(contract, tensors, constructed_components=["vae"]),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.code_topology(contract, [tensors[1], tensors[0]]),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.code_topology(contract, [tensors[0], tensors[0]]),
)
refuses("MALFORMED_DIGEST", lambda: tensorfs.code_topology("not-a-digest", tensors))
refuses(
    "DTYPE_UNKNOWN",
    lambda: tensorfs.code_topology(
        contract, [tensorfs.CodeTensor("dit", "weight", "", [1], "any")]
    ),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.code_topology(
        contract,
        [
            tensorfs.CodeTensor(
                "dit", "weight", "bf16", [1], [encoding_b, encoding_a]
            )
        ],
    ),
)

print("CodeTopology Python author seam green")
