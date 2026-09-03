#!/usr/bin/env python3
"""Live Python author-seam proof and red arms for canonical TensorRequirements bytes."""

from __future__ import annotations

import json
from collections.abc import Callable

import tensorfs


contract = "sha256:" + "11" * 32
encoding_a = "sha256:" + "22" * 32
encoding_b = "sha256:" + "33" * 32
tensors = [
    tensorfs.TensorRequirement("dit", "block.0.weight", "bf16", [2, 4], "any"),
    tensorfs.TensorRequirement(
        "dit", "block.1.weight", "f32", [4], [encoding_a, encoding_b]
    ),
]
raw = tensorfs.tensor_requirements(contract, tensors)
document: dict[str, object] = json.loads(raw)
assert document["contract"] == contract
assert document["constructed_components"] == ["dit"]
components = document["components"]
assert isinstance(components, dict)
assert list(components) == ["dit"]
assert "pinned" not in components["dit"]["block.0.weight"]
assert tensorfs.object_id(raw).startswith("sha256:")
pinned = tensorfs.tensor_requirements(
    contract, [tensorfs.TensorRequirement("dit", "ids", "i64", [77], "any", pinned=True)]
)
assert json.loads(pinned)["components"]["dit"]["ids"]["pinned"] is True


def refuses(code: str, call: Callable[[], bytes]) -> None:
    try:
        call()
    except tensorfs.errors.Refusal as error:
        assert error.code == code, (error.code, error.detail)
    else:
        raise AssertionError(f"TensorRequirements author accepted {code} red arm")


refuses("EMPTY_COMPONENTS", lambda: tensorfs.tensor_requirements(contract, []))
refuses(
    "MISSING_COMPONENT",
    lambda: tensorfs.tensor_requirements(contract, tensors, constructed_components=["vae"]),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.tensor_requirements(contract, [tensors[1], tensors[0]]),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.tensor_requirements(contract, [tensors[0], tensors[0]]),
)
refuses("MALFORMED_DIGEST", lambda: tensorfs.tensor_requirements("not-a-digest", tensors))
refuses(
    "DTYPE_UNKNOWN",
    lambda: tensorfs.tensor_requirements(
        contract, [tensorfs.TensorRequirement("dit", "weight", "", [1], "any")]
    ),
)
refuses(
    "SORT_ORDER",
    lambda: tensorfs.tensor_requirements(
        contract,
        [
            tensorfs.TensorRequirement(
                "dit", "weight", "bf16", [1], [encoding_b, encoding_a]
            )
        ],
    ),
)

print("TensorRequirements Python author seam green")
