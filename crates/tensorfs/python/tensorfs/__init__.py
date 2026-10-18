"""tensorfs — the typed facade over the compiled core (tensorfs.md §10, tfs-007).

Marshal only. Every byte-level fact — hashing, canonical bytes, geometry, verification and
leases — comes from the extension, and `scripts/fence.py` enforces that this package
computes none of them. What lives here is types and names.

TensorFS is a library used in TensorD and Runtime's executor processes. TensorD makes
weight sources and shared CPU buffers available and coordinates machine-wide memory
policy. Runtime constructs models and performs GPU operations through the local plane
and device APIs within those grants (`tensorfs.varena`), reads sources and writes derived
weights through `tensorfs.derived`, and directs fetching through `tensorfs.fetch`.
TensorFS never imports Torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import errors
from .records import FitResult, Header, Tensor
from ._ext import (
    STORE,
    _fit,
    Closure,
    ReadPlan,
    __version__,
    dtypes as _dtypes,
    manifest_max_bytes,
    object_id,
    parse_header as _parse_header,
    plan,
    seed_digests,
)

# The ordinary Python wheel always includes the store; a Rust build without it is
# plane-only and serves no Store.
if TYPE_CHECKING or STORE:
    from ._ext import DerivedWriter, ReadLease, Store, gc


def parse_header(data: bytes, closure: Closure | None = None) -> Header:
    """Parse a verified CozyTensors header into TensorFS's typed Python projection."""

    return _parse_header(data, closure)


@dataclass(frozen=True)
class TensorRequirement:
    """One tensor constructed for this request."""

    component: str
    key: str
    shape: Sequence[int]
    logical_dtype: str | None = None


def fit(
    requirements: Sequence[TensorRequirement],
    header: bytes,
    *,
    custody: str,
    encoded_leaves: bool,
    device: str | None = None,
    observations: bytes | None = None,
) -> FitResult:
    """Compare a transient factory census with one CozyTensors header."""
    rows = [
        (
            tensor.component,
            tensor.key,
            list(tensor.shape),
            tensor.logical_dtype,
        )
        for tensor in requirements
    ]
    return _fit(
        rows,
        header,
        custody=custody,
        encoded_leaves=encoded_leaves,
        device=device,
        observations=observations,
    )


#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = _dtypes()

__all__ = [
    "STORE",
    "DTYPES",
    "Closure",
    "FitResult",
    "Header",
    "ReadPlan",
    "Tensor",
    "TensorRequirement",
    "__version__",
    "errors",
    "fit",
    "manifest_max_bytes",
    "object_id",
    "parse_header",
    "plan",
    "seed_digests",
]
if STORE:
    __all__ += ["DerivedWriter", "ReadLease", "Store", "gc"]
