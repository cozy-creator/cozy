"""The weight plane (`crates/varena/API.md`), compiled in `tensorfs._ext.varena`.

Typed in `varena.pyi`. TensorD coordinates machine-wide memory policy and shared weight
lifetimes. Runtime constructs models and uses the plane/device APIs for GPU allocation,
copies and mappings within TensorD's grants. The library also supplies TensorD's shared
CPU-buffer mechanisms; it is not a separate daemon or policy authority.
"""

from typing import TYPE_CHECKING

from ._ext import STORE as _STORE
from ._ext.varena import (
    SEALED_HOLES,
    SEALED_WHILE_FILLING,
    BelowFloor,
    BudgetExceeded,
    Closed,
    CudaError,
    CudaUnavailable,
    Cursor,
    DeviceView,
    Invalid,
    IoError,
    Lease,
    LeaseViolation,
    Plane,
    PlaneError,
    Poisoned,
    Shortfall,
    Ticket,
    Unsupported,
    WeightSet,
    hold_tier,
    release_tier,
    reserve,
)

# Store-backed sources and owner-side CPU fills use the native store feature, enabled
# in every supported Python wheel. Managed executor access still requires supplied sources.
if TYPE_CHECKING or _STORE:
    from ._ext.varena import Source, fill_sealed

__all__ = [
    "SEALED_HOLES",
    "SEALED_WHILE_FILLING",
    "BelowFloor",
    "BudgetExceeded",
    "Closed",
    "CudaError",
    "CudaUnavailable",
    "Cursor",
    "DeviceView",
    "Invalid",
    "IoError",
    "Lease",
    "LeaseViolation",
    "Plane",
    "PlaneError",
    "Poisoned",
    "Shortfall",
    "Ticket",
    "Unsupported",
    "WeightSet",
    "hold_tier",
    "release_tier",
    "reserve",
]
if _STORE:
    __all__ += ["Source", "fill_sealed"]
