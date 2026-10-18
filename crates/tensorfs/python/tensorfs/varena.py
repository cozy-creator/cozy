"""The weight plane (`crates/varena/API.md`), compiled in `tensorfs._ext.varena`.

Typed in `varena.pyi`. TensorD coordinates machine-wide memory policy and shared weight
lifetimes. Runtime constructs models and uses the plane/device APIs for GPU allocation,
copies and mappings within TensorD's grants. The library also supplies TensorD's shared
CPU-buffer mechanisms; it is not a separate daemon or policy authority.
"""

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
    Source,
    Ticket,
    Unsupported,
    WeightSet,
    hold_tier,
    release_tier,
    reserve,
)

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
    "Source",
    "Ticket",
    "Unsupported",
    "WeightSet",
    "hold_tier",
    "release_tier",
    "reserve",
]
