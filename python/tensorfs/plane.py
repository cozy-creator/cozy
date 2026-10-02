"""The weight plane (`crates/tensorfs-plane/API.md`), compiled in `tensorfs._ext.plane`.

Typed in `plane.pyi`. The plane moves bytes; cozy-runtime decides.
"""

from ._ext.plane import (
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
    WeightSet,
    hold_tier,
    release_tier,
)

__all__ = [
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
    "WeightSet",
    "hold_tier",
    "release_tier",
]
