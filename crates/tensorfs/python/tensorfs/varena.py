"""The weight plane (`crates/varena/API.md`), compiled in `tensorfs._ext.varena`, typed in
`varena.pyi`. Mechanism only: callers want, acquire and drop; nothing here sets a budget.
"""

from ._ext.varena import (
    Closed,
    CudaError,
    Cursor,
    DeviceFull,
    DeviceView,
    Floor,
    Invalid,
    IoError,
    Lease,
    PageCache,
    Plane,
    PlaneError,
    Poisoned,
    SealedLayout,
    Ticket,
    WeightSet,
)

__all__ = [
    "Closed",
    "CudaError",
    "Cursor",
    "DeviceFull",
    "DeviceView",
    "Floor",
    "Invalid",
    "IoError",
    "Lease",
    "PageCache",
    "Plane",
    "PlaneError",
    "Poisoned",
    "SealedLayout",
    "Ticket",
    "WeightSet",
]
