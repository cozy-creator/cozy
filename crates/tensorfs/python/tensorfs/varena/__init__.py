"""varena, compiled in `tensorfs._ext.varena` and typed in `__init__.pyi`.

The memory of the process computing with model weights. `Gpu` places a checkpoint's regions on
a GPU: home pages, scratch slots, and pins that are tensor lifetimes. `Host` serves them from
host memory where there is no GPU. `Ledger` is the per-GPU table every process's core shares.
The host tier functions are the machine's (TensorD's) half.
"""

from typing import TYPE_CHECKING

from .._ext import STORE as _STORE
from .._ext.varena import (
    Closed,
    Counters,
    CudaError,
    CudaUnavailable,
    DeviceFull,
    Gpu,
    GpuStats,
    Host,
    Invalid,
    IoError,
    Part,
    Pinned,
    SetStats,
    Unsupported,
    VarenaError,
    WeightSet,
    hold_tier,
    release_tier,
    reserve,
)
from .._ext.varena import Admit, Ledger, LedgerError, Row, RowFacts, Victim

# Checkpoint-backed sets and owner-side fills need the native store feature, enabled in every
# supported Python wheel.
if TYPE_CHECKING or _STORE:
    from .._ext.varena import fill_sealed, seal

__all__ = [
    "Closed",
    "Counters",
    "CudaError",
    "CudaUnavailable",
    "DeviceFull",
    "Gpu",
    "GpuStats",
    "Host",
    "Invalid",
    "IoError",
    "Part",
    "Pinned",
    "SetStats",
    "Unsupported",
    "VarenaError",
    "WeightSet",
    "hold_tier",
    "release_tier",
    "reserve",
]
__all__ += ["Admit", "Ledger", "LedgerError", "Row", "RowFacts", "Victim"]
if _STORE:
    __all__ += ["fill_sealed", "seal"]
