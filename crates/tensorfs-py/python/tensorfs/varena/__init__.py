"""varena, compiled in `tensorfs._ext.varena` and typed in `__init__.pyi`.

The memory of the process computing with model weights. `Gpu` places a checkpoint's regions on
a GPU: home pages, scratch slots, and pins that are tensor lifetimes. `Host` serves them from
host memory where there is no GPU. `Ledger` is the per-GPU table every process's core shares.
The host tier functions are the machine's (TensorD's) half.
"""

from .._ext.varena import (
    Ceiling,
    Closed,
    Copy,
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
    ceiling,
    copies,
    hold_tier,
    release_tier,
    regions,
    seal,
)
from .._ext.varena import Admit, Ledger, LedgerError, Row, RowFacts, Victim

__all__ = [
    "Ceiling",
    "Closed",
    "Copy",
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
    "ceiling",
    "copies",
    "hold_tier",
    "release_tier",
    "regions",
    "seal",
]
__all__ += ["Admit", "Ledger", "LedgerError", "Row", "RowFacts", "Victim"]
