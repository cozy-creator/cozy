"""Physical format inspection; ordinary reads use Component.tensors."""

from collections.abc import Mapping, Sequence
from typing_extensions import Buffer

from ._ext import Checkpoint
from .api import DLPack

class Part:
    """One stored part: a plain tensor's data (role `""`) or an encoded tensor's part."""

    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def chunk(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def length(self) -> int: ...

def parts(checkpoint: Checkpoint, component: str, key: str) -> dict[str, Part]:
    """Stored ranges only; does not fetch or load array bytes."""

def decode_into(
    scheme: str,
    parts: Mapping[str, tuple[str, Sequence[int], Buffer | DLPack]],
    out: Buffer | DLPack,
    *,
    dtype: str,
    shape: Sequence[int],
    params: Mapping[str, object] | None = None,
) -> None:
    """Decode existing contiguous part storage into an exact logical destination.

    Part tuples describe format bytes; carriers may be raw byte views. The output
    must match logical dtype and shape, without casting or input/output aliasing.
    CUDA parts and output must be on one device; mixed placement is unsupported.
    """
