"""Physical format inspection; ordinary reads use Component.tensors."""

from ._ext import Checkpoint

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
