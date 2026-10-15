"""The executor's half of its fetch channel: Runtime directs what TensorFS fetches.

The owner (TensorD) hands each attempt one connected descriptor. Checkpoints are named by
manifest id (only those the attempt selected) and objects by digest and length; the
executor never sees a hub, a credential or a path. Calls may come from any thread: a
``wait`` blocks only its caller, and answers arrive as their objects land.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from ._derived_channel import ChannelRefusal, _Channel, _error
from .records import ObjectRef

__all__ = ["Fetch"]


class _Call:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.answer: dict[str, Any] = {}


class Fetch:
    """One attempt's fetch channel."""

    def __init__(self, fd: int) -> None:
        self._channel = _Channel(fd)
        self._sending = threading.Lock()
        self._lock = threading.Lock()
        self._next = 0
        self._calls: dict[int, _Call] = {}
        self._closed: BaseException | None = None
        threading.Thread(target=self._read, name="tensorfs-fetch", daemon=True).start()

    @classmethod
    def from_fd(cls, fd: int) -> Fetch:
        """Take ownership of a connected descriptor the owner handed this attempt."""
        return cls(fd)

    def schedule(self, manifest: str, objects: Sequence[ObjectRef], rank: int, behind: bool = False) -> None:
        """Want ``objects`` of ``manifest`` before this run's later ranks, or behind everything."""
        self._call("schedule", manifest, objects, rank=rank, behind=behind)

    def wait(self, manifest: str, objects: Sequence[ObjectRef]) -> None:
        """Raise ``objects`` to now and return once they are present."""
        self._call("wait", manifest, objects)

    def present(self, manifest: str, objects: Sequence[ObjectRef]) -> list[bool]:
        """Whether each of ``objects`` is present."""
        answer = self._call("present", manifest, objects)
        if not isinstance(answer, list) or len(answer) != len(objects):
            raise ChannelRefusal("WRONG_TYPE", "invalid present answer")
        return [bool(value) for value in answer]

    def close(self) -> None:
        """End the channel: the owner releases everything it wanted."""
        self._channel.close()

    def __enter__(self) -> Fetch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _call(self, op: str, manifest: str, objects: Sequence[ObjectRef], **fields: Any) -> Any:
        call = _Call()
        with self._lock:
            if self._closed is not None:
                raise ChannelRefusal("TRANSFER_FAILED", f"fetch channel closed: {self._closed}")
            self._next += 1
            ident = self._next
            self._calls[ident] = call
        rows = [{"sha256": o["sha256"].removeprefix("sha256:"), "length": o["length"]} for o in objects]
        request = {"id": ident, "op": op, "manifest": manifest, "objects": rows, **fields}
        try:
            with self._sending:
                self._channel.send(request)
        except BaseException:
            with self._lock:
                self._calls.pop(ident, None)
            raise
        call.done.wait()
        answer = call.answer
        if "error" in answer:
            raise _error(str(answer["error"]), str(answer.get("detail", "")))
        return answer.get("result")

    def _read(self) -> None:
        try:
            while True:
                value = self._channel.receive()
                if not isinstance(value, dict) or type(value.get("id")) is not int:
                    raise ChannelRefusal("WRONG_TYPE", "invalid fetch answer")
                with self._lock:
                    call = self._calls.pop(value["id"], None)
                if call is not None:
                    call.answer = value
                    call.done.set()
        except BaseException as exc:  # The channel ended: every pending call fails with it.
            with self._lock:
                self._closed = exc
                calls, self._calls = self._calls, {}
            for call in calls.values():
                call.answer = {"error": "TRANSFER_FAILED", "detail": f"fetch channel closed: {exc}"}
                call.done.set()
