"""One connected native writer's bounded local byte channel.

The server owns an already-authorized native writer. The peer cannot name a Store,
transaction, epoch, path or administrative operation. Pull-driven part streams can
perform nested source reads without whole-role spooling or a second reader thread.
No byte, checkpoint or receipt state is persisted here.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import threading
from collections.abc import Buffer, Callable, Mapping
from typing import Any, ContextManager, Literal, Protocol, cast

from . import errors
from ._ext import DerivedWriter

_HEADER = struct.Struct("!BI")
_JSON = 0
_BYTES = 1
_CHUNK = 1 << 20
_METADATA = 64 << 20
_READ = 64 << 20


class Reader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _BufferReader:
    def __init__(self, source: Buffer) -> None:
        self.view = memoryview(source).cast("B")

    def read(self, size: int = -1) -> bytes:
        chunk = self.view[:size] if size >= 0 else self.view
        self.view = self.view[len(chunk) :]
        return bytes(chunk)


def _reader(source: Reader | Buffer) -> Reader:
    return _BufferReader(source) if isinstance(source, Buffer) else source


class ChannelRefusal(errors.Refusal):
    def __init__(self, code: str, detail: str) -> None:
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


def _error(code: str, detail: str) -> Exception:
    kind = errors.CODES.get(code)
    if kind is None:
        return ChannelRefusal(code, detail)
    error = kind(f"{code}: {detail}")
    error.detail = detail
    return error


def _socket(fd: int) -> socket.socket:
    sock = socket.socket(fileno=fd)
    if sock.family != socket.AF_UNIX or sock.type != socket.SOCK_STREAM:
        sock.close()
        raise ChannelRefusal(
            "SOURCE_NOT_ALLOWED", "derived channel requires a local stream"
        )
    sock.getpeername()  # Must already be connected; never dial a path or endpoint.
    sock.set_inheritable(False)
    return sock


class _Channel:
    def __init__(self, fd: int) -> None:
        self.socket = _socket(fd)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()

    def _exact(self, length: int) -> bytes:
        buffer = bytearray(length)
        view = memoryview(buffer)
        while view:
            n = self.socket.recv_into(view)
            if not n:
                raise EOFError("derived channel disconnected")
            view = view[n:]
        return bytes(buffer)

    def send(self, value: Mapping[str, Any]) -> None:
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > _METADATA:
            raise ChannelRefusal("SIZE_CAP", "derived metadata exceeds its bound")
        self.socket.sendall(_HEADER.pack(_JSON, len(raw)))
        self.socket.sendall(raw)

    def data(self, value: bytes | memoryview) -> None:
        if len(value) > _CHUNK:
            raise ChannelRefusal("SIZE_CAP", "derived byte frame exceeds its bound")
        self.socket.sendall(_HEADER.pack(_BYTES, len(value)))
        self.socket.sendall(value)

    def receive(self) -> dict[str, Any] | bytes:
        kind, length = _HEADER.unpack(self._exact(_HEADER.size))
        if kind not in (_JSON, _BYTES) or length > (
            _METADATA if kind == _JSON else _CHUNK
        ):
            raise ChannelRefusal("SIZE_CAP", "invalid derived frame")
        raw = self._exact(length)
        if kind == _BYTES:
            return raw
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ChannelRefusal("WRONG_TYPE", "derived frame must be an object")
        return value

    def reply(self) -> dict[str, Any]:
        value = self.receive()
        if not isinstance(value, dict):
            raise ChannelRefusal("WRONG_TYPE", "expected derived metadata")
        if "error" in value:
            raise _error(str(value["error"]), str(value.get("detail", "")))
        return value


class DerivedTransaction:
    """Executor-side handle for one writer. FD ownership transfers to this object."""

    def __init__(self, fd: int) -> None:
        self._channel: _Channel | None = _Channel(fd)
        self._lock = threading.RLock()
        self.receipt: dict[str, Any] | None = None

    @classmethod
    def replay(cls, receipt: Mapping[str, Any]) -> DerivedTransaction:
        """The owner supplies a receipt only after validating retained native custody."""
        value = cls.__new__(cls)
        value._channel = None
        value._lock = threading.RLock()
        value.receipt = dict(receipt)
        return value

    @classmethod
    def accept(cls, fd: int) -> DerivedTransaction:
        """Receive the owner's live-writer or validated committed-result decision."""
        value = cls(fd)
        receipt = value._call("receipt")
        if receipt is not None:
            if not isinstance(receipt, dict):
                value.close()
                raise ChannelRefusal("WRONG_TYPE", "native receipt must be an object")
            value.receipt = receipt
            value.close()
        return value

    def _open(self) -> _Channel:
        if self._channel is None:
            raise ChannelRefusal("TRANSACTION_CLOSED", "derived handle is closed")
        return self._channel

    def _call(self, op: str, **fields: Any) -> Any:
        with self._lock:
            channel = self._open()
            try:
                channel.send({"op": op, **fields})
                result = channel.reply()
                if "result" not in result:
                    raise ChannelRefusal("MISSING_FIELD", "derived answer has no result")
                return result["result"]
            except BaseException:
                self.close()
                raise

    def completed_parts(self) -> list[tuple[str, str, str]]:
        return [tuple(row) for row in self._call("completed_parts")]

    def completed_configs(self) -> list[str]:
        return cast(list[str], self._call("completed_configs"))

    def checkpoint(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._call("checkpoint"))

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: object,
    ) -> None:
        view = memoryview(into).cast("B")  # type: ignore[arg-type]
        if view.readonly or not view.c_contiguous or not 0 < len(view) <= _READ:
            raise ChannelRefusal(
                "BUFFER_SIZE", "read needs bounded writable contiguous memory"
            )
        with self._lock:
            channel = self._open()
            try:
                channel.send(
                    {
                        "op": "read",
                        "source": source,
                        "component": component,
                        "key": key,
                        "role": role,
                        "offset": offset,
                        "length": len(view),
                    }
                )
                if channel.reply().get("length") != len(view):
                    raise ChannelRefusal(
                        "LENGTH_MISMATCH", "derived read length changed"
                    )
                while view:
                    data = channel.receive()
                    if not isinstance(data, bytes) or not data or len(data) > len(view):
                        raise ChannelRefusal(
                            "LENGTH_MISMATCH", "invalid derived read data"
                        )
                    view[: len(data)] = data
                    view = view[len(data) :]
            except BaseException:
                self.close()
                raise

    def _add(self, op: str, source: Reader | Buffer, **fields: str) -> Any:
        with self._lock:
            channel = self._open()
            reader = _reader(source)
            try:
                channel.send({"op": op, **fields})
                while True:
                    response = channel.reply()
                    if "result" in response:
                        return response["result"]
                    if (
                        "pull" not in response
                        or type(response["pull"]) is not int
                        or not 0 < response["pull"] <= _CHUNK
                    ):
                        raise ChannelRefusal("NUMBER_RANGE", "invalid derived pull")
                    chunk = reader.read(response["pull"])
                    if not isinstance(chunk, bytes) or len(chunk) > response["pull"]:
                        raise ChannelRefusal(
                            "LENGTH_MISMATCH", "producer exceeded requested read"
                        )
                    channel.data(chunk)
            except BaseException:
                self.close()
                raise

    def add_part(
        self, component: str, key: str, role: str, reader: Reader | Buffer
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._add("part", reader, component=component, key=key, role=role),
        )

    def add_config(self, name: str, reader: Reader | Buffer) -> None:
        self._add("config", reader, name=name)

    def commit(self) -> dict[str, Any]:
        if self.receipt is None:
            self.receipt = cast(dict[str, Any], self._call("commit"))
            self.close()
        return self.receipt

    def close(self) -> None:
        channel, self._channel = self._channel, None
        if channel is not None:
            channel.close()

    def fence(self) -> None:
        if self._channel is not None:
            try:
                self._call("fence")
            finally:
                self.close()

    def __enter__(self) -> DerivedTransaction:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


class _PullReader:
    def __init__(
        self,
        channel: _Channel,
        read: Callable[[dict[str, Any]], None],
        check_current: Callable[[], None],
    ) -> None:
        self.channel, self.read_source, self.check_current = (
            channel,
            read,
            check_current,
        )
        self.ended = False

    def read(self, size: int = -1) -> bytes:
        self.check_current()
        if self.ended:
            return b""
        size = min(size if size > 0 else _CHUNK, _CHUNK)
        self.channel.send({"pull": size})
        while True:
            value = self.channel.receive()
            if isinstance(value, bytes):
                if len(value) > size:
                    raise ChannelRefusal(
                        "LENGTH_MISMATCH", "producer exceeded native pull"
                    )
                self.ended = not value
                return value
            self.read_source(value)  # Only bounded reads may nest inside a part.


def _fields(frame: Mapping[str, Any], op: str, names: tuple[str, ...]) -> None:
    if frame.get("op") != op or set(frame) != {"op", *names}:
        raise ChannelRefusal(
            "UNKNOWN_FIELD", "operation is outside this derived handle"
        )


def serve_derived(
    writer: DerivedWriter,
    fd: int,
    *,
    operation_id: str,
    slot: str,
    checkpoint: tuple[str, int] | None,
    check_current: Callable[[], None],
    admit_io: Callable[[str, int], ContextManager[None]],
    record_checkpoint: Callable[[dict[str, Any]], None],
    record_receipt: Callable[[dict[str, Any]], None],
) -> None:
    """Serve one prebound writer until terminal/peer loss, then fence its live handle.

    All hooks are mandatory trusted-owner hooks. They cannot be supplied by the
    peer, and native writer validation is never replaced by their approval.
    """
    channel = _Channel(fd)

    def read_source(frame: dict[str, Any], *, borrowed: bool = False) -> None:
        _fields(
            frame, "read", ("source", "component", "key", "role", "offset", "length")
        )
        length, offset = frame["length"], frame["offset"]
        if (
            type(length) is not int
            or not 0 < length <= _READ
            or type(offset) is not int
            or offset < 0
        ):
            raise ChannelRefusal("RANGE_BOUNDS", "invalid derived source range")
        check_current()
        # Part admission covers its one bounded source-read window. Reacquiring
        # the global storage/capacity lock here would deadlock a lazy producer.
        with contextlib.nullcontext() if borrowed else admit_io("read", length):
            target = bytearray(length)
            writer.source_read_into(
                frame["source"],
                frame["component"],
                frame["key"],
                frame["role"],
                offset,
                target,
            )
            check_current()
            channel.send({"length": length})
            for start in range(0, length, _CHUNK):
                channel.data(memoryview(target)[start : start + _CHUNK])

    try:
        while True:
            frame = channel.receive()
            if not isinstance(frame, dict):
                raise ChannelRefusal("WRONG_TYPE", "expected a derived operation")
            check_current()
            op = frame.get("op")
            result: Any
            if op == "read":
                read_source(frame)
                continue
            if op in ("part", "config"):
                names = ("component", "key", "role") if op == "part" else ("name",)
                _fields(frame, op, names)
                bound = (
                    writer.part_write_bound(
                        frame["component"], frame["key"], frame["role"]
                    )
                    if op == "part"
                    else writer.config_write_bound(frame["name"])
                )
                with admit_io(op, bound):
                    check_current()
                    reader = _PullReader(
                        channel,
                        lambda value: read_source(value, borrowed=True),
                        check_current,
                    )
                    if op == "part":
                        result = writer.add_part(
                            frame["component"], frame["key"], frame["role"], reader
                        )
                    else:
                        writer.add_config(frame["name"], reader)
                        result = None
                    check_current()
            else:
                if op not in (
                    "receipt",
                    "completed_parts",
                    "completed_configs",
                    "checkpoint",
                    "commit",
                    "fence",
                ):
                    raise ChannelRefusal(
                        "SOURCE_NOT_ALLOWED", "operation is outside this derived handle"
                    )
                _fields(frame, op, ())
                with admit_io(op, 0):
                    check_current()
                    if op == "receipt":
                        result = None
                    elif op == "completed_parts":
                        result = writer.completed_parts()
                    elif op == "completed_configs":
                        result = writer.completed_configs()
                    elif op == "checkpoint":
                        result = writer.checkpoint(
                            operation_id, slot, previous=checkpoint
                        )
                        check_current()
                        record_checkpoint(result)
                        if result.get("head") is not None:
                            checkpoint = str(result["head"]), int(result["head_length"])
                    elif op == "commit":
                        result = writer.commit()
                        record_receipt(
                            result
                        )  # Reconcile a native winner even if cancellation raced.
                    else:
                        result = writer.fence()
            channel.send({"result": result})
            if op in ("commit", "fence"):
                return
    except EOFError:
        return
    except Exception as exc:
        with contextlib.suppress(OSError, EOFError):
            channel.send(
                {
                    "error": str(getattr(exc, "code", "IO_FAILED")),
                    "detail": str(
                        getattr(exc, "detail", "derived owner refused operation")
                    )[:1024],
                }
            )
        raise
    finally:
        channel.close()
        writer.fence()


def serve_derived_replay(
    receipt: Mapping[str, Any],
    fd: int,
    *,
    check_current: Callable[[], None],
    record_receipt: Callable[[dict[str, Any]], None],
) -> None:
    """Deliver retained native custody selected by the trusted execution owner."""
    channel = _Channel(fd)
    try:
        request = channel.receive()
        if not isinstance(request, dict):
            raise ChannelRefusal("WRONG_TYPE", "expected a receipt request")
        _fields(request, "receipt", ())
        check_current()
        facts = dict(receipt)
        record_receipt(facts)
        channel.send({"result": facts})
    except EOFError:
        return
    except Exception as exc:
        with contextlib.suppress(OSError, EOFError):
            channel.send({"error": str(getattr(exc, "code", "IO_FAILED")), "detail": "native receipt owner refused"})
        raise
    finally:
        channel.close()
