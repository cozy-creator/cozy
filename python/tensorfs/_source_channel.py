"""One exact source's metadata and native part-reader channel, with no Store escape."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from typing import Any, ContextManager

from . import _ext
from ._derived_channel import ChannelRefusal, _Channel, _CHUNK, _METADATA, _fields
from .derived import Source, SourceInspection

_READ_LIMIT = 64 << 20


class SourceClient:
    def __init__(self, fd: int, source: Source) -> None:
        self.channel = _Channel(fd)
        self.source = source
        self.lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()

    def _result(self, request: dict[str, Any]) -> Any:
        self.channel.send(request)
        answer = self.channel.reply()
        if "result" not in answer:
            raise ChannelRefusal("MISSING_FIELD", "source answer has no result")
        return answer["result"]

    def _receive_into(self, view: memoryview) -> None:
        while view:
            block = self.channel.receive()
            if not isinstance(block, bytes) or not block or len(block) > len(view):
                raise ChannelRefusal("LENGTH_MISMATCH", "invalid source byte stream")
            view[:len(block)] = block
            view = view[len(block):]

    def inspect(self, components: Sequence[str], configs: Sequence[str]) -> SourceInspection:
        with self.lock:
            try:
                value = self._result({"op": "inspect", "components": list(components), "configs": list(configs)})
                if not isinstance(value, dict):
                    raise ChannelRefusal("WRONG_TYPE", "invalid source inspection response")
                total = 0
                for config in value["configs"]:
                    length = config.pop("length")
                    if type(length) is not int or not 0 <= length <= _METADATA - total:
                        raise ChannelRefusal("SIZE_CAP", "source configurations exceed their bound")
                    total += length
                    data = bytearray(length)
                    self._receive_into(memoryview(data))
                    config["data"] = bytes(data)
                result = SourceInspection.from_native(value)
                if result.source != self.source:
                    raise ChannelRefusal("TRANSACTION_CONFLICT", "source inspection changed identity")
                return result
            except BaseException:
                self.channel.close()
                raise

    def check(self) -> None:
        with self.lock:
            try:
                if self._result({"op": "check", "manifest": self.source.manifest, "manifest_length": self.source.length}) is not None:
                    raise ChannelRefusal("WRONG_TYPE", "invalid source check response")
            except BaseException:
                self.channel.close()
                raise

    def read_part_into(self, component: str, key: str, role: str, offset: int, into: object) -> None:
        view = memoryview(into)  # type: ignore[arg-type]
        if view.readonly or not view.c_contiguous or view.nbytes > _READ_LIMIT:
            raise ChannelRefusal("SIZE_CAP", "source read needs a bounded writable contiguous buffer")
        view = view.cast("B")
        with self.lock:
            try:
                length = self._result({"op": "read", "manifest": self.source.manifest, "manifest_length": self.source.length, "component": component, "key": key,
                                       "role": role, "offset": offset, "length": len(view)})
                if type(length) is not int or length != len(view):
                    raise ChannelRefusal("LENGTH_MISMATCH", "source read changed its length")
                self._receive_into(view)
            except BaseException:
                self.channel.close()
                raise


def serve_source(
    store: _ext.Store,
    source: Source,
    fd: int,
    *,
    check_current: Callable[[], None],
    admit_io: Callable[[], ContextManager[None]],
) -> None:
    """Own a native read lease for one fixed source until descriptor close/fence.

    Metadata-only inspection leaves no live read lease that could block writer admission.
    Checks/part reads pin the source until descriptor close/fence.
    Only semantic source parts can be read. Native ReadLease validates header custody,
    geometry, ranges, part contents and revocation; no arbitrary object reader is exposed.
    """
    channel = _Channel(fd)
    lease: _ext.ReadLease | None = None
    raw_header: bytes | None = None
    header: dict[str, Any] | None = None
    try:
        while True:
            request = channel.receive()
            if not isinstance(request, dict):
                raise ChannelRefusal("WRONG_TYPE", "expected a scoped source operation")
            op = request.get("op")
            if op == "inspect":
                _fields(request, "inspect", ("components", "configs"))
            elif op == "read":
                _fields(request, "read", ("manifest", "manifest_length", "component", "key", "role", "offset", "length"))
                if type(request["offset"]) is not int or request["offset"] < 0:
                    raise ChannelRefusal("NUMBER_RANGE", "source read offset must be nonnegative")
                if type(request["length"]) is not int or not 0 <= request["length"] <= _READ_LIMIT:
                    raise ChannelRefusal("SIZE_CAP", "source read exceeds its buffer bound")
            elif op == "check":
                _fields(request, "check", ("manifest", "manifest_length"))
            else:
                raise ChannelRefusal("SOURCE_NOT_ALLOWED", "operation is outside this source handle")
            if op in {"read", "check"} and (request["manifest"] != source.manifest or type(request["manifest_length"]) is not int or request["manifest_length"] != source.length):
                raise ChannelRefusal("TRANSACTION_CONFLICT", "source operation changed its bound identity")
            check_current()
            # Acquiring/inspecting native metadata may write custody bookkeeping.
            # Reads/checks on an existing lease write no files and need no disk grant.
            with admit_io() if op == "inspect" or lease is None else contextlib.nullcontext():
                check_current()
                if raw_header is None:
                    raw_header = store.manifest(source.manifest)["header"]
                    if raw_header is None:
                        raise ChannelRefusal("MISSING_FIELD", "source has no tensor header")
                    header = _ext.parse_header(raw_header)
                    # Native exact-source validation runs even when the first operation is a read.
                    store.inspect_derived_source(source.manifest, source.length, list(header["components"]), [])
                assert header is not None and raw_header is not None
                if op in {"read", "check"} and lease is None:
                    lease = store.acquire_cozytensors(source.manifest)
                if lease is not None:
                    lease.recheck()
                if op == "inspect":
                    components, configs = request["components"], request["configs"]
                    if not isinstance(components, list) or not isinstance(configs, list):
                        raise ChannelRefusal("WRONG_TYPE", "source selections must be lists")
                    value = store.inspect_derived_source(
                        source.manifest, source.length,
                        components or list(header["components"]), configs or list(header.get("configs", {})),
                    )
                elif op == "read":
                    data = bytearray(request["length"])
                    assert lease is not None
                    lease.read_part_into(raw_header, request["component"], request["key"], request["role"], request["offset"], data)
                if lease is not None:
                    lease.recheck()
            check_current()
            if op == "inspect":
                configuration = value["configs"]
                metadata: dict[str, Any] = {
                    **value,
                    "configs": [{"name": row["name"], "length": len(row["data"])} for row in configuration],
                }
                channel.send({"result": metadata})
                for row in configuration:
                    view = memoryview(row["data"])
                    for start in range(0, len(view), _CHUNK):
                        check_current()
                        channel.data(view[start:start+_CHUNK])
            elif op == "read":
                channel.send({"result": len(data)})
                view = memoryview(data)
                for start in range(0, len(view), _CHUNK):
                    check_current()
                    channel.data(view[start:start+_CHUNK])
            else:
                channel.send({"result": None})
    except EOFError:
        return
    except Exception as exc:
        with contextlib.suppress(OSError, EOFError):
            channel.send({"error": str(getattr(exc, "code", "IO_FAILED")), "detail": "source owner refused operation"})
        raise
    finally:
        channel.close()
        if lease is not None:
            lease.release()
