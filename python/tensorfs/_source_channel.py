"""One exact source's bounded metadata/config channel, with no Store escape."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from typing import Any, ContextManager

from . import _ext
from ._derived_channel import ChannelRefusal, _Channel, _CHUNK, _METADATA, _fields
from .derived import Source, SourceInspection


class SourceClient:
    def __init__(self, fd: int, source: Source) -> None:
        self.channel = _Channel(fd)
        self.source = source
        self.lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()

    def inspect(self, components: Sequence[str], configs: Sequence[str]) -> SourceInspection:
        with self.lock:
            try:
                self.channel.send({"op": "inspect", "components": list(components), "configs": list(configs)})
                answer = self.channel.reply()
                if set(answer) != {"result"} or not isinstance(answer["result"], dict):
                    raise ChannelRefusal("WRONG_TYPE", "invalid source inspection response")
                value = answer["result"]
                total = 0
                for config in value["configs"]:
                    length = config.pop("length")
                    if type(length) is not int or not 0 <= length <= _METADATA - total:
                        raise ChannelRefusal("SIZE_CAP", "source configurations exceed their bound")
                    total += length
                    data = bytearray(length)
                    view = memoryview(data)
                    while view:
                        block = self.channel.receive()
                        if not isinstance(block, bytes) or not block or len(block) > len(view):
                            raise ChannelRefusal("LENGTH_MISMATCH", "invalid source configuration stream")
                        view[:len(block)] = block
                        view = view[len(block):]
                    config["data"] = bytes(data)
                result = SourceInspection.from_native(value)
                if result.source != self.source:
                    raise ChannelRefusal("TRANSACTION_CONFLICT", "source inspection changed identity")
                return result
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
    """The owner fixes the source; the peer can only select its native metadata."""
    channel = _Channel(fd)
    try:
        while True:
            request = channel.receive()
            if not isinstance(request, dict):
                raise ChannelRefusal("WRONG_TYPE", "expected a source inspection")
            _fields(request, "inspect", ("components", "configs"))
            check_current()
            components, configs = request["components"], request["configs"]
            if not isinstance(components, list) or not isinstance(configs, list):
                raise ChannelRefusal("WRONG_TYPE", "source selections must be lists")
            with admit_io():
                check_current()
                if not components or not configs:
                    raw_header = store.manifest(source.manifest)["header"]
                    if raw_header is None:
                        raise ChannelRefusal("MISSING_FIELD", "source has no tensor header")
                    header = _ext.parse_header(raw_header)
                    components = components or list(header["components"])
                    configs = configs or list(header.get("configs", {}))
                value = store.inspect_derived_source(source.manifest, source.length, components, configs)
            check_current()
            configuration = value["configs"]
            metadata: dict[str, Any] = {
                **value,
                "configs": [{"name": row["name"], "length": len(row["data"])} for row in configuration],
            }
            channel.send({"result": metadata})
            for row in configuration:
                data = memoryview(row["data"])
                for start in range(0, len(data), _CHUNK):
                    check_current()
                    channel.data(data[start:start+_CHUNK])
    except EOFError:
        return
    except Exception as exc:
        with contextlib.suppress(OSError, EOFError):
            channel.send({"error": str(getattr(exc, "code", "IO_FAILED")), "detail": "source owner refused inspection"})
        raise
    finally:
        channel.close()
