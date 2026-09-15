"""Lazy PyTorch tensor access over verified native TensorFS part readers."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata
from types import MappingProxyType
from typing import Any, Literal

from . import _ext
from .derived import Part, SourceCapability, Tensor

_PLAIN = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
_ROWWISE_FLAT = (
    "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890"
)
_ROWWISE_KEEPDIM = (
    "sha256:6773dfe384aeb380d8d32de9f0b384a59a3294a96da4f3f1c8b956e194e3ff2e"
)
_ROWWISE = frozenset({_ROWWISE_FLAT, _ROWWISE_KEEPDIM})
_DTYPES = {
    "f64": "float64",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "f8_e4m3fn": "float8_e4m3fn",
    "f8_e5m2": "float8_e5m2",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "bool": "bool",
}


class TensorAccessError(ValueError):
    """Unsupported tensor representation or incompatible caller-owned allocation."""


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Tensor access requires tensorfs[torch]; storage and metadata do not"
        ) from exc


def _float8() -> Any:
    try:
        installed = metadata.version("torchao")
        module = import_module("torchao")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ImportError("Rowwise FP8 tensor access requires tensorfs[torch]") from exc
    if (
        installed not in {"0.18.0", "0.18.0+cpu"}
        or getattr(module, "__version__", None) != installed
    ):
        raise TensorAccessError(
            "Rowwise FP8 tensor access requires the reviewed TorchAO 0.18.0"
        )
    return import_module(
        "torchao.quantization.quantize_.workflows.float8.float8_tensor"
    )


def _dtype(name: str) -> Any:
    if name not in _DTYPES:
        raise TensorAccessError(f"unsupported Torch dtype {name!r}")
    return getattr(_torch(), _DTYPES[name])


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    if any(type(size) is not int or size <= 0 for size in shape):
        raise TensorAccessError("stored tensor dimensions must be positive integers")
    return tuple(shape)


def _codec(spec: Tensor) -> str:
    shape = _shape(spec.shape)
    _dtype(spec.logical_dtype)
    if spec.encoding == _PLAIN:
        if (
            set(spec.parts) != {"value"}
            or spec.parts["value"].dtype != spec.logical_dtype
            or tuple(spec.parts["value"].shape) != shape
        ):
            raise TensorAccessError(
                "plain tensor must contain its exact logical value part"
            )
        return "plain"
    if spec.encoding in _ROWWISE:
        if (
            len(shape) != 2
            or not all(shape)
            or spec.logical_dtype not in {"f16", "bf16", "f32"}
        ):
            raise TensorAccessError("rowwise FP8 requires a nonempty floating matrix")
        if set(spec.parts) != {"data", "scale"}:
            raise TensorAccessError("rowwise FP8 requires data and scale parts")
        data, scale = spec.parts["data"], spec.parts["scale"]
        if (
            data.dtype != "f8_e4m3fn"
            or tuple(data.shape) != shape
            or scale.dtype != "f32"
            or tuple(scale.shape)
            != ((shape[0],) if spec.encoding == _ROWWISE_FLAT else (shape[0], 1))
        ):
            raise TensorAccessError(
                "rowwise FP8 requires E4M3 codes and FP32 per-row scales"
            )
        return "rowwise"
    raise TensorAccessError(
        f"unsupported tensor encoding {spec.encoding!r}; no conversion is implicit"
    )


def _options(
    codec: str, mm_config: Any, kernel_preference: Any, act_quant_kwargs: Any
) -> dict[str, Any]:
    if codec == "plain":
        if any(
            value is not None
            for value in (mm_config, kernel_preference, act_quant_kwargs)
        ):
            raise TensorAccessError("plain tensors do not accept FP8 execution options")
        return {}
    module = _float8()
    from torchao.float8.inference import Float8MMConfig  # type: ignore[import-not-found]
    from torchao.quantization.quantize_.common import KernelPreference  # type: ignore[import-not-found]

    if mm_config is not None and not isinstance(mm_config, Float8MMConfig):
        raise TensorAccessError("mm_config must be an upstream Float8MMConfig")
    if kernel_preference is not None and not isinstance(
        kernel_preference, KernelPreference
    ):
        raise TensorAccessError(
            "kernel_preference must be an upstream KernelPreference"
        )
    if act_quant_kwargs is not None and not isinstance(
        act_quant_kwargs, module.QuantizeTensorToFloat8Kwargs
    ):
        raise TensorAccessError(
            "act_quant_kwargs must be upstream Float8 activation configuration"
        )
    return dict(
        mm_config=mm_config,
        act_quant_kwargs=act_quant_kwargs,
        kernel_preference=KernelPreference.AUTO
        if kernel_preference is None
        else kernel_preference,
    )


def _ordinary(value: Any) -> bool:
    torch = _torch()
    return type(value) in (torch.Tensor, torch.nn.Parameter)


def _validate_part(
    value: Any, part: Part, device: Any = None, *, allow_grad: bool = False
) -> None:
    torch = _torch()
    if (
        not _ordinary(value)
        or value.layout != torch.strided
        or not value.is_contiguous()
        or (value.requires_grad and not allow_grad)
    ):
        raise TensorAccessError(
            "physical parts must be contiguous ordinary inference tensors"
        )
    if tuple(value.shape) != tuple(part.shape) or value.dtype != _dtype(part.dtype):
        raise TensorAccessError(
            "physical part shape or dtype differs from its stored contract"
        )
    if value.device.type not in {"cpu", "cuda"} or (
        device is not None and value.device != device
    ):
        raise TensorAccessError(
            "physical part is on an unsupported or different device"
        )


def from_parts(
    spec: Tensor,
    parts: Mapping[str, Any],
    *,
    mm_config: Any = None,
    kernel_preference: Any = None,
    act_quant_kwargs: Any = None,
) -> Any:
    """Adopt physical Torch allocations without copies, decoding or requantization.

    The caller owns supplied allocations. Execution options are upstream objects;
    checkpoint metadata cannot change the logical or physical tensor geometry.
    """
    codec = _codec(spec)
    options = _options(codec, mm_config, kernel_preference, act_quant_kwargs)
    if set(parts) != set(spec.parts):
        raise TensorAccessError("physical part names differ from the tensor contract")
    device = getattr(next(iter(parts.values())), "device", None) if parts else None
    for role, part in spec.parts.items():
        _validate_part(parts[role], part, device)
    if codec == "plain":
        return parts["value"]
    shape = tuple(spec.shape)
    return _float8().Float8Tensor(
        parts["data"],
        parts["scale"].view(shape[0], 1),
        block_size=[1, shape[1]],
        dtype=_dtype(spec.logical_dtype),
        **options,
    )


def physical_tensors(value: Any) -> tuple[Any, ...]:
    """Physical leaves of an ordinary Tensor or the supported upstream FP8 wrapper."""
    if _ordinary(value):
        return (value,)
    if type(value) is not _float8().Float8Tensor:
        raise TensorAccessError("unsupported tensor subclass for physical inspection")
    from torchao.utils import UnwrapTensorSubclass  # type: ignore[import-not-found]

    return tuple(UnwrapTensorSubclass().right_inverse(value))


@dataclass(frozen=True, slots=True)
class TensorAllocation:
    """One exact physical allocation requested from the caller's admitted allocator."""

    role: str
    shape: tuple[int, ...]
    dtype: Any
    device: Any


class TensorReader:
    """One metadata selection; tensor reads return owned, ready allocations."""

    def __init__(
        self,
        source: Any,
        manifest: str | None,
        components: Sequence[str],
        buffer_bytes: int,
    ) -> None:
        if type(buffer_bytes) is not int or not 0 < buffer_bytes <= 64 << 20:
            raise TensorAccessError("reader windows must be between 1 byte and 64 MiB")
        self._lock = threading.Lock()
        self._closed = False
        self._buffer_bytes = buffer_bytes
        self._source = source
        self._manifest = manifest
        self._raw_header: bytes | None = None
        self._header: dict[str, Any] = {}
        if isinstance(source, SourceCapability):
            if manifest is not None:
                raise TensorAccessError(
                    "a scoped source already names its exact manifest"
                )
            inspection = source.inspect(components=components)
            self._specs = dict(inspection.components)
        elif isinstance(source, _ext.Store):
            if not manifest:
                raise TensorAccessError(
                    "a Store tensor reader requires an exact manifest"
                )
            row = source.manifest(manifest)
            self._raw_header = row["header"]
            if self._raw_header is None:
                raise TensorAccessError("the manifest has no tensor header")
            self._header = _ext.parse_header(self._raw_header)
            selected = tuple(components) or tuple(self._header["components"])
            # Native source inspection validates the exact artifact and selection.
            from .derived import SourceInspection

            inspection = SourceInspection.from_native(
                source.inspect_derived_source(
                    manifest, len(row["manifest"]), selected, []
                )
            )
            self._specs = dict(inspection.components)
        else:
            raise TypeError(
                "tensor reader requires a TensorFS Store or scoped SourceCapability"
            )
        if components and set(components) != set(self._specs):
            raise TensorAccessError(
                "source did not provide the exact requested components"
            )
        self._specs = {
            component: {
                key: Tensor(
                    spec.logical_dtype,
                    tuple(spec.shape),
                    spec.encoding,
                    MappingProxyType(
                        {
                            role: Part(part.dtype, tuple(part.shape))
                            for role, part in spec.parts.items()
                        }
                    ),
                )
                for key, spec in tensors.items()
            }
            for component, tensors in self._specs.items()
        }

    def spec(self, component: str, key: str) -> Tensor:
        if self._closed:
            raise TensorAccessError("tensor reader is closed")
        try:
            return self._specs[component][key]
        except KeyError as exc:
            raise TensorAccessError(
                f"tensor {component}/{key} is outside this reader selection"
            ) from exc

    def _parts(
        self,
        spec: Tensor,
        out: Any,
        device: Any,
        allocator: Callable[[TensorAllocation], Any] | None,
    ) -> dict[str, Any]:
        torch = _torch()
        if out is not None and allocator is not None:
            raise TensorAccessError("out and allocator are mutually exclusive")
        if out is not None:
            if spec.encoding == _PLAIN:
                parts = {"value": out}
            else:
                if (
                    type(out) is not _float8().Float8Tensor
                    or tuple(out.shape) != tuple(spec.shape)
                    or out.dtype != _dtype(spec.logical_dtype)
                    or out.block_size != [1, spec.shape[1]]
                ):
                    raise TensorAccessError(
                        "out must have the exact rowwise FP8 tensor contract"
                    )
                parts = {
                    "data": out.qdata,
                    "scale": out.scale.view(spec.parts["scale"].shape),
                }
        else:
            parts = {}
            for role, part in spec.parts.items():
                ask = TensorAllocation(
                    role, _shape(part.shape), _dtype(part.dtype), device
                )
                parts[role] = (
                    allocator(ask)
                    if allocator is not None
                    else torch.empty(ask.shape, dtype=ask.dtype, device=ask.device)
                )
        for role, value in parts.items():
            _validate_part(value, spec.parts[role], device, allow_grad=out is not None)
        ranges = sorted(
            (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
            for value in parts.values()
            if value.numel()
        )
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise TensorAccessError("writable physical parts overlap")
        return parts

    @contextlib.contextmanager
    def _read(self, component: str, key: str) -> Any:
        if isinstance(self._source, SourceCapability):
            self._source.check()
            yield lambda role, offset, into: self._source.read_part_into(
                component, key, role, offset, into
            )
            self._source.check()
            return
        assert self._raw_header is not None and self._manifest is not None
        refs = {_ext.object_id(self._raw_header): len(self._raw_header)}
        for part in self._header["components"][component][key]["parts"].values():
            for segment in part.get("segments", ()):
                refs["sha256:" + segment["sha256"].removeprefix("sha256:")] = segment[
                    "length"
                ]
        with self._source.acquire(self._manifest, list(refs.items())) as lease:
            yield lambda role, offset, into: lease.read_part_into(
                self._raw_header, component, key, role, offset, into
            )
            lease.recheck()

    def get(
        self,
        component: str,
        key: str,
        *,
        device: Any = None,
        out: Any = None,
        allocator: Callable[[TensorAllocation], Any] | None = None,
        mm_config: Any = None,
        kernel_preference: Any = None,
        act_quant_kwargs: Any = None,
    ) -> Any:
        """Read one owned tensor; infer out.device when provided, otherwise default to CPU."""
        torch = _torch()
        with self._lock, torch.no_grad():
            spec = self.spec(component, key)
            codec = _codec(spec)
            options = _options(codec, mm_config, kernel_preference, act_quant_kwargs)
            if out is not None and codec == "rowwise":
                for name, requested in (
                    ("mm_config", mm_config),
                    ("kernel_preference", kernel_preference),
                    ("act_quant_kwargs", act_quant_kwargs),
                ):
                    if requested is not None and requested != getattr(out, name, None):
                        raise TensorAccessError(
                            "out already has different FP8 execution options"
                        )
            target = torch.device(
                device if device is not None else getattr(out, "device", "cpu")
            )
            if target.type not in {"cpu", "cuda"}:
                raise TensorAccessError(
                    "tensor reader supports CPU and CUDA destinations"
                )
            with self._read(component, key) as read:
                # Establish live source authority before CUDA initialization or
                # allocating any caller/default destination storage.
                if target.type == "cuda" and target.index is None:
                    target = torch.device("cuda", torch.cuda.current_device())
                parts = self._parts(spec, out, target, allocator)
                try:
                    for role, value in parts.items():
                        destination = value.detach().reshape(-1).view(torch.uint8)
                        if target.type == "cpu":
                            view = memoryview(destination.numpy())
                            for offset in range(0, len(view), self._buffer_bytes):
                                read(
                                    role,
                                    offset,
                                    view[offset : offset + self._buffer_bytes],
                                )
                        else:
                            count = min(destination.numel(), self._buffer_bytes)
                            if not count:
                                continue
                            staging = torch.empty(
                                count, dtype=torch.uint8, pin_memory=True
                            )
                            event = torch.cuda.Event()
                            for offset in range(0, destination.numel(), count):
                                size = min(count, destination.numel() - offset)
                                read(role, offset, memoryview(staging.numpy())[:size])
                                stream = torch.cuda.current_stream(target)
                                try:
                                    destination[offset : offset + size].copy_(
                                        staging[:size], non_blocking=True
                                    )
                                    event.record(stream)
                                    event.synchronize()
                                except BaseException:
                                    # Even a failed enqueue/record must not release host
                                    # staging beneath a copy that reached this stream.
                                    stream.synchronize()
                                    raise
                finally:
                    # Native CPU writes bypass PyTorch operators. Invalidate the
                    # actual physical parts even if a read wrote only a prefix.
                    torch.autograd.graph.increment_version(tuple(parts.values()))
            return out if out is not None else from_parts(spec, parts, **options)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> TensorReader:
        if self._closed:
            raise TensorAccessError("tensor reader is closed")
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def tensors(
    source: Any,
    manifest: str | None = None,
    *,
    components: Sequence[str] = (),
    buffer_bytes: int = 16 << 20,
) -> TensorReader:
    """Open the default inference reader without importing Torch until tensor access."""
    return TensorReader(source, manifest, components, buffer_bytes)
