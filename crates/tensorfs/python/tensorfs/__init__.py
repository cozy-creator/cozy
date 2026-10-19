"""TensorFS: self-contained model variants, read and written by component.

    ck = tensorfs.open("acme/sdxl", revision="1.0", variant="int8")
    unet = ck.components["unet"]
    tensors = unet.tensors.read_all(device="cpu")
    tokenizer_path = ck.components["tokenizer"].path

Arrays cross to NumPy, PyTorch, JAX or CuPy through DLPack. TensorFS never imports a framework.
"""

from . import errors
from ._ext import (
    Array,
    Checkpoint,
    Closure,
    Component,
    ComponentWriter,
    DecodeError,
    DeniedError,
    DiskFullError,
    DPoP,
    Endpoint,
    Encoding,
    Error,
    FormatError,
    Info,
    IntegrityError,
    InvalidError,
    NetworkError,
    NotFoundError,
    Part,
    Pipeline,
    StorageError,
    Store,
    UnknownSchemeError,
    UnsupportedError,
    Writer,
    __version__,
    parse_header,
    plan,
    regions,
)
from .api import ComponentWriter, Writer, configure, convert, export, hub_download, open, resolve_source, snapshot_download, upload_folder

__all__ = [
    "Array",
    "Checkpoint",
    "Closure",
    "Component",
    "ComponentWriter",
    "DecodeError",
    "DeniedError",
    "DiskFullError",
    "DPoP",
    "Endpoint",
    "Encoding",
    "Error",
    "FormatError",
    "Info",
    "IntegrityError",
    "InvalidError",
    "NetworkError",
    "NotFoundError",
    "Part",
    "Pipeline",
    "StorageError",
    "Store",
    "UnknownSchemeError",
    "UnsupportedError",
    "Writer",
    "__version__",
    "errors",
    "parse_header",
    "plan",
    "regions",
    "configure",
    "convert",
    "export",
    "hub_download",
    "resolve_source",
    "open",
    "snapshot_download",
    "upload_folder",
]
