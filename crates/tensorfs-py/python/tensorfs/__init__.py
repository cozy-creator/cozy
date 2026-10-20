"""TensorFS: self-contained model variants, read and written by component.

    ck = tensorfs.open("acme/sdxl", revision="1.0", variant="int8")
    unet = ck.components["unet"]
    tensors = unet.tensors.read(device="cpu")
    tokenizer_config = ck.components["tokenizer"].config

Arrays cross to NumPy, PyTorch, JAX or CuPy through DLPack. TensorFS never imports a framework.
"""

from collections.abc import Mapping

from ._ext import (
    Array,
    Checkpoint,
    Component,
    ConflictError,
    DecodeError,
    DeniedError,
    DiskFullError,
    DPoP,
    Encoded,
    Encoding,
    Error,
    FormatError,
    TensorMetadata,
    IntegrityError,
    InvalidError,
    NetworkError,
    NotFoundError,
    Pipeline,
    StorageError,
    Tensors,
    UnknownSchemeError,
    UnsupportedError,
    __version__,
)
from . import raw
from .api import ComponentWriter, Endpoint, Writer, configure, convert, export, hub_download, open, resolve_source, snapshot_download, upload_folder

Mapping.register(Tensors)

__all__ = [
    "Array",
    "Checkpoint",
    "Component",
    "ConflictError",
    "ComponentWriter",
    "DecodeError",
    "DeniedError",
    "DiskFullError",
    "DPoP",
    "Encoded",
    "Endpoint",
    "Encoding",
    "Error",
    "FormatError",
    "TensorMetadata",
    "IntegrityError",
    "InvalidError",
    "NetworkError",
    "NotFoundError",
    "Pipeline",
    "StorageError",
    "Tensors",
    "UnknownSchemeError",
    "UnsupportedError",
    "Writer",
    "__version__",
    "configure",
    "convert",
    "export",
    "hub_download",
    "resolve_source",
    "open",
    "snapshot_download",
    "upload_folder",
]
