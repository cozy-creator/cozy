"""TensorFS: self-contained model variants, read and written by component.

    ck = tensorfs.open("acme/sdxl", revision="1.0", variant="model.int8.cozytensors")
    unet = ck.components["unet"]
    tensors = unet.tensors.read(device="cpu")
    tokenizer_config = ck.components["tokenizer"].config

Arrays cross to NumPy, PyTorch, JAX or CuPy through DLPack. TensorFS never imports a framework.
"""

from collections.abc import Mapping

from ._ext import (
    Array,
    Ceiling,
    Checkpoint,
    ClassRef,
    Component,
    ConflictError,
    DecodeError,
    DeniedError,
    DiskFullError,
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
    VerificationReport,
    __version__,
    ceiling,
)
from . import raw
from .api import CheckpointFile, ComponentWriter, Endpoint, Writer, configure, convert_to_cozytensors, download_file, open, materialize, upload_folder

Mapping.register(Tensors)

__all__ = [
    "Array",
    "Ceiling",
    "Checkpoint",
    "CheckpointFile",
    "ClassRef",
    "Component",
    "ConflictError",
    "ComponentWriter",
    "DecodeError",
    "DeniedError",
    "DiskFullError",
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
    "VerificationReport",
    "Writer",
    "__version__",
    "ceiling",
    "configure",
    "convert_to_cozytensors",
    "download_file",
    "open",
    "materialize",
    "upload_folder",
]
