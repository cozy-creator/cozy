"""TensorFS: cozytensors/1 checkpoints in a content-addressed store.

    store = tensorfs.Store("~/.cache/tensorfs")          # created when absent
    ck = store.import_("hf://hf-internal-testing/tiny-sdxl-pipe")
    ck = store.pull("acme/sdxl@1.0", variant="int8")     # from tensorhub.com
    unet = ck.components["unet"]                         # .cls, .config, .tensors, .assets
    torch.from_dlpack(unet.tensors.read("conv_in.weight")) # encoded tensors decode
    store.export(ck, "out/")                             # a Diffusers/Transformers folder

Compiled in `tensorfs._ext`, typed in `_ext.pyi`; the weight plane is `tensorfs.varena`;
`parse_header` projects a CBOR header (`tensorfs.errors` refusals). TensorFS never imports torch.
"""

from collections.abc import Mapping

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
    Encoded,
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
    Tensors,
    UnknownSchemeError,
    UnsupportedError,
    Writer,
    __version__,
    parse_header,
    plan,
    regions,
)
from .api import ComponentWriter, Writer, configure, convert, export, open, snapshot_download, upload_folder

Mapping.register(Tensors)

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
    "Encoded",
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
    "Tensors",
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
    "open",
    "snapshot_download",
    "upload_folder",
]
