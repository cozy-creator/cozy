"""TensorFS: cozytensors/1 checkpoints in a content-addressed store (design §1).

    store = tensorfs.Store("~/.tensorfs")                # created when absent
    m = store.import_("hf://hf-internal-testing/tiny-sdxl-pipe")
    m = store.pull("acme/sdxl@1.0", lane="int8")  # from tensorhub.com
    ck = store.checkpoint(m)
    ck.tensors["unet"]["conv_in.weight"]                 # Info(dtype, shape, encoding, parts)
    torch.from_dlpack(ck.read("unet", "conv_in.weight")) # encoded tensors decode
    store.export(m, "out/")                              # a Diffusers/Transformers folder

Compiled in `tensorfs._ext`, typed in `_ext.pyi`; the weight plane is `tensorfs.varena`;
`parse_header` projects a CBOR header (`tensorfs.errors` refusals). TensorFS never imports torch.
"""

from . import errors
from ._ext import (
    Array,
    Checkpoint,
    Closure,
    DecodeError,
    DeniedError,
    DiskFullError,
    Encoding,
    Error,
    FormatError,
    Info,
    IntegrityError,
    InvalidError,
    Manifest,
    NetworkError,
    NotFoundError,
    Part,
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

__all__ = [
    "Array",
    "Checkpoint",
    "Closure",
    "DecodeError",
    "DeniedError",
    "DiskFullError",
    "Encoding",
    "Error",
    "FormatError",
    "Info",
    "IntegrityError",
    "InvalidError",
    "Manifest",
    "NetworkError",
    "NotFoundError",
    "Part",
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
]
