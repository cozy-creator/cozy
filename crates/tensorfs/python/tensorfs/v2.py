"""TensorFS v2 (design §1), compiled in `tensorfs._ext.v2` and typed in `v2.pyi`.

    store = tensorfs.v2.Store("~/.tensorfs")             # created when absent
    m = store.import_("hf://hf-internal-testing/tiny-sdxl-pipe")
    m = store.pull("https://models.example.com", "acme/sdxl@1.0", lane="int8")
    ck = store.checkpoint(m)
    ck.tensors["unet"]["conv_in.weight"]                  # Info(dtype, shape, encoding, parts)
    torch.from_dlpack(ck.read("unet", "conv_in.weight"))  # encoded tensors decode
    store.export(m, "out/")                               # a Diffusers/Transformers folder

It sits beside the v1 surface until xs-045 U10 removes that; then these are `tensorfs.*`.
"""

from ._ext.v2 import (
    Array,
    Checkpoint,
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
    regions,
)

__all__ = [
    "Array",
    "Checkpoint",
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
    "regions",
]
