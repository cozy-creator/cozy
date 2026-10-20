"""Pinned Triton3.8 CUDA ABI checks; stdlib/data only, no driver or compiler import.

Triton's AOT launcher appends global_scratch and profile_scratch even when both sizes
are zero: https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/tools/compile.py
The narrow CUDA ELF attribute decoder is checked against our emitted catalog.
"""

import re
import struct

PARAMETERS = 13


def _sections(image):
    if len(image) < 64 or image[:6] != b"\x7fELF\x02\x01":
        raise ValueError("kernel is not a little-endian ELF64 image")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", image)
    if header[2] != 190 or header[11] != 64 or not header[12]:
        raise ValueError("kernel has no supported CUDA section table")
    offset, count, names_index = header[6], header[12], header[13]
    if offset + count * 64 > len(image) or names_index >= count:
        raise ValueError("kernel section table exceeds image")
    records = [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * 64) for i in range(count)]

    def data(record):
        begin, size = record[4:6]
        if begin + size > len(image):
            raise ValueError("kernel section exceeds image")
        return image[begin : begin + size]

    names = data(records[names_index])
    sections = {}
    for record in records:
        if record[1] == 8:  # SHT_NOBITS has no file contents.
            continue
        begin = record[0]
        end = names.find(b"\0", begin)
        if begin >= len(names) or end < 0:
            raise ValueError("kernel section has an invalid name")
        sections[names[begin:end].decode()] = data(record)
    return sections


def check_image(image, symbol):
    """Require the actual emitted kernel's thirteen eight-byte parameter slots."""
    info = _sections(image).get(".nv.info." + symbol)
    if info is None:
        raise ValueError("kernel has no parameter attributes")
    parameters, bank_size, offset = [], None, 0
    while offset < len(info):
        if offset + 4 > len(info):
            raise ValueError("truncated kernel attribute header")
        form, attribute, size = struct.unpack_from("<BBH", info, offset)
        if form not in (1, 2, 3, 4):
            raise ValueError("unknown kernel attribute format")
        end = offset + 4 + (size if form == 4 else 0)
        if end > len(info):
            raise ValueError("truncated kernel attribute payload")
        if form == 3 and attribute == 0x19:  # EIATTR_CBANK_PARAM_SIZE
            bank_size = size
        if form == 4 and attribute == 0x17:  # EIATTR_KPARAM_INFO
            if size != 12:
                raise ValueError("unexpected kernel parameter attribute size")
            _, ordinal, position, flags = struct.unpack_from("<IHHI", info, offset + 4)
            parameters.append((ordinal, position, flags >> 18))
        offset = end
    expected = [(i, i * 8, 8) for i in range(PARAMETERS)]
    if sorted(parameters) != expected or bank_size != PARAMETERS * 8:
        raise ValueError(f"kernel parameter ABI is {sorted(parameters)}, bank={bank_size}; expected thirteen u64 slots")
    return PARAMETERS


def check_compiled(metadata, ptx, image):
    """Refuse scratch requirements unsupported by the native zero-scratch launcher."""
    for name in ("global_scratch_size", "profile_scratch_size"):
        size = getattr(metadata, name, None)
        if type(size) is not int or size != 0:
            raise ValueError(f"AOT kernel {metadata.name} requires unsupported {name}={size!r}")
    entry = re.search(r"\.entry\s+" + re.escape(metadata.name) + r"\s*\((.*?)\)", ptx, re.S)
    if entry is None:
        raise ValueError("compiled PTX entry was not found")
    declarations = entry.group(1).split(",")
    if len(declarations) != PARAMETERS or any(
        re.search(r"\.param\s+\.u64\s+\.ptr\b", value) is None for value in declarations
    ):
        raise ValueError("compiled PTX does not declare thirteen pointer parameters")
    check_image(image, metadata.name)
    return {"parameters": PARAMETERS, "global_scratch_size": 0, "profile_scratch_size": 0}
