#!/usr/bin/env python3
"""Repack one diffusers component as a single-file carrier with prefixed keys.

This packaging-convergence fixture rewrites only the carrier header and copies the data
region verbatim. It does not interpret tensor bytes and is not a TensorFS document codec.
"""

import json
import struct
import sys

source, destination, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
chunk = 8 << 20

with open(source, "rb") as reader:
    header_length = struct.unpack("<Q", reader.read(8))[0]
    header = json.loads(reader.read(header_length))
    metadata = header.pop("__metadata__", None)
    rewritten = {prefix + key: value for key, value in header.items()}
    if metadata is not None:
        rewritten = {"__metadata__": metadata, **rewritten}
    encoded = json.dumps(rewritten, separators=(",", ":")).encode()
    with open(destination, "wb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        while data := reader.read(chunk):
            writer.write(data)

print(f"{len(header)} keys prefixed {prefix!r}: {source} -> {destination}")
