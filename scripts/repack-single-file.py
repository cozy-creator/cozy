#!/usr/bin/env python3
"""Repack one diffusers component as a SINGLE-FILE carrier: same bytes, prefixed keys.

The packaging-convergence arm needs two carriers that differ only in how they are packaged.
This rewrites the header with a component prefix on every key and copies the data region
verbatim — so every tensor's bytes are literally the same bytes, and any difference in the
ingested result is a packaging artifact rather than a content one.

Not a codec: it does not interpret a single tensor. Header in, header out, body copied.
"""
import json
import struct
import sys

src, dst, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = 8 << 20

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    out = {prefix + k: v for k, v in header.items()}
    if meta is not None:
        out = {"__metadata__": meta, **out}
    blob = json.dumps(out, separators=(",", ":")).encode()
    with open(dst, "wb") as g:
        g.write(struct.pack("<Q", len(blob)))
        g.write(blob)
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            g.write(b)
print(f"{len(header)} keys prefixed {prefix!r}: {src} -> {dst}")
