#!/usr/bin/env python3
"""varena's host core through the installed wheel, CPU only, no third-party packages.

A checkpoint is registered from its chunks (read by TensorFS) and from a sealed host layout this
process filled (`seal`), and that layout adopted from the regions it carries. Every part
is read back through `Pinned.ptr`. Freshness, close and the GPU core's refusals are checked.

    python varena-host.py [workdir]
"""

import ctypes
import sys
import tempfile
from pathlib import Path

import tensorfs
from tensorfs import varena


def main(root: Path) -> None:
    w = tensorfs.Writer(cache_dir=root)
    unet = w.component("unet", cls="test.Unet", config={})
    want: dict[str, bytes] = {}
    for b in range(3):
        for p in range(4):
            data = bytes((b * 31 + p * 7 + i) % 251 for i in range(4096 * (p + 1) + 5))
            unet.put(f"blocks.{b}.p{p}", "U8", [len(data)], data)
            want[f"blocks.{b}.p{p}"] = data
    ck = w.commit()
    rs = varena.regions(ck, 1 << 20, [f"unet/blocks.{b}" for b in range(3)])
    fd = varena.seal(ck, rs, "host-smoke")
    for name, host_fd in (("chunks", None), ("sealed", fd), ("adopted", fd)):
        if name == "adopted":
            ws = varena.Host().adopt(name, ck, fd)
            assert ws.regions == rs, "a sealed layout carries the regions it was built from"
        else:
            ws = varena.Host().register(name, ck, rs, host_fd)
        assert ws.device is None and len(ws.spans) == len(rs) and len(ws.parts) == len(want)
        assert ws.kept() == ws.kept_quiescent() == [], "host sets retain no GPU pages"
        for r in range(len(rs)):
            pin = ws.bind(r)
            assert pin.fresh and not ws.bind(r).fresh, "fresh once per page-in"
            assert pin.__dlpack_device__() == (1, 0) and pin.writable == (host_fd is None)
            for part in (p for p in ws.parts if p.region == r):
                assert (part.dtype, part.shape) == ("U8", (part.nbytes,))
                assert ctypes.string_at(pin.ptr + part.offset, part.nbytes) == want[part.key], part.key
            pin.__dlpack__()  # an unconsumed capsule frees its share of the pin
        assert ws.evict() == 0
        ws.close()
        try:
            ws.bind(0)
            raise AssertionError("a closed set binds nothing")
        except varena.Closed:
            pass
    try:
        varena.Gpu(0, 3 << 20)
        raise AssertionError("chunks are whole 2 MiB pages")
    except varena.Invalid as e:
        assert e.code == "INVALID"
    print("varena host core: every part exact, from the chunks, a sealed layout and its adoption")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="varena-host-") as tmp:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp))
