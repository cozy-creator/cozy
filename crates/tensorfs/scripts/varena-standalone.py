"""The weight plane's standalone consumer (weight-residency-v2 §4), on one GPU: Store ->
Checkpoint -> regions -> Plane(PageCache) -> want -> acquire -> DLPack. No TensorD, no Runtime,
no budget. Every part of the first region and a spread of the others is checked against the
checkpoint's own CPU read.

    python scripts/varena-standalone.py <store root> <checkpoint ref>
"""

import sys

import numpy as np
import torch
from tensorfs import v2, varena


def main(root, ref):
    store = v2.Store(root)
    ck = store.checkpoint(ref)
    p = varena.Plane(devices=[0])
    ws = p.register(ck, v2.regions(ck, max_bytes=64 << 20))
    p.want(ws, 0).wait()
    stream = torch.cuda.current_stream().cuda_stream
    regions = ws.regions
    picked = sorted({0, len(regions) // 3, len(regions) // 2, len(regions) - 1})
    checked = 0
    for r in picked:
        with p.acquire(ws, 0, region=r, stream=stream) as lease:
            x = torch.from_dlpack(lease.view())
            base = regions[r]["offset"]
            for part in (q for q in ws.parts if q["region"] == r):
                info = ck.tensors[part["component"]][part["key"]]
                if info.encoding is not None:
                    continue
                want = torch.from_numpy(np.from_dlpack(ck.read(part["component"], part["key"])).view(np.uint8).reshape(-1).copy())
                got = x[part["offset"] - base : part["offset"] - base + part["nbytes"]].cpu()
                assert torch.equal(got, want), f"{part['component']}/{part['key']} differs on the GPU"
                checked += 1
            torch.cuda.current_stream().synchronize()
            del x  # a view keeps the weight set from closing
    st = p.stats()
    assert st["poisoned"] is None and st["devices"][0]["misses"] == 0, st
    freed = p.drop(ws, 0)
    assert freed == sum(r["span"] for r in regions) and p.stats()["devices"][0]["mapped"] == 0
    ws.close()
    p.close()
    print(f"standalone: {len(regions)} regions, {sum(r['nbytes'] for r in regions)} bytes, {checked} parts checked in regions {picked}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
