"""Pull time of each layout from R2, modeled from two real v1 lane pulls on one pod.

    python pullmodel.py result.json   # `cozy run show <speedgate run> --json`

Fit (rental methode, RunPod IS, 2026-10-18, transport log `walk` lines): SDXL int8 3,880 GETs, 4.08 GB in
14.4 s; Qwen-Image int8 3,863 GETs, 19.2 GB in 18.0 s. At ~70 MB/s per connection (POD-EXPERIMENTS.md) both
fit a 0.20 s TTFB over 59 connections; the aggregate is at least Qwen's 1.07 GB/s. Today's transport asks
8 MiB ranges; v2's asks one GET per chunk.
"""

import json
import sys

TTFB, PER_CONN, CONNS, AGGREGATE = 0.20, 70e6, 59, 1.07e9


def pull(sizes, rng):
    reqs = [min(rng, s - o) for s in sizes for o in range(0, s, rng)]
    return max(sum(TTFB + r / PER_CONN for r in reqs) / CONNS, sum(sizes) / AGGREGATE), len(reqs)


result = json.load(open(sys.argv[1]))["result"]["value"]
for name, model in result["models"].items():
    for layout, sizes in model["sizes"].items():
        (t8, n8), (t1, n1) = pull(sizes, 8 << 20), pull(sizes, 64 << 20)
        print(f"{name} {layout}: {len(sizes)} objects; 8 MiB ranges {t8:.1f} s ({n8} GETs); "
              f"one GET per object {t1:.1f} s ({n1}); floor {sum(sizes) / AGGREGATE:.1f} s")
