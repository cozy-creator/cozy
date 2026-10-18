"""Pull time of each layout from R2, modeled from measured transport parameters.

    python pullmodel.py result.json   # the `value` of a speedgate run (`cozy run show <run> --json`)

Parameters: TTFB 0.24 s, ~70 MB/s per connection, 1.53 GB/s at 64 connections (a RunPod pod,
outputs/tensorfs-download-tail-20261001/POD-EXPERIMENTS.md). Today's transport asks 8 MiB ranges;
v2's asks one GET per chunk.
"""

import json
import sys

TTFB, PER_CONN, CONNS, AGGREGATE = 0.24, 70e6, 64, 1.53e9


def pull(sizes, rng):
    reqs = [min(rng, s - o) for s in sizes for o in range(0, s, rng)]
    return max(sum(TTFB + r / PER_CONN for r in reqs) / CONNS, sum(sizes) / AGGREGATE), len(reqs)


result = json.load(open(sys.argv[1]))
result = result.get("value", result)
for name, model in result["models"].items():
    for layout, sizes in model["sizes"].items():
        (t8, n8), (t1, n1) = pull(sizes, 8 << 20), pull(sizes, 64 << 20)
        print(f"{name:10s} {layout:9s} objects {len(sizes):5d}  8 MiB ranges {t8:5.1f} s ({n8} GETs)"
              f"  one GET per object {t1:5.1f} s ({n1})  floor {sum(sizes) / AGGREGATE:5.1f} s")
