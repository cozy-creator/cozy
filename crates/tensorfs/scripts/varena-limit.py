#!/usr/bin/env python3
"""A `gpu_memory_limit`'s arithmetic on the CPU (memory-varena §5.6, xs-046 B2).

Under a ledger capacity below the card, this process's torch may reserve the capacity less its
CUDA context, every other row's held bytes and its own varena pages (mapped and idle chunks);
its allocator's fraction is that over the card's total.

    CUDA_VISIBLE_DEVICES= python scripts/varena-limit.py
Runs with any python that has torch and `PYTHONPATH=<worktree>/python`.
"""

import sys
from types import SimpleNamespace

from tensorfs.varena.torch import Memory, limited

GiB = 1 << 30
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


check(
    "the limit less context, others and own pages",
    limited(8 * GiB, GiB, 2 * GiB, 3 * GiB) == 2 * GiB,
)
check("never below nothing", limited(8 * GiB, GiB, 6 * GiB, 3 * GiB) == 0)

# `Memory._room` reads the ledger and the core: every other row's held, never its own
rows = [
    SimpleNamespace(slot=0, held=5 * GiB),
    SimpleNamespace(slot=1, held=GiB),
    SimpleNamespace(slot=2, held=GiB // 2),
]
table = SimpleNamespace(snapshot=lambda: rows)
memory = Memory.__new__(Memory)
memory.limit, memory.context, memory.row = 8 * GiB, GiB // 2, (table, SimpleNamespace(id=0))
memory.core = SimpleNamespace(stats=lambda: SimpleNamespace(mapped=3 * GiB, idle=GiB // 2))
room = memory._room()
check(
    "other rows' held only",
    room == 8 * GiB - GiB // 2 - (GiB + GiB // 2) - (3 * GiB + GiB // 2),
    room,
)
check(
    "torch's fraction on a 24 GiB card",
    abs(room / (24 * GiB) - 2.5 / 24) < 1e-12,
    room / (24 * GiB),
)
memory.limit = None
check("no limit, no room to keep to", memory._room() is None)

print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
