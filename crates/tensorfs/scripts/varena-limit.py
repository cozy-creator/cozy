#!/usr/bin/env python3
"""A `gpu_memory_limit`'s arithmetic on the CPU (memory-varena §5.6, xs-046 B2).

Under a ledger capacity below the card, this process's torch may reserve the capacity less every
byte the driver counts in use that is not that reserve: its CUDA context, NCCL, varena's pages
and any other process's, with a ledger row or not.

    CUDA_VISIBLE_DEVICES= python scripts/varena-limit.py
Runs with any python that has torch and `PYTHONPATH=<worktree>/python`.
"""

import sys
import torch

from tensorfs.varena.torch import Memory, limited

GiB = 1 << 30
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


check("the limit less what is in use elsewhere", limited(8 * GiB, 6 * GiB) == 2 * GiB)
check("never below nothing", limited(8 * GiB, 9 * GiB) == 0)

# `Memory._room` reads the driver: every byte in use but this torch's reserve, any process's
torch.cuda.mem_get_info = lambda _: (16 * GiB, 24 * GiB)  # 8 GiB in use on a 24 GiB card
torch.cuda.memory_reserved = lambda _: 3 * GiB
memory = Memory.__new__(Memory)
memory.limit, memory.gpu = 8 * GiB, 0
room = memory._room()
check("in use but not torch's reserve counts against the limit", room == 3 * GiB, room)
check("torch's fraction on a 24 GiB card", abs(room / (24 * GiB) - 3 / 24) < 1e-12, room)
memory.limit = None
check("no limit, no room to keep to", memory._room() is None)

print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
