#!/usr/bin/env python3
"""Balance product cases by measured cost while preserving their execution order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import sys


# Workflow 36718375142: fence 7s, build/vet 25s, cross-compile 73s,
# release acceptance 48s, types 2s. These once-only gates run on group zero.
GROUP_ZERO_OVERHEAD = 155.0


def measured(path: Path) -> dict[str, float]:
    costs: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        seconds, name = line.split("\t")
        value = float(seconds)
        if not name.startswith("Test") or name in costs or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid measured product cost: {line}")
        costs[name] = value
    if not costs:
        raise ValueError("measured product profile is empty")
    return costs


def partition(cases: Sequence[str], costs: Mapping[str, float], count: int) -> list[list[str]]:
    if count < 1 or count > len(cases) or len(set(cases)) != len(cases):
        raise ValueError("partitioning requires unique cases and nonempty groups")
    # New cases receive the rounded-up mean measured cost, rather than zero.
    default = max(1.0, float(math.ceil(sum(costs.values()) / len(costs))))
    # Go reports hundredths of a second. Integer costs keep equal-load tie breaks
    # independent of floating-point summation order, including zero-cost cases.
    units = {name: round(costs.get(name, default) * 100) for name in cases}
    loads = [round(GROUP_ZERO_OVERHEAD * 100) if index == 0 else 0 for index in range(count)]
    sizes = [0] * count
    assigned: dict[str, int] = {}
    ordered = sorted(cases, key=lambda case: (-units[case], case))
    for position, name in enumerate(ordered):
        empty = [group for group in range(count) if sizes[group] == 0]
        eligible = empty if len(ordered) - position == len(empty) else list(range(count))
        index = min(eligible, key=lambda group: (loads[group], sizes[group], group))
        assigned[name] = index
        loads[index] += units[name]
        sizes[index] += 1
    return [[name for name in cases if assigned[name] == index] for index in range(count)]


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: ci_product_partitions.py PROFILE CASES COUNT OUTPUT_PREFIX")
    profile = measured(Path(sys.argv[1]))
    cases = Path(sys.argv[2]).read_text().splitlines()
    count = int(sys.argv[3])
    groups = partition(cases, profile, count)
    default = max(1.0, float(math.ceil(sum(profile.values()) / len(profile))))
    for index, group in enumerate(groups):
        suffix = f"{index:0{max(2, len(str(count)))}d}"
        Path(sys.argv[4] + suffix).write_text("\n".join(group) + "\n")
        estimate = sum(profile.get(name, default) for name in group)
        overhead = GROUP_ZERO_OVERHEAD if index == 0 else 0.0
        print(f"group {index}: {len(group)} cases, estimated cases {estimate:.2f}s + gates {overhead:.0f}s")
