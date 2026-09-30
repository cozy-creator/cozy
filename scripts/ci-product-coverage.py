#!/usr/bin/env python3
"""Prove that the three CI partitions selected and completed every product case once."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


def names(path: Path) -> list[str]:
    return path.read_text().splitlines()


def audit(root: Path, full: bool) -> None:
    groups = sorted(root.glob("product-logs-*"))
    if len(groups) != 3:
        raise ValueError(f"expected three product log groups, received {len(groups)}")
    expected = names(groups[0] / "all-tests.txt")
    selected: list[str] = []
    outcomes: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, group in enumerate(groups):
        if names(group / "all-tests.txt") != expected:
            raise ValueError(f"{group.name}: product test census differs")
        chosen = names(group / "selected-tests.txt")
        if chosen != expected[index::3]:
            raise ValueError(f"{group.name}: round-robin partition changed")
        finished: list[str] = []
        for line in names(group / "timings.tsv"):
            _, status, test = line.split("\t")
            if status not in ("PASS", "FAIL", "SKIP"):
                raise ValueError(f"{group.name}: unknown outcome {status}")
            finished.append(test)
            outcomes[test] = status
            counts[status] += 1
        if Counter(finished) != Counter(chosen):
            raise ValueError(f"{group.name}: selected cases did not each finish once")
        selected.extend(chosen)
    if Counter(selected) != Counter(expected):
        raise ValueError("product partitions omitted or duplicated cases")
    if full:
        for required in (
            "TestStandaloneHostPreparesOrderedPrivateLoRAView",
            "TestNativeCollectedFailureCancelSurvivesDaemonRestart",
        ):
            if outcomes.get(required) != "PASS":
                raise ValueError(f"required native qualification did not pass: {required}")
    print(f"{len(expected)} cases: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped; complete three-partition coverage")
    if counts["FAIL"]:
        raise ValueError("a product case failed")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("true", "false"):
        raise SystemExit("usage: ci-product-coverage.py LOG_ROOT FULL")
    audit(Path(sys.argv[1]), sys.argv[2] == "true")
