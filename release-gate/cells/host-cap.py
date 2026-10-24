# /// script
# requires-python = ">=3.12"
# [tool.cozy]
# accelerator = false
# ///
"""The host memory this machine's executors may use: the tightest memory.max/high (cgroup v2) or memory.limit_in_bytes
(v1) on this process's cgroup path, with MemTotal and SwapTotal. The host-RAM cap of cells/stress-host.sh."""
import json
from pathlib import Path


def number(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def meminfo(key: str) -> int | None:
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return None


def main(ctx) -> str:
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        v2 = hierarchy == "0"
        if not v2 and "memory" not in controllers.split(","):
            continue
        base = Path("/sys/fs/cgroup") if v2 else Path("/sys/fs/cgroup/memory")
        names = ("memory.max", "memory.high") if v2 else ("memory.limit_in_bytes",)
        at = base / relative.lstrip("/")
        while True:
            limits += [n for n in (number(at / name) for name in names) if n]
            if at == base or base not in at.parents:
                break
            at = at.parent
    return json.dumps({"cap": min(limits, default=None), "mem_total": meminfo("MemTotal"), "swap_total": meminfo("SwapTotal")})
