# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena's native GPU suite on a rental: `cozy run native.py --on <rental> --gpus 1
--asset binary=<the gpu test binary> --asset suite=<suite.json>`.

`suite.json` is `{"label": "...", "cases": ["name", ...]}`. Each case runs in a fresh process of
the binary, one after another; it passes when the harness reports 1 passed, 0 failed, 0 ignored
and it printed `PASS ... <GPU UUID>` (libtest may put it on its `test <name> ... ` line). Every
case must name the same physical GPU. A case whose processes use no CPU across a whole STALL
window is stuck (a deadlock): it is killed and fails as stalled. The outputs hold each case's
log (stdout and stderr) and result.json, whichever cases failed: the caller judges from it.
"""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import tempfile
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
# Seconds: a case's longest wait (a gate, a fill, a child's start) lasts milliseconds to seconds,
# so a window this long with no CPU tick in its process group is a deadlock, not a wait.
STALL = 60


def cpu(group: int) -> int:
    """Clock ticks of CPU used so far by the processes of process group `group`."""
    ticks = 0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == group:
            ticks += int(fields[11]) + int(fields[12])
    return ticks


def run_case(program: Path, case: str, log: Path) -> tuple[int, bool]:
    """Runs one case to its end, or kills it once its process group stops using CPU."""
    with log.open("w") as out:
        proc = subprocess.Popen([str(program), "--ignored", "--exact", case, "--nocapture", "--test-threads=1"],
                                stdout=out, stderr=subprocess.STDOUT, cwd=program.parent, start_new_session=True)
        seen = None
        while True:
            try:
                return proc.wait(timeout=STALL), False
            except subprocess.TimeoutExpired:
                now = cpu(proc.pid)
                if now == seen:
                    os.killpg(proc.pid, signal.SIGKILL)
                    return proc.wait(), True
                seen = now


def main(*, binary: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
         suite: Annotated[FileAsset, AssetBound(max_bytes=1 << 16)], out: Outputs) -> Tree:
    selection = json.loads(suite.read_bytes())
    label, cases = selection["label"], selection["cases"]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("the suite names distinct cases")
    print(f"native suite {label}: {len(cases)} cases", flush=True)
    root = Path(tempfile.mkdtemp(prefix="varena-gpu-"))
    program = root / "gpu"
    program.write_bytes(binary.read_bytes())
    program.chmod(0o700)
    logs = root / "logs"
    logs.mkdir()
    passed, failed, stalled, gpus = [], [], [], set()
    for case in cases:
        log = logs / f"{case}.log"
        code, stuck = run_case(program, case, log)
        text = log.read_text(errors="replace")
        print(text, flush=True)
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        named = [m for line in text.splitlines() if "PASS " in line for m in UUID.findall(line)[-1:]]
        if stuck:
            stalled.append(case)
        if code or summary[-1:] != [("1", "0", "0")] or not named:
            failed.append(case)
            continue
        gpus.update(named)
        passed.append(case)
    result = {"label": label, "cases": cases, "passed": passed, "failed": failed, "stalled": stalled, "gpus": sorted(gpus),
              "ok": not failed and len(gpus) == 1, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    print(f"native suite {label}: passed {len(passed)}, failed {failed}, GPUs {sorted(gpus)}", flush=True)
    return out.save_tree(logs)
