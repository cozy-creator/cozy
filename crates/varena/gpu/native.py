# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena's native GPU suite on a rental: `cozy run native.py --on <rental> --gpus 1
--asset binary=<the gpu test binary> --asset suite=<suite.json>`.

`suite.json` is `{"label": "...", "cases": ["name", ...], "seconds": {"name": s, ...}}`, `seconds`
being each case's runtime on an earlier run. Each case runs in a fresh process of the binary, one
after another; it passes when the harness reports 1 passed, 0 failed, 0 ignored and it printed
`PASS ... <GPU UUID>` (libtest may put it on its `test <name> ... ` line). Every case must name the
same physical GPU. A case is looked at every interval of its own earlier runtime (else the longest
runtime known). At two looks in a row, if its processes together used no more CPU than as many idle
CUDA processes (the binary's `role_idle`, measured over the same interval) and it wrote nothing, it
is stuck: killed, it fails as stalled. Each case's start and verdict go to the run log as they
happen. The outputs hold each case's log and result.json with every runtime; the caller judges.
"""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, ScriptContext, Tree

UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def group_cpu(group: int) -> tuple[int, int]:
    """CPU nanoseconds used so far by every thread of process group `group`, and its processes."""
    ns, procs = 0, 0
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int((proc / "stat").read_text().rsplit(")", 1)[1].split()[2]) != group:
                continue
            procs += 1
            ns += sum(int((t / "schedstat").read_text().split()[0]) for t in (proc / "task").iterdir())
        except (OSError, ValueError, IndexError):
            continue
    return ns, procs


def run_case(program: Path, case: str, log: Path, interval: float | None, idle: int) -> tuple[int, bool]:
    """Runs one case to its end or, looked at every `interval`, until two looks in a row find its
    processes using no more CPU than idle CUDA processes (`idle`'s group) and writing nothing."""
    with log.open("w") as out:
        proc = subprocess.Popen([str(program), "--ignored", "--exact", case, "--nocapture", "--test-threads=1"],
                                stdout=out, stderr=subprocess.STDOUT, cwd=program.parent, start_new_session=True)
        last, still = None, 0
        while True:
            try:
                return proc.wait(timeout=interval), False
            except subprocess.TimeoutExpired:
                (used, procs), (reference, _) = group_cpu(proc.pid), group_cpu(idle)
                size = log.stat().st_size
                if last is not None:
                    busy = used - last[0] > procs * (reference - last[1])
                    still = 0 if busy or size != last[2] else still + 1
                if still == 2:
                    os.killpg(proc.pid, signal.SIGKILL)
                    return proc.wait(), True
                last = (used, reference, size)


def main(ctx: ScriptContext, *, binary: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
         suite: Annotated[FileAsset, AssetBound(max_bytes=1 << 16)], out: Outputs) -> Tree:
    selection = json.loads(suite.read_bytes())
    label, cases, seconds = selection["label"], selection["cases"], dict(selection.get("seconds", {}))
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("the suite names distinct cases")
    ctx.log(f"native suite {label}: {len(cases)} cases")
    root = Path(tempfile.mkdtemp(prefix="varena-gpu-"))
    program = root / "gpu"
    program.write_bytes(binary.read_bytes())
    program.chmod(0o700)
    logs = root / "logs"
    logs.mkdir()
    env = dict(os.environ, VARENA_GPU_ROLE="idle")
    idle = subprocess.Popen([str(program), "--ignored", "--exact", "role_idle", "--nocapture", "--test-threads=1"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=root, env=env,
                            start_new_session=True)
    for line in idle.stdout:
        if "@ ready" in line:
            break
    else:
        raise RuntimeError("the idle reference did not start")
    passed, failed, stalled, gpus, took = [], [], [], set(), {}
    for case in cases:
        log = logs / f"{case}.log"
        ctx.log(f"START {case}")
        start = time.monotonic()
        code, stuck = run_case(program, case, log, seconds.get(case) or max(seconds.values(), default=None), idle.pid)
        took[case] = seconds[case] = round(time.monotonic() - start, 3)
        text = log.read_text(errors="replace")
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        named = [m for line in text.splitlines() if "PASS " in line for m in UUID.findall(line)[-1:]]
        if stuck:
            stalled.append(case)
        ok = not code and summary[-1:] == [("1", "0", "0")] and bool(named)
        (passed if ok else failed).append(case)
        gpus.update(named if ok else [])
        verdict = "PASS" if ok else "STALLED" if stuck else "FAIL"
        ctx.log(f"{verdict} {case} in {took[case]} s", level="info" if ok else "error")
    idle.stdin.close()
    idle.wait()
    result = {"label": label, "cases": cases, "passed": passed, "failed": failed, "stalled": stalled,
              "seconds": took, "gpus": sorted(gpus), "ok": not failed and len(gpus) == 1,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    ctx.log(f"native suite {label}: passed {len(passed)}, failed {failed}, GPUs {sorted(gpus)}")
    return out.save_tree(logs)
