# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena's native GPU suite on a rental: `cozy run native.py --on <rental> --gpus 1
--asset binary=<bundle.tar of test binaries> --asset suite=<suite.json>`.

The bundle (gpu/testbins.sh) holds varena's `gpu` (its cases, and the `role_monitor` role) and
TensorFS's `tensorfs-io`; a case is named `<binary>/<test>`.

`suite.json`: `{"label", "cases": [name, ...], "steps": {name: s}}`, `steps` being each case's
longest step on an earlier run. Each case runs in a fresh process of the binary, one after another;
it passes when the harness reports 1 passed, 0 failed, 0 ignored and it printed `PASS ...
<GPU UUID>` (libtest may put it on its `test <name> ... ` line). Every case must name one GPU.

A case is judged only by its own progress: the lines it prints (`STEP <line> <ms>` after each
statement), the bytes its processes read and write (/proc/<pid>/io), and whether the GPU ran
kernels or moved memory (NVML, through the binary's `role_monitor`). It is looked at every
interval of its own longest step on an earlier run. Two looks in a row with nothing moved make it
stuck: a step's line can land just after one look, so only a second quiet look spans a whole
quiet step. It is then killed and fails as stalled. A case with no earlier run is never killed:
it runs unjudged. Each start
and verdict goes to the run log as it happens; a cancel kills the running case and returns the
logs so far. A case that fails also puts its log's tail in the run log, which reaches the caller
even when a cancelled run's outputs do not. The outputs hold every case's log and result.json;
the caller judges from it.
"""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import tarfile
import tempfile
import time
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, ScriptContext, Tree

UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
STEP = re.compile(r"STEP \d+ (\d+)")
# Seconds between checks for a cancel while a case runs unjudged: the executor allows 40 s.
CANCEL = 1.0


def group_io(group: int) -> int:
    """Bytes the processes of process group `group` have read and written so far."""
    total = 0
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int((proc / "stat").read_text().rsplit(")", 1)[1].split()[2]) != group:
                continue
            counters = dict(line.split(": ") for line in (proc / "io").read_text().splitlines())
            total += int(counters["rchar"]) + int(counters["wchar"])
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return total


class Monitor:
    """The binary's `role_monitor`: whether the GPU worked since the last look. A monitor that
    cannot answer gives no GPU signal; the case's lines and bytes still judge it."""

    def __init__(self, program: Path, log):
        env = dict(os.environ, VARENA_GPU_ROLE="monitor")
        self.log = log
        self.proc = subprocess.Popen([str(program), "--ignored", "--exact", "role_monitor", "--nocapture", "--test-threads=1"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     cwd=program.parent, env=env, start_new_session=True)
        self.alive = self.answer() is not None

    def answer(self) -> str | None:
        for line in self.proc.stdout:
            if "@ " in line:
                return line.split("@ ", 1)[1].strip()
        self.log("the GPU monitor ended: judging by lines and bytes only")
        self.alive = False
        return None

    def worked(self) -> bool:
        if not self.alive:
            return False
        try:
            self.proc.stdin.write("look\n")
            self.proc.stdin.flush()
        except OSError:
            self.alive = False
            return False
        answer = self.answer() or ""
        return answer.split()[:1] != ["unknown"] and "true" in answer.split()

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()


def longest_step(text: str, took: float) -> float | None:
    """Seconds of the case's longest step: the largest gap between consecutive STEP lines, or the
    time outside them (start-up before the first, teardown after the last), at least their 1 ms
    resolution."""
    ms = [int(m) for m in STEP.findall(text)]
    if not ms:
        return None
    gaps = [b - a for a, b in zip(ms, ms[1:])] + [took * 1000 - (ms[-1] - ms[0])]
    return max(max(gaps), 1) / 1000


def run_case(program: Path, case: str, log: Path, interval: float | None, monitor: Monitor, cancelled) -> tuple[int, str | None]:
    """Runs one case to its end; or ends it as `stalled` (two quiet looks) or `cancelled`."""
    with log.open("w") as out:
        proc = subprocess.Popen([str(program), "--ignored", "--exact", case, "--nocapture", "--test-threads=1"],
                                stdout=out, stderr=subprocess.STDOUT, cwd=program.parent, start_new_session=True)
        last, quiet = None, 0
        while True:
            try:
                return proc.wait(timeout=interval or CANCEL), None
            except subprocess.TimeoutExpired:
                if cancelled():
                    os.killpg(proc.pid, signal.SIGKILL)
                    return proc.wait(), "cancelled"
                if interval is None:
                    continue
                # Judged between its first step and its verdict: start-up and teardown are not steps.
                text = log.read_text(errors="replace")
                if not STEP.search(text) or "PASS " in text:
                    continue
                now = (len(text), group_io(proc.pid))
                moved = monitor.worked() or now != last
                quiet = 0 if moved else quiet + 1
                if quiet == 2:
                    os.killpg(proc.pid, signal.SIGKILL)
                    return proc.wait(), "stalled"
                last = now


def main(ctx: ScriptContext, *, binary: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
         suite: Annotated[FileAsset, AssetBound(max_bytes=1 << 16)], out: Outputs) -> Tree:
    selection = json.loads(suite.read_bytes())
    label, cases = selection["label"], selection["cases"]
    steps = selection.get("steps", {})
    if not cases or len(set(cases)) != len(cases) or not all("/" in c for c in cases):
        raise ValueError("the suite names distinct cases, each <binary>/<test>")
    ctx.log(f"native suite {label}: {len(cases)} cases", level="warn")
    root = Path(tempfile.mkdtemp(prefix="varena-gpu-"))
    bundle = root / "bundle.tar"
    bundle.write_bytes(binary.read_bytes())
    with tarfile.open(bundle) as tar:
        for member in tar.getmembers():
            if not member.isfile() or "/" in member.name or member.name.startswith("."):
                raise ValueError(f"the bundle holds only plain binaries, not {member.name!r}")
            (root / member.name).write_bytes(tar.extractfile(member).read())
            (root / member.name).chmod(0o700)
    program = root / "gpu"
    logs = root / "logs"
    logs.mkdir()
    monitor = Monitor(program, lambda m: ctx.log(m, level="warn"))
    passed, failed, ended, unjudged, gpus, took, longest = [], [], {}, [], set(), {}, {}
    for case in cases:
        if ctx.cancelled:
            break
        name, test = case.split("/", 1)
        log = logs / f"{name}-{test}.log"
        interval = steps.get(case)
        if interval is None:
            unjudged.append(case)
        ctx.log(f"START {case}" + (" (unjudged: no earlier runtime)" if interval is None else f" (looks every {interval:.3f} s)"), level="warn")
        start = time.monotonic()
        code, end = run_case(root / name, test, log, interval, monitor, lambda: ctx.cancelled)
        took[case] = round(time.monotonic() - start, 3)
        text = log.read_text(errors="replace")
        longest[case] = longest_step(text, took[case])
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        named = [m for line in text.splitlines() if "PASS " in line for m in UUID.findall(line)[-1:]]
        # Varena's cases name the GPU they ran on; other binaries' tests do not print one.
        ok = not end and not code and summary[-1:] == [("1", "0", "0")] and (bool(named) or name != "gpu")
        (passed if ok else failed).append(case)
        if end:
            ended[case] = end
        gpus.update(named if ok else [])
        verdict = "PASS" if ok else (end or "FAIL").upper()
        ctx.log(f"{verdict} {case} in {took[case]} s, longest step {longest[case]} s", level="warn" if ok else "error")
        if not ok:
            # The run log reaches the caller even when a cancelled run's outputs do not; it keeps a
            # message's first 200 characters, so the tail goes line by line, the steps but the last
            # left out.
            lines = text.splitlines()
            stepped = [l for l in lines if STEP.search(l)]
            tail = [l for l in lines if not STEP.search(l)][-40:] + stepped[-1:]
            for line in tail:
                for at in range(0, max(len(line), 1), 180):
                    ctx.log(f"TAIL {case[:40]}: {line[at:at + 180]}", level="error")
    monitor.close()
    result = {"label": label, "cases": cases, "passed": passed, "failed": failed, "ended": ended, "unjudged": unjudged,
              "seconds": took, "steps": {c: s for c, s in longest.items() if s}, "gpus": sorted(gpus),
              "ok": not failed and len(passed) == len(cases) and len(gpus) == 1, "cancelled": ctx.cancelled,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    ctx.log(f"native suite {label}: passed {len(passed)}/{len(cases)}, failed {failed}, GPUs {sorted(gpus)}", level="warn")
    return out.save_tree(logs)
