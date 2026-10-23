# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena's native GPU suite on a rental: `cozy run native.py --on <rental> --gpus 1
--asset binary=<bundle.tar of test binaries> --asset suite=<suite.json>`.

The bundle (gpu/testbins.sh) holds varena's `gpu` and `varena-pass-cost` and TensorFS's
`tensorfs-io` and `tensorfs-refill`; a case is named `<binary>/<test>`.

`suite.json`: `{"label", "cases": [name, ...], "steps": {name: s}}`, `steps` being each case's
longest step on an earlier run. Each case runs in a fresh process of the binary, one after another;
it passes when the harness reports 1 passed, 0 failed, 0 ignored and it printed `PASS ...
<GPU UUID>` (libtest may put it on its `test <name> ... ` line). Every case must name one GPU.

A case is judged only by its own steps: the lines it prints (`STEP <line> <ms>` after each
statement). Bytes moved or GPU work are not progress: a loop that spins on the GPU or the bus
without reaching its next step is stuck. It is looked at every interval of its longest step: the
longest of its longest step on an earlier run (with none, the suite's longest healthy step, on
earlier runs and in this one), and its slowest so far in this run (the start-up before its first
step, or the largest gap between steps), which measures this machine. Two looks in
a row with no new step make it stuck: a step's line can land just after one look, so only a second
quiet look spans a whole quiet step. It is then killed and fails as stalled, its log ending with
each of its threads' state (process, name, scheduler state, kernel wait channel, syscall). Each
start, verdict and `TIMINGS` line (a case's per-call figures) goes to the run log as it happens; a
cancel kills the running case, its threads' state logged too, and returns the logs so far. A case that fails also puts its log's tail in the
run log, which reaches the caller even when a cancelled run's outputs do not. The outputs hold
every case's log and result.json; the caller judges from it.
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
# Seconds between checks for a cancel and for new figures: the executor allows 40 s for a cancel.
CANCEL = 1.0
# The run log keeps a message's first 200 characters: longer lines go in pieces of this size.
PIECE = 180
# The run log keeps 512 lines: each case's failure tail gets an equal share of them.
LOG_LINES = 512
# libtest's own lines in a failure's tail, which say nothing of the failure.
HARNESS = re.compile(r"^(running \d+ tests?|failures:|test result: .*|note: run with .*|FAILED|    \w+|\s*)$")


def group(pgrp: int) -> list[Path]:
    """The /proc entries of process group `pgrp`'s processes."""
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int((proc / "stat").read_text().rsplit(")", 1)[1].split()[2]) == pgrp:
                found.append(proc)
        except (OSError, ValueError, IndexError):
            continue
    return found


def threads(pgrp: int) -> list[str]:
    """Each thread of process group `pgrp`: process, thread, name, scheduler state, kernel wait channel, syscall."""
    lines = []
    for proc in group(pgrp):
        for task in sorted(proc.glob("task/[0-9]*"), key=lambda t: int(t.name)):
            try:
                stat = (task / "stat").read_text()
                name, state = stat[stat.index("(") + 1:stat.rindex(")")], stat.rsplit(")", 1)[1].split()[0]
                wchan = (task / "wchan").read_text().strip() or "-"
                syscall = (task / "syscall").read_text().split()[0]
            except (OSError, ValueError, IndexError):
                continue
            lines.append(f"THREAD {proc.name}/{task.name} {name} {state} wchan={wchan} syscall={syscall}")
    return lines


def longest_step(text: str, took: float) -> float | None:
    """Seconds of the case's longest step: the largest gap between consecutive STEP lines, or the
    time outside them (start-up before the first, teardown after the last), at least their 1 ms
    resolution."""
    ms = [int(m) for m in STEP.findall(text)]
    if not ms:
        return None
    gaps = [b - a for a, b in zip(ms, ms[1:])] + [took * 1000 - (ms[-1] - ms[0])]
    return max(max(gaps), 1) / 1000


def own_interval(text: str, startup: float) -> float:
    """Seconds of the case's slowest moment so far in this run: the start-up before its first step or the largest gap
    between its steps."""
    ms = [int(m) for m in STEP.findall(text)]
    return max([startup] + [(b - a) / 1000 for a, b in zip(ms, ms[1:])])


def pieces(line: str, size: int = PIECE) -> list[str]:
    return [line[at:at + size] for at in range(0, max(len(line), 1), size)]


def tail(text: str, lines: int) -> list[str]:
    """A failure's last `lines` log pieces: the case's own lines but its steps, then its last step. A long line keeps
    its first and last piece."""
    own = [l for l in text.splitlines() if not STEP.search(l) and not HARNESS.match(l)]
    last = [l for l in text.splitlines() if STEP.search(l)][-1:]
    cut = [p for l in own for p in (pieces(l) if len(l) <= 2 * PIECE else [pieces(l)[0], "… " + pieces(l)[-1]])]
    return cut[-max(lines - len(last), 0):] + last if lines > len(last) else last


def run_case(program: Path, case: str, log: Path, interval: float, cancelled, say) -> tuple[int, str | None]:
    """Runs one case to its end; or ends it as `stalled` (two quiet looks) or `cancelled`. Its
    `TIMINGS` lines go to `say` as they appear."""
    told = 0

    def tell(text: str):
        nonlocal told
        figures = [l for l in text.splitlines() if "TIMINGS " in l]
        for line in figures[told:]:
            # Each piece and the case's name before it within the run log's 200 characters.
            for piece in pieces(line[line.index("TIMINGS "):], PIECE - len(case[:40]) - 2):
                say(piece)
        told = len(figures)

    def end(proc, why: str) -> tuple[int, str]:
        with log.open("a") as more:
            more.write("\n".join([f"{why.upper()}: its threads"] + threads(proc.pid)) + "\n")
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.wait(), why

    with log.open("w") as out:
        started = time.monotonic()
        proc = subprocess.Popen([str(program), "--ignored", "--exact", case, "--nocapture", "--test-threads=1"],
                                stdout=out, stderr=subprocess.STDOUT, cwd=program.parent, start_new_session=True)
        last, quiet, looked, every, startup = None, 0, started, None, None
        while True:
            # Until its first step it is looked at as often as its start-up so far, which measures the start-up
            # within a factor of two.
            wait = every if every is not None else time.monotonic() - started
            try:
                code = proc.wait(timeout=min(CANCEL, wait))
                tell(log.read_text(errors="replace"))
                return code, None
            except subprocess.TimeoutExpired:
                pass
            text = log.read_text(errors="replace")
            tell(text)
            if cancelled():
                return end(proc, "cancelled")
            # Judged between its first step and its verdict: start-up and teardown are not steps.
            if not STEP.search(text) or "PASS " in text:
                continue
            startup = startup if startup is not None else time.monotonic() - started
            every = max(interval, own_interval(text, startup))
            if time.monotonic() - looked < every:
                continue
            looked = time.monotonic()
            now = len(STEP.findall(text))
            quiet = 0 if now != last else quiet + 1
            if quiet == 2:
                return end(proc, "stalled")
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
    logs = root / "logs"
    logs.mkdir()
    passed, failed, ended, own, gpus, took, longest = [], [], {}, [], set(), {}, {}
    for case in cases:
        if ctx.cancelled:
            break
        name, test = case.split("/", 1)
        log = logs / f"{name}-{test}.log"
        interval = steps.get(case)
        if interval is None:
            own.append(case)
            # No earlier runtime: the longest healthy step of the suite, on earlier runs and in this one so far.
            interval = max([*steps.values(), *(longest[c] for c in passed if longest[c])], default=0.0)
        ctx.log(f"START {case} (looks every {interval:.3f} s at least" + (", the suite's longest step: no earlier runtime)" if case in own else ")"), level="warn")
        start = time.monotonic()
        code, end = run_case(root / name, test, log, interval, lambda: ctx.cancelled,
                             lambda m: ctx.log(f"{case[:40]}: {m}", level="warn"))
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
            # The run log reaches the caller even when a cancelled run's outputs do not; a case's start and verdict
            # take two lines of its share.
            for piece in tail(text, LOG_LINES // len(cases) - 2):
                ctx.log(f"TAIL {case[:40]}: {piece}", level="error")
    result = {"label": label, "cases": cases, "passed": passed, "failed": failed, "ended": ended, "own_steps": own,
              "seconds": took, "steps": {c: s for c, s in longest.items() if s}, "gpus": sorted(gpus),
              "ok": not failed and len(passed) == len(cases) and len(gpus) == 1, "cancelled": ctx.cancelled,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    ctx.log(f"native suite {label}: passed {len(passed)}/{len(cases)}, failed {failed}, GPUs {sorted(gpus)}", level="warn")
    return out.save_tree(logs)
