#!/usr/bin/env python3
"""Compile TensorD tests once, then run whole harnesses with at most two workers.

Each harness keeps Cargo's package working directory, inherited fixture environment,
and ordinary test arguments/thread settings. Varena and wheel/AOT jobs are unchanged.
The surrounding ci-delegated.sh remains responsible for the delegated cgroup lifetime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


class Interrupted(Exception):
    def __init__(self, number: int) -> None:
        self.number = number


@dataclass(frozen=True)
class Task:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    compile: bool = False
    order: int = 2


@dataclass
class Result:
    name: str
    code: int
    tests: list[Task]


class Runner:
    def __init__(self, jobs: int) -> None:
        if jobs not in (1, 2):
            raise ValueError("test workers must be one or two")
        self.jobs = jobs
        self.lock = threading.Lock()
        self.output = threading.Lock()
        self.stopped = threading.Event()
        self.children: set[subprocess.Popen[str]] = set()

    def print(self, text: str) -> None:
        with self.output:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)

    def execute(self, task: Task) -> Result:
        with self.lock:
            if self.stopped.is_set():
                return Result(task.name, 130, [])
            try:
                child = subprocess.Popen(
                    task.argv, cwd=task.cwd, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True,
                )
            except OSError as error:
                self.print(f"FAIL {task.name}: {error}")
                return Result(task.name, 127, [])
            self.children.add(child)
        self.print(f"Running {task.name}")
        tests: dict[str, Task] = {}
        lines: list[str] = []
        try:
            assert child.stdout is not None
            for line in child.stdout:
                if task.compile:
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        self.print(line)
                        continue
                    if not isinstance(message, dict):
                        self.print(line)
                        continue
                    if message.get("reason") == "compiler-message":
                        self.print(message["message"].get("rendered") or str(message["message"]))
                    if (message.get("reason") == "compiler-artifact"
                            and message.get("profile", {}).get("test")
                            and (executable := message.get("executable"))):
                        manifest = Path(message["manifest_path"])
                        tests[executable] = Task(
                            message["target"]["name"], (executable,), manifest.parent,
                            order={"lib": 0, "bin": 1}.get(message["target"].get("kind", ["test"])[0], 2),
                        )
                else:
                    lines.append(line)
            code = child.wait()
        finally:
            child.stdout.close()
            child.wait()
            with self.lock:
                self.children.discard(child)
        if lines:
            self.print(f"--- {task.name} ---\n{''.join(lines)}")
        self.print(f"{'PASS' if code == 0 else 'FAIL'} {task.name} ({code})")
        return Result(task.name, code, sorted(tests.values(), key=lambda test: (test.order, test.name)))

    def run(self, tasks: list[Task]) -> list[Result]:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs)
        try:
            pending = [pool.submit(self.execute, task) for task in tasks]
            return [future.result() for future in concurrent.futures.as_completed(pending)]
        finally:
            if sys.exc_info()[0] is not None:
                self.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    def cancel(self) -> None:
        # Spawn and registration share this lock, so cancellation cannot miss a
        # worker between its stop check and creation of the new process group.
        self.stopped.set()
        with self.lock:
            children = list(self.children)
        for child in children:
            self.signal_group(child, signal.SIGTERM)
        deadline = time.monotonic() + 1
        for child in children:
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=max(deadline - time.monotonic(), 0))
        for child in children:
            self.signal_group(child, signal.SIGKILL)
        for child in children:
            child.wait()

    @staticmethod
    def signal_group(child: subprocess.Popen[str], number: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, number)


def run(jobs: int) -> int:
    runner = Runner(jobs)

    def interrupt(number: int, _frame: object) -> None:
        # Cleanup runs outside a signal handler and before joining worker threads.
        if not runner.stopped.is_set():
            raise Interrupted(number)

    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        cargo = ("cargo", "test", "--locked", "--no-fail-fast", "-p", "tensord")
        compiled = runner.run([Task(
            "compile", (*cargo, "--no-run", "--message-format=json"), Path.cwd(), True,
        )])[0]
        if compiled.code:
            return 1
        if not compiled.tests:
            raise RuntimeError("Cargo emitted no test executables; refusing an empty test run")
        results = runner.run(compiled.tests)
        # --no-run does not emit doctest executables. Retain that ordinary Cargo
        # coverage, including after a failed harness (--no-fail-fast semantics).
        results += runner.run([Task("doctests", (*cargo, "--doc"), Path.cwd())])
        return int(any(result.code for result in results))
    except Interrupted as error:
        runner.cancel()
        return 128 + error.number
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def self_test() -> int:
    """Real child executables exercise scheduling, failure and signal boundaries."""
    import select
    import tempfile
    import textwrap
    import unittest

    class Tests(unittest.TestCase):
        def fixture(self, mode: str) -> None:
            temporary = tempfile.TemporaryDirectory(prefix="cargo-binaries-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
            self.events = self.root / "events"
            self.gate = self.root / "gate"
            os.mkfifo(self.events)
            os.mkfifo(self.gate)
            self.reader = os.open(self.events, os.O_RDWR | os.O_NONBLOCK)
            self.writer = os.open(self.gate, os.O_RDWR)
            self.addCleanup(os.close, self.reader)
            self.addCleanup(os.close, self.writer)
            self.buffer = b""
            self.records: list[dict[str, object]] = []
            self.env = {**os.environ, "CASE_DIR": str(self.root), "CASE_MODE": mode,
                        "FIXTURE_VALUE": "inherited", "PATH": f"{self.root}:{os.environ['PATH']}"}

            def executable(name: str, code: str) -> None:
                path = self.root / name
                path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(code))
                path.chmod(0o755)

            executable("cargo", '''
                import json, os, sys
                from pathlib import Path
                root = Path(os.environ["CASE_DIR"])
                if "--doc" in sys.argv:
                    (root / "doc").touch()
                    sys.exit(9)
                assert "--no-run" in sys.argv
                with (root / "compiles").open("a") as f: f.write("once\\n")
                if os.environ["CASE_MODE"] == "compile-cancel":
                    os.execv(str(root / "long0"), [str(root / "long0")])
                names = ["short0", "short1", "short2", "short3"] if os.environ["CASE_MODE"] == "fail" else ["long0", "long1"]
                for name in [*names, names[0]]:
                    print(json.dumps({"reason": "compiler-artifact", "profile": {"test": True},
                        "executable": str(root / name), "manifest_path": str(root / "Cargo.toml"),
                        "target": {"name": name}}))
                print(json.dumps({"reason": "compiler-artifact", "profile": {"test": False},
                    "executable": "/not/a/test"}))
            ''')
            for index in range(4):
                executable(f"short{index}", '''
                    import json, os, sys
                    from pathlib import Path
                    root = Path(os.environ["CASE_DIR"])
                    assert Path.cwd() == root and os.environ["FIXTURE_VALUE"] == "inherited"
                    def event(kind):
                        with (root / "events").open("w") as f:
                            f.write(json.dumps({"kind": kind, "pid": os.getpid()}) + "\\n")
                    event("start")
                    with (root / "gate").open("rb", buffering=0) as f: f.read(1)
                    event("end")
                    sys.exit(7 if Path(__file__).name == "short0" else 0)
                ''')
            grandchild = '''import os, signal, sys
from pathlib import Path
def term(*_):
    Path(sys.argv[1]).touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
print("ready", flush=True)
signal.pause()
'''
            for index in range(2):
                executable(f"long{index}", '''
                    import json, os, signal, subprocess, sys
                    from pathlib import Path
                    root = Path(os.environ["CASE_DIR"])
                    child = subprocess.Popen([sys.executable, "-c", GRANDCHILD,
                        str(root / f"term-{os.getpid()}")], stdout=subprocess.PIPE, text=True)
                    assert child.stdout.readline().strip() == "ready"
                    def term(*_):
                        try: child.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            (root / "missed-group").touch()
                            child.kill(); child.wait()
                        signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    signal.signal(signal.SIGTERM, term)
                    with (root / "events").open("w") as f:
                        f.write(json.dumps({"kind": "start", "pid": os.getpid(), "child": child.pid}) + "\\n")
                    while True: signal.pause()
                '''.replace("GRANDCHILD", repr(grandchild)))
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())], env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.addCleanup(self.cleanup_process)

        def cleanup_process(self) -> None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    for record in self.records:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(int(record["pid"]), signal.SIGKILL)
                    self.process.kill()
                    self.process.communicate()

        def record(self) -> dict[str, object]:
            deadline = time.monotonic() + 10
            while b"\n" not in self.buffer:
                ready = select.select([self.reader], [], [], max(deadline - time.monotonic(), 0))[0]
                self.assertTrue(ready, "child did not reach its controlled boundary")
                self.buffer += os.read(self.reader, 65536)
            line, self.buffer = self.buffer.split(b"\n", 1)
            row = json.loads(line)
            self.records.append(row)
            return row

        def test_two_workers_and_all_failures(self) -> None:
            self.fixture("fail")
            first = [self.record(), self.record()]
            self.assertEqual([r["kind"] for r in first], ["start", "start"])
            self.assertFalse(select.select([self.reader], [], [], 0)[0], "third child exceeded worker bound")
            os.write(self.writer, b"xxxx")
            active = {r["pid"] for r in first}
            maximum = len(active)
            ended = 0
            while ended != 4:
                row = self.record()
                if row["kind"] == "start":
                    active.add(row["pid"])
                else:
                    active.remove(row["pid"])
                    ended += 1
                maximum = max(maximum, len(active))
            output, _ = self.process.communicate(timeout=5)
            self.assertEqual(maximum, 2)
            self.assertEqual(self.process.returncode, 1)
            self.assertIn("FAIL short0 (7)", output)
            self.assertIn("FAIL doctests (9)", output)
            self.assertEqual((self.root / "compiles").read_text(), "once\n")
            self.assertTrue((self.root / "doc").exists())

        def canceled(self, mode: str, count: int) -> None:
            self.fixture(mode)
            records = [self.record() for _ in range(count)]
            self.process.terminate()
            self.process.communicate(timeout=5)
            self.assertEqual(self.process.returncode, 128 + signal.SIGTERM)
            self.assertFalse((self.root / "missed-group").exists())
            for row in records:
                self.assertTrue((self.root / f"term-{row['pid']}").exists())
                for pid in (row["pid"], row["child"]):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(pid), 0)

        def test_cancellation_reaps_binary_groups(self) -> None:
            self.canceled("cancel", 2)

        def test_cancellation_reaps_compiler_group(self) -> None:
            self.canceled("compile-cancel", 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    sys.exit(self_test() if args.self_test else run(args.jobs))
