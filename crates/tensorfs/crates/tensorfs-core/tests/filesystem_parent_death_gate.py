"""Actual native process/credential CPU gates; only the privileged cases need root."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def creator(binary: str, root: str, configured_signal: str, reader: int) -> None:
    child = subprocess.Popen([binary, "wait", root, configured_signal], stdin=reader)
    os.close(reader)
    print(json.dumps({"child_pid": child.pid}), flush=True)
    sys.stdin.buffer.read(1)
    os._exit(0)


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "fixture subreaper")

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="tfs-parent-death.")
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name).chmod(0o755)
        self.root = Path(self.directory.name) / "store"
        subprocess.run(
            [BINARY, "setup", str(self.root)], check=True, capture_output=True
        )

    def owner(self, uid: int) -> None:
        for current, directories, files in os.walk(self.root):
            os.chown(current, uid, uid)
            for name in (*directories, *files):
                os.chown(Path(current) / name, uid, uid)

    def check(
        self, mode: str, configured_signal: int, *, uid: int | None = None
    ) -> dict:
        kwargs = {} if uid is None else {"user": uid, "group": uid, "extra_groups": []}
        result = subprocess.run(
            [BINARY, mode, str(self.root), str(configured_signal)],
            check=True,
            capture_output=True,
            text=True,
            **kwargs,
        )
        facts = json.loads(result.stdout)
        self.assertEqual(facts["before"], configured_signal)
        self.assertEqual(facts["in_scope"], configured_signal)
        self.assertEqual(facts["after"], configured_signal)
        self.assertFalse(facts["cuda_mapped"])
        return facts

    def test_same_owner_preserves_signal_and_disabled(self) -> None:
        for configured_signal in (0, 9, 15):
            self.check("success", configured_signal)

    @unittest.skipUnless(os.geteuid() == 0, "root foreign Store-owner fixture")
    def test_foreign_owner_success_error_unwind_and_native_store(self) -> None:
        self.owner(65532)
        for configured_signal in (0, 9, 15):
            for mode in ("success", "error", "unwind", "store"):
                facts = self.check(mode, configured_signal)
                if mode == "success":
                    self.assertEqual(facts["file_uid"], 65532)

    @unittest.skipUnless(os.geteuid() == 0, "root nonroot credential fixture")
    def test_nonroot_does_not_adopt_foreign_root(self) -> None:
        self.root.chmod(0o777)  # writable fixture, still owned by root
        for configured_signal in (0, 9):
            self.assertEqual(
                self.check("success", configured_signal, uid=65532)["file_uid"], 65532
            )

    def death_in_scope(self, configured_signal: int) -> None:
        reader, writer = os.pipe()
        child_pid = None
        reaped = False
        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--creator",
                BINARY,
                str(self.root),
                str(configured_signal),
                str(reader),
            ],
            pass_fds=(reader,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(reader)
        try:
            assert process.stdout is not None and process.stdin is not None
            frames = []
            for _ in range(2):
                frame = json.loads(process.stdout.readline())
                frames.append(frame)
                if "child_pid" in frame:
                    child_pid = frame["child_pid"]
            ready = next(frame for frame in frames if "ready" in frame)
            assert child_pid is not None
            self.assertEqual(ready["guard"], configured_signal)
            process.stdin.close()  # explicit creator exit; no deadline or timer
            self.assertEqual(process.wait(), 0)
            if configured_signal == 0:
                os.kill(child_pid, 0)  # still alive; disabled signal stays disabled
                os.write(writer, b"continue")
                terminal = json.loads(process.stdout.readline())
                self.assertEqual(terminal["after"], 0)
            _, status = os.waitpid(child_pid, 0)
            reaped = True
            self.assertEqual(os.waitstatus_to_exitcode(status), -configured_signal)
        finally:
            os.close(writer)
            if process.poll() is None:
                assert process.stdin is not None
                process.stdin.close()
                process.wait()
            if child_pid is not None and not reaped:
                try:
                    os.kill(child_pid, signal.SIGKILL)  # explicit owned-fixture cleanup
                except ProcessLookupError:
                    pass
                os.waitpid(child_pid, 0)
            process.stdout.close()
            process.stderr.close()

    def test_same_owner_actual_parent_death_and_disabled(self) -> None:
        for configured_signal in (9, 0):
            self.death_in_scope(configured_signal)

    @unittest.skipUnless(os.geteuid() == 0, "root foreign Store-owner fixture")
    def test_foreign_owner_actual_parent_death_and_disabled(self) -> None:
        self.owner(65532)
        for configured_signal in (9, 0):
            self.death_in_scope(configured_signal)


if __name__ == "__main__":
    if sys.argv[1] == "--creator":
        creator(*sys.argv[2:5], int(sys.argv[5]))
    else:
        BINARY = str(Path(sys.argv.pop(1)).resolve())
        unittest.main(verbosity=2)
