"""Bound only a newly-created proof child session; retain file-backed output on timeout."""
import os
from pathlib import Path
import signal
import subprocess
import time

SETUP_SECONDS = 60
CONSUMER_SECONDS = 180
KERNEL_SECONDS = 300
TERM_SECONDS = 3
KILL_SECONDS = 10


def run_owned(command, stdout, stderr, *, timeout, env=None, cwd=None,
              term_seconds=TERM_SECONDS, kill_seconds=KILL_SECONDS):
    """Never signal the parent/session or an existing process: pgid is this Popen's pid.

    File-backed streams avoid a descendant keeping a PIPE open after its parent exits.
    SIGTERM gets a short grace, then SIGKILL and a bounded direct-child reap. A kernel
    uninterruptible child can remain unreaped: record failure instead of waiting forever.
    """
    started = time.monotonic()
    with Path(stdout).open('wb') as output, Path(stderr).open('wb') as errors:
        child = subprocess.Popen(command, stdout=output, stderr=errors, env=env, cwd=cwd,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        pgid = child.pid

        def group_alive():
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False

        def send(sig):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass

        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        lingering = not timed_out and group_alive()
        signals = []
        if timed_out or lingering:
            send(signal.SIGTERM)
            signals.append('SIGTERM')
            deadline = time.monotonic() + term_seconds
            while time.monotonic() < deadline and group_alive():
                child.poll()
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            if group_alive():
                send(signal.SIGKILL)
                signals.append('SIGKILL')
        try:
            rc = child.wait(timeout=kill_seconds)
            reaped = True
        except subprocess.TimeoutExpired:
            rc, reaped = None, False
    return {'command': list(command), 'pid': child.pid, 'pgid': pgid, 'rc': rc,
            'timeout_s': timeout, 'elapsed_s': time.monotonic() - started,
            'timed_out': timed_out, 'lingering_descendants': lingering,
            'signals': signals, 'reaped': reaped,
            'ok': rc == 0 and reaped and not timed_out and not lingering}
