#!/usr/bin/env python3
"""The per-GPU ledger through `tensorfs.varena`, across real processes, CPU only.

This process creates a ledger and holds a row. A child adopts the memfd by descriptor, claims a
row, publishes, and answers an ask. Then it is SIGKILLed: its row reads dead, is reaped, and its
bytes count against admission until they are returned.

    PYTHONPATH=python scripts/varena-ledger.py
"""

import os
import signal
import subprocess
import sys
import time

from tensorfs import varena

MIB = 1 << 20
GIB = 1 << 30


def child(fd: int) -> None:
    ledger = varena.Ledger.adopt(fd)
    row = ledger.claim()
    resident, cold = 4 * GIB, 3 * GIB
    row.publish(context=512 * MIB, resident=resident, cold=cold, torch_reserved=GIB, torch_allocated=GIB,
                last_use=time.monotonic_ns())
    print(f"@{row.id}", flush=True)
    want = row.wait_want()
    freed = min(want, cold)
    row.publish(context=512 * MIB, resident=resident - freed, cold=cold - freed, torch_reserved=GIB,
                torch_allocated=GIB, last_use=time.monotonic_ns())
    row.ack(freed)
    print(f"@{want} {freed}", flush=True)
    sys.stdin.readline()


def row_of(ledger: varena.Ledger, pid: int) -> varena.RowFacts:
    return next(r for r in ledger.snapshot() if r.pid == pid)


def main() -> None:
    ledger = varena.Ledger.create("GPU-py", 8 * GIB)
    assert (ledger.gpu_uuid, ledger.capacity) == ("GPU-py", 8 * GIB)
    me = ledger.claim()
    me.publish(context=500 * MIB, torch_reserved=2 * GIB, torch_allocated=GIB, active_peak=2 * GIB,
               last_use=time.monotonic_ns())
    proc = subprocess.Popen([sys.executable, __file__, "--child", str(ledger.fd)], pass_fds=[ledger.fd],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    def reply() -> list[int]:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("@"), f"child said {line!r}"
        return [int(w) for w in line[1:].split()]

    [slot] = reply()
    theirs, mine = row_of(ledger, proc.pid), row_of(ledger, os.getpid())
    assert (theirs.slot, theirs.state, theirs.cold, theirs.held) == (slot, "live", 3 * GIB, 512 * MIB + 5 * GIB)
    assert (theirs.unevictable, theirs.reclaimable) == (512 * MIB + 2 * GIB, 3 * GIB)
    assert (mine.state, mine.held, mine.reclaimable) == ("live", 500 * MIB + 2 * GIB, 0)
    free = 8 * GIB - theirs.held - mine.held
    assert ledger.admit(free).now

    admit = ledger.admit(free + GIB)
    assert not admit.now and admit.short == 0
    assert [(v.pid, v.bytes) for v in admit.victims] == [(proc.pid, GIB)]

    victims = ledger.ask(GIB, asker=me.id)
    assert [(v.pid, v.bytes) for v in victims] == [(proc.pid, GIB)]
    ledger.wait(victims[0])
    assert ledger.answered(victims[0])
    assert reply() == [GIB, GIB]
    theirs = row_of(ledger, proc.pid)
    assert (theirs.want, theirs.acked, theirs.cold, theirs.progress) == (0, GIB, 2 * GIB, 3)
    assert ledger.stalled() == []

    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    dead = row_of(ledger, proc.pid)
    assert (dead.state, dead.held, dead.reclaimable) == ("dead", 512 * MIB + 4 * GIB, 0)
    assert ledger.ask(GIB) == []
    room = 8 * GIB - mine.held
    admit = ledger.admit(room)
    assert not admit.now and admit.victims == [] and admit.short == dead.held
    assert ledger.reap(proc.pid) == dead.held
    assert row_of(ledger, proc.pid).state == "reaping"
    assert ledger.admit(room).short == dead.held
    ledger.returned(proc.pid)
    assert [r.pid for r in ledger.snapshot()] == [os.getpid()]
    assert ledger.admit(room).now

    try:
        ledger.reap(os.getpid())
    except varena.LedgerError as e:
        assert e.code == "INVALID", e.code
    else:
        raise AssertionError("a live process was reaped")
    print("varena ledger: ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(int(sys.argv[2]))
    else:
        main()
