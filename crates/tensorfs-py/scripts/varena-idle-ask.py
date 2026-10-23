#!/usr/bin/env python3
"""An idle tenant's ask answers what its row offers, and it offers nothing else (tfs-133). Real GPU,
real ledger, two processes.

A child memory holds an LM-style cache (a tensor its shed frees) and stays idle, as character-describer
did beside H3 on the release smoke (171 asks of it answered 0 B). This process asks it:
- idle, it offers its cache and nothing else (no torch bytes);
- an ask gets the cache back to the driver, and the offer drops;
- busy (a request running), it offers no torch bytes, and an ask is answered at once;
- a cache whose owner gives nothing is out of the offer after one ask.

    PYTHONPATH=python scripts/varena-idle-ask.py
"""

import subprocess
import sys

import torch

from tensorfs.varena import Ledger
from tensorfs.varena.torch import LedgerGrant, Memory

MIB = 1 << 20
CACHE = 512 * MIB
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{f': {detail}' if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def child(fd: int, events: int) -> None:
    memory = Memory.open(torch.device("cuda", 0), LedgerGrant(0, fd, events))
    held: list[torch.Tensor] = []

    def cache(gives: bool) -> None:
        held.append(torch.ones(CACHE, dtype=torch.uint8, device="cuda"))

        def shed(_: int) -> int:
            if not gives:  # its owner is busy: nothing goes
                return 0
            nbytes = sum(t.numel() for t in held)
            held.clear()
            return nbytes

        memory.cache("lm:test", shed).nbytes = CACHE

    with memory.request("warm"):  # its end publishes the idle offer
        cache(gives=True)
    print("ready", flush=True)
    request = None
    for line in sys.stdin:
        match line.split():
            case ["busy"]:
                cache(gives=True)
                request = memory.request("busy")
                request.__enter__()
            case ["idle"]:
                request.__exit__(None, None, None)
                memory.caches.clear()
                held.clear()
                with memory.request("stubborn"):
                    cache(gives=False)
            case _:
                return
        print("ok", flush=True)


def main() -> int:
    torch.cuda.init()
    total = torch.cuda.mem_get_info()[1]
    table = Ledger.create(str(torch.cuda.get_device_properties(0).uuid), total)
    events = table.events_fd
    assert events is not None
    proc = subprocess.Popen(
        [sys.executable, __file__, "--child", str(table.fd), str(events)],
        pass_fds=(table.fd, events), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    say = lambda command: (proc.stdin.write(command + "\n"), proc.stdin.flush(), proc.stdout.readline())  # noqa: E731
    assert proc.stdout.readline().strip() == "ready"
    memory = Memory.open(torch.device("cuda", 0), LedgerGrant(0, table.fd, events))
    row = lambda: next(r for r in table.snapshot() if r.pid == proc.pid and r.state == "live")  # noqa: E731

    offered = row().reclaimable
    check("idle, it offers its cache and no torch bytes", offered == CACHE, f"{offered} B offered")
    got = memory._ask(memory._free() + CACHE)
    check("an ask gets the cache back to the driver", got >= CACHE * 9 // 10, f"got {got} B")
    check("and the offer drops", row().reclaimable == 0, f"{row().reclaimable} B offered")

    say("busy")
    offered = row().reclaimable
    check("busy, it offers no torch bytes", offered == 0, f"{offered} B offered")
    got = memory._ask(memory._free() + CACHE)
    check("a busy row is answered at once with what it offered", got == 0, f"got {got} B")

    say("idle")
    check("a stubborn cache is offered while untried", row().reclaimable == CACHE, f"{row().reclaimable} B")
    got = memory._ask(memory._free() + CACHE)
    check("its owner gives nothing", got == 0, f"got {got} B")
    check("and it is out of the offer after one ask", row().reclaimable == 0, f"{row().reclaimable} B")

    proc.stdin.close()
    proc.wait()
    print(f"{'FAILED ' + ', '.join(FAILED) if FAILED else 'ALL PASSED'}", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(int(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(main())
