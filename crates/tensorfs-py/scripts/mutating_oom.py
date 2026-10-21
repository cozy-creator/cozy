"""Real CPU mutations followed by an injected allocation failure must never replay.

The mode runs the native operation first, so the same exception is delivered after
actual out-resize/in-place side effects. CUDA's unchanged real quota oracle is separate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def load_source(path):
    spec = importlib.util.spec_from_file_location("mutating_oom_cuts", path)
    assert spec is not None and spec.loader is not None
    cuts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cuts
    spec.loader.exec_module(cuts)
    return cuts


class Room:
    def __init__(self):
        self.actions = []

    def free(self, device, need):
        self.actions.append("free")
        return True

    def lower(self, device, need):
        self.actions.append("lower")
        return False

    def available(self, device):
        return 0

    def mapped_output(self, like, device):
        self.actions.append("mapped")
        raise AssertionError("a mutating operation cannot acquire replacement output")

    def exhausted(self, device, need, name):
        return torch.OutOfMemoryError(f"{name} needs {need} B")


class FailAfterMutation(TorchDispatchMode):
    def __init__(self, target, error):
        super().__init__()
        self.target, self.error, self.attempts = target, error, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is self.target:
            self.attempts += 1
            if self.attempts == 1:
                raise self.error
        return out


def unchanged_error(cuts, run, target, room, *, nested):
    original = torch.OutOfMemoryError("Tried to allocate 32 MiB")
    mode = FailAfterMutation(target, original)
    try:
        with torch.no_grad(), mode, cuts.caught():
            if nested:
                cuts._settle(lambda _: run(), 4, 1, lambda n: n // 2,
                             torch.device("cpu"), "mutating outer", key=("mutating",))
            else:
                run()
    except torch.OutOfMemoryError as caught:
        assert caught is original, "the primary allocation error was replaced"
        assert caught.__traceback__ is not None, "the original traceback was erased"
        assert mode.attempts == 1, f"a partially mutating op ran {mode.attempts} times"
    else:
        raise AssertionError("the failing mutating op was silently retried")
    assert not room.actions, room.actions
    assert cuts._local.layer is None and not cuts._local.scopes


def main():
    from tensorfs.varena.torch import Memory

    path = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            Path(__file__).resolve().parents[1] / "python/tensorfs/varena/cuts.py")
    cuts = load_source(path)
    for nested in [False, True]:
        room, memo = Room(), cuts.Memo()
        cuts.serve(memo, room)
        source, destination = torch.arange(8, dtype=torch.float32), torch.empty(0)
        unchanged_error(cuts, lambda: torch.ops.aten.silu.out(source, out=destination),
                        torch.ops.aten.silu.out, room, nested=nested)
        assert destination.shape == source.shape, "the mode did not exercise actual native out resize"
        torch.testing.assert_close(destination, torch.ops.aten.silu.default(source), rtol=0, atol=0)
        assert memo.retries == 0
        value = torch.zeros(8)
        unchanged_error(cuts, lambda: torch.ops.aten.add_.Tensor(value, 1),
                        torch.ops.aten.add_.Tensor, room, nested=nested)
        assert torch.equal(value, torch.ones(8)), "an in-place mutation was applied twice"
        assert memo.retries == 0
        with torch.no_grad(), cuts.caught():
            assert torch.equal(torch.ops.aten.add.Tensor(value, 2), torch.full((8,), 3.0))

    room, memo = Room(), cuts.Memo()
    cuts.serve(memo, room)
    source, destination = torch.arange(8, dtype=torch.float32), torch.empty(0)
    site = cuts.Site(("empty-out", 0), torch.device("cpu"), cuts.Track("empty-out"))

    def forward(value):
        return torch.ops.aten.silu.out(value, out=destination)

    with torch.no_grad():
        cuts.run(site, forward, (source,), {})
    assert not memo.blocks[site.key].bare, "an empty out destination made its block falsely replayable"
    unchanged_error(cuts, lambda: cuts.run(site, forward, (source,), {}),
                    torch.ops.aten.silu.out, room, nested=True)
    assert memo.retries == 0

    # Functional results remain replayable: this mode discards only a fresh result.
    room, memo = Room(), cuts.Memo()
    cuts.serve(memo, room)
    fresh = torch.ops.aten.add.Tensor
    mode = FailAfterMutation(fresh, torch.OutOfMemoryError("Tried to allocate 32 MiB"))
    with torch.no_grad(), mode, cuts.caught():
        actual = fresh(source, 2)
    assert mode.attempts == 2 and room.actions == ["free"] and memo.retries == 1
    torch.testing.assert_close(actual, source + 2, rtol=0, atol=0)

    # Exercise the public request cleanup with the same escaping primary error.
    # This is CPU control-flow coverage; the real CUDA refusal remains root's gate.
    memory = Memory.open(torch.device("cpu"))
    original = torch.OutOfMemoryError("Tried to allocate 32 MiB")
    mode = FailAfterMutation(torch.ops.aten.silu.out, original)
    try:
        try:
            with memory.request("failed-mutation") as request, torch.no_grad(), mode, cuts.caught():
                torch.ops.aten.silu.out(source, out=destination)
        except torch.OutOfMemoryError as caught:
            assert caught is original and mode.attempts == 1
        else:
            raise AssertionError("failed mutation did not leave its request")
        assert request.stats is not None and not request.cleanup_errors
        assert not memory.active and cuts._local.layer is None and not cuts._local.scopes
        with memory.request("healthy-after-mutation") as healthy, torch.no_grad(), cuts.caught():
            actual = fresh(source, 2)
        assert healthy.stats is not None and not healthy.cleanup_errors
        torch.testing.assert_close(actual, source + 2, rtol=0, atol=0)
    finally:
        memory.close()
    print("PASS mutating OOM: native out/in-place single attempt, original error, nested scopes, warm dispatch and healthy reuse")
    print("PASS functional OOM: fresh results retain bounded retry")


if __name__ == "__main__":
    main()
