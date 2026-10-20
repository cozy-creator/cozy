"""CPU checks of the exact binding boundary; actual page pressure is a CUDA Rust test."""

from __future__ import annotations

import ast
import queue
import sys
import threading
import types
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path


def boundary(source: Path) -> types.ModuleType:
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = [node for node in classes["Memory"].body
               if getattr(node, "name", "") in ("_enter", "_exit")]
    module = types.ModuleType("fill_binding_source")
    module.__dict__.update(
        dataclass=dataclass, field=field, threading=threading,
        _disable_current_modes=nullcontext,
        Unbound=type("Unbound", (RuntimeError,), {}),
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(
            current_stream=lambda _: "stream", is_current_stream_capturing=lambda: False,
        )),
        _unbind=lambda block: setattr(block, "stream", None),
    )
    sys.modules[module.__name__] = module
    code = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        classes["Block"],
        ast.ClassDef(name="Memory", bases=[], keywords=[], body=methods, decorator_list=[]),
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), module.__dict__)  # noqa: S102
    return module


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "python/tensorfs/varena/torch.py"
)
B = boundary(SOURCE)


class Changed(threading.Condition):
    def __init__(self) -> None:
        super().__init__(threading.RLock())
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


class Room:
    _enter = B.Memory._enter
    _exit = B.Memory._exit

    def __init__(self) -> None:
        self.changed = Changed()
        self.device, self.native = "cuda", None
        self.bind = lambda block: None

    def _bind(self, block, stream) -> None:
        self.bind(block)
        block.stream = stream


def run(call):
    result = queue.Queue()

    def execute():
        try:
            result.put((True, call()))
        except BaseException as error:
            result.put((False, error))

    # An old same-thread self-wait must fail this test, not hang its entire process.
    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    return worker, result


class BindingBoundary(unittest.TestCase):
    def test_self_block_refuses_and_reuses_without_harming_bound_reentry(self) -> None:
        room = Room()
        block = B.Block(types.SimpleNamespace(memory=room), -1, "", None)
        room.bind = lambda _: room._enter(block, record=False)
        worker, result = run(lambda: room._enter(block, record=False))
        ok, error = result.get(timeout=2)
        worker.join()
        self.assertFalse(ok)
        self.assertIsInstance(error, B.Unbound)
        self.assertIn("own unfinished binding", str(error))
        self.assertEqual((block.count, block.binding, block.binding_owner), (0, False, None))
        calls = []
        room.bind = calls.append
        room._enter(block, record=False)
        room._enter(block, record=False)
        self.assertEqual(block.count, 2)
        self.assertEqual(calls, [block])
        room._exit(block)
        room._exit(block)
        self.assertEqual((block.count, block.binding, block.binding_owner), (0, False, None))

    def test_another_threads_binding_still_waits_then_borrows(self) -> None:
        room = Room()
        block = B.Block(types.SimpleNamespace(memory=room), -1, "", None)
        entered, release = threading.Event(), threading.Event()

        def bind(_):
            entered.set()
            self.assertTrue(release.wait(2))

        room.bind = bind
        first, a = run(lambda: room._enter(block, record=False))
        self.assertTrue(entered.wait(2))
        second, b = run(lambda: room._enter(block, record=False))
        try:
            self.assertTrue(room.changed.waiting.wait(2))
            self.assertTrue(b.empty(), "the other thread borrowed an unfinished binding")
        finally:
            release.set()
        self.assertTrue(a.get(timeout=2)[0])
        self.assertTrue(b.get(timeout=2)[0])
        first.join()
        second.join()
        self.assertEqual(block.count, 2)
        room._exit(block)
        room._exit(block)

    def test_another_blocks_lazy_dependency_remains_valid(self) -> None:
        room = Room()
        model = types.SimpleNamespace(memory=room)
        outer, factor = B.Block(model, 0, "outer", None), B.Block(model, 1, "factor", None)
        calls = []

        def bind(block):
            calls.append(block.path)
            if block is outer:
                room._enter(factor, record=False)
                room._exit(factor)

        room.bind = bind
        room._enter(outer, record=False)
        self.assertEqual(calls, ["outer", "factor"])
        room._exit(outer)
        self.assertEqual((outer.count, factor.count), (0, 0))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
