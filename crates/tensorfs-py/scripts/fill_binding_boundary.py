"""CPU checks of the exact binding boundary; actual page pressure is a CUDA Rust test."""

from __future__ import annotations

import ast
import hashlib
import queue
import sys
import threading
import types
import unittest
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path


def boundary(source: Path) -> types.ModuleType:
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = [node for node in classes["Memory"].body
               if getattr(node, "name", "") in ("_enter", "_exit", "_register")]
    module = types.ModuleType("fill_binding_source")
    module.__dict__.update(
        dataclass=dataclass, field=field, threading=threading, hashlib=hashlib, weakref=weakref,
        regions_of=lambda _: [["weight"]],
        _disable_current_modes=nullcontext,
        Unbound=type("Unbound", (RuntimeError,), {}),
        torch=types.SimpleNamespace(
            from_dlpack=lambda raw: raw,
            cuda=types.SimpleNamespace(
                current_stream=lambda _: "stream", is_current_stream_capturing=lambda: False,
                ExternalStream=lambda stream, device: stream,
                stream=lambda _: nullcontext(),
            ),
        ),
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
    def test_registration_retains_the_fill_that_named_its_content(self) -> None:
        class Fill:
            def __init__(self, version):
                self.version, self.prepared, self.written = version, 0, []

            def prepare(self):
                self.prepared += 1
                return self.version

            def __call__(self, block, parts):
                self.written.append((block, parts))

        class Native:
            def __init__(self):
                self.registrations = []

            def register(self, *arguments):
                self.registrations.append(arguments)
                return types.SimpleNamespace(id=len(self.registrations))

        class Model:
            pass

        first, second = Fill("first"), Fill("second")
        model = Model()
        model.fill, model.name, model.module = first, "model", object()
        model.source = types.SimpleNamespace(fingerprint="source", ck=object())
        model.common = types.SimpleNamespace(regions=[0], parts=lambda _, raw: {"weight": raw})
        model.blocks = []
        room = types.SimpleNamespace(native=Native(), device="cuda", lock=threading.RLock(), _names={})
        B.Memory._register(room, model)
        old = room.native.registrations[0]
        self.assertEqual(model.version, "first")
        self.assertEqual(old[3], hashlib.sha256(repr(("source", "first")).encode()).hexdigest())
        model.fill = second
        old[4](0, "old-bytes", 123)
        self.assertEqual(first.written, [(model.common, {"weight": "old-bytes"})])
        self.assertEqual(second.written, [], "old registration used a later fill selection")
        B.Memory._register(room, model)
        new = room.native.registrations[1]
        self.assertNotEqual(old[3], new[3])
        self.assertEqual(model.version, "second")
        new[4](0, "new-bytes", 123)
        old[4](0, "still-old-bytes", 123)
        self.assertEqual(second.written, [(model.common, {"weight": "new-bytes"})])
        self.assertEqual(first.written[-1][1], {"weight": "still-old-bytes"})
        self.assertEqual((first.prepared, second.prepared), (1, 1))
        owner = weakref.ref(model)
        del model
        self.assertIsNone(owner(), "the callback must retain its fill, not the Model")
        old[4](0, "after-close", 123)
        self.assertEqual(len(first.written), 2)

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
    unittest.main(argv=[sys.argv[0], *sys.argv[2:]])
