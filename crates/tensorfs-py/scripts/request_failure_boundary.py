"""CPU control-flow checks of the exact request source; no Torch/native imports.

The plane and allocator boundaries record ordering and inject cleanup faults.
Actual allocator release is covered separately by cuda_request_failure.py.
"""

from __future__ import annotations

import ast
import builtins
import itertools
import sys
import threading
import traceback
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path


def boundary(source: Path) -> types.ModuleType:
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    request = next(node for node in classes["Memory"].body if getattr(node, "name", "") == "request")
    clear = next(node for node in tree.body if getattr(node, "name", "") == "_clear_frames")
    module = types.ModuleType("request_failure_source")
    module.__dict__.update(
        dataclass=dataclass, field=field, contextmanager=contextmanager,
        nullcontext=nullcontext, cuts=types.SimpleNamespace(caught=nullcontext),
        traceback=traceback, sys=sys, builtins=builtins,
    )
    sys.modules[module.__name__] = module
    code = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        classes["RequestStats"], classes["Request"], clear,
        ast.ClassDef(name="Memory", bases=[], keywords=[], body=[request], decorator_list=[]),
    ], type_ignores=[])
    # Execute only the parsed request boundary under test, without Torch/native imports.
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), module.__dict__)  # noqa: S102
    return module


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "python/tensorfs/varena/torch.py"
)
B = boundary(SOURCE)


class Room:
    request = B.Memory.request

    def __init__(self, cuda: bool = True) -> None:
        self.cuda, self.gpu = cuda, 0
        self.events: list[str] = []
        self.errors: dict[str, BaseException] = {}
        self.models: dict = {}
        self.active: dict = {}
        self.local = threading.local()
        self._ids = itertools.count(1)
        self._names: dict = {}
        self.memo = types.SimpleNamespace(retries=0)
        self.opened: set[int] = set()
        self.slack = 64
        self.native = types.SimpleNamespace(begin=self.begin, end=self.end) if cuda else None
        B.gc = types.SimpleNamespace(collect=lambda: self.stage("gc"))

    def stage(self, stage: str) -> None:
        self.events.append(stage)
        if stage in self.errors:
            raise self.errors[stage]

    def begin(self, id: int, *_: object) -> None:
        self.stage("begin")
        self.opened.add(id)

    def end(self, id: int) -> dict:
        self.opened.remove(id)
        self.stage("end")
        return {"events": [], "copies": [], "io": [], "peak": 0}

    def _measure(self) -> None:
        self.stage("measure")

    def _trim(self, need: int) -> int:
        assert need == 0
        self.stage("trim")
        before, self.slack = self.slack, 0
        return before

    def _uncached_bytes(self) -> int:
        if "stats" in self.errors:
            self.stage("stats")
        return 0

    def _plan(self, _: object) -> tuple:
        return (), []


class RequestFailure(unittest.TestCase):
    def assert_closed(self, room: Room) -> None:
        self.assertFalse(room.active)
        self.assertFalse(room.opened)
        self.assertFalse(room.local.requests)

    def test_success_keeps_idle_slack(self) -> None:
        room = Room()
        with room.request("ok") as request:
            pass
        self.assertEqual(room.slack, 64)
        self.assertEqual(room.events, ["measure", "begin", "gc", "measure", "end"])
        self.assertEqual(request.cleanup_errors, ())
        self.assert_closed(room)

    def test_returned_failure_is_idempotent_and_trims_before_publication(self) -> None:
        room = Room()
        with room.request("failed") as request:
            request.mark_failed()
            request.mark_failed()
            self.assertEqual(room.slack, 64, "marking does not run cleanup in the body")
        self.assertEqual(room.slack, 0)
        self.assertEqual(room.events, ["measure", "begin", "gc", "trim", "measure", "end"])
        self.assertEqual(request.cleanup_errors, ())
        self.assertIsNotNone(request.stats)
        self.assert_closed(room)

    def test_escaping_failure_remains_primary_and_next_request_works(self) -> None:
        room = Room()
        original = ValueError("payload failed")
        with self.assertRaises(ValueError) as caught, room.request("failed"):
            raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(room.slack, 0)
        self.assert_closed(room)
        with room.request("failed") as recovered:
            pass
        self.assertIsNotNone(recovered.stats)
        self.assert_closed(room)

    def test_cleanup_faults_do_not_replace_returned_failure_or_skip_end(self) -> None:
        room = Room()
        trim = RuntimeError("trim failed")
        measure = RuntimeError("measurement failed")
        with room.request("failed") as request:
            request.mark_failed()
            room.errors.update(trim=trim, measure=measure)
        self.assertEqual(request.cleanup_errors, (trim, measure))
        self.assertEqual(room.events[-3:], ["trim", "measure", "end"])
        for error in request.cleanup_errors:
            self.assertIsNone(error.__traceback__)
            self.assertIsNone(error.__cause__)
            self.assertIsNone(error.__context__)
        self.assert_closed(room)

    def test_escaping_failure_gets_cleanup_notes_without_a_reference_cycle(self) -> None:
        room = Room()
        original = ValueError("payload failed")
        with self.assertRaises(ValueError) as caught, room.request("failed") as request:
            room.errors["trim"] = RuntimeError("trim failed")
            raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(original.__notes__, ["request cleanup failed: trim failed"])
        self.assertIsNone(request.cleanup_errors[0].__traceback__)
        self.assert_closed(room)

    def test_cleanup_groups_release_frames_but_keep_fault_classification(self) -> None:
        class Payload:
            pass

        class DriverFault(RuntimeError):
            code = "CUDA"
            sticky = True

        def grouped() -> tuple:
            payload = Payload()
            pointer = weakref.ref(payload)
            try:
                raise DriverFault("sticky cleanup fault")
            except DriverFault as error:
                return ExceptionGroup("cleanup", [error]), pointer

        error, pointer = grouped()
        self.assertIsNotNone(pointer(), "the diagnostic begins with a retained frame")
        room = Room()
        with room.request("failed") as request:
            request.mark_failed()
            room.errors["trim"] = error
        self.assertIs(request.cleanup_errors[0], error)
        self.assertIsNone(pointer(), "cleanup observations must not retain frame payloads")
        child = error.exceptions[0]
        self.assertEqual((child.code, child.sticky), ("CUDA", True))
        self.assertIsNone(child.__traceback__)
        self.assert_closed(room)

    def test_successful_scope_still_raises_cleanup_failure_after_end(self) -> None:
        room = Room()
        failure = RuntimeError("measurement failed")
        with self.assertRaises(RuntimeError) as caught, room.request("ok") as request:
            room.errors["measure"] = failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(request.cleanup_errors, (), "raised faults must not own their frame")
        self.assert_closed(room)

    def test_gc_and_end_failures_are_observed_without_fabricated_stats(self) -> None:
        room = Room()
        with room.request("failed") as request:
            request.mark_failed()
            room.errors.update(gc=RuntimeError("gc failed"), end=RuntimeError("end failed"))
        self.assertEqual([str(error) for error in request.cleanup_errors], ["gc failed", "end failed"])
        self.assertEqual(room.events[-3:], ["trim", "measure", "end"])
        self.assertIsNone(request.stats)
        self.assert_closed(room)

    def test_post_end_stats_fault_preserves_returned_and_escaping_failures(self) -> None:
        for escaping in (False, True):
            with self.subTest(escaping=escaping):
                room = Room()
                original = ValueError("payload failed")
                failure = RuntimeError("stats unavailable")
                try:
                    with room.request("failed") as request:
                        room.errors["stats"] = failure
                        if escaping:
                            raise original
                        request.mark_failed()
                except ValueError as caught:
                    self.assertTrue(escaping)
                    self.assertIs(caught, original)
                    self.assertEqual(caught.__notes__, ["request cleanup failed: stats unavailable"])
                else:
                    self.assertFalse(escaping)
                self.assertIsNone(request.stats)
                self.assertEqual(request.cleanup_errors, (failure,))
                self.assertIsNone(failure.__traceback__)
                self.assertIsNone(failure.__context__)
                self.assertEqual(room.events.count("end"), 1)
                self.assert_closed(room)

    def test_successful_scope_raises_post_end_stats_fault(self) -> None:
        room = Room()
        failure = RuntimeError("stats unavailable")
        with self.assertRaises(RuntimeError) as caught, room.request("ok") as request:
            room.errors["stats"] = failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(room.events.count("end"), 1)
        self.assertIsNone(request.stats)
        self.assert_closed(room)

    def test_cpu_failure_never_calls_cuda_trim(self) -> None:
        room = Room(cuda=False)
        with room.request("cpu") as request:
            request.mark_failed()
        self.assertNotIn("trim", room.events)
        self.assertEqual(request.cleanup_errors, ())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
