"""Run the adapter's graph orchestration with a native stand-in, without importing CUDA/torch.

Only the selected classes' orchestration methods are loaded from the actual source. This keeps
these ordering regressions runnable alongside native HostBackend tests on a CPU builder.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
import weakref


class Binding:
    def __init__(self, model, blocks):
        self.model, self.blocks, self.closed = model, blocks, False


def adapter():
    path = Path(__file__).parents[1] / "python/tensorfs/varena/torch.py"
    source = ast.parse(path.read_text())
    selected = {"Model": {"retain"}, "Memory": {"_declare", "graph_region_blocks"}, "GraphRegion": None}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name in selected:
            methods = selected[node.name]
            if methods is not None:
                node.body = [method for method in node.body if getattr(method, "name", None) in methods]
            body.append(node)
    ns = {"Binding": Binding, "threading": threading, "contextmanager": contextmanager, "Unbound": RuntimeError, "_fill": lambda fill: fill.identity}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), ns)
    return ns


class Block:
    def __init__(self, path, region, name):
        self.path, self.region, self.tensors = path, region, {name: {}}


class GraphDependencies(unittest.TestCase):
    def setUp(self):
        self.api = adapter()

    def test_block_dependencies_include_common_and_tied_aliases_only(self):
        model = self.api["Model"]()
        model.common = Block("", 0, "bias")
        model.blocks = {name: Block(name, i + 1, f"{name}.weight") for i, name in enumerate("abc")}
        shared = []
        model.sites = {"a.weight": shared, "b.weight": shared, "c.weight": [], "bias": []}
        binding = model.retain(("a", "a"))
        self.assertEqual([block.region for block in binding.blocks], [0, 1, 2])
        self.assertNotIn(model.blocks["c"], binding.blocks)
        del model.blocks["b"]  # A checkpoint may store only one of the tied names.
        binding = model.retain(("b",))
        self.assertEqual([block.region for block in binding.blocks], [0, 1])

    def test_changed_fill_retires_graphs_before_any_native_rebind(self):
        order = []
        memory = self.api["Memory"]()
        model = self.api["Model"]()
        model.id, model.identity = 1, "bake"
        model.fill = NS(identity="per_step")
        model.holds = [NS(shed=lambda: order.append("step hold"))]
        model.graphs = weakref.WeakSet()
        memory.native = NS(fill=lambda *_: order.append("new fill"))
        for identity in ("per_step", "bake"):
            native = NS(pool=(1, 1), close=lambda: order.append("graph pins"))
            graph = self.api["GraphRegion"](memory, native, [model], lambda: order.append("readers and executable"))
            model.fill.identity = identity
            memory._declare(model)
            self.assertEqual(model.identity, identity)
            self.assertTrue(graph.closed)
            self.assertEqual(order[-4:], ["readers and executable", "graph pins", "step hold", "new fill"])

    def test_failed_retirement_does_not_publish_a_new_fill(self):
        memory = self.api["Memory"]()
        model = self.api["Model"]()
        model.id, model.identity = 1, "old"
        model.fill, model.holds, model.graphs = NS(identity="new"), [], weakref.WeakSet()
        memory.native = NS(fill=lambda *_: self.fail("published a fill while its graph still lives"))
        def failed():
            raise RuntimeError("reader completion failed")
        graph = self.api["GraphRegion"](memory, NS(pool=(1, 1), close=lambda: self.fail("pins released")), [model], failed)
        with self.assertRaisesRegex(RuntimeError, "reader completion"):
            memory._declare(model)
        self.assertEqual(model.identity, "old")
        self.assertFalse(graph.closed)
        self.assertTrue(graph.dropping)

    def test_retirement_waits_for_an_active_replay_to_release(self):
        order, entered, proceed, retired = [], threading.Event(), threading.Event(), threading.Event()
        memory = NS(fence_pool=lambda *_: order.append("fence"))
        native = NS(pool=(1, 1), acquire=lambda: True, release=lambda _: order.append("release"), close=lambda: order.append("close"))
        graph = self.api["GraphRegion"](memory, native)
        def replay():
            with graph.replaying(NS(cuda_stream=1)) as ok:
                self.assertTrue(ok)
                entered.set()
                self.assertTrue(proceed.wait(2))
        def retire():
            graph.retire()
            retired.set()
        worker = threading.Thread(target=replay)
        worker.start()
        self.assertTrue(entered.wait(2))
        closer = threading.Thread(target=retire)
        closer.start()
        self.assertFalse(retired.wait(0.02))
        proceed.set()
        worker.join(2)
        closer.join(2)
        self.assertTrue(retired.is_set())
        self.assertEqual(order, ["release", "fence", "close"])
        with graph.replaying(NS(cuda_stream=1)) as ok:
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
