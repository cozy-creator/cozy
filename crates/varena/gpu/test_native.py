"""CPU proof of native.py, no GPU: real processes of a stub test binary (a shell script that
speaks libtest's arguments and output) for a case that passes, one that fails, one that hangs
(judged by its earlier steps, so killed as stalled), one that hangs unjudged until a cancel, and
the GPU monitor. cozy_runtime's author API is stood in for by the few names native.py uses.

    python3 -I crates/varena/gpu/test_native.py
"""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import threading
import time
import types

STUB = r"""#!/bin/sh
case "$3" in
  role_monitor) echo "running 1 test"; echo "test role_monitor ... @ ready"; while read l; do echo "@ unknown no sample"; done ;;
  passes) echo "test passes ... STEP 1 0"; echo "STEP 2 5"; echo "PASS it passes GPU-00000000-0000-0000-0000-000000000000"; echo "test result: ok. 1 passed; 0 failed; 0 ignored" ;;
  fails) echo "test fails ... STEP 1 0"; echo "thread 'fails' panicked: on purpose" >&2; echo "test result: FAILED. 0 passed; 1 failed; 0 ignored"; exit 101 ;;
  hangs|waits) echo "test $3 ... STEP 1 0"; exec sleep 100000 ;;
esac
"""


def author() -> types.ModuleType:
    """The names native.py takes from cozy_runtime.author."""
    module = types.ModuleType("cozy_runtime.author")

    class FileAsset:
        def __init__(self, data: bytes):
            self.data = data

        def read_bytes(self) -> bytes:
            return self.data

    class Outputs:
        def save_tree(self, path):
            return {p.name: p.read_text() for p in Path(path).iterdir()}

    class ScriptContext:
        def __init__(self):
            self.cancelled = False
            self.lines = []

        def log(self, message, *, level="info"):
            self.lines.append((level, message))

    module.AssetBound = lambda **_: None
    module.FileAsset, module.Outputs, module.ScriptContext, module.Tree = FileAsset, Outputs, ScriptContext, dict
    package = types.ModuleType("cozy_runtime")
    package.author = module
    sys.modules["cozy_runtime"], sys.modules["cozy_runtime.author"] = package, module
    return module


def main():
    stand_in = author()
    spec = importlib.util.spec_from_file_location("native", Path(__file__).with_name("native.py"))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "stub"
        stub.write_text(STUB)
        stub.chmod(0o755)
        ctx = stand_in.ScriptContext()
        # "hangs" has an earlier longest step, so it is judged; "waits" has none: unjudged until the cancel.
        suite = {"label": "cpu", "cases": ["passes", "fails", "hangs", "waits", "passes2"], "steps": {"hangs": 0.2}}
        threading.Timer(3.0, lambda: setattr(ctx, "cancelled", True)).start()
        start = time.monotonic()
        tree = native.main(ctx, binary=stand_in.FileAsset(stub.read_bytes()),
                           suite=stand_in.FileAsset(json.dumps(suite).encode()), out=stand_in.Outputs())
        took = time.monotonic() - start
    result = json.loads(tree["result.json"])
    for level, line in ctx.lines:
        print(f"{level}: {line.splitlines()[0]}")
    assert result["passed"] == ["passes"], result
    assert result["failed"] == ["fails", "hangs", "waits"], result
    assert result["ended"] == {"hangs": "stalled", "waits": "cancelled"}, result
    assert result["unjudged"] == ["passes", "fails", "waits"], result
    assert result["cancelled"] and not result["ok"], result
    assert "passes2" not in result["passed"] + result["failed"], "a cancel stops the suite"
    assert any("TAIL fails" in m and "panicked: on purpose" in m for _, m in ctx.lines), "a failure's tail reaches the run log"
    assert any("TAIL waits" in m for _, m in ctx.lines), "a cancelled case's tail reaches the run log"
    assert took < 10, f"the hang and the cancel ended the suite in {took:.1f} s"
    print(f"PASS native.py on CPU: pass, fail, stalled, cancelled, in {took:.1f} s")


if __name__ == "__main__":
    main()
