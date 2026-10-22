"""CPU proof of native.py, no GPU: real processes of stub test binaries (shell scripts that speak
libtest's arguments and output), bundled as the runner takes them: `gpu` with a case that passes
(printing a long TIMINGS line), one that fails, one whose
failure floods the log (its tail keeps the case's share of the run log), one that hangs (judged by its earlier steps, so killed as stalled), one that
hangs unjudged until a cancel, and the GPU monitor; `tensorfs-io` with a test that prints no GPU. cozy_runtime's author API is stood in for by the few names native.py uses.

    python3 -I crates/varena/gpu/test_native.py
"""
from pathlib import Path
import importlib.util
import json
import sys
import tarfile
import tempfile
import threading
import time
import types

STUB = r"""#!/bin/sh
case "$3" in
  role_monitor) echo "running 1 test"; echo "test role_monitor ... @ ready"; while read l; do echo "@ unknown no sample"; done ;;
  passes) echo "test passes ... STEP 1 0"; echo "TIMINGS holder load: ok $(printf 'map 1x 1ms, %.0s' $(seq 30))" >&2; echo "STEP 2 5"; echo "PASS it passes GPU-00000000-0000-0000-0000-000000000000"; echo "test result: ok. 1 passed; 0 failed; 0 ignored" ;;
  fails) echo "test fails ... STEP 1 0"; echo "thread 'fails' panicked: on purpose" >&2; echo "test result: FAILED. 0 passed; 1 failed; 0 ignored"; exit 101 ;;
  floods) echo "test floods ... STEP 1 0"; seq 300; echo "thread 'floods' panicked: Exhausted { $(printf 'Ref { block: %s }, ' $(seq 5000)) }" >&2; echo "test result: FAILED. 0 passed; 1 failed; 0 ignored"; exit 101 ;;
  hangs|waits) echo "test $3 ... STEP 1 0"; exec sleep 100000 ;;
  plain) echo "test plain ... ok"; echo "test result: ok. 1 passed; 0 failed; 0 ignored" ;;
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
        bundle = Path(tmp) / "bundle.tar"
        with tarfile.open(bundle, "w") as tar:
            for name in ["gpu", "tensorfs-io"]:
                stub = Path(tmp) / name
                stub.write_text(STUB)
                stub.chmod(0o755)
                tar.add(stub, arcname=name)
        ctx = stand_in.ScriptContext()
        # "hangs" has an earlier longest step, so it is judged; "waits" has none: unjudged until the cancel.
        cases = ["gpu/passes", "tensorfs-io/plain", "gpu/fails", "gpu/floods", "gpu/hangs", "gpu/waits", "gpu/passes2"]
        suite = {"label": "cpu", "cases": cases, "steps": {"gpu/hangs": 0.2}}
        threading.Timer(3.0, lambda: setattr(ctx, "cancelled", True)).start()
        start = time.monotonic()
        tree = native.main(ctx, binary=stand_in.FileAsset(bundle.read_bytes()),
                           suite=stand_in.FileAsset(json.dumps(suite).encode()), out=stand_in.Outputs())
        took = time.monotonic() - start
    result = json.loads(tree["result.json"])
    for level, line in ctx.lines:
        print(f"{level}: {line.splitlines()[0]}")
    assert result["passed"] == ["gpu/passes", "tensorfs-io/plain"], result
    assert result["failed"] == ["gpu/fails", "gpu/floods", "gpu/hangs", "gpu/waits"], result
    assert result["ended"] == {"gpu/hangs": "stalled", "gpu/waits": "cancelled"}, result
    assert result["unjudged"] == ["gpu/passes", "tensorfs-io/plain", "gpu/fails", "gpu/floods", "gpu/waits"], result
    assert result["cancelled"] and not result["ok"], result
    assert "gpu/passes2" not in result["passed"] + result["failed"], "a cancel stops the suite"
    assert any(m.startswith("TAIL gpu/fails") and "panicked: on purpose" in m for _, m in ctx.lines), "a failure's tail reaches the run log"
    assert any(m.startswith("TAIL gpu/waits") for _, m in ctx.lines), "a cancelled case's tail reaches the run log"
    flood = [m for _, m in ctx.lines if m.startswith("TAIL gpu/floods")]
    share = native.LOG_LINES // len(cases) - 2
    assert len(flood) == share and "panicked: Exhausted" in flood[-3] and flood[-1].endswith("STEP 1 0"), flood[-4:]
    figures = [m for _, m in ctx.lines if m.startswith("gpu/passes: ")]
    assert len(figures) == 3 and figures[0].startswith("gpu/passes: TIMINGS holder load: ok map") and all(len(m) <= 200 for m in figures), figures
    assert took < 10, f"the hang and the cancel ended the suite in {took:.1f} s"
    print(f"PASS native.py on CPU: pass (two binaries, figures told), fail, stalled, cancelled, in {took:.1f} s")


if __name__ == "__main__":
    main()
