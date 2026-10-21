# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4"]
# ///
"""varena's native GPU suite on a rental: `cozy run native.py --on <rental> --gpus 1
--asset binary=<the gpu test binary> --asset suite=<suite.json>`.

`suite.json` is `{"label": "...", "cases": ["name", ...]}`. Each case runs in a fresh process of
the binary, one after another; it passes when the harness reports 1 passed, 0 failed, 0 ignored
and it printed `PASS ... <GPU UUID>` (libtest may put it on its `test <name> ... ` line). Every
case must name the same physical GPU. The outputs hold each case's log (stdout and stderr) and
result.json, whichever cases failed: the caller judges the suite from result.json.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def main(*, binary: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
         suite: Annotated[FileAsset, AssetBound(max_bytes=1 << 16)], out: Outputs) -> Tree:
    selection = json.loads(suite.read_bytes())
    label, cases = selection["label"], selection["cases"]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("the suite names distinct cases")
    print(f"native suite {label}: {len(cases)} cases", flush=True)
    root = Path(tempfile.mkdtemp(prefix="varena-gpu-"))
    program = root / "gpu"
    program.write_bytes(binary.read_bytes())
    program.chmod(0o700)
    logs = root / "logs"
    logs.mkdir()
    passed, failed, gpus = [], [], set()
    for case in cases:
        run = subprocess.run([str(program), "--ignored", "--exact", case, "--nocapture", "--test-threads=1"],
                             capture_output=True, text=True, cwd=root)
        text = run.stdout + run.stderr
        (logs / f"{case}.log").write_text(text)
        print(text, flush=True)
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        named = [m for line in text.splitlines() if "PASS " in line for m in UUID.findall(line)[-1:]]
        if run.returncode or summary[-1:] != [("1", "0", "0")] or not named:
            failed.append(case)
            continue
        gpus.update(named)
        passed.append(case)
    result = {"label": label, "cases": cases, "passed": passed, "failed": failed, "gpus": sorted(gpus),
              "ok": not failed and len(gpus) == 1, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    print(f"native suite {label}: passed {len(passed)}, failed {failed}, GPUs {sorted(gpus)}", flush=True)
    return out.save_tree(logs)
