# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime==0.27.4"]
# ///
"""Run the portable native tests through `cozy run --on RENTAL --gpus 1`.

The outer Runtime launches a standalone Rust binary; it does not replace the worker's SDK.
The binary checks its actual visible GPU and runs each case in a fresh process.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

CASES = {
    "pages_pin_evict_refill_and_survive_clean_owner_exit": "PASS independent pages, pin protection, RAM refill and kept-page import",
    "escaped_writable_view_retains_pages_and_recovers": "PASS escaped writable retention and same-process recovery",
    "read_only_fault_is_contained_in_its_process": "PASS read-only VMM protection and healthy replacement",
}


def main(*, binary: Annotated[FileAsset, AssetBound(max_bytes=67108864)], out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="varena-slim-gpu-"))
    executable = root / "slim-gpu"
    executable.write_bytes(binary.read_bytes())
    executable.chmod(0o700)
    logs = root / "logs"
    logs.mkdir()
    results = []
    for name, marker in CASES.items():
        command = [str(executable), "--ignored", "--exact", name, "--nocapture", "--test-threads=1"]
        done = subprocess.run(command, capture_output=True, text=True, timeout=180)
        text = done.stdout + done.stderr
        (logs / f"{name}.log").write_text(text)
        print(text, flush=True)
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        if done.returncode or marker not in text or not summary or summary[-1] != ("1", "0", "0"):
            raise RuntimeError(f"Native GPU case {name} failed (exit {done.returncode})")
        results.append(name)
    (logs / "result.json").write_text(json.dumps({"passed": results, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2))
    return out.save_tree(logs)
