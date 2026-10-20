"""Template body: rendered only after exact candidate wheel/requirements verification."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Annotated

from cozy_runtime.author import AssetBound, Context, FileAsset, Outputs, Tree


def main(ctx: Context, *, stage_client: Annotated[FileAsset, AssetBound(max_bytes=6281488)],
         metadata_bundle: Annotated[FileAsset, AssetBound(max_bytes=316402)],
         native_wheel: Annotated[FileAsset, AssetBound(max_bytes=NATIVE_ASSET_BYTES)],
         deadline_unix: int, out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="cozy-h3-proof-"))
    receipts, work = root / "receipts", root / "work"
    for part in ("source", "logs", "reports"):
        (receipts / part).mkdir(parents=True)
    work.mkdir()
    spec_path = receipts / "manifest.json"
    spec_path.write_text(json.dumps(SPEC, indent=2) + "\n")
    for name, source in FILES.items():
        path = receipts / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (receipts / "source/payload.py").write_bytes(Path(__file__).read_bytes())
    helper, rules = {}, {}
    exec(compile(FILES["owned_child.py"], "owned_child.py", "exec"), helper)
    exec(compile(FILES["contract.py"], "contract.py", "exec"), rules)
    identity = {"request_id": ctx.request_id, "context_device": str(ctx.device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "deadline_unix": deadline_unix, "started_unix": time.time(), "ok": False}
    operations = []

    def call(name, command, cap):
        started = time.time()
        timeout = min(cap, rules["deadline_budget"](started, deadline_unix))
        done = helper["run_owned"](command, receipts / f"logs/{name}.stdout", receipts / f"logs/{name}.stderr",
                                   timeout=timeout, env=env, cwd=work)
        operations.append({"name": name, "started_unix": started, "finished_unix": time.time(), **done})
        if not done["ok"]:
            raise RuntimeError(f"{name} failed: {done}")
        return receipts / f"logs/{name}.stdout"

    def asset(value, name, expected):
        data = value.read_bytes()
        rules["require"](len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"],
                         "wrong FileAsset: " + name)
        path = work / name
        path.write_bytes(data)
        return path

    try:
        rules["check_spec"](SPEC)
        rules["deadline_budget"](identity["started_unix"], deadline_unix)
        rules["require"](sys.flags.optimize == 0 and os.environ.get("TRITON_INTERPRET", "") in ("", "0"), "unsafe interpreter mode")
        env = dict(os.environ, PYTHONNOUSERSITE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        from cozy_runtime._build_provenance import COMMIT
        identity["runtime_source"] = COMMIT
        identity["runtime_version"] = importlib.metadata.version("cozy-runtime")
        rules["require"](COMMIT == SPEC["runtime"]["source"] and identity["runtime_version"] == SPEC["runtime"]["version"], "Runtime drift")
        runtime_root = Path(importlib.util.find_spec("cozy_runtime").origin).parent.parent
        identity["runtime_files"] = {name: hashlib.sha256((runtime_root / name).read_bytes()).hexdigest()
                                     for name in SPEC["runtime"]["files"]}
        rules["require"](identity["runtime_files"] == SPEC["runtime"]["files"], "Runtime file drift")
        identity["dependencies"] = {name: importlib.metadata.version(name) for name in SPEC["dependencies"]["versions"]}
        rules["require"](identity["dependencies"] == SPEC["dependencies"]["versions"], "dependency drift")
        client = asset(stage_client, "stage-client", SPEC["client"])
        client.chmod(0o700)
        bundle = asset(metadata_bundle, "metadata.tar.gz", SPEC["metadata"])
        wheel = asset(native_wheel, SPEC["native"]["file"], SPEC["native"])
        metadata_root = work / "metadata"
        metadata_root.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            rules["require"]({m.name for m in members} == set(SPEC["metadata_files"]) and len(members) == 23,
                             "metadata members differ")
            for member in members:
                rel = Path(member.name)
                rules["require"](member.isfile() and not rel.is_absolute() and ".." not in rel.parts, "unsafe metadata path")
                expected = SPEC["metadata_files"][member.name]
                rules["require"](member.size == expected["bytes"] and member.size <= 1 << 20, "metadata bound differs")
                data = archive.extractfile(member).read()
                rules["require"](hashlib.sha256(data).hexdigest() == expected["sha256"], "metadata bytes differ")
                dst = metadata_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        cache = work / "source-store"
        source = {"source_label": "paul/minimax-h3@1.0.0-rc.3/int8", "legacy_manifest_sha256": rules["MANIFEST"],
                  "header": str(metadata_root / f"d10/blobs/0f/03/{rules['HEADER']}"),
                  "header_sha256": rules["HEADER"], "cache_dir": str(cache), "tensor_count": 3858,
                  "stored_bytes": rules["PAYLOAD"]}
        source_path = work / "source.json"
        source_path.write_text(json.dumps(source))
        preflight = json.loads(call("metadata", [str(client), str(source_path)], 60).read_text())
        rules["require"](preflight.get("status") == "metadata_only" and not cache.exists(), "metadata preflight wrote cache")
        capacity = json.loads(call("capacity", [sys.executable, str(receipts / "source/probe.py"),
            "--spec", str(spec_path), "--harness", str(receipts / "source/h3_load.py"), "--wheel", str(wheel),
            "--work", str(work), "--deadline", str(deadline_unix)], 60).read_text())
        rules["check_capacity"](capacity)
        rules["require"](capacity["cuda_visible_devices"] == identity["cuda_visible_devices"], "child seal drift")
        (receipts / "reports/capacity.json").write_text(json.dumps(capacity, indent=2))
        stage = json.loads(call("stage", [str(client), str(source_path), "--download"], rules["MAX_WALL"]).read_text())
        rules["check_stage"](stage)
        (receipts / "reports/stage.json").write_text(json.dumps(stage, indent=2))
        result = receipts / "reports/h3.json"
        call("benchmark", [sys.executable, str(receipts / "source/h3_load.py"), "--source", str(source_path),
            "--work", str(work / "fixtures"), "--result", str(result), "--wheel", str(wheel),
            "--wheel-sha256", SPEC["native"]["sha256"], "--native-source", SPEC["native"]["source"],
            "--torch-version", SPEC["dependencies"]["versions"]["torch"], "--device", "cuda:0"], rules["MAX_WALL"])
        result_value = json.loads(result.read_text())
        identity["benchmark"] = rules["check_benchmark"](result_value, SPEC, capacity["gpu_uuid"])
        identity["ok"] = True
    except BaseException:
        (receipts / "logs/worker-failure.log").write_text(traceback.format_exc())
    identity["operations"] = operations
    identity["finished_unix"] = time.time()
    identity["ok"] = identity["ok"] and identity["finished_unix"] <= deadline_unix
    (receipts / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (receipts / "files.json").write_text(json.dumps({str(p.relative_to(receipts)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in receipts.rglob("*") if p.is_file()}, indent=2) + "\n")
    # Only this dedicated receipts root is returned; no cache, wheel or normalized fixture enters the Tree.
    return out.save_tree(receipts)
