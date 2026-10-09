"""Trusted development transport, invoked by Creator over an owned SSH connection.

This never runs in a package executor and never changes its fences. It submits one
Comfy prompt, keeps an independent observer, and serves immutable retained files.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def atomic(path, value):
    atomic_text(path, json.dumps(value, sort_keys=True))


def atomic_text(path, value):
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return None if fields[2] == "Z" else int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def status(root):
    result = root / "result.json"
    if result.exists():
        return json.loads(result.read_text())
    checkpoint = root / "checkpoint/state.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    owner = root / "process.json"
    process = json.loads(owner.read_text()) if owner.exists() else {}
    live = bool(process) and identity(process["pid"]) == process["start_ticks"]
    return {"status": "running" if live else "observer_stopped", "prompt_id": state.get("prompt_id", ""),
            "post_attempted": state.get("post_attempted", False), "process": process}


def serve(root):
    with (root / "observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # The original observer still owns the only submission path.
        if (root / "result.json").exists():
            return
        import importlib.util
        spec = importlib.util.spec_from_file_location("cozy_comfy_observer", root / "observer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        request = json.loads((root / "request.json").read_text())
        payload = dict(request["payload"])
        remaining = int(request["deadline_unix_s"] - time.time())
        if remaining <= 0:
            atomic(root / "result.json", {"status": "timed_out", "files": [],
                   "detail": "Original observation deadline expired; no new submission sent",
                   "prompt_id": status(root).get("prompt_id", "")})
            return
        payload["timeout_s"] = remaining
        try:
            result = module.observe(**payload, directory=root / "checkpoint",
                                    deadline=time.monotonic() + remaining + 1,
                                    require_existing_state=request.get("observer_attempts", 0) > 1,
                                    check_cancelled=lambda: None)
        except Exception as exc:
            result = {"status": "failed", "detail": f"{type(exc).__name__}: {exc}",
                      "prompt_id": status(root).get("prompt_id", "")}
        artifacts = []
        source = root / "checkpoint/artifacts"
        total = 0
        for path in sorted(source.rglob("*")) if source.exists() else []:
            if path.is_symlink():
                raise ValueError("retained artifacts cannot contain symlinks")
            if not path.is_file():
                continue
            length = path.stat().st_size
            total += length
            if total > 64 << 30 or len(artifacts) >= 8192:
                raise ValueError("retained artifacts exceed their bound")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            artifacts.append({"name": str(path.relative_to(source)), "bytes": length, "sha256": digest})
        result["files"] = artifacts
        result["finished_unix_s"] = time.time()
        atomic(root / "result.json", result)


def dispatch(request):
    if sys.version_info < (3, 11):
        raise ValueError("Comfy development observer requires Python3.11 or newer")
    operation = request["operation"]
    if not re.fullmatch(r"[a-f0-9]{64}", operation):
        raise ValueError("invalid operation identity")
    base = Path(request["state_root"])
    if not base.is_absolute():
        raise ValueError("remote state root must be absolute")
    root = base / operation
    action = request["action"]
    if action in {"prepare", "start"}:
        payload = request["payload"]
        if not isinstance(payload.get("graph_json"), str) or len(payload["graph_json"]) > 4 << 20:
            raise ValueError("graph_json must be a bounded JSON string")
        if not 1 <= payload.get("port", 8188) <= 65535 or not 1 <= payload.get("timeout_s", 3600) <= 7200:
            raise ValueError("port or deadline is out of bounds")
        # Canonical inputs are immutable under the operation key. The captured
        # observer itself is kept on replay, including after a CLI upgrade.
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if request.get("require_existing") and not (root / "request.json").is_file():
            raise ValueError("acknowledged remote receipt is missing; no new Comfy submission sent")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / "start.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = root / "request.json"
            if path.exists():
                if json.loads(path.read_text())["fingerprint"] != fingerprint:
                    raise ValueError("idempotency conflict: Comfy inputs changed")
            else:
                atomic_text(root / "observer.py", request["observer_source"])
                atomic_text(root / "driver.py", request["driver_source"])
                atomic(path, {"fingerprint": fingerprint, "payload": payload,
                              "deadline_unix_s": time.time() + payload.get("timeout_s", 3600)})
            if action == "prepare":
                return {"status": "prepared", "prompt_id": status(root).get("prompt_id", "")}
            state = status(root)
            if state["status"] == "observer_stopped":
                held = json.loads(path.read_text())
                attempted = held.get("observer_attempts", 0)
                if (attempted or (root / "process.json").exists()) and not (root / "checkpoint/state.json").is_file():
                    raise ValueError("Original submission checkpoint is missing; no new Comfy submission sent")
                held["observer_attempts"] = attempted + 1
                atomic(path, held)  # A crash before process.json is still a prior attempt.
                # A resumed observer reads the original durable post_attempted
                # marker. A lost POST acknowledgement can never POST again.
                with (root / "observer.log").open("ab") as log:
                    process = subprocess.Popen([sys.executable, str(root / "driver.py"), "serve", str(root)],
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               start_new_session=True)
                atomic(root / "process.json", {"pid": process.pid, "start_ticks": identity(process.pid)})
            return status(root)
    if not root.is_dir():
        raise ValueError("unknown Comfy operation")
    if action == "status":
        return status(root)
    if action == "fetch":
        result = status(root)
        record = next((item for item in result.get("files", []) if item["name"] == request["name"]), None)
        if record is None:
            raise ValueError("artifact is not in the retained manifest")
        path = root / "checkpoint/artifacts" / record["name"]
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to((root / "checkpoint/artifacts").resolve()):
            raise ValueError("artifact escapes its operation")
        offset = request.get("offset", 0)
        if type(offset) is not int or not 0 <= offset <= record["bytes"]:
            raise ValueError("invalid artifact offset")
        with path.open("rb") as stream:
            stream.seek(offset)
            while chunk := stream.read(1 << 20):
                sys.stdout.buffer.write(chunk)
        return None
    raise ValueError("unsupported Comfy development operation")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "serve":
        serve(Path(sys.argv[2]))
    else:
        # _cozy_request is parsed by the fixed SSH bootstrap, never interpolated
        # into shell source. Only the CLI's own embedded driver is executed here.
        try:
            result = dispatch(_cozy_request)
        except Exception as exc:
            if _cozy_request.get("action") == "fetch":
                raise
            result = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        if result is not None:
            print(json.dumps(result), flush=True)
