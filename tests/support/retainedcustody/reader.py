"""Bounded operator recovery adapter over the public TensorFS push CLI.

Each short-lived grant arrives on stdin. No capability is written to disk or printed.
This process never changes an output's native disposition or claims Hub verification.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import tensorfs

MAX_REQUEST = 64 << 10
OBJECT = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PreparedPush:
    manifest_digest: str
    object_id: str
    length: int
    command: tuple[str, ...]
    grant: str


def failure(exc: Exception) -> dict[str, Any]:
    code = re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(exc, "code", "recovery_refused")))[:128]
    return {"ok": False, "code": code, "error_type": type(exc).__name__}


def exact_ref(value: Any) -> tuple[str, int]:
    if not isinstance(value, dict) or set(value) != {"digest", "length"}:
        raise ValueError("invalid reference")
    digest, length = value["digest"], value["length"]
    if not isinstance(digest, str) or not OBJECT.fullmatch(digest) or type(length) is not int or length <= 0:
        raise ValueError("invalid reference")
    return digest, length


class Reader:
    def __init__(self, root: Path, tfs: Path, *, allow_local: bool = False) -> None:
        if not root.is_absolute() or not tfs.is_absolute() or not tfs.is_file():
            raise ValueError("Store and TensorFS CLI paths must be absolute")
        self.root, self.tfs = root, tfs
        self.store = tensorfs.Store.open(str(root))
        self.allow_local = allow_local
        self.manifest: tuple[str, int] | None = None
        self.manifest_bytes = b""
        self.objects: dict[str, int] = {}

    def close(self) -> None:
        self.manifest = None

    def prepare(self, value: Any) -> PreparedPush:
        if not isinstance(value, dict) or set(value) != {"manifest", "object", "grant"}:
            raise ValueError("invalid request")
        manifest = exact_ref(value["manifest"])
        object_id, length = exact_ref(value["object"])
        grant = value["grant"]
        if not isinstance(grant, dict) or set(grant) != {"url", "required_headers"}:
            raise ValueError("invalid grant")
        url, headers = grant["url"], grant["required_headers"]
        if not isinstance(url, str) or any(c in url for c in "\r\n\0") or not isinstance(headers, dict):
            raise ValueError("invalid grant")
        expected = {"if-none-match": "*", "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(object_id[7:])).decode()}
        if headers != expected:
            raise ValueError("grant headers differ from exact object")
        if self.manifest != manifest:
            self.close()
            raw = bytes(self.store.manifest(manifest[0])["manifest"])
            if len(raw) != manifest[1] or "sha256:" + hashlib.sha256(raw).hexdigest() != manifest[0]:
                raise ValueError("manifest bytes differ")
            # Native closure owns membership; the public push command verifies and
            # streams the one named object through its existing transport.
            objects = {str(row["id"]): int(row["length"]) for row in self.store.walk(manifest[0])}
            self.manifest, self.manifest_bytes, self.objects = manifest, raw, objects
        is_manifest = object_id == manifest[0] and length == manifest[1]
        if not is_manifest and self.objects.get(object_id) != length:
            raise ValueError("object outside exact native manifest closure")
        host = urlparse(url).hostname or ""
        command = [str(self.tfs), "push", str(self.root), object_id,
                   "--grant-file", "/dev/stdin", "--allow-hosts", host]
        if is_manifest:
            command.append("--manifest")
        if self.allow_local:
            command.append("--allow-local")
        grant_bytes = "\n".join([url, *(f"{k}: {v}" for k, v in headers.items())]) + "\n"
        return PreparedPush(manifest[0], object_id, length, tuple(command), grant_bytes)

    @staticmethod
    def push(prepared: PreparedPush) -> dict[str, Any]:
        object_id, length = prepared.object_id, prepared.length
        result = subprocess.run(prepared.command, input=prepared.grant, text=True, capture_output=True, env={})
        if result.returncode:
            # Native errors can quote capabilities; retain only exit/type information.
            raise RuntimeError("native push refused")
        lines = result.stdout.splitlines()
        if len(lines) != 3:
            raise ValueError("native push result shape")
        pushed = re.fullmatch(r"pushed\s+(sha256:[0-9a-f]{64})", lines[0])
        status = re.fullmatch(r"status\s+HTTP ([0-9]{3})", lines[1])
        sent = re.fullmatch(r"sent\s+([0-9]+) B(?: \(the immutable key already held these exact bytes\))?", lines[2])
        if not pushed or pushed[1] != object_id or not status or not sent:
            raise ValueError("native push result identity")
        http_status, transferred = int(status[1]), int(sent[1])
        if not (200 <= http_status < 300 or http_status == 412) or transferred > length or (http_status != 412 and transferred != length):
            raise ValueError("native push result length/status")
        # HTTP412 is only an already-present observation; Hub must verify custody.
        return {"ok": True, "manifest_digest": prepared.manifest_digest, "object_id": object_id, "length": length, "transferred_bytes": transferred, "http_status": http_status, "checksum_sha256": object_id if transferred == length else "", "etag": ""}

    def upload(self, value: Any) -> dict[str, Any] | list[dict[str, Any]]:
        if not isinstance(value, list):
            return self.push(self.prepare(value))
        if not 1 <= len(value) <= 4:
            raise ValueError("batch size outside bound")
        try:
            manifests = [exact_ref(row["manifest"]) for row in value]
            if any(manifest != manifests[0] for manifest in manifests):
                raise ValueError("batch spans manifests")
            # All native Store access and grant validation finish on this thread.
            prepared = [self.prepare(row) for row in value]
        except Exception as exc:
            return [failure(exc) for _ in value]

        def push_one(item: PreparedPush) -> dict[str, Any]:
            try:
                return self.push(item)
            except Exception as exc:
                return failure(exc)

        # Each child CLI owns its own native Store/lease. These threads only wait
        # for child processes; no native ReadLease is shared between Python threads.
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(push_one, prepared))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tfs", type=Path, required=True)
    parser.add_argument("--allow-local", action="store_true", help="isolated loopback proof only")
    args = parser.parse_args()
    reader = Reader(args.store, args.tfs, allow_local=args.allow_local)
    try:
        while raw := sys.stdin.buffer.readline(MAX_REQUEST + 1):
            try:
                if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                    raise ValueError("request exceeds bound")
                result = reader.upload(json.loads(raw))
            except Exception as exc:
                result = failure(exc)
            print(json.dumps(result, separators=(",", ":")), flush=True)
            if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                return 2
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
