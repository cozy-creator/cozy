#!/usr/bin/env python3
"""Exercise the standalone 0.1.2 -> 0.2.0 migration against real files."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import cbor2

ROOT = Path(__file__).resolve().parent.parent
MIGRATE = ROOT / "tools" / "migrate-0.1.2-to-0.2.0.py"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(root: Path, kind: str, data: bytes) -> tuple[str, int]:
    identity = digest(data)
    suffix = ".json" if kind == "manifests" else ""
    path = root / kind / identity[:2] / identity[2:4] / f"{identity}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return identity, len(data)


def main() -> int:
    tfs = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="tensorfs-migration-smoke-") as raw:
        root = Path(raw)
        for directory in ("repos/org", "blobs", "manifests", "tmp"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        config = b'{ "z": 1.0, "a": -0 }'
        config_id, config_length = put(root, "blobs", config)
        current = cbor2.loads(
            (ROOT / "conformance/cbor/vectors/green/minimal-inline.cbor").read_bytes()
        )
        current[1] = [
            ["model", bytes.fromhex(config_id), config_length, "application/json"]
        ]
        old_header = cbor2.dumps(current, canonical=True)
        old_header_id, old_header_length = put(root, "blobs", old_header)
        readme_id, readme_length = put(root, "blobs", b"sample model\n")
        manifest = canonical(
            {
                "entries": [
                    {
                        "blob": {"length": readme_length, "sha256": readme_id},
                        "kind": "file",
                        "path": "README.txt",
                    },
                    {
                        "blob": {
                            "length": old_header_length,
                            "sha256": old_header_id,
                        },
                        "kind": "cozytensors",
                        "path": "model.cozytensors",
                    },
                    {
                        "blob": {"length": config_length, "sha256": config_id},
                        "kind": "file",
                        "path": "model/config.json",
                    },
                ]
            }
        )
        manifest_id, manifest_length = put(root, "manifests", manifest)
        repository = canonical(
            {
                "checkpoints": [
                    {
                        "evidence": {
                            "config_digest": "sha256:" + "11" * 32,
                            "shape_digest": "sha256:" + "22" * 32,
                            "topology_digest": "sha256:" + "33" * 32,
                        },
                        "manifest": {
                            "length": manifest_length,
                            "sha256": manifest_id,
                        },
                        "published": True,
                    }
                ],
                "releases": [
                    {
                        "lanes": [
                            {
                                "lane": "main",
                                "manifest": {
                                    "length": manifest_length,
                                    "sha256": manifest_id,
                                },
                            }
                        ],
                        "revision": 1,
                        "version": "1.0.0",
                        "yanked": False,
                    }
                ],
                "repo": {"name": "model", "org": "org"},
            }
        )
        repo_path = root / "repos/org/model.json"
        repo_path.write_bytes(repository)
        connection = sqlite3.connect(root / "tensorfs.sqlite")
        connection.execute("PRAGMA user_version=1")
        connection.close()

        command = (sys.executable, str(MIGRATE), str(root), "--tfs", str(tfs))
        subprocess.run(command, check=True)
        assert repo_path.read_bytes() == repository
        subprocess.run((*command, "--apply"), check=True)
        migrated = json.loads(repo_path.read_bytes())
        assert "evidence" not in migrated["checkpoints"][0]
        new_manifest_id = migrated["checkpoints"][0]["manifest"]["sha256"]
        new_manifest = json.loads(
            (root / "manifests" / new_manifest_id[:2] / new_manifest_id[2:4] / f"{new_manifest_id}.json").read_bytes()
        )
        assert [entry["path"] for entry in new_manifest["entries"]] == [
            "README.txt",
            "model.cozytensors",
        ]
        header_ref = new_manifest["entries"][1]["blob"]
        header_path = (
            root
            / "blobs"
            / header_ref["sha256"][:2]
            / header_ref["sha256"][2:4]
            / header_ref["sha256"]
        )
        parsed = cbor2.loads(header_path.read_bytes())
        assert parsed[1] == [["model", b'{"a":0,"z":1}']]
        assert (
            root / "migration-backup-0.1.2/repos/org/model.json"
        ).read_bytes() == repository
        assert (root / "tensorfs.sqlite.0.1.2.bak").is_file()
        assert sqlite3.connect(root / "tensorfs.sqlite").execute(
            "PRAGMA user_version"
        ).fetchone()[0] == 2
        subprocess.run((*command, "--apply"), check=True)
    print("migration smoke green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
