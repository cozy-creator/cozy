#!/usr/bin/env python3
"""Exercise the standalone 0.1.2/0.2.x -> 0.3.0 migration against real files."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import cbor2

ROOT = Path(__file__).resolve().parent.parent
MIGRATE = ROOT / "tools" / "migrate-to-0.3.0.py"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_path(root: Path, identity: str) -> Path:
    return root / "blobs" / identity[:2] / identity[2:4] / identity


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
        for encoding in current[3]:
            encoding.insert(3, ["retired documentation"])
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

        intermediate = cbor2.loads(
            (ROOT / "conformance/cbor/vectors/green/minimal-inline.cbor").read_bytes()
        )
        intermediate[1] = [["model", b'{"hidden_size":1}']]
        for encoding in intermediate[3]:
            encoding.insert(3, ["retired documentation"])
        intermediate_header = cbor2.dumps(intermediate, canonical=True)
        intermediate_header_id, intermediate_header_length = put(
            root, "blobs", intermediate_header
        )
        intermediate_manifest = canonical(
            {
                "entries": [
                    {
                        "blob": {
                            "length": intermediate_header_length,
                            "sha256": intermediate_header_id,
                        },
                        "kind": "cozytensors",
                        "path": "model.cozytensors",
                    }
                ]
            }
        )
        intermediate_manifest_id, intermediate_manifest_length = put(
            root, "manifests", intermediate_manifest
        )
        intermediate_repository = canonical(
            {
                "checkpoints": [
                    {
                        "manifest": {
                            "length": intermediate_manifest_length,
                            "sha256": intermediate_manifest_id,
                        },
                        "published": True,
                    }
                ],
                "releases": [],
                "repo": {"name": "intermediate", "org": "org"},
            }
        )
        intermediate_repo_path = root / "repos/org/intermediate.json"
        intermediate_repo_path.write_bytes(intermediate_repository)

        def sortable_manifest(number: int) -> tuple[str, int]:
            sortable_config = canonical({"n": number})
            sortable_config_id, sortable_config_length = put(
                root, "blobs", sortable_config
            )
            sortable_header = cbor2.loads(
                (
                    ROOT / "conformance/cbor/vectors/green/minimal-inline.cbor"
                ).read_bytes()
            )
            for encoding in sortable_header[3]:
                encoding.insert(3, ["retired documentation"])
            sortable_header[1] = [
                [
                    "model",
                    bytes.fromhex(sortable_config_id),
                    sortable_config_length,
                    "application/json",
                ]
            ]
            sortable_header_bytes = cbor2.dumps(sortable_header, canonical=True)
            sortable_header_id, sortable_header_length = put(
                root, "blobs", sortable_header_bytes
            )
            return put(
                root,
                "manifests",
                canonical(
                    {
                        "entries": [
                            {
                                "blob": {
                                    "length": sortable_header_length,
                                    "sha256": sortable_header_id,
                                },
                                "kind": "cozytensors",
                                "path": "model.cozytensors",
                            },
                            {
                                "blob": {
                                    "length": sortable_config_length,
                                    "sha256": sortable_config_id,
                                },
                                "kind": "file",
                                "path": "model/config.json",
                            },
                        ]
                    }
                ),
            )

        sortable_refs = [sortable_manifest(0), sortable_manifest(2)]
        assert sortable_refs[0][0] < sortable_refs[1][0]
        sortable_repository = canonical(
            {
                "checkpoints": [
                    {
                        "manifest": {"length": length, "sha256": identity},
                        "published": True,
                    }
                    for identity, length in sortable_refs
                ],
                "releases": [
                    {
                        "lanes": [
                            {
                                "lane": lane,
                                "manifest": {
                                    "length": reference[1],
                                    "sha256": reference[0],
                                },
                            }
                            for lane, reference in zip(
                                ("a", "b"), sortable_refs, strict=True
                            )
                        ],
                        "revision": 1,
                        "version": "1.0.0",
                        "yanked": False,
                    }
                ],
                "repo": {"name": "sort-regression", "org": "org"},
            }
        )
        sortable_repo_path = root / "repos/org/sort-regression.json"
        sortable_repo_path.write_bytes(sortable_repository)
        connection = sqlite3.connect(root / "tensorfs.sqlite")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE tensorfs_derived_transactions "
            "(id TEXT PRIMARY KEY, bytes BLOB NOT NULL) STRICT"
        )
        historical = (
            ("sha256:" + "11" * 32, canonical({"state": "committed"})),
            ("sha256:" + "22" * 32, canonical({"state": "abandoned"})),
            (
                "sha256:" + "33" * 32,
                canonical({"current_session": None, "state": "open"}),
            ),
        )
        connection.executemany(
            "INSERT INTO tensorfs_derived_transactions(id,bytes) VALUES (?,?)",
            historical,
        )
        connection.commit()
        connection.close()

        command = (sys.executable, str(MIGRATE), str(root), "--tfs", str(tfs))
        for relative in (
            "tmp/leases/read-123-456.lease",
            "tmp/writers/123-456.writer",
            "tmp/writers/recovery.lock",
        ):
            marker = root / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("w+b") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                refused = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                assert refused.returncode != 0
                assert "TensorFS has a live lease/writer" in refused.stderr
            marker.unlink()
        subprocess.run(command, check=True)
        assert repo_path.read_bytes() == repository
        assert sqlite3.connect(root / "tensorfs.sqlite").execute(
            "SELECT count(*) FROM tensorfs_derived_transactions"
        ).fetchone()[0] == len(historical)
        subprocess.run((*command, "--apply"), check=True)
        migrated = json.loads(repo_path.read_bytes())
        assert "evidence" not in migrated["checkpoints"][0]
        new_manifest_id = migrated["checkpoints"][0]["manifest"]["sha256"]
        new_manifest = json.loads(
            (
                root
                / "manifests"
                / new_manifest_id[:2]
                / new_manifest_id[2:4]
                / f"{new_manifest_id}.json"
            ).read_bytes()
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
        assert all(len(encoding) == 4 for encoding in parsed[3])
        intermediate_migrated = json.loads(intermediate_repo_path.read_bytes())
        intermediate_ref = intermediate_migrated["checkpoints"][0]["manifest"]
        intermediate_document = json.loads(
            (
                root
                / "manifests"
                / intermediate_ref["sha256"][:2]
                / intermediate_ref["sha256"][2:4]
                / f"{intermediate_ref['sha256']}.json"
            ).read_bytes()
        )
        intermediate_header_ref = intermediate_document["entries"][0]["blob"]
        intermediate_final_bytes = object_path(
            root, intermediate_header_ref["sha256"]
        ).read_bytes()
        assert len(intermediate_final_bytes) == intermediate_header_ref["length"]
        assert digest(intermediate_final_bytes) == intermediate_header_ref["sha256"]
        intermediate_final = cbor2.loads(intermediate_final_bytes)
        assert intermediate_final[1] == [["model", b'{"hidden_size":1}']]
        assert all(len(encoding) == 4 for encoding in intermediate_final[3])
        sortable_migrated = json.loads(sortable_repo_path.read_bytes())
        sortable_checkpoints = [
            row["manifest"]["sha256"] for row in sortable_migrated["checkpoints"]
        ]
        sortable_lanes = {
            row["lane"]: row["manifest"]["sha256"]
            for row in sortable_migrated["releases"][0]["lanes"]
        }
        assert sortable_lanes["b"] < sortable_lanes["a"]
        assert sortable_checkpoints == [sortable_lanes["b"], sortable_lanes["a"]]
        assert (
            root / "migration-backup-0.3.0/repos/org/model.json"
        ).read_bytes() == repository
        assert (root / "tensorfs.sqlite.pre-0.3.0.bak").is_file()
        assert sqlite3.connect(root / "tensorfs.sqlite.pre-0.3.0.bak").execute(
            "SELECT count(*) FROM tensorfs_derived_transactions"
        ).fetchone()[0] == len(historical)
        assert (
            sqlite3.connect(root / "tensorfs.sqlite")
            .execute("PRAGMA user_version")
            .fetchone()[0]
            == 2
        )
        assert (
            sqlite3.connect(root / "tensorfs.sqlite")
            .execute("SELECT count(*) FROM tensorfs_derived_transactions")
            .fetchone()[0]
            == 0
        )
        assert len(list((root / "tmp").glob("migration-0.2.0-rows-*.jsonl"))) == 1
        rerun = subprocess.run(
            (*command, "--apply"), capture_output=True, text=True, check=True
        )
        assert (
            "preflight: 0 new CAS objects, 0 repository replacements, "
            "0 historical derived rows to archive"
        ) in rerun.stdout
    print("migration smoke green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
