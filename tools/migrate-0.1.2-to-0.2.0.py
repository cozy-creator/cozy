#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["cbor2==6.1.4"]
# ///
"""One-shot, restartable TensorFS 0.1.2 -> 0.2.0 repository migration.

The released TensorFS library reads only the current header. This standalone tool owns the
retired four-field ConfigRef row, stages every replacement before changing a repository root,
keeps the old repository documents and SQLite catalog as backups, and never rewrites payload
objects in place.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cbor2


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def object_path(root: Path, digest: str) -> Path:
    return root / "blobs" / digest[:2] / digest[2:4] / digest


def manifest_path(root: Path, digest: str) -> Path:
    return root / "manifests" / digest[:2] / digest[2:4] / f"{digest}.json"


def read_exact(path: Path, digest: str, length: int) -> bytes:
    data = path.read_bytes()
    if len(data) != length or sha256(data) != digest:
        raise ValueError(f"{path}: bytes do not match sha256:{digest} length {length}")
    return data


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def marker_is_live(path: Path) -> bool:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"coordination marker is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def require_quiescent(root: Path) -> int:
    catalog = root / "tensorfs.sqlite"
    markers = [
        marker
        for directory in ("leases", "writers")
        for marker in (root / "tmp" / directory).glob("*")
    ]
    if any(marker_is_live(marker) for marker in markers):
        raise RuntimeError("TensorFS has a live lease/writer; stop every owner first")
    if (root / "tensorfs.sqlite-wal").exists() or (root / "tensorfs.sqlite-shm").exists():
        raise RuntimeError("SQLite WAL/SHM files exist; stop TensorFS cleanly before migration")
    if not catalog.exists():
        return 0
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in (
            "tensorfs_holds",
            "tensorfs_operations",
            "tensorfs_source_preparations",
            "tensorfs_source_results",
        ):
            if table in tables and connection.execute(
                f"SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)"
            ).fetchone()[0]:
                raise RuntimeError(f"{table} is nonempty; finish or abandon active work first")
        historical_derived = (
            connection.execute("SELECT count(*) FROM tensorfs_derived_transactions").fetchone()[0]
            if "tensorfs_derived_transactions" in tables
            else 0
        )
    finally:
        connection.close()
    return historical_derived


def ref(row: object) -> tuple[str, int]:
    if not isinstance(row, dict) or set(row) != {"sha256", "length"}:
        raise ValueError("expected one exact ObjectRef")
    digest, length = row["sha256"], row["length"]
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(length, int):
        raise ValueError("malformed ObjectRef")
    return digest, length


@dataclass(frozen=True)
class Replacement:
    path: Path
    data: bytes


class Migration:
    def __init__(self, root: Path, tfs: Path) -> None:
        self.root = root
        self.tfs = tfs
        self.objects: dict[Path, bytes] = {}
        self.manifests: dict[str, tuple[str, int]] = {}

    def run(self, *args: str) -> None:
        subprocess.run((str(self.tfs), *args), check=True)

    def canonical_config(self, name: str, data: bytes) -> bytes:
        with tempfile.TemporaryDirectory(prefix="tfs-config-migrate-", dir=self.root / "tmp") as raw:
            work = Path(raw)
            source, output = work / "source.json", work / "canonical.json"
            source.write_bytes(data)
            self.run("config", "canonicalize", str(source), "--out", str(output))
            canonical = output.read_bytes()
        if not canonical:
            raise ValueError(f"config {name!r} canonicalized to empty bytes")
        return canonical

    def header(self, digest: str, length: int) -> tuple[str, int, list[tuple[str, str, int]]]:
        source = read_exact(object_path(self.root, digest), digest, length)
        document = cbor2.loads(source)
        if cbor2.dumps(document, canonical=True) != source:
            raise ValueError(f"sha256:{digest}: header is not canonical deterministic CBOR")
        if not isinstance(document, list) or len(document) != 5 or document[0] != "cozytensors/1":
            raise ValueError(f"sha256:{digest}: not a supported 0.1.2 CozyTensors header")
        configs = document[1]
        if not isinstance(configs, list):
            raise ValueError("header configs are not an array")
        if all(isinstance(row, list) and len(row) == 2 for row in configs):
            self.run("cbor", "check", str(object_path(self.root, digest)))
            return digest, length, []

        retired: list[tuple[str, str, int]] = []
        inline: list[list[object]] = []
        for row in configs:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("mixed or unknown config row shape")
            name, raw_digest, raw_length, media_type = row
            if (
                not isinstance(name, str)
                or not isinstance(raw_digest, bytes)
                or len(raw_digest) != 32
                or not isinstance(raw_length, int)
                or not isinstance(media_type, str)
            ):
                raise ValueError(f"config row {name!r} is not the 0.1.2 shape")
            config_digest = raw_digest.hex()
            config = read_exact(object_path(self.root, config_digest), config_digest, raw_length)
            inline.append([name, self.canonical_config(name, config)])
            retired.append((name, config_digest, raw_length))
        document[1] = inline
        encoded = cbor2.dumps(document, canonical=True)
        new_digest = sha256(encoded)
        self.objects[object_path(self.root, new_digest)] = encoded
        with tempfile.TemporaryDirectory(prefix="tfs-header-migrate-", dir=self.root / "tmp") as raw:
            candidate = Path(raw) / "header.cbor"
            candidate.write_bytes(encoded)
            self.run("cbor", "check", str(candidate))
        return new_digest, len(encoded), retired

    def manifest(self, digest: str, length: int) -> tuple[str, int]:
        if digest in self.manifests:
            return self.manifests[digest]
        source = read_exact(manifest_path(self.root, digest), digest, length)
        document = json.loads(source)
        if canonical_json(document) != source:
            raise ValueError(f"sha256:{digest}: Manifest is not canonical JSON")
        entries = document.get("entries") if isinstance(document, dict) else None
        if not isinstance(entries, list):
            raise ValueError(f"sha256:{digest}: malformed Manifest")
        selected = [entry for entry in entries if entry.get("kind") == "cozytensors"]
        if not selected:
            self.manifests[digest] = (digest, length)
            return digest, length
        if len(selected) != 1:
            raise ValueError(f"sha256:{digest}: model Manifest has {len(selected)} headers")
        old_header, old_length = ref(selected[0].get("blob"))
        new_header, new_length, retired = self.header(old_header, old_length)
        retired_paths = {
            (f"{name}/config.json", config_digest, config_length)
            for name, config_digest, config_length in retired
        }
        kept = []
        for entry in entries:
            entry_digest, entry_length = ref(entry.get("blob"))
            if entry.get("path") in {
                "evidence/converter-recipe",
                "evidence/verification-report",
            }:
                continue
            if (
                entry.get("kind") == "file"
                and (entry.get("path"), entry_digest, entry_length) in retired_paths
            ):
                continue
            if entry is selected[0]:
                entry = {
                    "blob": {"length": new_length, "sha256": new_header},
                    "kind": "cozytensors",
                    "path": entry["path"],
                }
            kept.append(entry)
        document["entries"] = kept
        encoded = canonical_json(document)
        new_digest = sha256(encoded)
        if encoded != source:
            with tempfile.TemporaryDirectory(prefix="tfs-manifest-migrate-", dir=self.root / "tmp") as raw:
                candidate, refs = Path(raw) / "manifest.json", Path(raw) / "refs.jsonl"
                candidate.write_bytes(encoded)
                self.run("manifest", "inspect", str(candidate), "--refs", str(refs))
            self.objects[manifest_path(self.root, new_digest)] = encoded
        self.manifests[digest] = (new_digest, len(encoded))
        return new_digest, len(encoded)

    def repositories(self) -> list[Replacement]:
        replacements: list[Replacement] = []
        for path in sorted((self.root / "repos").glob("*/*.json")):
            source = path.read_bytes()
            document = json.loads(source)
            if canonical_json(document) != source:
                raise ValueError(f"{path}: repository is not canonical JSON")
            for checkpoint in document.get("checkpoints", []):
                digest, length = ref(checkpoint["manifest"])
                migrated, migrated_length = self.manifest(digest, length)
                checkpoint["manifest"] = {"length": migrated_length, "sha256": migrated}
                checkpoint.pop("evidence", None)
            for release in document.get("releases", []):
                for lane in release.get("lanes", []):
                    digest, length = ref(lane["manifest"])
                    migrated, migrated_length = self.manifest(digest, length)
                    lane["manifest"] = {"length": migrated_length, "sha256": migrated}
            encoded = canonical_json(document)
            with tempfile.TemporaryDirectory(prefix="tfs-repo-migrate-", dir=self.root / "tmp") as raw:
                candidate, rows = Path(raw) / "repo.json", Path(raw) / "rows.jsonl"
                candidate.write_bytes(encoded)
                self.run("repo", "inspect", str(candidate), "--rows", str(rows))
            if encoded != source:
                replacements.append(Replacement(path, encoded))
        return replacements


def install_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace unequal CAS object {path}")
        return
    temporary = path.with_name(f".{path.name}.migrate-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def replace_repository(root: Path, item: Replacement) -> None:
    backup = root / "migration-backup-0.1.2" / item.path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() and backup.read_bytes() != item.path.read_bytes():
        raise FileExistsError(f"unequal migration backup already exists: {backup}")
    if not backup.exists():
        os.link(item.path, backup)
    temporary = item.path.with_name(f".{item.path.name}.migrate-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(item.data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, item.path)
    fsync_directory(item.path.parent)


def rebuild_catalog(root: Path, tfs: Path) -> None:
    catalog = root / "tensorfs.sqlite"
    backup = root / "tensorfs.sqlite.0.1.2.bak"
    if catalog.exists():
        version = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True).execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        if version == 2:
            return
        if version != 1 or backup.exists():
            raise ValueError(f"refusing unexpected catalog version/backup state: {version}")
        os.replace(catalog, backup)
        fsync_directory(root)
    rows = root / "tmp" / f"migration-0.2.0-rows-{os.getpid()}.jsonl"
    subprocess.run(
        (str(tfs), "store", "rebuild", str(root), "--rows", str(rows), "--sqlite", str(catalog)),
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--tfs", type=Path, required=True, help="TensorFS 0.2.0 tfs binary")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    historical_derived = require_quiescent(root)
    migration = Migration(root, args.tfs.resolve(strict=True))
    replacements = migration.repositories()
    print(
        f"preflight: {len(migration.objects)} new CAS objects, "
        f"{len(replacements)} repository replacements, "
        f"{historical_derived} historical derived rows to archive"
    )
    if not args.apply:
        print("dry run; pass --apply after reviewing the counts")
        return 0
    for path, data in sorted(migration.objects.items(), key=lambda item: str(item[0])):
        install_new(path, data)
    for replacement in replacements:
        replace_repository(root, replacement)
    rebuild_catalog(root, migration.tfs)
    print("migration complete; 0.1.2 repository and catalog backups retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
