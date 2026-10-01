#!/usr/bin/env python3
"""Qualify keyed cache roots through the installed wheel: put, replay, list, drop, GC."""
from pathlib import Path
import json
import tempfile

import tensorfs
from tensorfs.errors import Refusal


def tree(directory: Path, body: bytes) -> tuple[bytes, list[tuple[str, Path]]]:
    source = directory / f"src-{tensorfs.object_id(body)[7:15]}"
    source.write_bytes(body)
    manifest = json.dumps({"entries": [{
        "blob": {"sha256": tensorfs.object_id(body).removeprefix("sha256:"), "length": len(body)},
        "kind": "file", "path": "out.bin",
    }]}, sort_keys=True, separators=(",", ":")).encode()
    return manifest, [("out.bin", source)]


def main() -> None:
    assert "keyed-roots/1" in tensorfs.CAPABILITIES
    with tempfile.TemporaryDirectory(prefix="tfs-keyed-roots-") as temporary:
        directory = Path(temporary)
        store = tensorfs.Store.init(directory / "store")
        space, first, second = "encoder", "1" * 64, "2" * 64
        manifest, files = tree(directory, b"first output")
        root = store.put_keyed_root(space, first, manifest, files)
        assert root == {
            "space": space, "key": first, "manifest_digest": tensorfs.object_id(manifest),
            "manifest_length": len(manifest), "bytes": len(manifest) + 12,
            "created_unix_ms": root["created_unix_ms"],
        }, root
        assert store.put_keyed_root(space, first, manifest, files) == root
        assert store.keyed_root(space, first) == root
        other, other_files = tree(directory, b"second output")
        try:
            store.put_keyed_root(space, first, other, other_files)
        except Refusal as error:
            assert error.code == "TRANSACTION_CONFLICT", error
        else:
            raise AssertionError("a different tree replaced a keyed root")
        newer = store.put_keyed_root(space, second, other, other_files)
        assert store.keyed_roots(space) == [root, newer]
        assert store.drop_keyed_root(space, first) and not store.drop_keyed_root(space, first)
        assert store.keyed_root(space, first) is None
        assert sorted(p.name for p in (Path(store.root) / "roots/keyed" / space).iterdir()) == [
            f"{second}.json"]
        assert tensorfs.gc(store.root)["reclaimed_bytes"] == root["bytes"]
        assert store.keyed_roots(space) == [newer]
        assert store.put_keyed_root(space, first, manifest, files)["key"] == first
        print("keyed-roots: put/replay/conflict/list/drop and GC reclamation pass")


if __name__ == "__main__":
    main()
