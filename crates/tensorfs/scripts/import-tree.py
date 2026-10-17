#!/usr/bin/env python3
"""Qualify the installed local-tree importer and no-source replay."""
from pathlib import Path
import json
import tempfile

import tensorfs
from tensorfs.errors import Refusal


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tfs-import-tree-") as temporary:
        directory = Path(temporary)
        store = tensorfs.Store.init(directory / "store")
        source = directory / "source"
        body = b"exact imported output"
        source.write_bytes(body)
        manifest = json.dumps({"entries": [{
            "blob": {"sha256": tensorfs.object_id(body).removeprefix("sha256:"), "length": len(body)},
            "kind": "file", "path": "nested/output.bin",
        }]}, sort_keys=True, separators=(",", ":")).encode()
        owner = tensorfs.object_id(b"local output producer")
        files = [("nested/output.bin", source)]
        result = store.import_tree(owner, manifest, files)
        assert result["complete"] and not result["released"]
        assert result["manifest_digest"] == tensorfs.object_id(manifest)
        source.unlink()
        assert store.import_tree(owner, manifest, files) == result
        assert store.manifest(result["manifest_digest"])["header"] is None
        with store.acquire_manifest(result["manifest_digest"]) as lease:
            actual = bytearray(len(body))
            lease.read_into(tensorfs.object_id(body), len(body), 0, len(body), actual)
            assert actual == body
        destination = directory / "checkout"
        store.checkout(result["manifest_digest"], destination, symlink=False)
        assert (destination / "nested/output.bin").read_bytes() == body
        for selected in (store.walk_cozytensors, store.acquire_cozytensors):
            try:
                selected(result["manifest_digest"])
            except Refusal as error:
                assert error.code == "ATTACHMENT_CARDINALITY", error
            else:
                raise AssertionError("ordinary files were accepted as a tensor checkpoint")
        for roster in ([], [files[0], files[0]], [("extra", source)]):
            try:
                store.import_tree(owner, manifest, roster)
            except Refusal as error:
                assert error.code == "TRANSACTION_CONFLICT", error
            else:
                raise AssertionError("inexact local member roster was accepted")
        print("import-tree: native import/replay/roster pass")


if __name__ == "__main__":
    main()
