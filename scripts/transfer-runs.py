#!/usr/bin/env python3
"""Live proof for the TensorFS-owned exact Store-to-Store transfer door."""

from __future__ import annotations

import hashlib
import io
import pathlib
import tempfile

import tensorfs


def object_id(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def object_path(store: tensorfs.Store, identity: str) -> pathlib.Path:
    """Test-only access to plant corruption inside the layout-owning repository."""

    digest = identity.removeprefix("sha256:")
    return pathlib.Path(store.root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tensorfs-transfer-runs-") as raw_root:
        root = pathlib.Path(raw_root)
        raw = b"one exact object crossing between TensorFS stores"
        identity = object_id(raw)
        source = tensorfs.Store.init(str(root / "source"))
        destination = tensorfs.Store.init(str(root / "destination"))
        source.put_reader(io.BytesIO(raw), identity, len(raw))

        admitted = tensorfs.transfer_object(source, destination, identity, len(raw))
        skipped = tensorfs.transfer_object(source, destination, identity, len(raw))
        assert admitted == {"id": identity, "length": len(raw), "admitted": True}
        assert skipped == {"id": identity, "length": len(raw), "admitted": False}
        assert destination.document(identity) == raw
        assert destination.verify(identity)["verdict"] == "verified"
        print("PASS exact transfer admits once and skips an existing exact key")

        wrong_length = tensorfs.Store.init(str(root / "wrong-length"))
        try:
            tensorfs.transfer_object(source, wrong_length, identity, len(raw) + 1)
        except tensorfs.errors.LengthMismatch:
            assert not wrong_length.contains(identity)
            print("PASS expected-length disagreement commits nothing")
        else:
            raise AssertionError("wrong expected length transferred")

        corrupt_destination = tensorfs.Store.init(str(root / "corrupt-destination"))
        corrupt_destination.put_reader(io.BytesIO(raw), identity, len(raw))
        planted_destination = object_path(corrupt_destination, identity)
        planted_destination.chmod(0o644)
        planted_destination.write_bytes(b"Y" * len(raw))
        planted_destination.chmod(0o444)
        try:
            tensorfs.transfer_object(source, corrupt_destination, identity, len(raw))
        except tensorfs.errors.ObjectCorrupt:
            assert not corrupt_destination.contains(identity)
            assert list((pathlib.Path(corrupt_destination.root) / "quarantine").iterdir())
            print("PASS an existing wrong destination is quarantined, never overwritten")
        else:
            raise AssertionError("wrong destination was treated as an exact skip")

        corrupt_source = tensorfs.Store.init(str(root / "corrupt-source"))
        corrupt_source.put_reader(io.BytesIO(raw), identity, len(raw))
        planted = object_path(corrupt_source, identity)
        planted.chmod(0o644)
        planted.write_bytes(b"X" * len(raw))
        planted.chmod(0o444)
        clean_destination = tensorfs.Store.init(str(root / "clean-destination"))
        try:
            tensorfs.transfer_object(corrupt_source, clean_destination, identity, len(raw))
        except tensorfs.errors.ObjectCorrupt:
            assert not corrupt_source.contains(identity)
            assert not clean_destination.contains(identity)
            assert list((pathlib.Path(corrupt_source.root) / "quarantine").iterdir())
            print("PASS corrupt source is quarantined and never reaches destination")
        else:
            raise AssertionError("corrupt source transferred")

        absent = tensorfs.Store.init(str(root / "absent"))
        try:
            tensorfs.transfer_object(absent, destination, identity, len(raw))
        except tensorfs.errors.ObjectAbsent:
            print("PASS absent source refuses through the Store boundary")
        else:
            raise AssertionError("absent source transferred")

    print("store-to-store transfer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
