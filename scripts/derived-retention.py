#!/usr/bin/env python3
"""Independent result custody through the installed native wheel, including real GC/readback."""

import io
import json
import tempfile

import tensorfs


def identity(value):
    return "sha256:" + value * 32


def main():
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    targets = {"model": {"drop": [], "add": {
        "weight": {"logical_dtype": "f32", "shape": [512], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": [512]}}},
    }}}
    original = identity("31")
    first, second = identity("41"), identity("42")
    data = bytes([0x11]) * 2048
    with tempfile.TemporaryDirectory(prefix="tensorfs-result-retention-") as root:
        store = tensorfs.Store.init(root)
        writer = store.begin_derived(original, 1, {}, targets, {}, [("model", "weight")],
                                     len(data), work_fingerprint=identity("51"))
        writer.add_part("model", "weight", "value", io.BytesIO(data))
        receipt = writer.commit()
        digest = tensorfs.object_id(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
        a = store.retain_derived_result(original, digest, first)
        b = store.retain_derived_result(original, digest, second)
        assert a["manifest"] == b["manifest"] == receipt["manifest"]
        assert a["receipt_digest"] == b["receipt_digest"] == digest
        store.derived_dispose(original)
        store = tensorfs.Store.open(root)
        assert store.retain_derived_result(original, digest, first) == a
        store.release_derived_retention(original, digest, first)
        tensorfs.gc(root)
        payload = receipt["added_objects"][0]
        into = bytearray(len(data))
        with store.acquire_cozytensors(receipt["manifest"]["sha256"]) as lease:
            lease.read_into(payload["sha256"], payload["length"], 0, len(into), into)
        assert bytes(into) == data
        stopped = store.release_derived_retention(original, digest, second)
        assert stopped["released"]
        assert store.release_derived_retention(original, digest, second) == stopped
        tensorfs.gc(root)
        assert not store.contains(payload["sha256"])
        try:
            store.retain_derived_result(original, digest, second)
        except tensorfs.errors.Refusal as error:
            assert error.code == "TRANSACTION_CLOSED", error
        else:
            raise AssertionError("released retention was resurrected")
        print("native result retention green: original receipt preserved, independent owners, "
              "restart, GC, exact tensor readback, final release, no resurrection")


if __name__ == "__main__":
    main()
