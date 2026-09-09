#!/usr/bin/env python3
"""Read held model parts without creating a derived output or writer."""

import io
import tempfile
from pathlib import Path

import tensorfs


def refused(call):
    try:
        call()
    except tensorfs.errors.Refusal:
        return
    raise AssertionError("an ungranted model-part read succeeded")


def main():
    plain = next(value for name, value in tensorfs.seed_digests() if name == "plain/1")
    targets = {"model": {"drop": [], "add": {
        "weight": {"logical_dtype": "f32", "shape": [512], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": [512]}}},
        "scalar": {"logical_dtype": "f32", "shape": [], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": []}}},
    }}}
    payload = bytes(range(256)) * 8
    with tempfile.TemporaryDirectory(prefix="tensorfs-model-part-") as root:
        store = tensorfs.Store.init(root)

        def fixture(identity, scalar):
            writer = store.begin_derived(
                identity, 1, {}, targets, {}, [("model", "weight"), ("model", "scalar")],
                4096, work_fingerprint="sha256:" + "82" * 32,
            )
            writer.add_part("model", "weight", "value", io.BytesIO(payload))
            writer.add_part("model", "scalar", "value", io.BytesIO(scalar))
            return writer.commit()["manifest"]

        first_id, second_id = "sha256:" + "83" * 32, "sha256:" + "84" * 32
        first = fixture(first_id, b"\0\0\x80\x3f")
        second = fixture(second_id, b"\0\0\0\x40")
        header = store.manifest(first["sha256"])["header"]
        other_header = store.manifest(second["sha256"])["header"]
        lease = store.acquire_cozytensors(first["sha256"])
        target = bytearray(773)
        lease.read_part_into(header, "model", "weight", "value", 19, target)
        assert target == payload[19:792]
        scalar = bytearray(4)
        lease.read_part_into(header, "model", "scalar", "value", 0, scalar)
        assert scalar == b"\0\0\x80\x3f"
        refused(lambda: lease.read_part_into(other_header, "model", "scalar", "value", 0, scalar))
        refused(lambda: lease.read_part_into(header, "model", "weight", "scale", 0, scalar))
        refused(lambda: lease.read_part_into(header, "model", "weight", "value", 2047, scalar))
        refused(lambda: lease.read_part_into(header, "model", "scalar", "value", 0, bytes(4)))
        refused(lambda: lease.read_part_into(b"corrupt header", "model", "scalar", "value", 0, scalar))

        store.derived_dispose(first_id)
        refused(lambda: tensorfs.gc(root))  # the active native reader fences GC
        lease.read_part_into(header, "model", "weight", "value", 19, target)
        assert target == payload[19:792], "independent reader custody was lost with the producer"
        markers = [path for path in (Path(root) / "tmp/leases").glob("*")
                   if lease.hold_id in path.name]
        assert len(markers) == 1
        marker = markers[0]
        saved = marker.read_bytes()
        marker.unlink()
        refused(lambda: lease.read_part_into(header, "model", "scalar", "value", 0, scalar))
        marker.write_bytes(saved)
        lease.release()
        refused(lambda: lease.read_part_into(header, "model", "scalar", "value", 0, scalar))
        store.derived_dispose(second_id)
        tensorfs.gc(root)
    print("native model-part reads: ranges, scalar, exact header, bounds, GC custody and revocation pass")


if __name__ == "__main__":
    main()
