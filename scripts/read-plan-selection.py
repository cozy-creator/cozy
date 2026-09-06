#!/usr/bin/env python3
"""Select native part windows from a complete plan and stream exact bytes through its lease."""

import io
import tempfile
from pathlib import Path

import tensorfs


def refused(code, operation):
    try:
        operation()
    except tensorfs.errors.Refusal as error:
        assert error.code == code, error
    else:
        raise AssertionError(f"expected {code}")


def main():
    seeds = {alias: next(digest for name, digest in tensorfs.seed_digests() if name == alias)
             for alias in ("plain/1", "fp8-rowwise/1")}
    data, unused, bias, scale = bytes([0x31]) * 2048, bytes([0x72]) * 2048, b"\x00\x00\x80?", bytes(256)
    with tempfile.TemporaryDirectory(prefix="tensorfs-plan-select-") as directory:
        store = tensorfs.Store.init(Path(directory) / "store")
        writer = store.begin_derived("sha256:" + "85" * 32, 1, {}, {"model": {"drop": [], "add": {
            "weight": {"logical_dtype": "bf16", "shape": [64, 32], "encoding": seeds["fp8-rowwise/1"],
                       "parts": {"data": {"dtype": "f8_e4m3fn", "shape": [64, 32]},
                                 "scale": {"dtype": "f32", "shape": [64]}}},
            "unused": {"logical_dtype": "f32", "shape": [512], "encoding": seeds["plain/1"],
                       "parts": {"value": {"dtype": "f32", "shape": [512]}}},
            "bias": {"logical_dtype": "f32", "shape": [1], "encoding": seeds["plain/1"],
                     "parts": {"value": {"dtype": "f32", "shape": [1]}}},
        }}}, {}, [("model", "weight"), ("model", "unused"), ("model", "bias")], 8192,
            work_fingerprint="sha256:" + "86" * 32)
        for key, role, raw in (("weight", "data", data), ("weight", "scale", scale),
                               ("unused", "value", unused), ("bias", "value", bias)):
            writer.add_part("model", key, role, io.BytesIO(raw))
        receipt = writer.commit()
        manifest = "sha256:" + receipt["manifest"]["sha256"]
        header = store.manifest(manifest)["header"]
        order = [("model", "weight"), ("model", "unused"), ("model", "bias")]
        refused("TRAVERSAL_ORDER_MISMATCH", lambda: tensorfs.plan(
            header, [("model", "weight"), ("model", "bias")], components=["model"], window=128))
        full = tensorfs.plan(header, order, components=["model"], window=128)
        original = full.items()
        targets = ["model/bias#value", "model/weight#data"]
        selected = full.select(targets)
        expected = [row for row in original if row["what"] in targets]
        actual = selected.items()
        assert len(actual) == 17
        offset = 0
        for before, after in zip(expected, actual, strict=True):
            assert after == {**before, "dest_off": offset}
            offset += after["len"]
        assert full.items() == original
        assert (selected.bytes, selected.io_bytes, selected.inline_parts) == (2052, 2048, 1)
        assert selected.order == full.order
        refused("DUPLICATE_KEY", lambda: full.select([targets[0], targets[0]]))
        refused("MISSING_FIELD", lambda: full.select(["model/no-such-part#value"]))
        empty = full.select([])
        assert len(empty) == empty.bytes == empty.io_bytes == empty.inline_parts == empty.batches(512) == 0
        slots = [memoryview(bytearray(512)) for _ in range(2)]
        batches = {}

        def receive(batch):
            batches[batch.index] = bytes(slots[batch.slot][:batch.nbytes])
            batch.release()

        with store.acquire_cozytensors(manifest) as lease:
            result = lease.stream(selected, slots, receive, readers=2)
            assert b"".join(batches[index] for index in sorted(batches)) == data + bias
            assert (result["bytes"], result["io_bytes"], result["items"]) == (2052, 2048, 17)
            assert lease.stream(empty, slots, receive, readers=2)["bytes"] == 0
        print("read plan selection green: complete census retained; 16 data windows + inline bias; "
              "native sources/order unchanged, compact destinations, exact leased stream, empty/duplicate/unknown arms")


if __name__ == "__main__":
    main()
