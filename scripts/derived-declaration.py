#!/usr/bin/env python3
"""Native intent declarations and committed receipts agree through the installed wheel."""

import io
import json
import tempfile
from pathlib import Path

import tensorfs


def main():
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    fp8 = next(digest for alias, digest in tensorfs.seed_digests() if alias == "fp8-rowwise/1")
    targets = {"model": {"drop": [], "add": {
        "table": {"logical_dtype": "f32", "shape": [512], "encoding": plain,
                  "parts": {"value": {"dtype": "f32", "shape": [512]}}},
        "weight": {"logical_dtype": "bf16", "shape": [16, 32], "encoding": fp8,
                   "parts": {"data": {"dtype": "f8_e4m3fn", "shape": [16, 32]},
                             "scale": {"dtype": "f32", "shape": [16]}}},
    }}}
    configs = {"model.json": {"kind": "add"}}
    order = [("model", "table"), ("model", "weight")]
    transaction = "sha256:" + "82" * 32
    with tempfile.TemporaryDirectory(prefix="tensorfs-native-declaration-") as root:
        store = tensorfs.Store.init(root)
        declaration = store.derived_declaration({}, targets, configs, order, 4096)
        assert isinstance(declaration, bytes)
        assert store.derived_lookup(transaction)["state"] == "absent"
        expected = json.loads(declaration)
        assert expected["components"]["model"]["add"]["weight"]["logical"]["dtype"] == "bf16"
        assert "work_fingerprint" not in expected
        def filled_writer(transaction):
            writer = store.begin_derived(transaction, 1, {}, targets, configs, order, 4096)
            writer.add_part("model", "table", "value", io.BytesIO(bytes(2048)))
            writer.add_part("model", "weight", "data", io.BytesIO(bytes([0x31]) * 512))
            writer.add_part("model", "weight", "scale", io.BytesIO(bytes(64)))
            writer.add_config("model.json", io.BytesIO(b'{"a":1}'))
            return writer

        writer = filled_writer(transaction)
        receipt = writer.commit()
        assert receipt["declaration"] == expected
        assert store.derived_lookup(transaction)["receipt"] == receipt
        store.derived_dispose(transaction)
        print("native declaration green: source-free table + FP8 data/scale + config; exact receipt equality; existing writer ABI")

        for arm, suffix in (("revoked", "83"), ("fenced", "84")):
            transaction = "sha256:" + suffix * 32
            writer = filled_writer(transaction)
            marker = None
            if arm == "revoked":
                markers = list((Path(root) / "tmp/leases").glob("derived-writer-*.lease"))
                assert len(markers) == 1
                marker = markers[0]
                saved = marker.read_bytes()
                marker.unlink()  # Revoke only this isolated test writer's hold.
            else:
                assert writer.fence()
            try:
                writer.commit()
            except tensorfs.errors.LeaseRevoked:
                pass
            else:
                raise AssertionError(f"{arm} writer committed without its live hold")
            if marker is not None:
                marker.write_bytes(saved)
                assert writer.fence()
            store.derived_abandon(transaction)
            assert store.derived_lookup(transaction)["state"] == "abandoned"
        print("writer hold green: revoked and released guards refuse commit; no receipt minted")


if __name__ == "__main__":
    main()
