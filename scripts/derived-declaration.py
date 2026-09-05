#!/usr/bin/env python3
"""Native intent declarations and committed receipts agree through the installed wheel."""

import io
import json
import tempfile

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
        writer = store.begin_derived(transaction, 1, {}, targets, configs, order, 4096)
        writer.add_part("model", "table", "value", io.BytesIO(bytes(2048)))
        writer.add_part("model", "weight", "data", io.BytesIO(bytes([0x31]) * 512))
        writer.add_part("model", "weight", "scale", io.BytesIO(bytes(64)))
        writer.add_config("model.json", io.BytesIO(b'{"a":1}'))
        receipt = writer.commit()
        assert receipt["declaration"] == expected
        assert store.derived_lookup(transaction)["receipt"] == receipt
        store.derived_dispose(transaction)
        print("native declaration green: source-free table + FP8 data/scale + config; exact receipt equality; existing writer ABI")


if __name__ == "__main__":
    main()
