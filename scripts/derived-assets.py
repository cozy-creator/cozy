#!/usr/bin/env python3
"""Header-only tokenizer assets survive derivation through the installed full wheel.

Optionally pass a directory containing merges.txt and vocab.json for exact real-data
readback. The default fixture is self-contained and exceeds the inline file limit.
"""

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import tensorfs


def main():
    actual = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    data = {name: (actual / name).read_bytes() if actual else byte * 70_001
            for name, byte in (("merges.txt", b"m"), ("vocab.json", b"v"))}
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    targets = {"model": {"drop": [], "add": {
        "weight": {"logical_dtype": "f32", "shape": [512], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": [512]}}},
    }}}
    original, derived, work = ("sha256:" + value * 32 for value in ("c1", "c2", "c3"))
    with tempfile.TemporaryDirectory(prefix="tensorfs-header-assets-") as temporary:
        root = Path(temporary)
        store = tensorfs.Store.init(root / "store")
        files = {}
        for name, body in data.items():
            path = root / name
            path.write_bytes(body)
            put = store.put_file(path)
            files[name] = (put["id"], put["length"])
        writer = store.begin_derived(original, 1, {}, targets, {}, [("model", "weight")],
                                     2048, files, work_fingerprint=work)
        writer.add_part("model", "weight", "value", io.BytesIO(b"w" * 2048))
        receipt = writer.commit()
        # Use the real asset-update producer: its output manifest has only the header.
        source = store.update_model_assets(
            receipt["manifest"]["sha256"], receipt["manifest"]["length"],
            copies=[(prefix + "/" + name, name, "text/plain" if name.endswith(".txt") else "application/json")
                    for prefix in ("tokenizer", "tokenizer_2") for name in data],
            drops=list(data),
        )
        current = store.replace_local(None, "asset-source", work, source["manifest_id"], source["manifest_length"])
        store.derived_dispose(original)
        source_row = store.manifest(source["manifest_id"])
        assert [e["path"] for e in tensorfs.parse_manifest(source_row["manifest"])["entries"]] == ["model.cozytensors"]
        source_header = tensorfs.parse_header(source_row["header"])
        inherited = {"model": {"source": "source", "source_component": "model", "drop": [], "add": {}}}
        marker = b"derived marker\n"
        writer = store.begin_derived(derived, 1,
            {"source": (source["manifest_id"], source["manifest_length"])}, inherited, {},
            [("model", "weight")], len(marker), {"marker.txt": base64.b64encode(marker).decode()},
            work_fingerprint=work)
        result = writer.commit()
        assert result["added_objects"] == [{"sha256": hashlib.sha256(marker).hexdigest(), "length": len(marker)}]
        assert result["inherit_observation"]["reads"] == result["inherit_observation"]["hashes"] == 0
        row = store.manifest(result["manifest"]["sha256"])
        header = tensorfs.parse_header(row["header"])
        assert {name: asset for name, asset in header["assets"].items() if name != "marker.txt"} == source_header["assets"]
        assert [e["path"] for e in tensorfs.parse_manifest(row["manifest"])["entries"]] == ["marker.txt", "model.cozytensors"]
        store.derived_adopt(derived, "retained-assets")
        store.remove_local(current, "asset-source")
        tensorfs.gc(store.root)
        with store.acquire_cozytensors(result["manifest"]["sha256"]) as lease:
            for name in source_header["assets"]:
                assert lease.read_asset(row["header"], name) == data[name.split("/")[-1]]
        assert store.derived_lookup(derived)["receipt"] == result
        store.derived_dispose(derived)
        tensorfs.gc(store.root)
        assert all(not store.contains(object_id) for object_id, _ in files.values())
        print(json.dumps({"passed": True, "real_assets": str(actual) if actual else None,
                          "assets": source_header["assets"], "inherit_observation": result["inherit_observation"],
                          "added_objects": result["added_objects"]}, sort_keys=True))


if __name__ == "__main__":
    main()
