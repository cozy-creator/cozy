#!/usr/bin/env python3
"""Check the reviewed instrument profiles against real pinned carrier headers only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    vectors = ROOT / "vectors/instrument-headers"
    fixtures = json.loads((vectors / "MANIFEST.json").read_bytes())
    registry_path = ROOT / "vectors/fingerprints/registry.json"
    registry = json.loads(registry_path.read_bytes())
    changes = []
    header_bytes = 0
    with tempfile.TemporaryDirectory(prefix="instrument-headers-") as temporary:
        area = Path(temporary)
        for fixture in fixtures:
            directory = area / fixture["name"]
            for member in fixture["members"]:
                header = (vectors / member["header_file"]).read_bytes()
                assert len(header) == member["header_bytes"]
                assert hashlib.sha256(header).hexdigest() == member["header_sha256"]
                header_bytes += len(header)
                path = directory / member["member"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    stream.write(header)
                    # Only sparse geometry. This driver never reads a payload or converts it.
                    stream.truncate(member["full_length"])
            matches = [row for row in registry["entries"]
                       if row["component"] == fixture["component"]
                       and row["provenance"]["source"] == fixture["source"]]
            assert len(matches) == 1, fixture["name"]
            row = matches[0]
            bank = area / (fixture["name"] + ".json")
            subprocess.run([
                str(args.tfs.resolve()), "ingest", "bank", row["dialect"], row["converter"],
                "--source", f"{fixture['component']}={directory / fixture['carrier']}",
                "--registry", str(bank), "--source-name", fixture["source"],
            ], check=True, stdout=subprocess.DEVNULL)
            computed = json.loads(bank.read_bytes())["entries"][0]
            for field in row:
                if field != "tensor_schema_digest" and row[field] != computed[field]:
                    raise SystemExit(f"{fixture['name']}: reviewed {field} changed")
            if row["tensor_schema_digest"] != computed["tensor_schema_digest"]:
                changes.append({"source": fixture["name"], "old": row["tensor_schema_digest"],
                                "new": computed["tensor_schema_digest"]})
                row["tensor_schema_digest"] = computed["tensor_schema_digest"]
            profile = next(p for p in registry["source_profiles"] if p["name"] == fixture["profile"])
            assert len(profile["components"]) == 1
            component = profile["components"][0]
            assert component["component"] == fixture["component"]
            assert component["source_member"] == fixture["carrier"]
            assert component["target_encoding"] == "plain/1"
            assert len(component["variants"]) == 1
            variant = component["variants"][0]
            assert variant["keyset_digest"] == computed["keyset_digest"]
            assert len(variant["construction_order"]) == fixture["logical_keys"]
            order = json.dumps(variant["construction_order"], separators=(",", ":")).encode()
            assert hashlib.sha256(order).hexdigest() == fixture["construction_order_sha256"]
    print(json.dumps({"checked": len(fixtures), "header_bytes": header_bytes,
                      "tensor_payload_bytes_read": 0, "changes": changes}, indent=2))
    if changes and not args.write:
        raise SystemExit("instrument registry is stale; review changes before --write")
    if changes:
        registry_path.write_text(json.dumps(registry, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
