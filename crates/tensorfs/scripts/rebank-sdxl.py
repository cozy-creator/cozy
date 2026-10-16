#!/usr/bin/env python3
"""Verify the four original SDXL source observations against the native schema hash.

The corpus contains bounded real header reads, never model payloads. Every banked
field other than tensor_schema_digest must remain identical, including the original
header/geometry source identity. This does not widen reviewed source acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "stabilityai/stable-diffusion-xl-base-1.0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    registry_path = ROOT / "vectors/fingerprints/registry.json"
    registry = json.loads(registry_path.read_text())
    headers = ROOT / "vectors/sdxl-headers"
    changes = []
    with tempfile.TemporaryDirectory(prefix="sdxl-headers-") as temporary:
        area = Path(temporary)
        bank = area / "bank.json"
        for carrier in json.loads((headers / "MANIFEST.json").read_text()):
            component = carrier["component"]
            raw = (headers / carrier["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != carrier["header_sha256"]:
                raise SystemExit(f"{component}: header corpus identity changed")
            path = area / carrier["member"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                stream.write(raw)
                stream.truncate(carrier["full_length"])
            matches = [entry for entry in registry["entries"]
                       if entry["component"] == component and entry["provenance"]["source"] == SOURCE]
            if len(matches) != 1:
                raise SystemExit(f"{component}: expected one original reviewed observation")
            row = matches[0]
            subprocess.run([
                str(args.tfs.resolve()), "ingest", "bank", row["dialect"], row["converter"],
                "--source", f"{component}={path}", "--registry", str(bank),
                "--source-name", SOURCE,
            ], check=True, stdout=subprocess.DEVNULL)
            computed = next(entry for entry in json.loads(bank.read_text())["entries"]
                            if entry["component"] == component)
            for field in row:
                if field != "tensor_schema_digest" and row[field] != computed[field]:
                    raise SystemExit(f"{component}: reviewed {field} changed; refusing re-bank")
            if row["tensor_schema_digest"] != computed["tensor_schema_digest"]:
                changes.append({"component": component, "old": row["tensor_schema_digest"],
                                "new": computed["tensor_schema_digest"]})
                row["tensor_schema_digest"] = computed["tensor_schema_digest"]
    print(json.dumps({"checked": 4, "changes": changes}, indent=2))
    if changes and not args.write:
        raise SystemExit("SDXL registry is stale; review changes and use --write to re-bank")
    if changes:
        registry_path.write_text(json.dumps(registry, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
