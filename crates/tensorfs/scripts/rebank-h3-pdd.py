#!/usr/bin/env python3
"""Verify the two PDD adapter source profiles from their retained real headers."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ROOT / "vectors/h3-pdd-headers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    registry_path = ROOT / "vectors/fingerprints/registry.json"
    original = json.loads(registry_path.read_text())
    with tempfile.TemporaryDirectory(prefix="h3-pdd-bank-") as temporary:
        area = Path(temporary)
        registry = area / "registry.json"
        registry.write_text(json.dumps(original))
        for row in json.loads((HEADERS / "MANIFEST.json").read_text()):
            task = "fl2va" if "-FL2VA-" in row["member"] else "ref2va"
            component = task + "_adapter"
            raw = (HEADERS / row["member"]).read_bytes()
            header = json.loads(raw[8:])
            keys = sorted(key for key in header if key != "__metadata__")
            assert len(keys) == row["logical_keys"] == 728
            assert {header[key]["dtype"] for key in keys} == {"BF16"}
            assert header["__metadata__"]["lora_rank"] == "64"
            assert header["__metadata__"]["lora_alpha"] == "64.0"
            assert header["__metadata__"]["pdd_num_steps"] == "32"
            assert header["__metadata__"]["pdd_block_size"] == "4"
            carrier = area / row["member"]
            with carrier.open("wb") as stream:
                stream.write(raw)
                # Geometry for header-only banking; these holes are not tensor evidence.
                stream.truncate(row["full_length"])
            order = area / "order.json"
            order.write_text(json.dumps([[component, key] for key in keys]))
            subprocess.run([
                str(args.tfs.resolve()), "ingest", "bank",
                "safetensors.diffusers", "diffusers.identity/1",
                "--source", f"{component}={carrier}",
                "--registry", str(registry), "--source-name", row["source"],
                "--source-profile", f"hf/minimax-h3/pdd-{task}-bf16/1",
                "--target", f"{component}=plain/1", "--order", str(order),
                "--member", f"{component}={row['member']}",
            ], check=True)
        updated = json.loads(registry.read_text())
        if args.write:
            registry_path.write_text(json.dumps(updated, sort_keys=True, separators=(",", ":")))
        elif updated != original:
            raise SystemExit("PDD source profiles differ; review and use --write")
    print("Verified two 728-tensor PDD adapter source profiles")


if __name__ == "__main__":
    main()
