#!/usr/bin/env python3
"""Check or re-bank the five official H3 entries using TensorFS and real headers."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"
MEMBERS = {
    "fl2va_dit": "FL2VA/transformer/model.safetensors.index.json",
    "ref2va_dit": "Ref2VA/transformer/model.safetensors.index.json",
    "text_encoder": "text_encoder/model.safetensors.index.json",
    "audio_vae": "audio_vae/diffusion_pytorch_model.safetensors",
    "video_vae": "vae/diffusion_pytorch_model.safetensors.index.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    registry_path = ROOT / "crates/tensorfs-core/vectors/fingerprints/registry.json"
    registry = json.loads(registry_path.read_text())
    headers = ROOT / "vectors/h3-headers"
    changes = []
    with tempfile.TemporaryDirectory(prefix="h3-headers-") as temporary:
        area = Path(temporary)
        for row in json.loads((headers / "MANIFEST.json").read_text()):
            path = area / row["member"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                stream.write((headers / row["member"].replace("/", "__")).read_bytes())
                # Sparse geometry only: no payload is read or used as weight evidence.
                stream.truncate(row["full_length"])
        bank = area / "bank.json"
        for component, member in MEMBERS.items():
            matches = [row for row in registry["entries"]
                       if row["component"] == component and row["provenance"]["source"] == SOURCE]
            if len(matches) != 1:
                raise SystemExit(f"{component}: expected one reviewed official H3 entry")
            row = matches[0]
            subprocess.run([
                str(args.tfs.resolve()), "ingest", "bank", row["dialect"], row["converter"],
                "--source", f"{component}={area / member}", "--registry", str(bank),
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
    print(json.dumps({"checked": len(MEMBERS), "changes": changes}, indent=2))
    if changes and not args.write:
        raise SystemExit("H3 registry is stale; review the changes and use --write to re-bank")
    if changes:
        registry_path.write_text(json.dumps(registry, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
