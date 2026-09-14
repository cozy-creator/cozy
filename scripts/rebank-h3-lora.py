#!/usr/bin/env python3
"""Reproduce reviewed H3 style LoRA fingerprints from immutable real headers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ROOT / "vectors/h3-lora-headers"


def canonical_keys(header: dict) -> list[str]:
    keys = set()
    for key in header:
        if key == "__metadata__":
            continue
        key = key.removeprefix("diffusion_model.")
        if key.startswith("lora_unet_blocks_"):
            match = re.fullmatch(r"lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)(\..*)", key)
            assert match, key
            leaf = {"attn_qkv_proj": "attn.qkv_proj", "attn_out_proj": "attn.out_proj", "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}[match[2]]
            key = f"blocks.{match[1]}.{leaf}{match[3]}".replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
        if key.startswith("blocks."):
            key = "transformer_blocks." + key.removeprefix("blocks.")
        else:
            key = key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if ".attn.qkv_proj." in key:
            keys.update(key.replace(".attn.qkv_proj.", f".attn.to_{part}.") for part in "qkv")
        else:
            keys.add(key.replace(".attn.out_proj.", ".attn.to_out.0.").replace(".mlp.fc1.", ".ff.net.0.proj.").replace(".mlp.fc2.", ".ff.net.2."))
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    registry_path = ROOT / "vectors/fingerprints/registry.json"
    original = json.loads(registry_path.read_text())
    with tempfile.TemporaryDirectory(prefix="h3-lora-bank-") as temporary:
        area = Path(temporary)
        registry = area / "registry.json"
        registry.write_text(json.dumps(original))
        # Same topology in each version: one representative per source dialect.
        rows = json.loads((HEADERS / "MANIFEST.json").read_text())
        for row in [rows[0], rows[2]]:
            raw = (HEADERS / row["member"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["header_sha256"]
            header = json.loads(raw[8:])
            assert len(header) - 1 == row["logical_keys"]
            carrier = area / row["member"]
            with carrier.open("wb") as stream:
                stream.write(raw)
                stream.truncate(row["full_length"])  # Header proof only; holes are never read.
            order = area / "order.json"
            order.write_text(json.dumps([["adapter", key] for key in canonical_keys(header)]))
            subprocess.run([
                str(args.tfs.resolve()), "ingest", "bank", "safetensors.h3_lora", "h3.lora/1",
                "--source", f"adapter={carrier}", "--registry", str(registry),
                "--source-name", row["source"] + "/" + row["member"],
                "--source-profile", "h3/native-lora-a-b/1" if row["logical_keys"] == 416 else "h3/native-lora-kohya/1", "--target", "adapter=plain/1",
                "--order", str(order),
            ], check=True)
        updated = json.loads(registry.read_text())
        for profile in updated["source_profiles"]:
            if profile["name"].startswith("h3/native-lora-"):
                # Explicit opt-in: same reviewed structure under any carrier name.
                profile["components"][0]["source_member_prefix"] = ""
        if args.write:
            registry_path.write_text(json.dumps(updated, sort_keys=True, separators=(",", ":")))
        elif updated != original:
            raise SystemExit("H3 LoRA profiles differ; review before --write")
    print("Verified two source dialects for canonical H3 LoRA factors")


if __name__ == "__main__":
    main()
