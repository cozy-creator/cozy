#!/usr/bin/env python3
"""Check the reviewed single-file SDXL header against today's native schema projection."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sparse(path, header, full_length):
    with path.open("wb") as target:
        target.write(header)
        target.truncate(full_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfs", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    binary = str(args.tfs.resolve())
    corpus = ROOT / "vectors/civitai-sdxl-headers"
    observed = json.loads((corpus / "MANIFEST.json").read_text())
    raw = (corpus / observed["file"]).read_bytes()
    assert len(raw) == observed["header_bytes"] == 8 + struct.unpack("<Q", raw[:8])[0]
    assert hashlib.sha256(raw).hexdigest() == observed["header_sha256"]
    registry_path = ROOT / "crates/tensorfs-core/vectors/fingerprints/registry.json"
    registry = json.loads(registry_path.read_text())
    routes = json.loads((ROOT / "crates/tensorfs-core/src/ingest/routes/sdxl.routes.json").read_text())
    converter = routes["converter"]
    components = ("text_encoder", "text_encoder_2", "unet", "vae")
    changes = []
    with tempfile.TemporaryDirectory(prefix="civitai-sdxl-header-") as temporary:
        area = Path(temporary)
        carrier = area / "header-only.safetensors"
        sparse(carrier, raw, observed["full_bytes"])
        bank = area / "bank.json"
        command = [binary, "ingest", "bank", "safetensors.single_file", converter,
                   "--registry", str(bank), "--source-name", observed["source_name"]]
        for component in components:
            command.extend(("--source", f"{component}={carrier}"))
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        computed = json.loads(bank.read_text())["entries"]
        assert len(computed) == len(components)
        for current in computed:
            matches = [entry for entry in registry["entries"]
                       if entry["component"] == current["component"]
                       and entry["provenance"]["source"] == observed["source_name"]]
            assert len(matches) == 1
            reviewed = matches[0]
            for field in reviewed:
                if field != "tensor_schema_digest" and reviewed[field] != current[field]:
                    raise SystemExit(f"{current['component']}: reviewed {field} changed; refusing re-bank")
            if reviewed["tensor_schema_digest"] != current["tensor_schema_digest"]:
                changes.append({"component": current["component"],
                                "old": reviewed["tensor_schema_digest"],
                                "new": current["tensor_schema_digest"]})
                reviewed["tensor_schema_digest"] = current["tensor_schema_digest"]
        if changes and not args.write:
            raise SystemExit("Civitai SDXL registry is stale: " + json.dumps(changes))
        if changes:
            registry["entries"].sort(key=lambda row: (
                row["keyset_digest"], row["component"], row["tensor_schema_digest"]))
            registry_path.write_text(json.dumps(registry, sort_keys=True, separators=(",", ":")))

        def plan(path, member=observed["member"], compiled=False):
            registry = [] if compiled else ["--registry", str(registry_path)]
            return subprocess.run(
                [binary, "ingest", "source-plan", "--carrier", f"{member}={path}",
                 *registry, "--source-profile", observed["profile"]],
                capture_output=True, text=True,
            )

        accepted = plan(carrier)
        assert accepted.returncode == 0, accepted.stderr
        planned = json.loads(accepted.stdout)
        assert planned["profile"] == observed["profile"]
        assert planned["converter"] == converter
        assert planned["construction_order"] == [
            [c["component"], row[0]] for c in routes["components"] for row in c["routes"]]
        assert {row["component"] for row in planned["sources"]} == set(components)
        assert all(row["projected"] and row["source_member"] == observed["member"]
                   for row in planned["sources"])
        assert json.loads(plan(carrier).stdout) == planned
        native = plan(carrier, compiled=True)
        assert native.returncode == 0, native.stderr
        native_plan = json.loads(native.stdout)
        assert native_plan["construction_order"] == planned["construction_order"]
        assert native_plan["sources"] == planned["sources"]
        assert native_plan["target"] == planned["target"]

        header = json.loads(raw[8:])
        for arm in ("same-width-dtype", "missing-key", "provider-namespace"):
            changed = copy.deepcopy(header)
            member = observed["member"]
            if arm == "same-width-dtype":
                key = next(key for key in changed if key.startswith("model.diffusion_model."))
                assert changed[key]["dtype"] == "F16"
                changed[key]["dtype"] = "BF16"
            elif arm == "missing-key":
                # Remove the final byte run so the changed carrier remains gap-free and
                # reaches fingerprint refusal rather than a malformed-offset refusal.
                key = max((key for key in changed if key != "__metadata__"),
                          key=lambda key: changed[key]["data_offsets"][1])
                del changed[key]
            else:
                member = "unreviewed/files/92696"
            body = json.dumps(changed, separators=(",", ":")).encode()
            maximum = max(row["data_offsets"][1] for key, row in changed.items()
                          if key != "__metadata__")
            altered = area / (arm + ".safetensors")
            sparse(altered, struct.pack("<Q", len(body)) + body, 8 + len(body) + maximum)
            result = plan(altered, member, compiled=True)
            if arm == "same-width-dtype":
                # The reviewed key set classifies; conversion validates the actual dtypes.
                assert result.returncode == 0, (arm, result.stderr)
                continue
            # The reviewed profile is named, so a carrier it does not fit refuses; unnamed,
            # the same carrier would be stored as-is.
            assert result.returncode != 0 and "UNREGISTERED_FINGERPRINT" in result.stderr, (
                arm, result.stdout, result.stderr)
    print(json.dumps({"checked": 4, "changes": changes,
                      "profile": observed["profile"], "tensor_payload_bytes_read": 0,
                      "classifies": ["same-width-dtype"],
                      "strict_refusals": ["missing-key", "provider-namespace"]}))


if __name__ == "__main__":
    main()
