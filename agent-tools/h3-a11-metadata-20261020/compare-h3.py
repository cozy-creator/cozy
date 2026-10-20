#!/usr/bin/env python3
"""Independent offline check of the one selected H3 .12 and staged D10 metadata."""
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import cbor2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return cbor2.dumps(value, canonical=True)


def raw_header(data, arity):
    value = cbor2.loads(data)
    assert isinstance(value, list) and len(value) == arity
    assert value[0] == "cozytensors/1"
    assert canonical(value) == data, "header is not deterministic CBOR"
    return value


def main():
    bundle_path, destination, reference_path = map(Path, sys.argv[1:4])
    bundle = json.loads(bundle_path.read_text())
    assert len(bundle["repos"]) == len(bundle["source_read_lists"]) == 1
    assert bundle["repos"][0]["name"] == "paul/minimax-h3"
    source = bundle_path.parent / "objects"
    reference_bytes = reference_path.read_bytes()
    bits = next(
        ast.literal_eval(node.value)
        for node in ast.parse(reference_bytes).body
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == "BITS" for name in node.targets)
    )

    def get(at, original=False):
        digest = at["sha256"]
        path = source / digest if original else destination / "blobs" / digest[:2] / digest[2:4] / digest
        data = path.read_bytes()
        assert len(data) == at["length"] and sha(data) == digest
        return data

    reports = []
    union = set()
    for selected in bundle["source_read_lists"]:
        name = selected["name"]
        repo = json.loads((destination / "refs" / (name + ".json")).read_text())
        release = repo["releases"][0]
        assert len(repo["releases"]) == 1 and not repo.get("checkpoints", [])
        tree = json.loads(get(release["tree"]))
        new_at = tree["files"]["model.int8.cozytensors"]
        old_at = selected["files"]["model.cozytensors"]
        old = raw_header(get(old_at, original=True), 5)
        new = raw_header(get(new_at), 6)
        assert len(old[1]) == 1676
        assert sum(len(t) for _, t in old[3]) == 3858
        old_chunks = [(digest.hex(), length) for digest, length in old[1]]
        new_chunks = [(digest.hex(), length) for digest, length in new[1]]
        assert len(set(old_chunks)) == len(old_chunks)
        assert set(old_chunks) == set(new_chunks), "the complete SHA/length chunk sets differ"
        assert old_chunks == new_chunks, "chunk-table order changed, invalidating location indices"
        assert canonical(old[1]) == canonical(new[1]), "chunk-table bytes changed"
        assert canonical(old[2]) == canonical(new[2]), "encoding table changed"
        old_components = {component[0]: component[1] for component in old[3]}
        new_components = {component[0]: component[3] for component in new[4]}
        assert len(old_components) == len(old[3]) and len(new_components) == len(new[4])
        assert [c for c in new_components if c in old_components] == list(old_components)
        assert {name for name, tensors in old_components.items() if tensors} == {
            name for name, tensors in new_components.items() if tensors
        }
        components = {}
        for component, tensors in old_components.items():
            assert canonical(tensors) == canonical(new_components[component]), component
            assert len({tensor[0] for tensor in tensors}) == len(tensors)
            logical = stored = parts = 0
            dtypes = {}
            for key, dtype, shape, encoding, body in tensors:
                assert dtype in bits
                logical += math.prod(shape) * bits[dtype] // 8
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                if encoding == 0:
                    stored += body[2]
                    parts += 1
                else:
                    stored += sum(part[3][2] for part in body)
                    parts += len(body)
            components[component] = {
                "tensors": len(tensors), "encoded_tensors": sum(t[3] != 0 for t in tensors),
                "stored_parts": parts, "dtypes": dtypes, "logical_bytes": logical,
                "stored_tensor_bytes": stored, "whole_tensor_records_equal": True,
                "canonical_cbor_tensor_records_sha256": sha(canonical(tensors)),
            }
        configs = {}
        if new[3]:
            configs["model_index.json"] = new[3][1]
        assets = {}
        for component in new[4]:
            cname, _, config, _, asset_rows = component[:5]
            config_file = (component[5] if len(component) > 5 else None) or "config.json"
            prefix = "" if not new[3] and cname == "model" else cname + "/"
            configs[prefix + config_file] = config
            for path, (digest, length) in asset_rows.items():
                assets[prefix + path] = {"sha256": digest.hex(), "length": length}
        file_checks = []
        for path, at in selected["files"].items():
            if path == "model.cozytensors":
                continue
            original = get(at, original=True)
            if path in configs:
                assert json.loads(original) == configs[path], path
                kind = "parsed-config-equal"
            else:
                now = assets.get(path, tree["files"].get(path))
                assert now == at and get(now) == original, path
                kind = "asset-or-repo-file-byte-identical"
            file_checks.append({"path": path, "object": at, "proof": kind})
        pinned = "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"
        prefix = "/MiniMaxAI/MiniMax-H3/resolve/42ed227ee7df40d41602854ae760620d6eb651fe/"
        docs = {}
        for replay_path, at in bundle["replay"].items():
            if not replay_path.startswith(prefix):
                continue
            path = replay_path.removeprefix(prefix)
            assert path in ("LICENSE", "README.md")
            assert tree["files"].get(path) == at
            assert get(at) == get(at, original=True)
            docs[path] = at
        assert set(docs) == {"README.md", "LICENSE"}
        assert new[5]["source"] == pinned
        assert new[5]["filled"]["from"] == pinned
        assert sorted(new[5]["filled"]["files"]) == sorted(docs)
        classes = {component[0]: component[1] for component in new[4]}
        assert classes == {
            "audio_vae": "diffusers.AutoencoderKLMiniMaxH3Audio",
            "text_encoder": "transformers.Qwen3VLForConditionalGeneration",
            "transformer": "diffusers.MiniMaxH3Transformer3DModel",
            "transformer_ref": "diffusers.MiniMaxH3Transformer3DModel",
            "vae": "diffusers.AutoencoderKLMiniMaxH3",
        }
        for digest, length in old_chunks:
            assert not (destination / "blobs" / digest[:2] / digest[2:4] / digest).exists()
        union.update(old_chunks)
        reports.append({
            "repo": name, "release": release["version"], "staged_revision": release["revision"],
            "original_manifest": selected["manifest"], "original_header": old_at,
            "staged_tree": release["tree"], "staged_header": new_at,
            "chunk_count": len(old_chunks), "chunk_bytes": sum(length for _, length in old_chunks),
            "all_chunk_sha_length_pairs_equal": True, "chunk_table_order_equal": True,
            "chunk_set_sha256": sha(json.dumps(sorted(old_chunks), separators=(",", ":")).encode()),
            "canonical_cbor_chunk_table_sha256": sha(canonical(old[1])),
            "encoding_table_equal": True, "encoding_count": len(old[2]),
            "canonical_cbor_encoding_table_sha256": sha(canonical(old[2])),
            "tensor_count": sum(len(t) for t in old_components.values()),
            "components": components, "original_run_files": file_checks,
            "local_chunk_files": 0, "pinned_docs": docs, "component_classes": classes,
            "source": pinned, "original_metadata_sha256": sha(canonical(old[4])),
        })
    assert sum(
        component["stored_parts"]
        for report in reports
        for component in report["components"].values()
    ) == 4834, "the selected H3 lane has 2882 plain plus 1952 encoded stored parts"
    receipt = {
        "format": "cozy.r4.selected-metadata-provenance/1",
        "selection": "one exact recorded H3 int8 lane in own staging destination; no live update",
        "method": "independent cbor2 decoder/canonical encoder, same package used by existing reference reader; no TensorFS native or Torch import",
        "cbor2_version": importlib.metadata.version("cbor2"),
        "reference_reader": str(reference_path), "reference_reader_sha256": sha(reference_bytes),
        "source_bundle_sha256": sha(bundle_path.read_bytes()),
        "comparison_script_sha256": sha(Path(__file__).read_bytes()),
        "datasets": reports, "unique_retained_chunks": len(union),
        "unique_retained_chunk_bytes": sum(length for _, length in union),
        "retained_chunk_set_sha256": sha(json.dumps(sorted(union), separators=(",", ":")).encode()),
        "payload_downloads": 0, "source_mutations": 0,
    }
    output = bundle_path.parent / "independent-comparison.json"
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
