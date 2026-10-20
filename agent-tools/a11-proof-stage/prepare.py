"""Assemble unchanged A11 trees as thin folders; no network or tensor-chunk reads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def checked(path, at):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != at["length"]:
        raise RuntimeError("metadata file type/length mismatch")
    if at["length"] > 32 << 20 or hashlib.sha256(path.read_bytes()).hexdigest() != at["sha256"]:
        raise RuntimeError("metadata identity/cap mismatch")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage, output = args.stage.resolve(), args.output.resolve()
    receipt = json.loads((stage / "receipt.json").read_text())
    assert receipt["tensor_chunk_downloads"] == receipt["hub_mutations"] == 0
    assert receipt["a11_source"] == "e2c8ded23669b7b18185808c1c1c0934693def3b"
    output.mkdir()  # Fresh output only.
    repos = {}
    for row in receipt["staged"]:
        source = row["repo"]
        name = source.removeprefix("fidika/")
        assert source == "fidika/" + name and name and all(c.isalnum() or c in "-._" for c in name)
        repo = repos.setdefault(source, {"source": source, "source_release": row["release"],
            "destination": "fidika/varena-slim-20261020-" + name, "tree": row["tree"],
            "folder": str(output / "folders" / name), "variants": {}, "objects": {}})
        assert repo["tree"] == row["tree"] and repo["source_release"] == row["release"]
        repo["variants"][row["variant"]] = row["header"]
        folder = Path(repo["folder"])
        (folder / "chunks").mkdir(parents=True, exist_ok=True)
        # Stage-only output contains header assets and newly generated inline packs only.
        original_chunks = {x["sha256"] for x in row["chunks"]}
        generated = {x["sha256"] for x in row["generated_inline_packs"]}
        for asset in (Path(row["folder"]) / "chunks").iterdir():
            assert asset.name not in original_chunks or asset.name in generated
            at = {"sha256": asset.name, "length": asset.stat().st_size}
            shutil.copyfile(checked(asset, at), folder / "chunks" / asset.name)
            repo["objects"][asset.name] = at
    for repo in repos.values():
        tree_path = checked(stage / "objects" / repo["tree"]["sha256"], repo["tree"])
        tree = json.loads(tree_path.read_text())
        assert tree["format"] == "tensorfs.tree/1"
        folder = Path(repo["folder"])
        for name, at in tree["files"].items():
            parts = Path(name).parts
            assert parts and not name.startswith("/") and ".." not in parts and parts[0] != "chunks"
            paths = [stage / "objects" / at["sha256"], folder / "chunks" / at["sha256"]]
            for row in receipt["staged"]:
                if row["repo"] == repo["source"]:
                    paths += [Path(row["folder"]) / name, Path(row["folder"]) / "chunks" / at["sha256"]]
            source_path = next((p for p in paths if p.is_file()), None)
            assert source_path is not None
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked(source_path, at), target)
            repo["objects"][at["sha256"]] = at
        repo["files"] = tree["files"]
    plan = {"format": "cozy.private-d10-proof/1", "origin": "https://tensorhub.com",
        "a11_source": receipt["a11_source"], "repos": list(repos.values()), "tensor_chunk_downloads": 0}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(output / "plan.json"), "repos": len(repos),
        "variants": sum(len(r["variants"]) for r in repos.values()), "tensor_chunk_reads": 0}))


if __name__ == "__main__":
    main()
