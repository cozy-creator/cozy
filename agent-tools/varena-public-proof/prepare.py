"""Copy verified public-source fixtures, adding only VARENA_PROOF.md for unique public trees."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def ref(body):
    return {"sha256": hashlib.sha256(body).hexdigest(), "length": len(body)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_stage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads((args.private_stage / "plan.json").read_text())
    models = json.loads((args.private_stage / "models.json").read_text())
    assert source["format"] == "cozy.private-d10-proof/1"
    assert json.loads((args.private_stage / "run-001/receipt.json").read_text())["complete"] is True
    output = args.output.resolve()
    output.mkdir()
    (output / "trees").mkdir()
    rows = []
    for old in source["repos"]:
        name = old["source"].removeprefix("fidika/")
        assert old["source"] == "fidika/" + name and name in models
        row = dict(old)
        folder = output / "folders" / name
        shutil.copytree(old["folder"], folder, symlinks=False)
        proof = ("# Varena public test metadata\n\n"
                 f"Source: {old['source']}@{old['source_release']}.\n\n"
                 "This tree is test metadata for the Varena slim model proof. It retains the source\n"
                 "README, license, model headers, assets and tensor bytes. VARENA_PROOF.md gives\n"
                 "this public fixture its own tree identity. Source attribution and licenses apply.\n").encode()
        assert not (folder / "VARENA_PROOF.md").exists()
        (folder / "VARENA_PROOF.md").write_bytes(proof)
        row["files"] = dict(old["files"], **{"VARENA_PROOF.md": ref(proof)})
        tree = canonical({"format": "tensorfs.tree/1", "files": row["files"]})
        row["private_repo"], row["private_tree"] = old["destination"], old["tree"]
        row["destination"] = "fidika/varena-slim-public-20261020-" + name
        row["folder"], row["tree"] = str(folder), ref(tree)
        assert row["tree"] != old["tree"]
        tree_path = output / "trees" / row["tree"]["sha256"]
        tree_path.write_bytes(tree)
        row["tree_path"], row["proof"] = str(tree_path), ref(proof)
        row["probe_variant"] = models[name]["variant"]
        row["objects"] = dict(old["objects"])
        row["objects"][row["proof"]["sha256"]] = row["proof"]
        for path, at in old["files"].items():
            assert ref((folder / path).read_bytes()) == at
        rows.append(row)
    plan = {"format": "cozy.public-d10-proof/1", "origin": "https://tensorhub.com", "release": "0.1.0", "repos": rows}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(output / "plan.json"), "repos": len(rows), "variants": sum(len(r["variants"]) for r in rows), "tensor_reads": 0}))


if __name__ == "__main__":
    main()
