#!/usr/bin/env python3
"""Render only from caller-reviewed candidate pins and exact local wheel files; no installs."""
import argparse
import ast
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

from contract import HARNESS_SOURCE, HARNESS_SHA, check_spec, require, template

HERE = Path(__file__).resolve().parent
HELPER_SOURCE = "1a221c31d6f38ba4628caf340d48522e1a2bf102"
HELPER_PATH = "crates/tensorfs-py/gates/consumers/owned_child.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wheel(path, name, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected["sha256"], name + " wheel hash differs")
    require(re.fullmatch("[0-9a-f]{40}", expected["source"]), name + " source is unresolved")
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "duplicate wheel members")
        require(all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist()), "unsafe wheel member")
        rows = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
        require(len(rows) == 1, "wheel metadata count")
        info = Parser().parsestr(z.read(rows[0]).decode())
        require(info["Name"].lower().replace("_", "-") == name, "wheel name differs")
        require(bool(info["Version"]) and bool(info["Requires-Python"]), "wheel metadata incomplete")
        tag = Parser().parsestr(z.read(rows[0].removesuffix("METADATA") + "WHEEL").decode())
        require(tag["Root-Is-Purelib"] == "false", "native wheel required")
        prefix = "cozy_runtime/" if name == "cozy-runtime" else "tensorfs/"
        files = {n: digest(z.read(n)) for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")}
        if name == "cozy-runtime":
            nodes = ast.parse(z.read("cozy_runtime/_build_provenance.py")).body
            commits = [ast.literal_eval(n.value) for n in nodes if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "COMMIT" for t in n.targets)]
            require(commits == [expected["source"]], "Runtime COMMIT differs")
            modules = {n for n in files if n.startswith("cozy_runtime/_kernels/") and n.endswith((".so", ".pyd"))}
            require(modules == {f"cozy_runtime/_kernels/{n}.abi3.so" for n in ("_gemm", "_norm", "_fa3")}, "Runtime kernels differ")
            require(tag.get_all("Tag") == ["cp312-abi3-manylinux_2_28_x86_64"], "Runtime platform differs")
        else:
            modules = {n for n in files if n.startswith("tensorfs/_ext") and n.endswith((".so", ".pyd"))}
            require(len(modules) == 1, "TensorFS extension missing")
        for module in modules:
            image = z.read(module)
            require(image[:6] == b"\x7fELF\x02\x01" and image[18:20] == b"\x3e\x00", "wrong native architecture")
    return {**expected, "file": path.name, "bytes": len(raw), "version": info["Version"],
            "requires_python": info["Requires-Python"], "files": files}


def requirements(path):
    raw = path.read_bytes()
    versions, pins = {}, []
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9!+_.-]+)", line)
        require(m is not None, "requirements must be exact name==version pins")
        name = m[1].lower().replace("_", "-")
        require(name not in versions and name not in ("tensorfs", "cozy-runtime"), "duplicate/artifact-owned dependency")
        versions[name] = m[2]
        pins.append(line)
    return {"versions": versions, "requirements": pins, "sha256": digest(raw)}, raw.decode()


def generate(args):
    expected = json.loads(args.candidate.read_text())
    # A supplied receipt is an external review input, never synthesized hardware evidence.
    require(expected.get("pair_receipt_sha256") == digest(args.pair_receipt.read_bytes()), "candidate review receipt differs")
    for name in ("runtime", "native"):
        require(isinstance(expected.get(name), dict) and re.fullmatch("[0-9a-f]{64}", str(expected[name].get("sha256"))),
                "unresolved " + name + " candidate")
    spec = template(json.loads((HERE / "metadata-files.json").read_text()))
    spec["pair_receipt_sha256"] = expected["pair_receipt_sha256"]
    spec["runtime"] = wheel(args.runtime_wheel, "cozy-runtime", expected["runtime"])
    spec["native"] = wheel(args.native_wheel, "tensorfs", expected["native"])
    spec["dependencies"], pins_text = requirements(args.requirements)
    require(spec["runtime"]["requires_python"] == spec["native"]["requires_python"], "Python requirements need review")
    def source(commit, path):
        return subprocess.check_output(["git", "-C", str(args.creator_repo), "show", f"{commit}:{path}"], timeout=15).decode()
    files = {name: (HERE / name).read_text() for name in ("contract.py", "probe.py")}
    files["requirements.txt"] = pins_text
    files["h3_load.py"] = source(HARNESS_SOURCE, "crates/tensorfs-py/gates/speed/h3_load.py")
    require(digest(files["h3_load.py"].encode()) == HARNESS_SHA, "frozen harness differs")
    # Derive/check the inventory against literal constants in the pinned implementation.
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(files["h3_load.py"]).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ("REPS", "READERS")}
    require(assignments == {"REPS": 5, "READERS": __import__("contract").READERS}, "harness schedule constants drift")
    files["owned_child.py"] = source(HELPER_SOURCE, HELPER_PATH)
    spec["sources"] = {name: digest(value.encode()) for name, value in files.items()}
    spec["proof_source"] = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(HERE), "diff", "--exit-code", "HEAD", "--", "."], check=True)
    prefix = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-prefix"], text=True).strip()
    for name in ("worker.py", "contract.py", "probe.py", "render.py", "metadata-files.json"):
        committed = subprocess.check_output(["git", "-C", str(HERE), "show", f"HEAD:{prefix}{name}"])
        require(committed == (HERE / name).read_bytes(), "render only committed reviewed source: " + name)
    check_spec(spec)
    paths = {"cozy-runtime": os.path.relpath(args.runtime_wheel.resolve(), args.output.parent.resolve()),
             "tensorfs": os.path.relpath(args.native_wheel.resolve(), args.output.parent.resolve())}
    header = "# /// script\n# requires-python = " + json.dumps(spec["runtime"]["requires_python"]) + "\n"
    header += "# dependencies = " + json.dumps([f"cozy-runtime=={spec['runtime']['version']}",
        f"tensorfs=={spec['native']['version']}", *spec["dependencies"]["requirements"]]) + "\n# [tool.uv.sources]\n"
    header += "".join(f"# {name} = {{ path = {json.dumps(path)} }}\n" for name, path in paths.items())
    header += '# torch = { index = "pytorch-cu130" }\n# [[tool.uv.index]]\n# name = "pytorch-cu130"\n'
    header += '# url = "https://download.pytorch.org/whl/cu130"\n# explicit = true\n# ///\n'
    body = (HERE / "worker.py").read_text().replace("NATIVE_ASSET_BYTES", str(spec["native"]["bytes"]))
    payload = header + "\nSPEC = " + repr(spec) + "\nFILES = " + repr(files) + "\n\n" + body
    ast.parse(payload)
    require(len(payload.encode()) <= 32 << 20, "public script size limit")
    manifest = args.output.with_suffix(".manifest.json")
    require(not args.output.exists() and not manifest.exists(), "choose fresh payload paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    manifest.write_text(json.dumps({"spec": spec, "payload_sha256": digest(payload.encode())}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "pair-receipt", "runtime-wheel", "native-wheel", "requirements", "creator-repo", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    generate(p.parse_args())
