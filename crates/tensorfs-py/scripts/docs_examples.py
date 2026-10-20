#!/usr/bin/env python3
"""Run every Python, shell and Rust example in the given Markdown documents against small real
models, and exit 1 when any fails. CI runs it on the built wheel (publish-tensorfs.yaml):

    docs_examples.py [--wheels DIR] [--work DIR] DOC.md...

The fixture: `me/model`, release `1.0`, is hf-internal-testing/tiny-sdxl-pipe converted by TensorFS,
in a `bf16` variant and an `int8` one whose UNet matrices are `int8.rowwise`, with a README and
LICENSE. It is uploaded to a `file://` endpoint; each document gets its own copy of it, named by
TENSORFS_ENDPOINT, and an example's `https://tensorhub.com` means that copy. Each document runs in
its own process and working directory with an empty TENSORFS_HOME. Its Python blocks share one namespace,
in which `my_token` is a placeholder token and `private_ca_der` a real CA certificate. Shell blocks
run with bash, where `uv pip install tensorfs` and `pip install tensorfs` install the wheels in
--wheels. A document's Rust blocks run last, together, as one `cargo run --example` program.
An example whose code (not its comments) names CUDA is skipped where torch sees no CUDA device.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import traceback

import numpy as np

import tensorfs

ROOT = pathlib.Path(__file__).resolve().parents[3]
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
LANGS = {"python": "python", "bash": "shell", "sh": "shell", "shell": "shell", "rust": "rust"}
TENSORHUB = "https://tensorhub.com"
SOURCE = "hf://hf-internal-testing/tiny-sdxl-pipe"


def blocks(doc: pathlib.Path) -> list[dict]:
    text = doc.read_text()
    found = []
    for m in FENCE.finditer(text):
        lang = LANGS.get(m.group(1))
        if lang:
            found.append({"line": text.count("\n", 0, m.start()) + 1, "lang": lang, "code": m.group(2)})
    return found


def fixture(work: pathlib.Path) -> pathlib.Path:
    """`me/model@1.0` on a directory endpoint, returned: variant `bf16` is the tiny SDXL rounded to
    BF16; `int8` also stores the UNet's matrices as `int8.rowwise`."""
    store, folder, endpoint = work / "fixture-store", work / "fixture-folder", work / "endpoint"
    folder.mkdir()
    source = tensorfs.convert_to_cozytensors(SOURCE, cache_dir=store)
    for variant in ("bf16", "int8"):
        with tensorfs.Writer(folder / f"model.{variant}.cozytensors", cache_dir=store) as writer:
            writer.metadata["source"] = SOURCE
            writer.pipeline(source.pipeline.cls, source.pipeline.config)
            for name, component in source.components.items():
                out = writer.component(name, cls=component.cls, config=component.config, config_file=component.config_file)
                for key, info in component.tensors.items():
                    data, shape = component.tensors.read(key).tobytes(), list(info.shape)
                    if info.dtype != "F32":
                        out.put(key, data, dtype=info.dtype, shape=shape)
                        continue
                    values = np.frombuffer(data, np.float32).reshape(shape)
                    if variant == "int8" and name == "unet" and len(shape) == 2:
                        scale = np.abs(values).max(axis=1) / 127
                        scale[scale == 0] = 1
                        weight = np.clip(np.round(values / scale[:, None]), -127, 127).astype(np.int8)
                        parts = {"weight": ("I8", shape, weight.tobytes()), "scale": ("F32", shape[:1], scale.astype(np.float32).tobytes())}
                        out.put_encoded(key, dtype="BF16", shape=shape, scheme="int8.rowwise",
                                        params={"axis": 0, "scale_dtype": "F32"}, parts=parts)
                    else:
                        bits = values.view(np.uint32)
                        out.put(key, ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes(), dtype="BF16", shape=shape)
                for asset in component.assets:
                    out.asset(asset, component.asset(asset))
    (folder / "README.md").write_text(f"# me/model\n\n{SOURCE} converted by TensorFS, in bf16 and int8 variants.\n")
    (folder / "LICENSE").write_text(f"The terms of {SOURCE.replace('hf://', 'https://huggingface.co/')} apply.\n")
    tensorfs.upload_folder(repo_id="me/model", revision="1.0", folder_path=folder, endpoint=endpoint.as_uri(), cache_dir=store)
    print(f"fixture: me/model@1.0 (bf16, int8) from {SOURCE} at {endpoint}", flush=True)
    return endpoint


def placeholders() -> dict:
    """The values a reader supplies: a token, and their private CA (any real one will do)."""
    pem = pathlib.Path(ssl.get_default_verify_paths().cafile or "/etc/ssl/certs/ca-certificates.crt").read_text()
    first = pem[: pem.index("-----END CERTIFICATE-----") + len("-----END CERTIFICATE-----")]
    return {"my_token": "docs-example-token", "private_ca_der": ssl.PEM_cert_to_DER_cert(first)}


def child(spec_path: str) -> None:
    """One document's Python and shell blocks, in order; one result line each."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    os.chdir(spec["cwd"])
    namespace = {"__name__": "__main__", **placeholders()}
    with open(spec["results"], "a") as results:
        for block in spec["blocks"]:
            status, detail = "pass", ""
            if block["skip"]:
                status, detail = "skip", block["skip"]
            elif block["lang"] == "python":
                try:
                    # Leading newlines keep a traceback's line numbers the document's.
                    exec(compile("\n" * block["line"] + block["code"], spec["doc"], "exec"), namespace)
                except BaseException as error:  # noqa: BLE001 - every failure is the example's
                    status = "fail"
                    detail = "".join(traceback.format_exception_only(type(error), error)).strip()
                    traceback.print_exc()
            else:
                ran = subprocess.run(["bash", "-eo", "pipefail", "-c", block["code"]], check=False)
                if ran.returncode:
                    status, detail = "fail", f"bash exited {ran.returncode}"
            print(f"{spec['doc']}:{block['line']} {block['lang']}: {status} {detail}".rstrip(), flush=True)
            results.write(json.dumps({"line": block["line"], "lang": block["lang"], "status": status, "detail": detail}) + "\n")
            results.flush()


def rust(doc: pathlib.Path, code: list[dict], cwd: pathlib.Path, env: dict) -> dict:
    """A document's Rust blocks, in order, as the body of one async main in a tensorfs example."""
    name = "docs_" + re.sub(r"\W", "_", str(doc.relative_to(ROOT)).removesuffix(".md"))
    example = ROOT / "crates/tensorfs/examples" / f"{name}.rs"
    body = "\n".join(block["code"] for block in code)
    example.parent.mkdir(exist_ok=True)
    example.write_text(
        f"//! Generated from {doc.relative_to(ROOT)} by docs_examples.py; not committed.\n"
        "#[tokio::main]\nasync fn main() -> Result<(), Box<dyn std::error::Error>> {\n"
        f"{body}\nOk(())\n}}\n"
    )
    try:
        ran = subprocess.run(
            ["cargo", "run", "--locked", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"), "-p", "tensorfs", "--example", name],
            cwd=cwd, env=env, capture_output=True, text=True, check=False,
        )
    finally:
        example.unlink()
    print(ran.stdout + ran.stderr, flush=True)
    lines = ", ".join(str(block["line"]) for block in code)
    detail = "" if ran.returncode == 0 else f"blocks at lines {lines}: " + " | ".join(ran.stderr.strip().splitlines()[-8:])
    return {"line": code[0]["line"], "lang": "rust", "status": "fail" if ran.returncode else "pass", "detail": detail}


def overrides(wheels: pathlib.Path | None, work: pathlib.Path) -> dict:
    """`uv pip install tensorfs` and `pip install tensorfs` take the wheels under test."""
    if wheels is None:
        return {}
    lines = []
    for wheel in sorted(wheels.glob("tensorfs-*.whl")):
        marker = 'python_version < "3.11"' if "-cp310-cp310-" in wheel.name else 'python_version >= "3.11"'
        lines.append(f"tensorfs @ {wheel.resolve().as_uri()} ; {marker}")
    path = work / "overrides.txt"
    path.write_text("\n".join(lines) + "\n")
    return {"UV_OVERRIDE": str(path), "PIP_FIND_LINKS": str(wheels.resolve()), "PIP_NO_INDEX": "1"}


def gpu() -> bool:
    """Torch sees a CUDA device (torch is the examples' GPU consumer)."""
    probe = "import torch, sys; sys.exit(not torch.cuda.is_available())"
    return subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("docs", nargs="*", type=pathlib.Path)
    parser.add_argument("--wheels", type=pathlib.Path)
    parser.add_argument("--work", type=pathlib.Path)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return 0
    work = (args.work or pathlib.Path(tempfile.mkdtemp(prefix="tensorfs-docs-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    seeded = fixture(work)
    has_gpu = gpu()
    rows = []
    for doc in (d.resolve() for d in args.docs):
        cwd = work / re.sub(r"\W", "_", str(doc.relative_to(ROOT)))
        cwd.mkdir()
        endpoint = shutil.copytree(seeded, cwd / ".endpoint").as_uri()
        env = {
            **os.environ,
            **overrides(args.wheels, work),
            "TENSORFS_HOME": str(cwd / ".tensorfs"),
            "TENSORFS_ENDPOINT": endpoint,
            "PATH": os.pathsep.join([str(pathlib.Path(sys.executable).parent), os.environ["PATH"]]),
        }
        found = blocks(doc)
        for block in found:
            block["code"] = block["code"].replace(TENSORHUB, endpoint)
            uses_cuda = re.search(r"\bcuda\b", re.sub(r"#.*", "", block["code"]))
            block["skip"] = "needs CUDA" if uses_cuda and not has_gpu else ""
        spec = {"doc": str(doc), "cwd": str(cwd), "results": str(cwd / ".results.jsonl"),
                "blocks": [b for b in found if b["lang"] != "rust"]}
        (cwd / ".spec.json").write_text(json.dumps(spec))
        ran = subprocess.run([sys.executable, __file__, "--child", str(cwd / ".spec.json")], env=env, check=False)
        done = [json.loads(line) for line in pathlib.Path(spec["results"]).read_text().splitlines()] if pathlib.Path(spec["results"]).exists() else []
        for block in spec["blocks"][len(done):]:
            done.append({"line": block["line"], "lang": block["lang"], "status": "fail", "detail": f"not run: the document's process exited {ran.returncode}"})
        code = [b for b in found if b["lang"] == "rust"]
        if code:
            done.append(rust(doc, code, cwd, env))
        rows += [{"doc": str(doc.relative_to(ROOT)), **row} for row in done]
    failed = [r for r in rows if r["status"] == "fail"]
    table = ["| example | language | result | detail |", "|---|---|---|---|"]
    table += [f"| {r['doc']}:{r['line']} | {r['lang']} | {r['status']} | {r['detail'].replace('|', '/')[:300]} |" for r in rows]
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pass", "fail", "skip")}
    summary = f"docs examples: {counts['pass']} pass, {counts['fail']} fail, {counts['skip']} skipped"
    print("\n".join(["", summary, *table]))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write("\n".join([f"## {summary}", "", *table, ""]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
