#!/usr/bin/env python3
"""The standalone e2e (design §11): no Cozy, no account, run by `e2e.sh` in a fresh venv.

1. CLI: `tfs convert hf://hf-internal-testing/tiny-sdxl-pipe`; `tfs export --safetensors`, which is
   every file of the repository and imports again as a fixed point (so do a Transformers model with
   its tokenizer, a sharded model and a sharded fp16 variant), and from which
   Diffusers renders exactly the upstream repo's image (CPU, fixed seed); upload to a directory
   endpoint, serve it by `python -m http.server`, then `tfs download` into a second store,
   byte-identical; a read-only audit detects a corrupt object, then its discarded copy is refetched.
2. The same through `tensorfs`, plus DLPack reads equal to upstream's own tensors, stored ranges,
   the component writer (a config with floats and None, an encoded tensor) and typed errors.

    e2e.py <workdir>
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import subprocess
import sys
from collections.abc import Iterator

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import tensorfs as tfs
from tensorfs import raw as _raw

REPO = "hf-internal-testing/tiny-sdxl-pipe"
#: Sources whose export keeps every file and imports as a fixed point, and their variant.
SOURCES = [
    (REPO, None),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", None),
    ("hf-internal-testing/tiny-random-bert-sharded-safetensors", None),
    ("hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format", "fp16"),
]
OTHER_WEIGHTS = (".bin", ".onnx", ".msgpack", ".h5")
NAME = "local/tiny-sdxl"
TFS = str(pathlib.Path(sys.executable).with_name("tfs"))
DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}


def cli(store: pathlib.Path, *args: str, ok: bool = True) -> dict:
    out = subprocess.run([TFS, "--store", str(store), "--json", *args], capture_output=True, text=True)
    if ok and out.returncode:
        raise SystemExit(f"tfs {' '.join(args)} failed:\n{out.stdout}{out.stderr}")
    return json.loads(out.stdout)


def render(source: str | pathlib.Path) -> tuple[np.ndarray, StableDiffusionXLPipeline]:
    pipe = StableDiffusionXLPipeline.from_pretrained(source, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator("cpu").manual_seed(0)
    image = pipe("a red cube on a table", num_inference_steps=2, generator=generator, output_type="np")
    return image.images[0], pipe


def same_image(folder: pathlib.Path, reference: np.ndarray) -> None:
    image, _ = render(folder)
    diff = float(np.abs(image - reference).max())
    print(f"render {folder.name}: {image.shape}, max |diff| vs upstream {diff}")
    assert diff == 0.0, f"{folder}: the image differs from upstream's by {diff}"


def same_folder(store: pathlib.Path, name: str, folder: pathlib.Path, repo: str) -> None:
    """The export holds every non-weight file of the repository byte for byte, and an index only
    of shards it wrote; imported, exported and imported again, it is the same checkpoint."""
    def kept(path: str) -> bool:
        return ".safetensors" not in path and not (path.endswith(OTHER_WEIGHTS) or ".bin.index" in path)

    allow = [p for p in HfApi().list_repo_files(repo) if kept(p)]
    upstream = pathlib.Path(snapshot_download(repo, allow_patterns=allow))
    files = {str(p.relative_to(folder)): p for p in folder.rglob("*") if p.is_file()}
    assert sorted(p for p in files if kept(p)) == sorted(allow), f"{repo}: {sorted(files)}"
    differ = [p for p in allow if files[p].read_bytes() != (upstream / p).read_bytes()]
    assert not differ, f"{repo}: {differ} differ from upstream's"
    for path in (p for p in files if ".safetensors.index" in p):
        shards = set(json.loads(files[path].read_text())["weight_map"].values())
        assert all((files[path].parent / s).is_file() for s in shards), f"{repo}: {path}"
    again = [cli(store, "convert", str(folder), "--name", f"{name}-again@1")["header"]]
    cli(store, "export", f"{name}-again@1", str(folder), "--safetensors")
    again.append(cli(store, "convert", str(folder), "--name", f"{name}-twice@1")["header"])
    assert again[0] == again[1], f"{repo}: import of an export is a fixed point"
    print(f"export of {repo}: its {len(allow)} other files byte for byte; re-import a fixed point")


def same_files(folder: pathlib.Path) -> None:
    """Every exported weights file is the repository's own, byte for byte: export lays a file out
    as safetensors writes it, so its tensors sit where upstream's do (CPU kernels round
    differently on differently aligned weights)."""
    files = sorted(p for p in folder.rglob("*.safetensors"))
    for p in files:
        upstream = pathlib.Path(hf_hub_download(REPO, str(p.relative_to(folder))))
        assert p.read_bytes() == upstream.read_bytes(), f"{p} differs from {REPO}'s"
    print(f"export: {len(files)} weights files byte-identical to upstream")


@contextlib.contextmanager
def serve(folder: pathlib.Path) -> Iterator[str]:
    args = [sys.executable, "-u", "-m", "http.server", "0", "--bind", "127.0.0.1", "--directory", str(folder)]
    server = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        line = server.stdout.readline()  # Serving HTTP on 127.0.0.1 port <n> (...)
        port = int(line.split(" port ")[1].split()[0])
        yield f"http://127.0.0.1:{port}"
    finally:
        server.kill()
        server.wait()


def blobs(store: pathlib.Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (store / "blobs").rglob("*") if p.is_file()}


def tree(folder: pathlib.Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def corrupt_and_refetch(store: pathlib.Path, origin: str) -> None:
    # Pick a declared chunk, not a repository object outside checkpoint.verify's scope.
    with tfs.open(NAME, revision="1", cache_dir=store, endpoint=origin) as checkpoint:
        digest, length = max(checkpoint.chunks, key=lambda obj: obj[1])
    victim = store / "blobs" / digest[:2] / digest[2:4] / digest
    data = bytearray(victim.read_bytes())
    assert len(data) == length
    data[len(data) // 2] ^= 0xFF
    victim.unlink()
    victim.write_bytes(data)
    verified = cli(store, "--endpoint", origin, "verify", f"{NAME}@1", ok=False)
    assert verified["error"]["code"] == "integrity" and digest in verified["error"]["detail"], verified
    with tfs.open(NAME, revision="1", cache_dir=store, endpoint=origin) as checkpoint:
        report = checkpoint.verify()
        assert report.corrupt == [(digest, length)] and report.missing == [], (report.corrupt, report.missing)
    assert victim.read_bytes() == data, "verify is an audit and never repairs or removes bytes"
    # Explicitly discard this test's bad copy; normal verified admission repairs the cache.
    victim.unlink()
    cli(store, "--endpoint", origin, "download", f"{NAME}@1")
    assert hashlib.sha256(victim.read_bytes()).hexdigest() == digest
    cli(store, "--endpoint", origin, "verify", f"{NAME}@1")
    print(f"corrupted {digest[:12]} ({length} B): reported without mutation, discarded, fetched again")


def main() -> None:
    work = pathlib.Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    reference, upstream = render(REPO)
    print(f"upstream {REPO}: {reference.shape}")

    # 1. The CLI.
    a, b = work / "a", work / "b"
    named = f"{NAME}@1"
    imported = cli(a, "convert", f"hf://{REPO}", "--name", named)
    inspected = cli(a, "inspect", named)
    assert inspected["header"] == imported["header"]
    names = [c["name"] for c in inspected["components"]]  # header order (#624)
    assert {"text_encoder", "text_encoder_2", "unet", "vae", "tokenizer", "tokenizer_2", "scheduler"} <= set(names), names
    cli(a, "export", named, str(work / "out"), "--safetensors")
    same_files(work / "out")
    for i, (repo, variant) in enumerate(SOURCES):
        flags = ["--variant", variant] if variant else []
        cli(work / "sources", "convert", f"hf://{repo}", "--name", f"local/source-{i}@1", *flags)
        cli(work / "sources", "export", f"local/source-{i}@1", str(work / f"source-{i}"), "--safetensors")
        same_folder(work / "sources", f"local/source-{i}", work / f"source-{i}", repo)
    same_image(work / "out", reference)
    cli(a, "export", named, str(work / "portable"))
    cli(a, "--endpoint", (work / "origin").as_uri(), "upload", str(work / "portable"), named)
    with serve(work / "origin") as url:
        cli(b, "--endpoint", url, "download", named)
        pulled = cli(b, "--endpoint", url, "inspect", named)
        assert pulled["header"] == imported["header"], (pulled, imported)
        expected = blobs(work / "origin")
        assert blobs(b) == expected, "every published object is downloaded byte-identically"
        corrupt_and_refetch(b, url)
        assert blobs(b) == expected
    print(f"CLI: convert, export, upload, download: {len(expected)} objects identical")

    # 2. The Python API.
    tfs.configure(cache_dir=work / "c")
    m = tfs.convert_to_cozytensors(f"hf://{REPO}")
    folder = tfs.materialize(m, local_dir=work / "python-portable")
    tfs.upload_folder(repo_id=NAME, revision="1", folder_path=folder)
    assert m.id == f"sha256:{imported['header']['sha256']}", "imports are deterministic"
    ck = tfs.open(NAME, revision="1", local_files_only=True)
    assert (ck.repo_id, ck.revision) == (NAME, "1")
    assert (ck.id, ck.tree) == (m.id, m.tree)
    compared = 0
    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        want = getattr(upstream, component).state_dict()
        for key, info in ck.components[component].tensors.items():
            assert info.encoding is None and not hasattr(info, "parts"), (key, info)
            stored = _raw.parts(ck, component, key)
            assert set(stored) == {""}, (component, key, stored)
            part = stored[""]
            assert (part.dtype, part.shape) == (info.dtype, info.shape)
            got = torch.from_dlpack(ck.components[component].tensors.read(key, decode=True))
            assert tuple(got.shape) == info.shape and got.dtype == DTYPES[info.dtype], (key, got.dtype, info)
            if key in want:
                assert torch.equal(got, want[key]), f"{component}/{key} differs from upstream"
                compared += 1
    assert compared > 100, compared
    conv = ck.components["unet"].tensors.read("conv_in.weight", decode=True)
    want_np = upstream.unet.state_dict()["conv_in.weight"].numpy()
    assert np.array_equal(np.from_dlpack(conv), want_np), "numpy reads it too"
    assert conv.tobytes() == want_np.tobytes()
    print(f"Python: {compared} tensors and their stored array metadata equal upstream's through DLPack")

    written = pathlib.Path(tfs.materialize(m, local_dir=work / "out2", format="safetensors"))
    assert (written / "unet/diffusion_pytorch_model.safetensors").is_file()
    assert tree(work / "out2") == tree(work / "out"), "the Python export is the CLI's, byte for byte"
    same_image(work / "out2", reference)
    origin = (work / "origin2").as_uri()
    remote_name = "conformance/tiny-sdxl"
    tfs.upload_folder(repo_id=remote_name, revision="1", folder_path=folder, endpoint=origin)
    with serve(work / "origin2") as url:
        downloaded = tfs.materialize(remote_name, revision="1", endpoint=url, cache_dir=work / "d")
        with tfs.open(downloaded) as remote:
            assert remote.id == m.id

    w = tfs.Writer(base=m)
    config = {"scale": 0.18215, "eps": 1e-05, "act": None, "blocks": [1, 2.5], "name": "x"}
    extra = w.component("extra", cls="diffusers.Extra", config=config)
    extra.put("x.weight", np.arange(4, dtype=np.float32).reshape(2, 2))
    parts = {"weight": ("F8_E4M3", [2, 16], bytes([0x38]) * 32), "scale": ("F32", [2], np.array([0.5, 2.0], dtype=np.float32))}
    extra.put_encoded("q.weight", dtype="F32", shape=[2, 16], scheme="fp8.rowwise", parts=parts, params={"axis": 0, "scale_dtype": "F32"})
    extra.asset("vocab.txt", b"a\nb\n")
    assert extra.has("q.weight") and w.has("extra", "x.weight")
    ck3 = w.commit()
    assert ck3.components["unet"].tensors.keys() == ck.components["unet"].tensors.keys(), "the base carries over"
    e = ck3.components["extra"]
    assert (e.cls, e.config) == ("diffusers.Extra", config), (e.cls, e.config)
    assert e.assets["vocab.txt"][1] == 4 and e.asset("vocab.txt") == b"a\nb\n"
    q = ck3.components["extra"].tensors["q.weight"]
    assert q.encoding is not None and q.encoding.scheme == "fp8.rowwise" and set(_raw.parts(ck3, "extra", "q.weight")) == {"weight", "scale"}
    assert str(q.encoding) == 'fp8.rowwise {axis: 0, scale_dtype: "F32"}', q.encoding
    assert torch.equal(torch.from_dlpack(ck3.components["extra"].tensors.read("q.weight", decode=True)), torch.tensor([[0.5] * 16, [2.0] * 16]))
    try:
        tfs.open("local/nothing", local_files_only=True)
        raise AssertionError("no such ref")
    except tfs.NotFoundError as e:
        assert e.code == "not_found" and isinstance(e, tfs.Error), e

    print("e2e: all green")


if __name__ == "__main__":
    main()
