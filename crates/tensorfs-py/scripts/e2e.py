#!/usr/bin/env python3
"""The standalone e2e (design §11): no Cozy, no account, run by `e2e.sh` in a fresh venv.

1. CLI: `tfs import hf://hf-internal-testing/tiny-sdxl-pipe`; `tfs export --safetensors`, which is
   every file of the repository and imports again as a fixed point (so do a Transformers model with
   its tokenizer, a sharded model and a sharded fp16 variant), and from which
   Diffusers renders exactly the upstream repo's image (CPU, fixed seed); `tfs export
   --origin`, served by `python -m http.server`, then `tfs pull` into a second store,
   byte-identical; a corrupted object is removed by `tfs verify` and fetched again.
2. The same through `tensorfs`, plus DLPack reads equal to upstream's own tensors, regions,
   the component writer (a config with floats and None, an encoded tensor) and typed errors.
3. A parsed header carries the stored checkpoint's pipeline, components and tensors, with no
   store behind it.

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
from tensorfs import varena

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
    again = [cli(store, "import", str(folder), "--name", f"{name}-again")["header"]]
    cli(store, "export", f"{name}-again", str(folder), "--safetensors")
    again.append(cli(store, "import", str(folder), "--name", f"{name}-twice")["header"])
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
    victim = max((p for p in (store / "blobs").rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.unlink()
    victim.write_bytes(data)
    verified = cli(store, "verify", ok=False)
    assert verified["error"]["code"] == "integrity" and verified["corrupt"] == [victim.name], verified
    assert not victim.exists(), "verify removes a corrupt object"
    again = cli(store, "pull", origin, NAME)
    assert again["fetched"] == len(data), again
    cli(store, "verify")
    print(f"corrupted {victim.name[:12]} ({len(data)} B): refused, removed, fetched again")


def main() -> None:
    work = pathlib.Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    reference, upstream = render(REPO)
    print(f"upstream {REPO}: {reference.shape}")

    # 1. The CLI.
    a, b = work / "a", work / "b"
    imported = cli(a, "import", f"hf://{REPO}", "--name", NAME)
    assert imported["name"] == NAME and imported["release"] == "1", imported
    inspected = cli(a, "inspect", NAME)
    names = [c["name"] for c in inspected["components"]]  # header order (#624)
    assert {"text_encoder", "text_encoder_2", "unet", "vae", "tokenizer", "tokenizer_2", "scheduler"} <= set(names), names
    cli(a, "export", NAME, str(work / "out"), "--safetensors")
    same_files(work / "out")
    for i, (repo, variant) in enumerate(SOURCES):
        flags = ["--variant", variant] if variant else []
        cli(work / "sources", "import", f"hf://{repo}", "--name", f"local/source-{i}", *flags)
        cli(work / "sources", "export", f"local/source-{i}", str(work / f"source-{i}"), "--safetensors")
        same_folder(work / "sources", f"local/source-{i}", work / f"source-{i}", repo)
    same_image(work / "out", reference)
    cli(a, "export", NAME, str(work / "origin"), "--origin")
    with serve(work / "origin") as url:
        pulled = cli(b, "pull", url, NAME)
        assert pulled["header"] == imported["header"], (pulled, imported)
        assert blobs(b) == blobs(a), "the pulled store is byte-identical"
        corrupt_and_refetch(b, url)
        assert blobs(b) == blobs(a)
    print(f"CLI: import, export, origin, pull: {len(blobs(a))} objects identical")

    # 2. The Python API.
    tfs.configure(cache_dir=work / "c")
    m = tfs.convert(f"hf://{REPO}")
    folder = tfs.snapshot_download(m.tree, local_files_only=True)
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
            got = torch.from_dlpack(ck.components[component].tensors.read(key))
            assert tuple(got.shape) == info.shape and got.dtype == DTYPES[info.dtype], (key, got.dtype, info)
            if key in want:
                assert torch.equal(got, want[key]), f"{component}/{key} differs from upstream"
                compared += 1
    assert compared > 100, compared
    conv = ck.components["unet"].tensors.read("conv_in.weight")
    want_np = upstream.unet.state_dict()["conv_in.weight"].numpy()
    assert np.array_equal(np.from_dlpack(conv), want_np), "numpy reads it too"
    assert conv.tobytes() == want_np.tobytes()
    grouped = varena.regions(ck, 1 << 20)
    names = [n for r in grouped for n in r]
    assert sorted(names) == sorted(f"{c}/{k}" for c, component in ck.components.items() for k in component.tensors), "every tensor in one region"
    print(f"Python: {compared} tensors equal upstream's through DLPack; {len(grouped)} regions")

    written = tfs.export(m, work / "out2")
    assert "unet/diffusion_pytorch_model.safetensors" in written
    assert tree(work / "out2") == tree(work / "out"), "the Python export is the CLI's, byte for byte"
    same_image(work / "out2", reference)
    origin = (work / "origin2").as_uri()
    remote_name = "conformance/tiny-sdxl"
    tfs.upload_folder(repo_id=remote_name, revision="1", folder_path=folder, endpoint=origin)
    downloaded = tfs.snapshot_download(remote_name, revision="1", endpoint=origin, cache_dir=work / "d")
    assert tfs.open(downloaded).id == m.id

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
    assert torch.equal(torch.from_dlpack(ck3.components["extra"].tensors.read("q.weight")), torch.tensor([[0.5] * 16, [2.0] * 16]))
    try:
        tfs.open("local/nothing", local_files_only=True)
        raise AssertionError("no such ref")
    except tfs.NotFoundError as e:
        assert e.code == "not_found" and isinstance(e, tfs.Error), e

    # 3. A parsed header.
    h = ck3.id.removeprefix("sha256:")
    parsed = tfs.Checkpoint.parse((work / "c" / "blobs" / h[:2] / h[2:4] / h).read_bytes())
    assert parsed.id is None and parsed.chunks == ck3.chunks and parsed.metadata == ck3.metadata
    assert (parsed.pipeline is None) == (ck3.pipeline is None)
    for name, c in ck3.components.items():
        p = parsed.components[name]
        assert (p.cls, p.config, p.assets) == (c.cls, c.config, c.assets), name
    try:
        parsed.components["extra"].asset("vocab.txt")
        raise AssertionError("a parsed header has no assets here")
    except tfs.UnsupportedError:
        pass
    for component, stored_component in ck.components.items():
        tensors = stored_component.tensors
        assert list(parsed.components[component].tensors) == list(tensors), component
        for key, info in tensors.items():
            assert (parsed.components[component].tensors[key].dtype, parsed.components[component].tensors[key].shape) == (info.dtype, info.shape)
    try:
        parsed.components["unet"].tensors.read("conv_in.weight")
        raise AssertionError("a parsed header has no chunks")
    except tfs.UnsupportedError:
        pass
    print(f"parse: {sum(len(c.tensors) for c in parsed.components.values())} tensors, as stored")
    print("e2e: all green")


if __name__ == "__main__":
    main()
