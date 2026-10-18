#!/usr/bin/env python3
"""The standalone e2e (design §11): no Cozy, no account, run by `e2e.sh` in a fresh venv.

1. CLI: `tfs import hf://hf-internal-testing/tiny-sdxl-pipe`; `tfs export --safetensors`, from
   which Diffusers renders exactly the upstream repo's image (CPU, fixed seed); `tfs export
   --origin`, served by `python -m http.server`, then `tfs pull` into a second store,
   byte-identical; a corrupted object is removed by `tfs verify` and fetched again.
2. The same through `tensorfs.v2`, plus DLPack reads equal to upstream's own tensors, regions,
   the writer (with an encoded tensor) and typed errors.

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
from huggingface_hub import hf_hub_download

import tensorfs.v2 as tfs

REPO = "hf-internal-testing/tiny-sdxl-pipe"
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
    assert set(names) == {"text_encoder", "text_encoder_2", "unet", "vae"}, names
    cli(a, "export", NAME, str(work / "out"), "--safetensors")
    same_files(work / "out")
    same_image(work / "out", reference)
    cli(a, "export", NAME, str(work / "origin"), "--origin")
    with serve(work / "origin") as url:
        pulled = cli(b, "pull", url, NAME)
        assert pulled["manifest"] == imported["manifest"], (pulled, imported)
        assert blobs(b) == blobs(a), "the pulled store is byte-identical"
        corrupt_and_refetch(b, url)
        assert blobs(b) == blobs(a)
    print(f"CLI: import, export, origin, pull: {len(blobs(a))} objects identical")

    # 2. The Python API.
    store = tfs.Store(work / "c")
    m = store.import_(f"hf://{REPO}", name=NAME)
    assert (m.sha256, m.length) == (imported["manifest"]["sha256"], imported["manifest"]["length"]), "imports are deterministic"
    assert (m.name, m.release) == (NAME, "1")
    ck = store.checkpoint(NAME)
    assert ck.manifest == m
    compared = 0
    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        want = getattr(upstream, component).state_dict()
        for key, info in ck.tensors[component].items():
            assert info.encoding is None and list(info.parts) == [""], (key, info)
            got = torch.from_dlpack(ck.read(component, key))
            assert tuple(got.shape) == info.shape and got.dtype == DTYPES[info.dtype], (key, got.dtype, info)
            if key in want:
                assert torch.equal(got, want[key]), f"{component}/{key} differs from upstream"
                compared += 1
    assert compared > 100, compared
    conv = ck.read("unet", "conv_in.weight")
    want_np = upstream.unet.state_dict()["conv_in.weight"].numpy()
    assert np.array_equal(np.from_dlpack(conv), want_np), "numpy reads it too"
    assert conv.tobytes() == want_np.tobytes()
    grouped = tfs.regions(ck, 1 << 20)
    names = [n for r in grouped for n in r]
    assert sorted(names) == sorted(f"{c}/{k}" for c, t in ck.tensors.items() for k in t), "every tensor in one region"
    print(f"Python: {compared} tensors equal upstream's through DLPack; {len(grouped)} regions")

    written = store.export(m, work / "out2")
    assert "unet/diffusion_pytorch_model.safetensors" in written
    assert tree(work / "out2") == tree(work / "out"), "the Python export is the CLI's, byte for byte"
    same_image(work / "out2", reference)
    store.export(NAME, work / "origin2", safetensors=False, origin=True)
    with serve(work / "origin2") as url:
        second = tfs.Store(work / "d")
        assert second.pull(url, NAME) == m
        assert blobs(work / "d") == blobs(work / "c")

    w = store.writer(base=m)
    w.put("extra", "x.weight", "F32", [2, 2], np.arange(4, dtype=np.float32))
    parts = {"weight": ("F8_E4M3", [2, 16], bytes([0x38]) * 32), "scale": ("F32", [2], np.array([0.5, 2.0], dtype=np.float32))}
    w.put_encoded("extra", "q.weight", "F32", [2, 16], "fp8.rowwise", parts, {"axis": 0})
    m3 = w.commit()
    ck3 = store.checkpoint(m3)
    assert ck3.tensors["unet"].keys() == ck.tensors["unet"].keys(), "the base carries over"
    q = ck3.tensors["extra"]["q.weight"]
    assert q.encoding is not None and q.encoding.scheme == "fp8.rowwise" and set(q.parts) == {"weight", "scale"}
    assert str(q.encoding) == "fp8.rowwise {axis: 0}", q.encoding
    assert torch.equal(torch.from_dlpack(ck3.read("extra", "q.weight")), torch.tensor([[0.5] * 16, [2.0] * 16]))
    try:
        store.checkpoint("local/nothing")
        raise AssertionError("no such ref")
    except tfs.NotFoundError as e:
        assert e.code == "not_found" and isinstance(e, tfs.Error), e
    assert store.gc()["objects"] == 0
    print("e2e: all green")


if __name__ == "__main__":
    main()
