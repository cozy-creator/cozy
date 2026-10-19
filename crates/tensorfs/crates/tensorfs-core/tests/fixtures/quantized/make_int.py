#!/usr/bin/env python3
"""Writes the real int4 fixtures (xs-049 U5): row bands of a few layers of one real GPTQ, AWQ and two compressed-tensors
w4a16 checkpoints, by HTTP range at pinned revisions, each beside the same rows of its bf16 original.

<case>/model.safetensors holds the producer's tensors for the band (its own names, dtypes and bytes; a GPTQ or AWQ
band keeps its packed word columns, a compressed-tensors band its rows and `weight_shape` states the band), and
<case>/config.json is the checkpoint's own. <case>.expect.safetensors holds `source.<module>.weight`, the bf16 rows.
AWQ folds each module's input scale into the norm before it, so its case also holds that norm from both checkpoints:
the decoded columns times `norm_awq / norm_bf16` are the original's.
real-int-provenance.json records revisions, header SHA-256s, bands and each tensor's SHA-256.

    python3 -I tests/fixtures/quantized/make_int.py      (stdlib only)
"""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co/{}/resolve/{}/{}"
SIZE = {"I32": 4, "I64": 8, "F16": 2, "BF16": 2}
CASES = [
    ("real-gptq-int4", "gptq", ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "c34a4a91629f09f73a285f32dbd26106b033c654"),
     ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")),
    ("real-awq-int4", "awq", ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "db09cd27ead7fee40cdee309693cf83601b9c899"),
     ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")),
    ("real-ct-int4-asym", "ct", ("RedHatAI/Qwen3-0.6B-quantized.w4a16", "3bceedd23534ac8ef59952f72c94790b27938e42"),
     ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")),
    ("real-ct-int4-act-order", "ct", ("RedHatAI/Qwen2.5-0.5B-quantized.w4a16", "81996352fa5c5b2154927b2fc0b3f58cbe59ddcc"),
     ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")),
]
# (module, first row, rows, the norm AWQ folded its input scale into): rows are outputs (N), on whole packed words.
BANDS = [("model.layers.0.self_attn.q_proj", 0, 16, None), ("model.layers.12.mlp.down_proj", 64, 8, None),
         ("model.layers.23.self_attn.o_proj", 128, 16, None)]
# AWQ scales a module's inputs and divides the norm before it; Qwen2.5-0.5B's o_proj (GQA) is left unscaled.
AWQ_BANDS = [("model.layers.0.self_attn.q_proj", 0, 16, "model.layers.0.input_layernorm"),
             ("model.layers.12.mlp.gate_proj", 64, 8, "model.layers.12.post_attention_layernorm"),
             ("model.layers.23.self_attn.o_proj", 128, 16, None)]


def get(repo, rev, path, start=None, end=None) -> bytes:
    req = urllib.request.Request(HF.format(repo, rev, path))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert start is None or len(data) == end - start, "short read"
    return data


class File:
    def __init__(self, repo, rev):
        self.repo, self.rev = repo, rev
        n = struct.unpack("<Q", get(repo, rev, "model.safetensors", 0, 8))[0]
        raw = get(repo, rev, "model.safetensors", 8, 8 + n)
        self.sha = hashlib.sha256(raw).hexdigest()
        self.hdr, self.off = json.loads(raw), 8 + n

    def has(self, key):
        return key in self.hdr

    def rows(self, key, r0, r1):
        """Rows r0..r1 of a tensor (its first axis), as (dtype, shape, bytes)."""
        t = self.hdr[key]
        width = SIZE[t["dtype"]]
        for e in t["shape"][1:]:
            width *= e
        a = self.off + t["data_offsets"][0]
        return t["dtype"], [r1 - r0, *t["shape"][1:]], get(self.repo, self.rev, "model.safetensors",
                                                          a + r0 * width, a + r1 * width)

    def whole(self, key):
        return self.rows(key, 0, self.hdr[key]["shape"][0])


def columns(t, c0, c1):
    """Columns c0..c1 of a 2-D (dtype, shape, bytes)."""
    dtype, (rows, width), data = t
    size = SIZE[dtype]
    out = b"".join(data[(r * width + c0) * size:(r * width + c1) * size] for r in range(rows))
    return dtype, [rows, c1 - c0], out


def band(q, family, module, r0, n):
    """The producer's tensors for output rows r0..r0+n of `module`."""
    if family == "gptq":
        out = {s: columns(q.whole(f"{module}.{s}"), r0, r0 + n) for s in ("qweight", "scales")}
        out["qzeros"] = columns(q.whole(f"{module}.qzeros"), r0 // 8, (r0 + n) // 8)
        out["g_idx"] = q.whole(f"{module}.g_idx")
    elif family == "awq":
        out = {s: columns(q.whole(f"{module}.{s}"), r0 // 8, (r0 + n) // 8) for s in ("qweight", "qzeros")}
        out["scales"] = columns(q.whole(f"{module}.scales"), r0, r0 + n)
    else:
        out = {s: q.rows(f"{module}.{s}", r0, r0 + n) for s in ("weight_packed", "weight_scale")}
        if q.has(f"{module}.weight_zero_point"):
            out["weight_zero_point"] = q.rows(f"{module}.weight_zero_point", r0 // 8, (r0 + n) // 8)
        if q.has(f"{module}.weight_g_idx"):
            out["weight_g_idx"] = q.whole(f"{module}.weight_g_idx")
        k = q.hdr[f"{module}.weight_packed"]["shape"][1] * 8
        dt = q.hdr[f"{module}.weight_shape"]["dtype"]
        out["weight_shape"] = (dt, [2], struct.pack("<2q" if dt == "I64" else "<2i", n, k))
    if q.has(f"{module}.bias"):
        out["bias"] = q.rows(f"{module}.bias", r0, r0 + n)
    return {f"{module}.{s}": t for s, t in out.items()}


def write(path, tensors):
    hdr, data = {"__metadata__": {"format": "pt"}}, b""
    for key, (dtype, shape, raw) in tensors.items():
        hdr[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    head = json.dumps(hdr, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(head)) + head + data)


def main():
    provenance = []
    for case, family, (repo, rev), (ref_repo, ref_rev) in CASES:
        q, ref = File(repo, rev), File(ref_repo, ref_rev)
        tensors, expect = {}, {}
        bands = AWQ_BANDS if family == "awq" else BANDS
        for module, r0, n, norm in bands:
            tensors.update(band(q, family, module, r0, n))
            expect[f"source.{module}.weight"] = ref.rows(f"{module}.weight", r0, r0 + n)
            if norm:
                tensors[f"{norm}.weight"] = q.whole(f"{norm}.weight")
                expect[f"source.{norm}.weight"] = ref.whole(f"{norm}.weight")
        write(HERE / case / "model.safetensors", tensors)
        (HERE / case / "config.json").write_bytes(get(repo, rev, "config.json"))
        write(HERE / f"{case}.expect.safetensors", expect)
        provenance.append({
            "case": case, "source": f"{repo}@{rev}/model.safetensors", "source_header_sha256": q.sha,
            "reference": f"{ref_repo}@{ref_rev}/model.safetensors", "reference_header_sha256": ref.sha,
            "bands": [{"module": m, "rows": [r0, r0 + n]} for m, r0, n, _ in bands],
            "sha256": {k: hashlib.sha256(t[2]).hexdigest() for k, t in {**tensors, **expect}.items()},
        })
    (HERE / "real-int-provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    total = sum(p.stat().st_size for c in CASES for p in [*(HERE / c[0]).iterdir(), HERE / f"{c[0]}.expect.safetensors"])
    print(f"{len(CASES)} cases, {total} bytes")


if __name__ == "__main__":
    main()
