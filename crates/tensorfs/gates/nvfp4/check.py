"""nvfp4 lane numeric check: does a lane's nvfp4 decode to its model's bf16 weights?

For each model, about `--samples` nvfp4 tensors spread over the lane (header order) are read
through Tensorhub by byte range (a 128-row band each: whole cuBLAS tiles, no full download),
decoded by TensorFS's reference decoder, and compared with the same rows of the bf16 lane.
Prints relative L2 per tensor and the median. A correct layout lands at 9-12% (xs-049, real
files); 25% or more is a wrong layout, flagged.

    python check.py --tensorhub local paul/flux-3-klein paul/minimax-h3@1.0.0-rc.3
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from tensorfs import v2

PASS, WRONG = 0.15, 0.25
BAND = 128


def hub_url(name: str) -> str:
    """A URL, `tensorhub`, or a Hub name from cozy's `~/.cozy/config.toml`."""
    if name.startswith(("http://", "https://")):
        return name.rstrip("/")
    if name == "tensorhub":
        return "https://tensorhub.com"
    hubs = tomllib.loads((Path.home() / ".cozy/config.toml").read_text()).get("hubs", {})
    if name not in hubs:
        sys.exit(f"no Hub {name!r} in ~/.cozy/config.toml (it names {sorted(hubs)})")
    return hubs[name].rstrip("/")


class Hub:
    def __init__(self, base: str, token: str | None):
        self.base, self.token = base, token

    def call(self, path: str, body: dict) -> dict:
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            sys.exit(f"{path} {body.get('ref', '')}: HTTP {e.code} {e.read()[:500].decode(errors='replace')}")

    def get(self, url: str, start: int, length: int) -> bytes:
        req = urllib.request.Request(url, headers={"range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        if r.status == 200 and len(body) != length:
            body = body[start : start + length]
        if len(body) != length:
            raise RuntimeError(f"{len(body)} of {length} bytes from {url[:80]}")
        return body


class Lane:
    """A lane: its manifest and header in a scratch store, where the library parses them; its
    chunks are read by range from presigned URLs."""

    def __init__(self, hub: Hub, store: v2.Store, ref: str, lane: str):
        page = hub.call("/v1/tensorfs/closure", {"ref": ref, "lane": lane})
        self.hub, self.store, self.root = hub, store, Path(store.root)
        self.manifest = page["manifest"]
        self.bound = page["presign_max_digests"]
        self.name = f"{page['model']}@{page['release']}/{page['lane'] or lane}"
        self.urls: dict[str, str] = {}
        doc = json.loads(self.whole(self.manifest))
        if "files" not in doc:
            sys.exit(f"{self.name}: manifest {self.manifest['sha256']} is not tensorfs.manifest/1")
        self.whole(doc["files"]["model.cozytensors"])
        self.ck = store.checkpoint(f"sha256:{self.manifest['sha256']}")
        self.chunks = [(Path(path).name, length) for path, length in self.ck.chunks]

    def presign(self, digests) -> None:
        missing = sorted({d for d in digests if d not in self.urls})
        for i in range(0, len(missing), self.bound):
            body = {"digests": missing[i : i + self.bound], "manifest": self.manifest["sha256"]}
            self.urls.update(self.hub.call("/v1/tensorfs/presign", body)["urls"])

    def whole(self, ref: dict) -> bytes:
        """An object, checked against its digest, into the scratch store."""
        self.presign([ref["sha256"]])
        data = self.hub.get(self.urls[ref["sha256"]], 0, ref["length"])
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise RuntimeError(f"{ref['sha256']}: the bytes served do not match")
        path = self.root / "blobs" / ref["sha256"][:2] / ref["sha256"][2:4] / ref["sha256"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def spans(self, part: v2.Part, start: int, length: int):
        """Where bytes [start, start + length) of `part` live: it runs on into later chunks."""
        chunk, offset = part.chunk, part.offset + start
        while offset >= self.chunks[chunk][1]:
            offset -= self.chunks[chunk][1]
            chunk += 1
        while length:
            digest, size = self.chunks[chunk]
            n = min(length, size - offset)
            yield digest, offset, n
            chunk, offset, length = chunk + 1, 0, length - n

    def read(self, part: v2.Part, start: int = 0, length: int | None = None) -> bytes:
        length = part.length - start if length is None else length
        spans = list(self.spans(part, start, length))
        self.presign(d for d, _, _ in spans)
        return b"".join(self.hub.get(self.urls[d], o, n) for d, o, n in spans)


def band(shape) -> tuple[int, int, int]:
    """The rows read of a `[*lead, C]` tensor: a middle band of whole 128-row tiles."""
    rows = int(np.prod(shape[:-1]))
    start = (rows // BAND // 2) * BAND if rows >= BAND else 0
    return start, min(BAND, rows), int(shape[-1])


def nvfp4_band(lane: Lane, info: v2.Info) -> tuple[dict, list[int]]:
    """The band's parts, a valid nvfp4 tensor of their own: weight rows, their block scales
    (row-major rows, or the 128-row band of cuBLAS tiles), and `scale_2`."""
    start, rows, cols = band(info.shape)
    p, params = info.parts, info.encoding.params
    w = cols // 16
    weight = lane.read(p["weight"], start * cols // 2, rows * cols // 2)
    if params["scale_layout"] == "row_major":
        scale, scale_shape = lane.read(p["scale"], start * w, rows * w), [rows, w]
    else:
        width = -(-w // 4) * 4
        tiles = -(-rows // 128) * 128
        scale, scale_shape = lane.read(p["scale"], start * width, tiles * width), [tiles, width]
    s2 = p["scale_2"]
    return {
        "weight": ("F4", [rows, cols], weight),
        "scale": ("F8_E4M3", scale_shape, scale),
        "scale_2": (s2.dtype, list(s2.shape), lane.read(s2)),
    }, [rows, cols]


def floats(dtype: str, data: bytes) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return np.frombuffer(data, {"F16": "<f2", "F32": "<f4"}[dtype]).astype(np.float64)


def check(hub: Hub, store: v2.Store, ref: str, lane: str, reference: str, samples: int) -> tuple[str, list]:
    q, r = Lane(hub, store, ref, lane), Lane(hub, store, ref, reference)
    found = [
        (c, k, info)
        for c, tensors in q.ck.tensors.items()
        for k, info in tensors.items()
        if info.encoding is not None and info.encoding.scheme == "nvfp4"
    ]
    if not found:
        return f"{q.name}: no nvfp4 tensor", []
    picked = sorted({round(i * (len(found) - 1) / max(samples - 1, 1)) for i in range(samples)})
    writer, rows = store.writer(), []
    for i in picked:
        component, key, info = found[i]
        want = r.ck.tensors.get(component, {}).get(key)
        if want is None or tuple(want.shape) != tuple(info.shape) or want.encoding is not None:
            rows.append((component, key, None, f"no plain {reference} twin ({want and want.encoding})"))
            continue
        parts, shape = nvfp4_band(q, info)
        writer.put_encoded(component, key, info.dtype, shape, "nvfp4", parts, info.encoding.params)
        start, n, cols = band(info.shape)
        size = {"BF16": 2, "F16": 2, "F32": 4}[want.dtype]
        source = floats(want.dtype, r.read(want.parts[""], start * cols * size, n * cols * size))
        rows.append((component, key, (info.dtype, source), str(info.encoding)))
    decoded = store.checkpoint(writer.commit())
    results = []
    for component, key, source, note in rows:
        if source is None:
            results.append((f"{component}/{key}", None, note))
            continue
        d = floats(source[0], decoded.read(component, key).tobytes())
        rel = float(np.linalg.norm(d - source[1]) / np.linalg.norm(source[1]))
        results.append((f"{component}/{key}", rel, note))
    return f"{q.name} against {r.name}", results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("refs", nargs="+", help="org/name[@release]")
    ap.add_argument("--tensorhub", default="tensorhub", help="a cozy Hub name or URL")
    ap.add_argument("--lane", default="nvfp4")
    ap.add_argument("--reference-lane", default="bf16")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    hub = Hub(hub_url(args.tensorhub), os.environ.get("TENSORHUB_TOKEN"))
    work = Path(tempfile.mkdtemp(prefix="nvfp4-check-"))
    report, failed = [], False
    try:
        for ref in args.refs:
            store = v2.Store(work / ref.replace("/", "_").replace("@", "_"))
            title, results = check(hub, store, ref, args.lane, args.reference_lane, args.samples)
            errors = [rel for _, rel, _ in results if rel is not None]
            median = statistics.median(errors) if errors else None
            wrong = [name for name, rel, _ in results if rel is not None and rel >= WRONG]
            ok = median is not None and median <= PASS and not wrong
            failed |= not ok
            report.append({"case": title, "median": median, "pass": ok, "wrong_layout": wrong,
                           "tensors": [{"tensor": n, "rel_l2": e, "note": note} for n, e, note in results]})
            if not args.json:
                print(f"\n{title}")
                for name, rel, note in results:
                    flag = "  WRONG LAYOUT" if rel is not None and rel >= WRONG else ""
                    shown = "  skipped" if rel is None else f"{rel:7.2%}"
                    print(f"  {shown}  {name}  {note}{flag}")
                verdict = "PASS" if ok else "FAIL"
                print(f"  median {median:.2%}: {verdict}" if median is not None else f"  {verdict}: nothing compared")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.json:
        print(json.dumps(report, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
