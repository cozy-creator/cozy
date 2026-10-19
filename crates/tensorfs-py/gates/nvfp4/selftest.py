"""check.py against a tiny local Tensorhub: a bf16 lane and nvfp4 lanes quantized from it, in
ModelOpt's layout, in ComfyUI's, and ComfyUI's bytes declared as ModelOpt's (a wrong layout).
The correct lanes must pass and the wrong one must fail with every tensor flagged.

    python selftest.py
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tensorfs

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
E4M3 = np.array([m * 2.0**-9 if c >> 3 == 0 else (1 + m / 8) * 2.0 ** ((c >> 3) - 7) for c in range(127) for m in [c & 7]])


def encode(table, sign, x):
    """Nearest code, ties to even, saturating."""
    a = np.abs(x)
    hi = np.clip(np.searchsorted(table, a), 1, len(table) - 1)
    lo = hi - 1
    near = np.where(a - table[lo] < table[hi] - a, lo, np.where(a - table[lo] > table[hi] - a, hi, np.where(lo % 2 == 0, lo, hi)))
    return (np.where(a >= table[-1], len(table) - 1, near) | np.where(x < 0, sign, 0)).astype(np.uint8)


def tiles(s):
    """comfy/float.py `to_blocked`: cuBLAS 128x4 tiles, zero padded."""
    r, c = s.shape
    nr, nc = -(-r // 128), -(-c // 4)
    p = np.zeros((nr * 128, nc * 4), s.dtype)
    p[:r, :c] = s
    return p.reshape(nr, 128, nc, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3).reshape(nr * 128, nc * 4)


def nvfp4(w, comfy):
    rows = w.reshape(-1, w.shape[-1])
    g = np.float32(np.abs(w).max() / 2688)
    blocks = rows.reshape(len(rows), -1, 16)
    s = encode(E4M3, 0x80, np.abs(blocks).max(-1) / 6 / g)
    q = encode(E2M1, 8, (blocks / (E4M3[s & 0x7F] * g)[..., None]).reshape(rows.shape))
    if comfy:
        return (q[:, 0::2] << 4) | q[:, 1::2], tiles(s), g
    return q[:, 0::2] | (q[:, 1::2] << 4), s.reshape(*w.shape[:-1], -1), g


def bf16(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def lanes(store):
    rng = np.random.default_rng(0)
    shapes = {f"blocks.{i}.attn.to_q.weight": (384, 256) for i in range(4)} | {"blocks.4.experts.weight": (2, 128, 256)}
    weights = {k: rng.standard_normal(s).astype(np.float32) * 0.05 for k, s in shapes.items()}
    out = {}
    w = store.writer()
    for k, x in weights.items():
        w.put("transformer", k, "BF16", x.shape, bf16(x).tobytes())
        w.put("transformer", "norm." + k, "BF16", x.shape[-1:], bf16(x.reshape(-1)[: x.shape[-1]]).tobytes())
    out["bf16"] = w.commit()
    for lane, comfy, declared in [("nvfp4", False, False), ("nvfp4-comfy", True, True), ("nvfp4-wrong", True, False)]:
        w = store.writer()
        for k, x in weights.items():
            values = (np.frombuffer(bf16(x).tobytes(), "<u2").astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
            packed, scale, g = nvfp4(values, comfy)
            layout = ("high_first", "cublas_128x4") if declared else ("low_first", "row_major")
            parts = {"weight": ("F4", x.shape, packed.tobytes()), "scale": ("F8_E4M3", scale.shape, scale.tobytes()),
                     "scale_2": ("F32", (), np.array(g, np.float32).tobytes())}
            params = {"block_size": 16, "global_scale": "multiply", "nibble_order": layout[0], "scale_layout": layout[1]}
            w.put_encoded("transformer", k, "BF16", x.shape, "nvfp4", parts, params)
            w.put("transformer", "norm." + k, "BF16", x.shape[-1:], bf16(x.reshape(-1)[: x.shape[-1]]).tobytes())
        out[lane] = w.commit()
    return out


def serve(store, lanes):
    root = Path(store.root)

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, doc, status=200):
            body = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/v1/tensorfs/closure":
                m = lanes.get(body["lane"])
                if m is None:
                    return self.reply({"error": {"code": "tensorfs.closure_denied", "message": "no lane"}}, 404)
                return self.reply({"complete": True, "lane": body["lane"], "manifest": {"sha256": m.sha256, "length": m.length},
                                   "model": "acme/tiny", "objects": [], "presign_max_digests": 2, "public": True,
                                   "release": "1.0.0", "scope": "runtime"})
            base = f"http://127.0.0.1:{self.server.server_port}"
            self.reply({"urls": {d: f"{base}/blobs/{d}" for d in body["digests"]}, "expires_at_unix": 0, "server_time_unix": 0})

        def do_GET(self):
            d = self.path.rsplit("/", 1)[1]
            data = (root / "blobs" / d[:2] / d[2:4] / d).read_bytes()
            a, b = self.headers["range"].removeprefix("bytes=").split("-")
            part = data[int(a) : int(b) + 1]
            self.send_response(206)
            self.send_header("content-length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = tensorfs.Store(Path(tmp) / "hub")
        server = serve(store, lanes(store))
        url = f"http://127.0.0.1:{server.server_port}"
        check = Path(__file__).with_name("check.py")
        results = {}
        for lane in ["nvfp4", "nvfp4-comfy", "nvfp4-wrong"]:
            started = time.monotonic()
            run = subprocess.run([sys.executable, "-I", str(check), "--tensorhub", url, "--lane", lane, "--json", "acme/tiny@1.0.0"],
                                 capture_output=True, text=True)
            [case] = json.loads(run.stdout)
            results[lane] = (run.returncode, case)
            errors = ", ".join(f"{t['rel_l2']:.1%}" for t in case["tensors"])
            print(f"{lane}: exit {run.returncode}, median {case['median']:.2%} [{errors}] in {time.monotonic() - started:.1f} s")
        server.shutdown()
    assert results["nvfp4"][0] == 0 and results["nvfp4-comfy"][0] == 0, results
    assert results["nvfp4-wrong"][0] == 1 and len(results["nvfp4-wrong"][1]["wrong_layout"]) == 5, results
    assert all(len(case["tensors"]) == 5 for _, case in results.values())
    print("selftest: pass")


if __name__ == "__main__":
    main()
