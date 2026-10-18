"""TensorFS speed gate: cozytensors/2 against safetensors, disk -> pinned -> GPU (xs-045 F1, U8).

    python speedgate.py <work dir> <result.json> [--build-only] [--plane=readers=16,staging_buffers=32]

The machine's v1 store supplies the real SDXL int8 and Qwen-Image int8 lane bytes. Each lane is
written once into a TensorFS v2 store by its writer (the real chunk layout) and once as one
safetensors file per component. Then every (reader, cell) is timed in interleaved repetitions:
cold (pages dropped) and warm, the whole checkpoint and one paging window, from first open to
the last byte resident on the GPU.

Readers:
- `T64+rust`: `tensorfs.varena` (the plane's Rust chunk reader): checkpoint, regions, register,
  want, wait.
- `T64+fst`: the same chunk files read by fastsafetensors' own C++ engine.
- `T64`: the F1 Python reader over the same chunk files.
- `fastsafetensors`: fastsafetensors 0.5.0 over the safetensors files (the gate's reference).
"""

import ctypes
import glob
import json
import os
import re
import resource
import shutil
import statistics
import sys
import threading
import time

import cbor2
import fastsafetensors.copier.nogds as fst_nogds
import torch
from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import cpp as fstcpp
from fastsafetensors.frameworks import get_framework_op
from tensorfs import v2, varena

# fastsafetensors dlopens an unversioned libcudart.so; torch's wheels ship only the versioned one.
_cudart = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "**", "libcudart.so*"), recursive=True))
if _cudart:
    fst_nogds.resolve_runtime_lib_name = lambda framework=None: os.path.realpath(_cudart[0])

MiB = 1 << 20
ALIGN = 4 << 10
SLOT = 16 * MiB  # one read request and its pinned staging buffer
THREADS = 16  # fastsafetensors' defaults: 16 threads, 16 MiB buffers
LANES = {  # v1 manifest, window component, window key prefix, repetitions
    "sdxl-int8": ("3eb5674846e82e5b8d3e20d73f083a6d4aaa57ea699425287d5669906cfe6064", "unet", "up_blocks.1.", 5),
    "qwen-int8": ("e7c79ee942b892d76e778fb873fbdce55deaadb1fe669fca6e7851980f747493", "transformer", "transformer_blocks.30.", 5),
}
LAYOUTS = ("T64+rust", "T64+fst", "T64", "fastsafetensors")


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def store_root():
    for c in (os.environ.get("COZY_TENSORFS_ROOT"), "/var/lib/tensorfs", os.path.expanduser("~/.tensorfs")):
        if c and os.path.isdir(os.path.join(c, "manifests")):
            return c
    raise RuntimeError("no TensorFS v1 store with manifests/ on this machine")


def v1_parts(root, manifest, wcomp, wprefix):
    """(component, key, role, length, inline bytes | None, [(object path, length)], in window), in execution order."""
    blob = lambda d: f"{root}/blobs/{d[:2]}/{d[2:4]}/{d}"
    m = glob.glob(f"{root}/manifests/{manifest[:2]}/{manifest[2:4]}/{manifest}*")[0]
    e = next(x for x in json.load(open(m))["entries"] if x["kind"] == "cozytensors")["blob"]["sha256"]
    out = []
    for comp, tensors in cbor2.loads(open(blob(e), "rb").read())[4]:
        for t in sorted(tensors, key=lambda t: natkey(t[0])):
            for role, _codec, _shape, body in t[4]:
                w = comp == wcomp and t[0].startswith(wprefix)
                if isinstance(body, bytes):
                    out.append((comp, t[0], role, len(body), body, [], w))
                else:
                    segs = [(blob(bytes(d).hex()), n) for d, n in body]
                    out.append((comp, t[0], role, sum(n for _, n in segs), None, segs, w))
    return out


def part_bytes(p):
    return p[4] if p[4] is not None else b"".join(open(path, "rb").read() for path, _ in p[5])


def write_v2(parts, root):
    """The lane through TensorFS's own writer, each part a U8 tensor `key#role`: the manifest digest."""
    w = v2.Store(root).writer()
    for p in parts:
        w.put(p[0], f"{p[1]}#{p[2]}", "U8", [p[3]], part_bytes(p))
    return w.commit().sha256


def build_safetensors(parts, d):
    """One file per component, parts in execution order as U8 tensors named `key#role`."""
    os.makedirs(d, exist_ok=True)
    for comp in dict.fromkeys(p[0] for p in parts):
        cp = [p for p in parts if p[0] == comp]
        meta, off = {}, 0
        for p in cp:
            meta[f"{p[1]}#{p[2]}"] = {"dtype": "U8", "shape": [p[3]], "data_offsets": [off, off + p[3]]}
            off += p[3]
        hb = json.dumps(meta, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(os.path.join(d, f"{comp}.safetensors"), "wb") as f:
            f.write(len(hb).to_bytes(8, "little") + hb)
            for p in cp:
                f.write(part_bytes(p))


class Reader:
    """F1's Python reader: disk -> pinned slot -> GPU, each batch one slot and one copy. THREADS
    workers, two slots, two events and a stream each: a read overlaps the previous copy."""

    def __init__(self):
        n = 2 * THREADS
        self.host = torch.empty(n * SLOT + ALIGN, dtype=torch.uint8, pin_memory=True)
        base = -self.host.data_ptr() % ALIGN
        self.slots = [self.host[base + i * SLOT : base + (i + 1) * SLOT] for i in range(n)]
        self.views = [memoryview((ctypes.c_char * SLOT).from_address(s.data_ptr())).cast("B") for s in self.slots]
        self.streams = [torch.cuda.Stream() for _ in range(THREADS)]
        self.events = [torch.cuda.Event() for _ in range(n)]

    def run(self, spans, direct):
        """spans: [(path, offset, length)], whole chunks. Returns bytes read."""
        batches = [(path, off, min(SLOT, n - off)) for path, _, n in spans for off in range(0, n, SLOT)]
        offsets, total = [], 0
        for b in batches:
            offsets.append(total)
            total += -(-b[2] // 256) * 256
        dst = torch.empty(max(total, 1), dtype=torch.uint8, device="cuda")
        work, lock, errors = iter(zip(offsets, batches)), threading.Lock(), []
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)

        def worker(t):
            k = 0
            try:
                while True:
                    with lock:
                        job = next(work, None)
                    if job is None:
                        break
                    dst_off, (path, off, n) = job
                    i = 2 * t + k
                    self.events[i].synchronize()
                    fd = os.open(path, flags)
                    try:
                        os.preadv(fd, [self.views[i][: -(-n // ALIGN) * ALIGN]], off)
                    finally:
                        os.close(fd)
                    with torch.cuda.stream(self.streams[t]):
                        dst[dst_off : dst_off + n].copy_(self.slots[i][:n], non_blocking=True)
                        self.events[i].record()
                    k ^= 1
            except BaseException as e:
                errors.append(e)

        ts = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        torch.cuda.synchronize()
        if errors:
            raise errors[0]
        return total


class Engine:
    """fastsafetensors' nogds reader (C++, 16 threads, 16 MiB pinned bounce buffers) over any file ranges."""

    def __init__(self):
        fst_nogds.load_library_func(get_framework_op("pytorch"))
        self.reader = fstcpp.nogds_file_reader(False, 16 << 10, 16, True, 0, -1)

    def run(self, spans, direct):
        offsets, total = [], 0
        for _, _, n in spans:
            offsets.append(total)
            total += -(-n // 256) * 256
        dst = torch.empty(max(total, 1), dtype=torch.uint8, device="cuda")
        buf = fstcpp.gds_device_buffer(dst.data_ptr(), dst.numel(), True)
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        reqs = []
        for (path, off, n), o in zip(spans, offsets):
            fd = os.open(path, flags)
            reqs.append((self.reader.submit_read(fd, buf, off, n, o), fd))
        failed = 0
        for req, fd in reqs:
            failed += req < 0 or self.reader.wait_read(req) == 0
            os.close(fd)
        torch.cuda.synchronize()
        if failed:
            raise RuntimeError(f"{failed} engine reads failed")
        return sum(n for _, _, n in spans)


class Model:
    def __init__(self, name, root, work):
        manifest, self.wcomp, self.wprefix, self.reps = LANES[name]
        self.name, self.dir = name, os.path.join(work, name)
        self.parts = v1_parts(root, manifest, self.wcomp, self.wprefix)
        self.bytes = sum(p[3] for p in self.parts)
        self.window_bytes = sum(p[3] for p in self.parts if p[6])
        self.store = os.path.join(self.dir, "v2")

    def build(self, log):
        done = os.path.join(self.dir, "DONE")
        if not os.path.exists(done):
            shutil.rmtree(self.dir, ignore_errors=True)
            t0 = time.perf_counter()
            digest = write_v2(self.parts, self.store)
            log(f"{self.name}: v2 checkpoint {digest} written in {time.perf_counter() - t0:.0f} s")
            build_safetensors(self.parts, os.path.join(self.dir, "st"))
            os.sync()
            open(done, "w").write(digest)
        self.manifest = open(done).read()
        ck = self.checkpoint()
        self.chunks = ck.chunks
        self.st = [os.path.join(self.dir, "st", f) for f in sorted(os.listdir(os.path.join(self.dir, "st")))]

    def checkpoint(self):
        return v2.Store(self.store).checkpoint(f"sha256:{self.manifest}")

    def in_window(self, comp, key):
        return comp == self.wcomp and key.startswith(self.wprefix)

    def window_chunks(self, ck):
        want = set()
        for comp, ts in ck.tensors.items():
            for key, info in ts.items():
                if self.in_window(comp, key):
                    for part in info.parts.values():
                        c, left = part.chunk, part.offset + part.length
                        while left > 0:
                            want.add(c)
                            left -= self.chunks[c][1]
                            c += 1
        return sorted(want)

    def files(self, layout):
        return self.st if layout == "fastsafetensors" else [p for p, _ in self.chunks]

    def rust(self, plane, window):
        """Checkpoint, regions, register, want and wait: what a standalone consumer does."""
        ck = self.checkpoint()
        regions = v2.regions(ck, 64 << 20)
        ws = plane.register(ck, regions, varena.PageCache())
        picked = None
        if window:
            keep = set(self.window_chunks(ck))
            picked = [i for i, r in enumerate(regions) if keep & set(r)]
        plane.want(ws, 0, picked).wait()
        moved = sum(ws.regions[i]["nbytes"] for i in (picked if picked is not None else range(len(regions))))
        return moved, ws

    def spans(self, window):
        ck = self.checkpoint()
        chunks = self.window_chunks(ck) if window else range(len(self.chunks))
        return [(self.chunks[c][0], 0, self.chunks[c][1]) for c in chunks]

    def safetensors(self, window):
        keys = [f"{p[1]}#{p[2]}" for p in self.parts if p[6] or not window]
        loader = SafeTensorsFileLoader(None, "cuda:0", nogds=True)
        if window:
            keep = set(keys)
            loader.set_tensor_filter(lambda n: n in keep)
        loader.add_filenames({0: self.st})
        fb = loader.copy_files_to_device()
        out = [fb.get_tensor(k) for k in keys]
        fb.close()
        loader.close()
        torch.cuda.synchronize()
        return sum(t.numel() for t in out)

    def once(self, readers, layout, window, cold):
        ws = None
        t0 = time.perf_counter()
        if layout == "fastsafetensors":
            moved = self.safetensors(window)
        elif layout == "T64+rust":
            moved, ws = self.rust(readers["plane"], window)
        else:
            moved = readers[layout].run(self.spans(window), direct=cold)
        s = time.perf_counter() - t0
        if ws is not None:
            ws.close()
        torch.cuda.empty_cache()
        return s, moved


def drop(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def bench(model, readers, log, warm_full):
    res = {}
    for rep in range(model.reps):
        order = LAYOUTS[rep % len(LAYOUTS) :] + LAYOUTS[: rep % len(LAYOUTS)]
        for layout in order:
            files = model.files(layout)
            for scope, window in (("full", False), ("window", True)):
                for cold in (True, False):
                    if not cold and scope == "full" and not warm_full:
                        continue
                    if cold:
                        drop(files)
                    else:
                        model.once(readers, layout, window, cold=False)  # populate the page cache
                    s, moved = model.once(readers, layout, window, cold=cold)
                    cell = f"{layout} {scope} {'cold' if cold else 'warm'}"
                    res.setdefault(cell, []).append((s, moved))
                    log(f"{model.name} {cell}: {1000 * s:.0f} ms {moved / 1e6:.0f} MB")
            drop(files)
        log(f"{model.name}: rep {rep + 1}/{model.reps} done")
    return res


def summary(model, res):
    cells = sorted({k.split(" ", 1)[1] for k in res}, key=lambda c: ("window" in c, "warm" in c))
    lines = [f"{model.name}: {model.bytes / 1e9:.2f} GB in {len(model.chunks)} chunks, window {model.window_bytes / 1e6:.0f} MB; ms median (best)"]
    lines.append("reader | " + " | ".join(cells))
    med = lambda xs: statistics.median(x[0] for x in xs)
    for layout in LAYOUTS:
        row = []
        for cell in cells:
            xs = sorted(res.get(f"{layout} {cell}", []))
            row.append(f"{1000 * med(xs):.0f} ({1000 * xs[0][0]:.0f}) {xs[0][1] / 1e6:.0f}MB" if xs else "-")
        lines.append(f"{layout} | " + " | ".join(row))
    gate = {}
    for cell in cells:
        rust, fst = med(res[f"T64+rust {cell}"]), med(res[f"fastsafetensors {cell}"])
        engine = med(res[f"T64+fst {cell}"])
        gate[cell] = {"rust_ms": round(1000 * rust), "fastsafetensors_ms": round(1000 * fst), "engine_ms": round(1000 * engine),
                      "margin": round(fst / rust - 1, 3), "pass": rust <= fst}
        lines.append(f"gate {cell}: T64+rust {1000 * rust:.0f} ms vs fastsafetensors {1000 * fst:.0f} ms ({100 * (fst / rust - 1):+.0f}%), "
                     f"vs its engine on the same chunks {1000 * engine:.0f} ms ({100 * (engine / rust - 1):+.0f}%) -> {'PASS' if rust <= fst else 'FAIL'}")
    return lines, gate


def environment(root, work):
    mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith(("MemTotal", "MemAvailable"))}
    limit = None  # the container's memory (page cache included) is what a warm read can keep
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.exists(p) and open(p).read().strip().isdigit():
            limit = int(open(p).read())
            break
    return {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "cpus": len(os.sched_getaffinity(0)),
            "mem_total": mem["MemTotal"], "mem_available": min(mem["MemAvailable"], limit or mem["MemTotal"]),
            "store": root, "work": work}


def main(work, out_path, build_only=False, plane=None):
    root = store_root()
    os.makedirs(work, exist_ok=True)
    env = environment(root, work)

    def log(msg):
        print(msg, flush=True)

    log(json.dumps(env))
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]  # the engine holds every file open
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard if 0 <= hard < 1 << 20 else 1 << 20, hard))
    readers = {"T64": Reader(), "T64+fst": Engine(), "plane": varena.Plane(devices=[0], **(plane or {}))}
    env["plane"] = plane or {}
    out = {"env": env, "models": {}}
    for name in LANES:
        model = Model(name, root, work)
        model.build(log)
        if build_only:
            out["models"][name] = {"bytes": model.bytes, "manifest": model.manifest, "store": model.store, "table": [], "gate": {}}
            json.dump(out, open(out_path, "w"))
            continue
        warm_full = env["mem_available"] > 1.3 * model.bytes + 2 * SLOT * THREADS
        res = bench(model, readers, log, warm_full)
        lines, gate = summary(model, res)
        for line in lines:
            log(line)
        out["models"][name] = {"bytes": model.bytes, "chunks": len(model.chunks), "window_bytes": model.window_bytes,
                               "manifest": model.manifest, "store": model.store,
                               "results": {k: [[round(s, 4), b] for s, b in v] for k, v in res.items()}, "table": lines, "gate": gate}
        json.dump(out, open(out_path, "w"))
    return out


if __name__ == "__main__":
    opts = sys.argv[3:]
    plane = next((dict((k, int(v)) for k, v in (kv.split("=") for kv in o.removeprefix("--plane=").split(","))) for o in opts if o.startswith("--plane=")), None)
    main(sys.argv[1], sys.argv[2], build_only="--build-only" in opts, plane=plane)
