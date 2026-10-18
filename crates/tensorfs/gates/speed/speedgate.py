# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.1", "torch==2.14.0", "safetensors==0.8.0", "fastsafetensors==0.5.0", "cbor2", "numpy"]
# ///
"""xs-045 F1 speed gate: cozytensors/2 chunk layouts against safetensors, disk -> pinned -> GPU.

    cozy model download fidika/sdxl@1.0.0/int8 --on <machine>
    cozy model download fidika/qwen-image-2.1@0.1.0/int8 --on <machine>
    cozy run crates/tensorfs/gates/speed/speedgate.py --on <machine>

The machine's v1 store supplies the bytes. Each layout is written beside it from those bytes, then
every (layout, cell) is timed in interleaved repetitions: cold (pages dropped) and warm, the whole
checkpoint and one paging window, from first open to the last byte resident on the GPU.
"""

import ctypes
import glob
import json
import os
import re
import resource
import shutil
import statistics
import threading
import time

import cbor2
import fastsafetensors.copier.nogds as fst_nogds
import torch
from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import cpp as fstcpp
from fastsafetensors.frameworks import get_framework_op
from safetensors import safe_open

# fastsafetensors dlopens an unversioned libcudart.so; torch's wheels ship only the versioned one.
_cudart = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "**", "libcudart.so*"), recursive=True))
if _cudart:
    fst_nogds.resolve_runtime_lib_name = lambda framework=None: os.path.realpath(_cudart[0])

MiB = 1 << 20
ALIGN = 4 << 10  # a part's offset in its chunk
LARGE = 16 * MiB  # parts this size or more stand alone, split on a 64 MiB grid (v1's segments)
# v2 layouts: pack target and whether packs close at a block boundary once 16 MiB or more (design §2).
# T32 is not built: block closing makes it T64 (SDXL int8 identical, Qwen int8 8 of 544 chunks differ).
V2 = {"T16": (16 * MiB, True), "T64": (64 * MiB, True), "T64-open": (64 * MiB, False)}
SLOT = 16 * MiB  # one read request and its pinned staging buffer, whatever the object size
THREADS = 16  # fastsafetensors' defaults: 16 threads, 16 MiB buffers
LANES = {  # v1 manifest, window component, window key prefix, repetitions
    "sdxl-int8": ("3eb5674846e82e5b8d3e20d73f083a6d4aaa57ea699425287d5669906cfe6064", "unet", "up_blocks.1.", 5),
    "qwen-int8": ("e7c79ee942b892d76e778fb873fbdce55deaadb1fe669fca6e7851980f747493", "transformer", "transformer_blocks.30.", 5),
}
# `+fst`: the layout read by fastsafetensors' own C++ engine, so it differs from `fastsafetensors` in layout only.
LAYOUTS = ("v1", *V2, "contiguous", "v1+fst", "T64+fst", "st-cuda", "st-cpu-cuda", "fastsafetensors")
SAFETENSORS = ("st-cuda", "st-cpu-cuda", "fastsafetensors")
OPTIONAL = ("v1+fst", "T64+fst", "fastsafetensors")  # need fastsafetensors to load here


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def block(key):
    """The deepest numbered key prefix: `transformer_blocks.12`."""
    p = key.split(".")
    i = [k for k, x in enumerate(p) if x.isdigit()]
    return ".".join(p[: i[-1] + 1]) if i else ""


def store_root():
    for c in (os.environ.get("COZY_TENSORFS_ROOT"), "/var/lib/tensorfs", os.path.expanduser("~/.tensorfs")):
        if c and os.path.isdir(os.path.join(c, "manifests")):
            return c
    raise RuntimeError("no TensorFS store with manifests/ on this machine")


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
    if p[4] is not None:
        yield p[4]
    for path, _ in p[5]:
        with open(path, "rb") as f:
            while c := f.read(16 * MiB):
                yield c


def build_v2(parts, target, closing, d):
    """The design §2 writer rule; writes the chunks and header.json, returns the object sizes."""
    os.makedirs(d, exist_ok=True)
    chunks, tensors = [], {}
    cur = None  # [file, chunk index, length]
    cur_comp = cur_block = None

    def close():
        nonlocal cur
        if cur:
            cur[0].close()
            chunks[cur[1]] = cur[2]
            cur = None

    for p in parts:
        comp, key, role, n = p[:4]
        entry = tensors.setdefault(comp, {}).setdefault(key, {})
        if n >= LARGE:
            entry[role] = {"chunk": len(chunks), "offset": 0, "length": n}
            for src, length in p[5]:
                shutil.copyfile(src, os.path.join(d, f"{len(chunks):05d}"))
                chunks.append(length)
            continue
        blk = (comp, block(key))
        if cur and (comp != cur_comp or (closing and blk != cur_block and cur[2] >= LARGE)):
            close()
        start = -(-cur[2] // ALIGN) * ALIGN if cur else 0
        if cur and start + n > target:
            close()
            start = 0
        if cur is None:
            cur = [open(os.path.join(d, f"{len(chunks):05d}"), "wb"), len(chunks), 0]
            chunks.append(0)
        cur[0].write(b"\0" * (start - cur[2]))
        for c in part_bytes(p):
            cur[0].write(c)
        entry[role] = {"chunk": cur[1], "offset": start, "length": n}
        cur[2] = start + n
        cur_comp, cur_block = comp, blk
    close()
    header = {"format": "cozytensors/2", "chunks": [{"length": n} for n in chunks],
              "components": {c: {"tensors": {k: {"parts": r} for k, r in ts.items()}} for c, ts in tensors.items()}}
    with open(os.path.join(d, "header.json"), "w") as f:
        json.dump(header, f, separators=(",", ":"), sort_keys=True)
    return chunks


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
                for c in part_bytes(p):
                    f.write(c)


class Reader:
    """Disk -> pinned slot -> GPU: each batch fills one pinned slot and moves in one copy.
    THREADS workers, two slots, two events and a stream each: a read overlaps the previous copy."""

    def __init__(self):
        n = 2 * THREADS
        self.host = torch.empty(n * SLOT + ALIGN, dtype=torch.uint8, pin_memory=True)
        base = -self.host.data_ptr() % ALIGN
        self.slots = [self.host[base + i * SLOT : base + (i + 1) * SLOT] for i in range(n)]
        self.views = [memoryview((ctypes.c_char * SLOT).from_address(s.data_ptr())).cast("B") for s in self.slots]
        self.streams = [torch.cuda.Stream() for _ in range(THREADS)]
        self.events = [torch.cuda.Event() for _ in range(n)]

    def run(self, batches, direct):
        """batches: [(copy_start, copy_length, [(path, offset, length, slot_offset)])]. Returns bytes read."""
        offsets, total = [], 0
        for b in batches:
            offsets.append(total)
            total += -(-b[1] // 256) * 256
        dst = torch.empty(max(total, 1), dtype=torch.uint8, device="cuda")
        work, lock, read, errors = iter(zip(offsets, batches)), threading.Lock(), [0] * THREADS, []
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)

        def worker(t):
            k = 0
            try:
                while True:
                    with lock:
                        job = next(work, None)
                    if job is None:
                        break
                    dst_off, (start, length, reads) = job
                    i = 2 * t + k
                    self.events[i].synchronize()
                    for path, off, n, slot_off in reads:
                        a = off - off % ALIGN
                        want = -(-(off + n - a) // ALIGN) * ALIGN
                        fd = os.open(path, flags)
                        try:
                            while want > 0:
                                got = os.preadv(fd, [self.views[i][slot_off : slot_off + want]], a)
                                read[t] += got
                                if got == want or got == 0 or got % ALIGN:  # whole, or the end of the file
                                    break
                                a, slot_off, want = a + got, slot_off + got, want - got
                        finally:
                            os.close(fd)
                    with torch.cuda.stream(self.streams[t]):
                        dst[dst_off : dst_off + length].copy_(self.slots[i][start : start + length], non_blocking=True)
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
        return sum(read)


class Engine:
    """fastsafetensors' nogds reader (C++, 16 threads, 16 MiB pinned bounce buffers) over any file ranges."""

    def __init__(self):
        fst_nogds.load_library_func(get_framework_op("pytorch"))
        self.reader = fstcpp.nogds_file_reader(False, 16 << 10, 16, True, 0, -1)

    def run(self, batches, direct):
        spans = merge3([(path, off, n) for _, _, reads in batches for path, off, n, _ in reads])
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


def merge3(spans):
    """Adjacent ranges of one file as one."""
    out = []
    for p, o, n in spans:
        if out and out[-1][0] == p and out[-1][1] + out[-1][2] == o:
            out[-1] = (p, out[-1][1], out[-1][2] + n)
        else:
            out.append((p, o, n))
    return out


def pack(objects):
    """Batches of small objects [(path, length)], each at a 4 KiB slot offset; a large one is ranges."""
    batches, reads, used = [], [], 0
    for path, n in objects + [(None, SLOT)]:
        if reads and (used + -(-n // ALIGN) * ALIGN > SLOT):
            batches.append((0, reads[-1][3] + reads[-1][2], reads))
            reads, used = [], 0
        if path is None:
            break
        if n >= SLOT:
            batches += ranges(path, [(0, n)])
            continue
        reads.append((path, 0, n, used))
        used += -(-n // ALIGN) * ALIGN
    return batches


def ranges(path, spans):
    """Batches over byte ranges of one file, one slot each; after the first, reads are aligned."""
    out = []
    for off, n in spans:
        while n > 0:
            take = min(n, SLOT - off % ALIGN)
            out.append((off % ALIGN, take, [(path, off, take, 0)]))
            off, n = off + take, n - take
    return out


def merge(spans):
    out = []
    for o, n in spans:
        if out and out[-1][0] + out[-1][1] == o:
            out[-1] = (out[-1][0], out[-1][1] + n)
        else:
            out.append((o, n))
    return out


class Model:
    def __init__(self, name, root, work):
        manifest, self.wcomp, wprefix, self.reps = LANES[name]
        self.name, self.dir = name, os.path.join(work, name)
        self.parts = v1_parts(root, manifest, self.wcomp, wprefix)
        self.bytes = sum(p[3] for p in self.parts)
        self.window_bytes = sum(p[3] for p in self.parts if p[6])
        self.sizes = {"v1": [n for p in self.parts for _, n in p[5]]}

    def build(self, log):
        need = 4 * self.bytes + (1 << 30)
        if not os.path.exists(os.path.join(self.dir, "DONE")):
            free = shutil.disk_usage(os.path.dirname(self.dir)).free
            if free < need:
                raise RuntimeError(f"{self.name}: {free / 1e9:.0f} GB free, layouts need {need / 1e9:.0f} GB")
            shutil.rmtree(self.dir, ignore_errors=True)
            for layout, (target, closing) in V2.items():
                t0 = time.perf_counter()
                build_v2(self.parts, target, closing, os.path.join(self.dir, layout))
                log(f"{self.name}: built {layout} in {time.perf_counter() - t0:.0f} s")
            build_safetensors(self.parts, os.path.join(self.dir, "st"))
            os.sync()
            open(os.path.join(self.dir, "DONE"), "w").close()
        for layout in V2:
            h = json.load(open(os.path.join(self.dir, layout, "header.json")))
            self.sizes[layout] = [c["length"] for c in h["chunks"]]
        self.st = []  # (path, data start, {name: (offset, length, in window)})
        for comp in dict.fromkeys(p[0] for p in self.parts):
            path = os.path.join(self.dir, "st", f"{comp}.safetensors")
            with open(path, "rb") as f:
                hl = int.from_bytes(f.read(8), "little")
                meta = json.loads(f.read(hl))
            win = {f"{p[1]}#{p[2]}" for p in self.parts if p[0] == comp and p[6]}
            self.st.append((path, 8 + hl, {k: (v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], k in win) for k, v in meta.items()}))

    def files(self, layout):
        layout = layout.removesuffix("+fst")
        if layout == "v1":
            return [path for p in self.parts for path, _ in p[5]]
        if layout.startswith("T"):
            d = os.path.join(self.dir, layout)
            return [os.path.join(d, x) for x in os.listdir(d)]
        return [s[0] for s in self.st]

    def plan(self, layout, window):
        """The read plan, made inside the timed region (v2's includes parsing its header)."""
        if layout == "v1":
            return pack([o for p in self.parts if p[6] or not window for o in p[5]])
        if layout.startswith("T"):
            d = os.path.join(self.dir, layout)
            h = json.load(open(os.path.join(d, "header.json")))
            lengths, want = [c["length"] for c in h["chunks"]], set()
            for comp, ts in h["components"].items():
                for key, t in ts["tensors"].items():
                    if window and (comp != self.wcomp or not key.startswith(LANES[self.name][2])):
                        continue
                    for r in t["parts"].values():
                        c, left = r["chunk"], r["offset"] + r["length"]
                        while left > 0:
                            want.add(c)
                            left -= lengths[c]
                            c += 1
            return [b for c in sorted(want) for b in ranges(os.path.join(d, f"{c:05d}"), [(0, lengths[c])])]
        return [b for path, start, ts in self.st for b in ranges(path, merge(sorted((start + o, n) for o, n, w in ts.values() if w or not window)))]

    def safetensors(self, layout, window):
        files = [(path, [k for k, (_, _, w) in ts.items() if w or not window]) for path, _, ts in self.st]
        files = [(path, names) for path, names in files if names]
        if layout == "fastsafetensors":  # one loader over every file, as a model load asks it
            loader = SafeTensorsFileLoader(None, "cuda:0", nogds=True)
            if window:
                keep = {k for _, names in files for k in names}
                loader.set_tensor_filter(lambda n: n in keep)
            loader.add_filenames({0: [path for path, _ in files]})
            fb = loader.copy_files_to_device()
            out = [fb.get_tensor(k) for _, names in files for k in names]
            fb.close()
            loader.close()
        else:
            out = []
            for path, names in files:
                with safe_open(path, framework="pt", device="cuda:0" if layout == "st-cuda" else "cpu") as f:
                    out += [f.get_tensor(k) if layout == "st-cuda" else f.get_tensor(k).to("cuda:0") for k in names]
        torch.cuda.synchronize()
        return sum(t.numel() for t in out)

    def once(self, readers, layout, window, cold):
        t0 = time.perf_counter()
        if layout in SAFETENSORS:
            moved = self.safetensors(layout, window)
        else:
            base, _, engine = layout.partition("+")
            moved = readers[bool(engine)].run(self.plan(base, window), direct=cold)
        s = time.perf_counter() - t0
        torch.cuda.empty_cache()
        return s, moved


def drop(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def bench(model, reader, log, warm_full):
    res, failed = {}, {}
    cells = [("full", False), ("window", True)]
    for rep in range(model.reps):
        order = LAYOUTS[rep % len(LAYOUTS) :] + LAYOUTS[: rep % len(LAYOUTS)]
        for layout in (l for l in order if l not in failed):
            files = model.files(layout)
            try:
                for scope, window in cells:
                    for cold in (True, False):
                        if not cold and scope == "full" and not warm_full:
                            continue
                        if cold:
                            drop(files)
                        else:
                            model.once(reader, layout, window, cold=False)  # populate the page cache
                        s, moved = model.once(reader, layout, window, cold=cold)
                        cell = f"{layout} {scope} {'cold' if cold else 'warm'}"
                        res.setdefault(cell, []).append((s, moved))
                        log(f"{model.name} {cell}: {1000 * s:.0f} ms {moved / 1e6:.0f} MB")
            except Exception as e:
                if layout not in OPTIONAL:
                    raise
                failed[layout] = repr(e)  # reported, not fatal
                log(f"{model.name}: {layout} failed: {e!r}")
                torch.cuda.empty_cache()
            drop(files)
        log(f"{model.name}: rep {rep + 1}/{model.reps} done")
    return res, failed


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


def summary(model, res):
    lines, gate = [], {}
    cells = sorted({k.split(" ", 1)[1] for k in res}, key=lambda c: ("window" in c, "warm" in c))
    lines.append(f"{model.name}: {model.bytes / 1e9:.2f} GB, window {model.window_bytes / 1e6:.0f} MB; ms median (best)")
    lines.append("layout (objects) | " + " | ".join(cells))
    for layout in LAYOUTS:
        objs = len(model.sizes.get(layout.removesuffix("+fst"), [])) or len(model.st)
        row = []
        for cell in cells:
            xs = res.get(f"{layout} {cell}")
            if not xs:
                row.append("-")
                continue
            s = sorted(x[0] for x in xs)
            row.append(f"{1000 * statistics.median(s):.0f} ({1000 * s[0]:.0f}) {xs[0][1] / 1e6:.0f}MB")
        lines.append(f"{layout} ({objs}) | " + " | ".join(row))
    for cell in cells:
        st = {l: statistics.median(x[0] for x in res[f"{l} {cell}"]) for l in SAFETENSORS if res.get(f"{l} {cell}")}
        v2 = {l: statistics.median(x[0] for x in res[f"{l} {cell}"]) for l in ("T64", "T64+fst") if res.get(f"{l} {cell}")}
        best, mine = min(st, key=st.get), min(v2, key=v2.get)
        t64, other = v2[mine], st[best]
        gate[cell] = {"v2": mine, "v2_ms": round(1000 * t64), "safetensors": best, "safetensors_ms": round(1000 * other),
                      "margin": round(other / t64 - 1, 3), "pass": t64 <= other}
        lines.append(f"gate {cell}: {mine} {1000 * t64:.0f} ms vs {best} {1000 * other:.0f} ms -> "
                     f"{'PASS' if t64 <= other else 'FAIL'} ({100 * (other / t64 - 1):+.0f}%)")
    return lines, gate


def main(ctx) -> dict:
    root = store_root()
    work = os.path.join(os.path.dirname(root.rstrip("/")), "f1-speedgate")
    os.makedirs(work, exist_ok=True)
    env = environment(root, work)

    def log(msg):
        print(msg, flush=True)
        ctx.log(msg)

    log(json.dumps(env))
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]  # the engine holds every v1 object open
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard if 0 <= hard < 1 << 20 else 1 << 20, hard))
    reader = (Reader(), Engine())
    out = {"env": env, "models": {}}
    for name in LANES:
        model = Model(name, root, work)
        model.build(log)
        warm_full = env["mem_available"] > 1.3 * model.bytes + 2 * SLOT * THREADS
        res, failed = bench(model, reader, log, warm_full)
        lines, gate = summary(model, res)
        for line in lines:
            log(line)
        out["models"][name] = {"bytes": model.bytes, "window_bytes": model.window_bytes, "sizes": model.sizes,
                               "results": {k: [[round(s, 4), b] for s, b in v] for k, v in res.items()},
                               "table": lines, "gate": gate, "failed": failed}
        ctx.raise_if_cancelled()
    return out
