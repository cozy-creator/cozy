"""F1/R4 serialized-byte loading: public TensorFS against stock safetensors readers.

    python speedgate.py WORK RESULT.json --sources=SOURCES.json

The same real INT8 parts become U8 tensors in both formats, outside the clock. Measure
first open through completed GPU transfer, five rotated repetitions of full/window ×
cold/warm. Preparation, engine initialization, decoding and inference are not timed.
The supplied checkpoint refs must already be local; this script never downloads weights.
"""

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import sys
import time

import fastsafetensors.copier.nogds as fst_nogds
import tensorfs
import torch
from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors.frameworks import get_framework_op
from safetensors import safe_open
from tensorfs import varena

# fastsafetensors dlopens libcudart.so; Torch wheels carry the versioned filename.
_cudart = sorted(glob.glob(str(Path(torch.__file__).parent / ".." / "nvidia" / "**" / "libcudart.so*"), recursive=True))
if _cudart:
    fst_nogds.resolve_runtime_lib_name = lambda framework=None: os.path.realpath(_cudart[0])

REPS = 5
LANES = {
    "sdxl-int8": {"component": "unet", "prefix": "up_blocks.1.",
                  "legacy_manifest": "3eb5674846e82e5b8d3e20d73f083a6d4aaa57ea699425287d5669906cfe6064"},
    "qwen-int8": {"component": "transformer", "prefix": "transformer_blocks.30.",
                  "legacy_manifest": "e7c79ee942b892d76e778fb873fbdce55deaadb1fe669fca6e7851980f747493"},
}
LAYOUTS = ("T64+rust", "fastsafetensors", "st-cuda", "st-cpu-cuda")
SAFETENSORS = LAYOUTS[1:]
SCOPES = ("full", "window")
CELLS = ("full cold", "full warm", "window cold", "window warm")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def natkey(key):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key)]


def safetensors_key(component, key, role):
    return f"{component}/{key}#{role}"


def copy_rows():
    """Drain existing shared-engine observations; this does not initiate a probe."""
    names = ("kind", "gpu", "bytes", "ns", "ceiling", "link", "busy_ns", "pin_ns", "staged", "cached")
    return [{name: getattr(row, name) for name in names} for row in varena.copies()]


def transfer_metrics(rows):
    h2d = [row for row in rows if row["kind"] == "h2d"]
    measured = bool(h2d) and all(row["ceiling"] > 0 and row["ns"] > 0 and row["link"] for row in h2d)
    nbytes, ns = sum(row["bytes"] for row in h2d), sum(row["ns"] for row in h2d)
    # Byte/ns is decimal GB/s. Aggregate each batch's measured wire-time denominator.
    wire_ns = sum(row["bytes"] / row["ceiling"] for row in h2d) if measured else None
    return {"records": rows, "h2d_gbps": nbytes / ns if ns else None,
            "pinned_ceiling_ratio": wire_ns / ns if measured else None, "measured": measured}


class Model:
    def __init__(self, name, spec, cache_dir, work):
        self.name, self.spec, self.cache_dir = name, spec, cache_dir
        self.dir = Path(work) / name
        self.store = self.dir / "tensorfs"
        self.wcomp, self.wprefix = LANES[name]["component"], LANES[name]["prefix"]
        ref = spec["checkpoint"]
        if re.fullmatch(r"sha256:[0-9a-f]{64}", ref) is None:
            raise ValueError(f"{name}: checkpoint must be an immutable local header ref")
        self.source = tensorfs.open(ref, cache_dir=cache_dir, local_files_only=True)
        if self.source.id != ref:
            raise ValueError(f"{name}: checkpoint is not the supplied header ref")
        self.parts = []
        for component, held in self.source.components.items():
            for key in sorted(held.tensors, key=natkey):
                for role, part in held.tensors[key].parts.items():
                    self.parts.append((component, key, role, part.length, part.dtype, list(part.shape)))
        self.bytes = sum(part[3] for part in self.parts)
        self.window_bytes = sum(part[3] for part in self.parts if self.in_window(part[0], part[1]))
        if not self.bytes or not self.window_bytes:
            raise ValueError(f"{name}: full payload and declared F1 window must both exist")
        names = [safetensors_key(component, key, role) for component, key, role, *_ in self.parts]
        if len(names) != len(set(names)):
            raise ValueError(f"{name}: safetensors' shared key namespace has duplicate part names")
        self.geometry = hashlib.sha256(canonical(self.parts)).hexdigest()

    def in_window(self, component, key):
        return component == self.wcomp and key.startswith(self.wprefix)

    def checkpoint(self):
        return tensorfs.open(self.header, cache_dir=self.store, local_files_only=True)

    def build(self, log):
        """Both formats receive the same public host-read buffer for each stored part."""
        marker = self.dir / "DONE.json"
        identity = {"format": "tensorfs.speed-fixture/3", "source": self.source.id,
                    "geometry": self.geometry, "extension_sha256": digest(tensorfs._ext.__file__)}
        if marker.exists():
            record = json.loads(marker.read_bytes())
            if record["identity"] != identity:
                raise ValueError(f"{self.dir}: existing fixture belongs to another source or wheel")
            self.header, self.part_hashes = record["header"], record["part_hashes"]
        else:
            # Never erase an earlier benchmark or another owner's files to rebuild a fixture.
            self.dir.mkdir(parents=True, exist_ok=False)
            st_dir = self.dir / "st"
            st_dir.mkdir()
            started = time.perf_counter()
            writer = tensorfs.Writer(cache_dir=self.store)
            self.part_hashes = []
            for component in dict.fromkeys(part[0] for part in self.parts):
                parts = [part for part in self.parts if part[0] == component]
                meta, offset = {}, 0
                for _, key, role, length, *_ in parts:
                    meta[safetensors_key(component, key, role)] = {
                        "dtype": "U8", "shape": [length],
                        "data_offsets": [offset, offset + length],
                    }
                    offset += length
                header = json.dumps(meta, separators=(",", ":")).encode()
                header += b" " * (-len(header) % 8)
                target = writer.component(component, cls="", config={})
                with open(st_dir / f"{component}.safetensors", "xb") as stream:
                    stream.write(len(header).to_bytes(8, "little") + header)
                    for _, key, role, length, *_ in parts:
                        data = bytearray(length)
                        self.source.components[component].tensors.read_into(key, data, role=role)
                        target.put(f"{key}#{role}", data, dtype="U8", shape=[length])
                        stream.write(data)
                        self.part_hashes.append((component, key, role, hashlib.sha256(data).hexdigest()))
            committed = writer.commit()
            self.header = committed.id
            committed.close()
            os.sync()
            marker.write_bytes(canonical({"identity": identity, "header": self.header,
                                          "part_hashes": self.part_hashes}))
            log(f"{self.name}: matched fixtures written in {time.perf_counter() - started:.1f}s")
        with self.checkpoint() as ck:
            self.chunks = list(ck.chunks)
            self.keys = {component: list(held.tensors) for component, held in ck.components.items()}
        self.st = sorted((self.dir / "st").glob("*.safetensors"))

    def selected(self, window):
        return {component: [key for key in keys if not window or self.in_window(component, key)]
                for component, keys in self.keys.items()}

    def drop(self, layout):
        """F1 cold semantics: request payload eviction, leaving metadata caches alone."""
        if layout == "T64+rust":
            with self.checkpoint() as ck:
                for index in range(len(ck.chunks)):
                    fd = ck.open_chunk(index)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
        else:
            for path in self.st:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)

    def public(self, window):
        started = time.perf_counter()
        ck = self.checkpoint()
        # Select before read_all allocates a destination or reads a tensor.
        selected = self.selected(window)
        opened = time.perf_counter() - started
        arrays = {component: ck.components[component].tensors.read_all(keys, device="cuda:0", decode=False)
                  for component, keys in selected.items() if keys}
        torch.cuda.synchronize()
        moved = sum(array.nbytes for tensors in arrays.values() for array in tensors.values())
        return moved, opened, (arrays, ck)

    def safetensors(self, layout, window):
        keys = [safetensors_key(component, key, role) for component, key, role, *_ in self.parts
                if not window or self.in_window(component, key)]
        started, opened = time.perf_counter(), 0.0
        if layout == "fastsafetensors":
            loader = SafeTensorsFileLoader(None, "cuda:0", nogds=True)
            if window:
                keep = set(keys)
                loader.set_tensor_filter(lambda name: name in keep)
            loader.add_filenames({0: [str(path) for path in self.st]})
            opened = time.perf_counter() - started
            files = loader.copy_files_to_device()
            arrays = [files.get_tensor(key) for key in keys]
            cleanup = (files, loader)
        else:
            arrays, cleanup = [], ()
            for path in self.st:
                component = path.stem
                names = [safetensors_key(comp, key, role) for comp, key, role, *_ in self.parts
                         if comp == component and (not window or self.in_window(comp, key))]
                started = time.perf_counter()
                with safe_open(path, framework="pt", device="cuda:0" if layout == "st-cuda" else "cpu") as source:
                    opened += time.perf_counter() - started
                    arrays.extend(source.get_tensor(key) if layout == "st-cuda"
                                  else source.get_tensor(key).to("cuda:0") for key in names)
        torch.cuda.synchronize()
        return sum(array.numel() for array in arrays), opened, (arrays, cleanup)

    def once(self, layout, window, collect=True):
        copy_rows()
        started = time.perf_counter()
        moved, opened, keep = self.public(window) if layout == "T64+rust" else self.safetensors(layout, window)
        seconds = time.perf_counter() - started
        rows = copy_rows()
        expected = self.window_bytes if window else self.bytes
        if moved != expected:
            raise ValueError(f"{self.name}/{layout}: moved {moved} bytes, expected {expected}")
        # Release after the first-open -> transfer-complete interval for every arm.
        if layout == "T64+rust":
            arrays, ck = keep
            arrays.clear()
            ck.close()
        else:
            arrays, cleanup = keep
            arrays.clear()
            for owner in cleanup:
                owner.close()
        torch.cuda.empty_cache()
        return {"seconds": seconds, "bytes": moved, "open_seconds": opened,
                "engine": transfer_metrics(rows) if collect and layout == "T64+rust" else None}

    def provenance(self):
        return {"source": self.spec, "header": self.header, "geometry_sha256": self.geometry,
                "bytes": self.bytes, "window_bytes": self.window_bytes, "chunks": len(self.chunks),
                "window": {"component": self.wcomp, "prefix": self.wprefix},
                "part_sha256": self.part_hashes, "legacy_manifest": LANES[self.name]["legacy_manifest"]}


def summary(results):
    gate = {}
    for cell in CELLS:
        own = results.get(f"T64+rust {cell}", [])
        controls = {layout: results.get(f"{layout} {cell}", []) for layout in SAFETENSORS}
        complete = len(own) == REPS and all(len(rows) == REPS for rows in controls.values())
        if not complete:
            gate[cell] = {"complete": False, "pass": False, "reason": "missing reader/repetition"}
            continue
        med = lambda rows: statistics.median(row["seconds"] for row in rows)
        best = min(controls, key=lambda layout: med(controls[layout]))
        seconds, baseline = med(own), med(controls[best])
        gate[cell] = {"complete": True, "pass": seconds <= baseline,
                      "rust_ms": round(1000 * seconds, 3), "safetensors": best,
                      "safetensors_ms": round(1000 * baseline, 3), "margin": baseline / seconds - 1,
                      "rust_open_ms": 1000 * statistics.median(row["open_seconds"] for row in own),
                      "rust_read_ms": 1000 * statistics.median(row["seconds"] - row["open_seconds"] for row in own)}
    warm = results.get("T64+rust full warm", [])
    link_measured = len(warm) == REPS and all(row["engine"]["measured"] for row in warm)
    ratios = [row["engine"]["pinned_ceiling_ratio"] for row in warm if row["engine"]["measured"]]
    return {"cells": gate, "link_measured": link_measured,
            "full_warm_pinned_ceiling_ratio_median": statistics.median(ratios) if ratios else None,
            "link_threshold": None, "parity_pass": all(cell["pass"] for cell in gate.values()),
            "r4_link_review_required": True,
            "note": "F1 parity only. Link ratios need parent review; no numeric threshold is invented."}


def environment(cache_dir, work):
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
           if line.startswith(("MemTotal", "MemAvailable"))}
    limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(path).exists() and Path(path).read_text().strip().isdigit():
            limit = int(Path(path).read_text())
            break
    identity = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    volume = subprocess.run(["df", "-T", str(work)], capture_output=True, text=True, check=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(0), "gpu_identity": identity, "volume": volume,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_ordinal": 0,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "versions": {name: importlib.metadata.version(name) for name in ("tensorfs", "safetensors", "fastsafetensors")},
            "extension_sha256": digest(tensorfs._ext.__file__), "harness_sha256": digest(__file__),
            "cpus": len(os.sched_getaffinity(0)), "mem_total": mem["MemTotal"],
            "mem_available": min(mem["MemAvailable"], limit or mem["MemTotal"]),
            "cache_dir": str(cache_dir), "work": str(work), "repetitions": REPS,
            "normalization": "identical stored INT8 parts as plain U8 tensors; no dequantization",
            "cold": "POSIX_FADV_DONTNEED requested for payload fds; metadata is not evicted",
            "warm": "one untimed read immediately before each measured read",
            "engine_options": "public API defaults of the recorded extension; no Plane tuning"}


def main(args):
    plan = json.loads(Path(args.sources).read_bytes())
    if set(plan["models"]) != set(LANES):
        raise ValueError("sources must pin exactly sdxl-int8 and qwen-int8")
    if importlib.metadata.version("fastsafetensors") != "0.5.0" or importlib.metadata.version("safetensors") != "0.8.0":
        raise ValueError("F1 controls require fastsafetensors0.5.0 and safetensors0.8.0")
    cache_dir, work = Path(plan["cache_dir"]).expanduser(), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = {"format": "tensorfs.speed-proof/2", "sources": plan, "models": {},
           "parity_pass": False, "link_measurements_complete": False,
           "r4_link_review_required": True, "r4_lifecycle_review_required": True}
    # A failed or partial rerun must never leave an earlier successful result at this path.
    with open(args.result, "xb") as result:
        result.write(canonical(out))
    models = [Model(name, plan["models"][name], cache_dir, work) for name in LANES]
    for model in models:
        model.build(lambda message: print(message, flush=True))
    if args.build_only:
        out["status"] = "fixtures_only_unmeasured"
        out["models"] = {model.name: model.provenance() for model in models}
        Path(args.result).write_bytes(canonical(out))
        return 0
    torch.cuda.init()
    fst_nogds.load_library_func(get_framework_op("pytorch"))
    env = environment(cache_dir, work)
    out["env"] = env
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard if 0 <= hard < 1 << 20 else 1 << 20, hard))
    for model in models:
        if env["mem_available"] <= 1.3 * model.bytes + (512 << 20):
            raise ValueError(f"{model.name}: insufficient page-cache headroom for required full-warm cell")
    # Public reads initialize the existing shared engine/probe outside every timed cell.
    out["initialization"] = {model.name: model.once("T64+rust", True)["engine"] for model in models}
    for model in models:
        results = {}
        for rep in range(REPS):
            order = args.readers[rep % len(args.readers):] + args.readers[:rep % len(args.readers)]
            for layout in order:
                for scope in args.cells:
                    for cold in (True, False):
                        window = scope == "window"
                        if cold:
                            model.drop(layout)
                        else:
                            model.once(layout, window, collect=False)
                        row = model.once(layout, window)
                        cell = f"{layout} {scope} {'cold' if cold else 'warm'}"
                        results.setdefault(cell, []).append(row)
                        print(f"{model.name} rep{rep + 1} {cell}: {1000 * row['seconds']:.3f}ms {row['bytes']}B", flush=True)
                model.drop(layout)
        out["models"][model.name] = {**model.provenance(), "results": results, "gate": summary(results)}
        Path(args.result).write_bytes(canonical(out))
    out["parity_pass"] = all(row["gate"]["parity_pass"] for row in out["models"].values())
    out["link_measurements_complete"] = all(row["gate"]["link_measured"] for row in out["models"].values())
    Path(args.result).write_bytes(canonical(out))
    print(json.dumps({name: row["gate"] for name, row in out["models"].items()}, indent=2), flush=True)
    # Success supplies F1 parity and complete observations, never an R4 complete verdict.
    return 0 if out["parity_pass"] and out["link_measurements_complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work")
    parser.add_argument("result")
    parser.add_argument("--sources", required=True, help="JSON cache_dir + models with immutable, already-local checkpoint refs")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--readers", type=lambda value: tuple(value.split(",")), default=LAYOUTS)
    parser.add_argument("--cells", type=lambda value: tuple(value.split(",")), default=SCOPES)
    args = parser.parse_args()
    if not args.readers or len(set(args.readers)) != len(args.readers) or set(args.readers) - set(LAYOUTS):
        parser.error(f"readers must be distinct members of {LAYOUTS}")
    if not args.cells or len(set(args.cells)) != len(args.cells) or set(args.cells) - set(SCOPES):
        parser.error(f"cells must be distinct members of {SCOPES}")
    sys.exit(main(args))
