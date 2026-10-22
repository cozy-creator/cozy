#!/usr/bin/env python3
"""Whole-H3 stored-byte load comparison; never acquires model data.

python h3_load.py --source source.json --work NEW_DIR --result NEW.json \
  --wheel TENSORFS.whl --wheel-sha256 HEX --native-source COMMIT

source.json pins source_label, legacy_manifest_sha256, header (local path),
header_sha256, cache_dir, tensor_count and stored_bytes. Paths are relative to
that JSON. Supply a staged CURRENT header, not the old five-field H3 header.
All five components are required. Preparation makes equivalent plain-U8
TensorFS/safetensors fixtures from every stored part. Five rotated repetitions
measure full host/GPU cold/warm loads, including allocation and completion.
Checksums, fixture preparation, engine initialization and cleanup are untimed.
This is one visible GPU at cuda:0. For the known H3 geometry the conservative
owned-output plus warm-source host guard is about156GB, derived at runtime.
This is a byte-loading benchmark, not decoding, inference or model loading.
Only stdlib imports occur until main explicitly loads the installed cohort.
"""

import argparse
from contextlib import ExitStack
from email.parser import BytesParser
import gc
import hashlib
import importlib
from importlib import machinery, metadata, util
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

COMPONENTS = ("audio_vae", "text_encoder", "transformer", "transformer_ref", "vae")
LEGACY_MANIFEST = "bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff"
SOURCE_LABEL = "paul/minimax-h3@1.0.0-rc.3/int8"
TENSOR_COUNT = 3858
REPS = 5
READERS = {"cpu": ("tensorfs", "safetensors-copy"),
           "cuda": ("tensorfs", "fastsafetensors", "safetensors-cuda", "safetensors-cpu-cuda")}
GIB = 1 << 30
INIT_OVERHEAD = 64 << 20
STEADY_OVERHEAD = 2 << 20
RECOVERY_SECONDS = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hex_digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def source_plan(path):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    require(set(plan) == {"source_label", "legacy_manifest_sha256", "header", "header_sha256",
                          "cache_dir", "tensor_count", "stored_bytes"}, "unexpected or missing source fields")
    require(plan["source_label"] == SOURCE_LABEL and plan["legacy_manifest_sha256"] == LEGACY_MANIFEST,
            "source must identify the pinned complete H3 INT8 checkpoint")
    require(hex_digest(plan["header_sha256"]), "header_sha256 must be a lowercase SHA256")
    require(type(plan["tensor_count"]) is int and plan["tensor_count"] == TENSOR_COUNT,
            "the complete H3 checkpoint must contain 3858 tensors")
    require(type(plan["stored_bytes"]) is int and plan["stored_bytes"] > 0, "stored_bytes must be positive")
    for name in ("header", "cache_dir"):
        plan[name] = str((path.parent / plan[name]).resolve(strict=True))
    require(Path(plan["header"]).is_file() and Path(plan["cache_dir"]).is_dir(), "source paths must be local")
    require(digest(plan["header"]) == plan["header_sha256"], "staged header SHA256 mismatch")
    return plan


def validate_geometry(parts, counts, expected_bytes):
    require(set(counts) == set(COMPONENTS) and all(counts[c] > 0 for c in COMPONENTS),
            "all five H3 components, and no extra components, are required")
    require(sum(counts.values()) == TENSOR_COUNT, "full H3 tensor count mismatch")
    require(parts and all(type(p["length"]) is int and p["length"] > 0 for p in parts), "empty stored part")
    require(sum(p["length"] for p in parts) == expected_bytes, "full stored-byte count mismatch")
    names = [(p["component"], p["key"], p["role"]) for p in parts]
    require(len(names) == len(set(names)), "duplicate stored-part identity")


def capacity(payload, largest_part, host_available, gpu_available, disk_available=None, part_count=0):
    # An owned destination plus the warm file working set and bounded verification scratch.
    host_need = 2 * payload + 2 * largest_part + 2 * GIB
    gpu_need = payload + max(2 * GIB, payload // 20)
    require(host_available >= host_need, f"insufficient host/cgroup memory: {host_available} < {host_need}")
    require(gpu_available >= gpu_need, f"insufficient GPU memory: {gpu_available} < {gpu_need}; no offload/window fallback")
    if disk_available is not None:
        disk_need = 2 * payload + 4096 * part_count + 2 * GIB
        require(disk_available >= disk_need, f"insufficient fixture disk space: {disk_available} < {disk_need}")
    return {"host_required": host_need, "gpu_required": gpu_need}


def device_namespace(visible_count, device):
    require(visible_count == 1 and device == "cuda:0",
            f"benchmark requires exactly one visible GPU and cuda:0; got {visible_count}, {device}")


def recover_memory(free_bytes, floor, *, clock=time.monotonic, pause=time.sleep):
    """Untimed, fixed-floor wait for asynchronous native frees; never relax the target."""
    start, samples = clock(), 0
    while True:
        free = free_bytes()
        samples += 1
        elapsed = clock() - start
        if free >= floor:
            return {"floor": floor, "settled_free": free, "seconds": elapsed,
                    "samples": samples, "deadline_seconds": RECOVERY_SECONDS}
        require(elapsed < RECOVERY_SECONDS,
                f"native cleanup did not recover GPU memory: free={free}, fixed floor={floor}, elapsed={elapsed}")
        pause(0.01)  # Release the GIL while the native cleanup worker returns buffers.


def schedule():
    for rep in range(REPS):
        for destination, readers in READERS.items():
            shift = rep % len(readers)
            for reader in readers[shift:] + readers[:shift]:
                for cold in (True, False):
                    yield rep, destination, reader, cold


def summarize(rows, payload):
    cells = {}
    for destination, readers in READERS.items():
        for cold in (True, False):
            selected = {reader: [r for r in rows if (r["destination"], r["reader"], r["cold"])
                                == (destination, reader, cold)] for reader in readers}
            complete = all(len(rs) == REPS and {r["rep"] for r in rs} == set(range(REPS))
                           and all(r["verified"] and r["bytes"] == payload and r["seconds"] > 0 for r in rs)
                           for rs in selected.values())
            result = {"complete": complete, "parity_pass": False}
            if complete:
                medians = {name: statistics.median(r["seconds"] for r in rs) for name, rs in selected.items()}
                best = min(readers[1:], key=medians.__getitem__)
                result.update(medians_seconds=medians, best_control=best,
                              parity_pass=medians["tensorfs"] <= medians[best])
            cells[f"{destination} {'cold' if cold else 'warm'}"] = result
    expected = len(list(schedule()))
    observed = [r for r in rows if r["reader"] == "tensorfs" and r["destination"] == "cuda"]
    observations_complete = len(observed) == 2 * REPS and all(
        r["metrics"]["measured"] and r["metrics"]["bytes"] >= payload for r in observed)
    return {"cells": cells, "complete": len(rows) == expected and all(c["complete"] for c in cells.values()),
            "parity_pass": len(rows) == expected and all(c["parity_pass"] for c in cells.values()),
            "observations_complete": observations_complete, "link_threshold": None,
            "r4_link_review_required": True, "r4_lifecycle_review_required": True}


def rate_metrics(nbytes, seconds, destination, ceiling, payload):
    """A TensorFS read's rate against TensorFS's measured ceiling (`tensorfs.ceiling`): GPU reads against pinned H2D,
    host reads against the volume. A raw read is no varena request, so it has no Stats; the ceiling is the measure."""
    kind = "disk" if destination == "cpu" else "h2d"
    limit = ceiling.get(f"{kind}_gbps") or 0
    gbps = nbytes / seconds / 1e9 if seconds > 0 else None
    measured = nbytes >= payload and gbps is not None and limit > 0
    return {"kind": kind, "measured": measured, "bytes": nbytes, "gbps": gbps, "ceiling_gbps": limit or None,
            "ceiling_ratio": gbps / limit if measured else None}


def installed_provenance(wheel, expected, native_source):
    require(hex_digest(expected) and hex_digest(native_source, 40), "exact wheel SHA256 and source commit required")
    require(digest(wheel) == expected, "TensorFS wheel SHA256 mismatch")
    dist = metadata.distribution("tensorfs")
    package = Path(dist.locate_file("tensorfs")).resolve()
    spec = util.find_spec("tensorfs")
    require(spec and Path(spec.origin).resolve() == package / "__init__.py", "TensorFS source overlay detected")
    installed = {}
    with zipfile.ZipFile(wheel) as archive:
        infos = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        require(len(infos) == 1, "wheel must contain one distribution")
        info = BytesParser().parsebytes(archive.read(infos[0]))
        require(info["Name"] == "tensorfs" and info["Version"] == dist.version, "installed TensorFS version mismatch")
        for name in archive.namelist():
            if not name.startswith("tensorfs/") or name.endswith("/"):
                continue
            relative = Path(name).relative_to("tensorfs")
            require(".." not in relative.parts, "unsafe wheel member")
            local = (package / relative).resolve()
            require(local.is_relative_to(package), "installed wheel member escapes package")
            with archive.open(name) as shipped:
                expected_file = hashlib.file_digest(shipped, "sha256").hexdigest()
            require(digest(local) == expected_file, f"installed file differs from wheel: {name}")
            installed[name] = expected_file
    native = [n for n in installed if any(n.endswith(s) for s in machinery.EXTENSION_SUFFIXES)]
    require(any(Path(n).name.startswith("_ext.") for n in native), "native TensorFS extension missing")
    stamp = re.search(r"\+dev\.g([0-9a-f]{8,40})(?:\.|$)", dist.version)
    require(not stamp or native_source.startswith(stamp[1]), "wheel source stamp mismatch")
    return {"wheel": str(Path(wheel).resolve()), "wheel_sha256": expected, "version": dist.version,
            "declared_native_source": native_source, "source_evidence": "caller build receipt; dev stamp checked when present",
            "installed_file_sha256": installed, "package": str(package)}


def load_cohort(torch_version):
    require(re.fullmatch(r"[0-9.]+\+cu[0-9]+", torch_version), "full CUDA Torch version required")
    versions = {name: metadata.version(name) for name in ("torch", "safetensors", "fastsafetensors")}
    require(versions == {"torch": torch_version, "safetensors": "0.8.0", "fastsafetensors": "0.5.0"},
            f"installed dependency mismatch: {versions}")
    torch = importlib.import_module("torch")
    tfs = importlib.import_module("tensorfs")
    importlib.import_module("tensorfs.raw")
    importlib.import_module("safetensors")
    importlib.import_module("fastsafetensors")
    require(isinstance(tfs._ext.__loader__, machinery.ExtensionFileLoader), "TensorFS is not a native extension")
    require(torch.cuda.is_available(), "CUDA required; no CPU fallback")
    return torch, tfs, versions


def memory_state():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    state = {"mem_available": available}
    # Resolve this process's v2 cgroup, falling back to the mounted namespace root.
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup")
            nested = root / line[3:].lstrip("/")
            root = nested if (nested / "memory.max").exists() else root
            limit = (root / "memory.max").read_text().strip()
            used = int((root / "memory.current").read_text())
            state.update(cgroup_path=str(root), cgroup_max=limit, cgroup_current=used)
            if limit != "max":
                available = min(available, max(0, int(limit) - used))
            break
    state["available"] = available
    return state


def physical_files(checkpoint, cache_dir):
    files = []
    for index, (sha, length) in enumerate(checkpoint.chunks):
        # The store is plain files: blobs/<aa>/<bb>/<sha256>.
        path = Path(cache_dir) / "blobs" / sha[:2] / sha[2:4] / sha
        stat = os.stat(path)
        require(stat.st_size == length, f"chunk {index} has wrong local length")
        files.append({"index": index, "sha256": sha, "length": length,
                      "path": str(path), "device": stat.st_dev, "inode": stat.st_ino})
    return files


class Fixture:
    def __init__(self, plan, work, tfs):
        self.tfs, self.work, self.ceiling = tfs, Path(work), {}
        self.source = tfs.open(plan["header"], cache_dir=plan["cache_dir"], local_files_only=True)
        try:
            require(self.source.id == "sha256:" + plan["header_sha256"], "opened header identity mismatch")
            report = self.source.verify()
            require(report.ok, f"source is not completely staged: missing={report.missing}, corrupt={report.corrupt}")
            self.parts, counts = [], {}
            for component, held in self.source.components.items():
                counts[component] = len(held.tensors)
                for key in sorted(held.tensors):
                    for role, part in tfs.raw.parts(self.source, component, key).items():
                        self.parts.append({"component": component, "key": key, "role": role,
                                           "dtype": part.dtype, "shape": list(part.shape), "length": part.length,
                                           "chunk": part.chunk, "offset": part.offset})
            validate_geometry(self.parts, counts, plan["stored_bytes"])
            for index, part in enumerate(self.parts):
                part["name"] = f"part{index:06d}"
                part["control_key"] = part["component"] + "/" + part["name"]
            self.counts, self.payload = counts, plan["stored_bytes"]
            self.largest = max(p["length"] for p in self.parts)
            self.source_files = physical_files(self.source, plan["cache_dir"])
            self.header, self.cache = self.work / "normalized.cozytensors", self.work / "tensorfs"
            self.st_files = [self.work / f"{c}.safetensors" for c in COMPONENTS]
        except BaseException:
            self.source.close()
            raise

    def build(self):
        self.work.mkdir(exist_ok=False)
        writer = self.tfs.Writer(self.header, cache_dir=self.cache)
        try:
            for component, path in zip(COMPONENTS, self.st_files):
                parts = [p for p in self.parts if p["component"] == component]
                offset, meta = 0, {}
                for p in parts:
                    meta[p["control_key"]] = {"dtype": "U8", "shape": [p["length"]],
                                                "data_offsets": [offset, offset + p["length"]]}
                    offset += p["length"]
                header = canonical(meta)
                header += b" " * (-len(header) % 8)
                target = writer.component(component, cls=("", "", ""), config={})
                with path.open("xb") as stream:
                    stream.write(len(header).to_bytes(8, "little") + header)
                    for p in parts:
                        data = bytearray(p["length"])
                        self.source.components[component].tensors.read_into(p["key"], data, role=p["role"])
                        p["sha256"] = hashlib.sha256(data).hexdigest()
                        target.put(p["name"], data, dtype="U8", shape=[p["length"]])
                        stream.write(data)
                        del data
            checkpoint = writer.commit()
            writer = None
            with checkpoint:
                report = checkpoint.verify()
                require(report.ok, "normalized TensorFS fixture failed verification")
                self.normalized_id = checkpoint.id
                self.normalized_files = physical_files(checkpoint, self.cache)
        except BaseException:
            if writer is not None:
                writer.abandon()
            raise
        finally:
            self.source.close()
        self.control_files = [{"path": str(p), "length": p.stat().st_size, "sha256": digest(p)} for p in self.st_files]
        record = {"parts": self.parts, "component_tensors": self.counts, "tensor_count": sum(self.counts.values()),
                  "part_count": len(self.parts), "largest_part": self.largest, "stored_bytes": self.payload,
                  "source_files": self.source_files, "normalized_id": self.normalized_id,
                  "normalized_files": self.normalized_files, "safetensors_files": self.control_files}
        (self.work / "fixture.json").write_bytes(canonical(record))
        os.sync()
        return record

    def drop_cache(self, reader):
        paths = [p["path"] for p in self.normalized_files] if reader == "tensorfs" else self.st_files
        for path in paths:
            with open(path, "rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def read(self, destination, reader, stack, torch, device):
        arrays = {}
        if reader == "tensorfs":
            checkpoint = stack.enter_context(self.tfs.open(self.header, cache_dir=self.cache, local_files_only=True))
            for component in COMPONENTS:
                selected = [p for p in self.parts if p["component"] == component]
                loaded = checkpoint.components[component].tensors.read([p["name"] for p in selected],
                                                                       device="cpu" if destination == "cpu" else device,
                                                                       decode=False)
                arrays.update((p["control_key"], loaded[p["name"]]) for p in selected)
        elif reader == "fastsafetensors":
            fst = importlib.import_module("fastsafetensors")
            loader = fst.SafeTensorsFileLoader(None, device, nogds=True)
            stack.callback(loader.close)
            loader.add_filenames({0: [str(p) for p in self.st_files]})
            files = loader.copy_files_to_device()
            stack.callback(files.close)
            arrays = {p["control_key"]: files.get_tensor(p["control_key"]) for p in self.parts}
        else:
            safe_open = importlib.import_module("safetensors").safe_open
            for component, path in zip(COMPONENTS, self.st_files):
                target = device if reader == "safetensors-cuda" else "cpu"
                with safe_open(path, framework="pt", device=target) as source:
                    for p in self.parts:
                        if p["component"] != component:
                            continue
                        tensor = source.get_tensor(p["control_key"])
                        if reader == "safetensors-copy":
                            tensor = tensor.clone()  # Own and fault every byte, unlike an mmap view.
                        elif reader == "safetensors-cpu-cuda":
                            tensor = tensor.to(device)
                        arrays[p["control_key"]] = tensor
        return arrays

    def validate(self, arrays, destination, reader, torch, ordinal):
        require(set(arrays) == {p["control_key"] for p in self.parts}, "returned part set is incomplete")
        for p in self.parts:
            array = arrays[p["control_key"]]
            require(tuple(array.shape) == (p["length"],), f"wrong output shape: {p['control_key']}")
            if reader == "tensorfs":
                require(array.dtype == "U8" and array.nbytes == p["length"], "wrong TensorFS output dtype/bytes")
                require(array.__dlpack_device__() == ((1, 0) if destination == "cpu" else (2, ordinal)),
                        "wrong TensorFS output device; no fallback allowed")
                actual = hashlib.sha256(array.tobytes()).hexdigest()
            else:
                require(array.dtype == torch.uint8 and array.is_contiguous(), "control is not contiguous U8")
                require(array.device.type == destination and (destination == "cpu" or array.device.index == ordinal),
                        "wrong control output device; no fallback allowed")
                host = array.detach().cpu().numpy()
                actual = hashlib.sha256(memoryview(host)).hexdigest()
            require(actual == p["sha256"], f"output SHA256 mismatch: {p['control_key']}")
        return self.payload

    def once(self, destination, reader, torch, device, ordinal, recovery_floor):
        before_free = torch.cuda.mem_get_info(ordinal)[0]
        require(before_free >= recovery_floor,
                f"GPU memory below fixed recovery floor before read: {before_free} < {recovery_floor}")
        arrays = {}
        try:
            with ExitStack() as stack:
                try:
                    started = time.perf_counter()
                    arrays = self.read(destination, reader, stack, torch, device)
                    if destination == "cuda":
                        torch.cuda.synchronize(ordinal)
                    seconds = time.perf_counter() - started
                    after_read_free = torch.cuda.mem_get_info(ordinal)[0]
                    # Every part remains owned through completion and untimed validation.
                    moved = self.validate(arrays, destination, reader, torch, ordinal)
                finally:
                    arrays.clear()
        finally:
            arrays.clear()
            gc.collect()
            torch.cuda.empty_cache()
        try:
            recovery = recover_memory(lambda: torch.cuda.mem_get_info(ordinal)[0], recovery_floor)
        except ValueError as error:
            raise ValueError(f"{error}; before_free={before_free}; after_read_free={after_read_free}") from error
        recovery.update(before_free=before_free, after_read_free=after_read_free)
        return {"seconds": seconds, "bytes": moved, "verified": True,
                "metrics": rate_metrics(moved, seconds, destination, self.ceiling, self.payload)
                if reader == "tensorfs" else None,
                "recovery": recovery}


def run(args, receipt):
    receipt["stage"] = "source/cohort validation"
    plan = source_plan(args.source)
    receipt["source"] = plan
    receipt["cohort"] = installed_provenance(args.wheel, args.wheel_sha256, args.native_source)
    torch, tfs, versions = load_cohort(args.torch_version)
    receipt["versions"] = versions
    visible = torch.cuda.device_count()
    device_namespace(visible, args.device)
    ordinal = 0
    torch.cuda.set_device(ordinal)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(ordinal)
    receipt["environment"] = {"python": sys.version, "platform": platform.platform(), "pid": os.getpid(),
                              "device": args.device, "device_name": torch.cuda.get_device_name(ordinal),
                              "device_uuid": str(props.uuid), "device_total_memory": props.total_memory,
                              "expected_visible_gpus": 1, "actual_visible_gpus": visible,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "cpus": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
                              "torch_cuda": torch.version.cuda, "harness_sha256": digest(__file__),
                              "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total",
                                                                      "--format=csv,noheader"], text=True, timeout=15),
                              "volume": subprocess.check_output(["df", "-T", str(Path(args.work).resolve().parent)], text=True)}
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(8192, hard if hard >= 0 else 8192)), hard))
    fixture = Fixture(plan, args.work, tfs)
    receipt["stage"] = "capacity/preparation"
    state = memory_state()
    free, total = torch.cuda.mem_get_info(ordinal)
    try:
        receipt["capacity"] = {**state, "gpu_free": free, "gpu_total": total,
                               **capacity(fixture.payload, fixture.largest, state["available"], free,
                                          shutil.disk_usage(Path(args.work).resolve().parent).free, len(fixture.parts))}
        receipt["fixture"] = fixture.build()
    finally:
        fixture.source.close()
    if args.prepare_only:
        receipt.update(stage="fixtures only", status="prepared_unmeasured")
        return True
    receipt["stage"] = "engine initialization"
    # Initialize the same shared TensorFS engine and fastsafetensors runtime outside the clock.
    nogds = importlib.import_module("fastsafetensors.copier.nogds")
    framework = importlib.import_module("fastsafetensors.frameworks")
    runtimes = sorted({str(p.resolve()) for p in Path(torch.__file__).parent.parent.glob("nvidia/**/libcudart.so*")})
    require(len(runtimes) == 1, f"one cohort CUDA runtime required, found {runtimes}")
    runtime = runtimes[0]
    receipt["environment"]["cudart"] = {"path": runtime, "sha256": digest(runtime)}
    nogds.resolve_runtime_lib_name = lambda framework=None: runtime
    nogds.load_library_func(framework.get_framework_op("pytorch"))
    # TensorFS's ceiling for this volume and GPU, measured (or kept this boot) outside every timed cell.
    ceiling = tfs.ceiling(ordinal, path=fixture.work)
    fixture.ceiling = {"disk_gbps": ceiling.disk_gbps, "disk_direct": ceiling.disk_direct,
                       "h2d_gbps": ceiling.h2d_gbps, "streams": ceiling.streams}
    receipt["ceiling"] = fixture.ceiling
    initial_free = torch.cuda.mem_get_info(ordinal)[0]
    initial_floor = initial_free - INIT_OVERHEAD
    receipt["initialization"] = []
    for destination, readers in READERS.items():
        for reader in readers:
            row = fixture.once(destination, reader, torch, args.device, ordinal, initial_floor)
            receipt["initialization"].append({"destination": destination, "reader": reader,
                                              "recovery": row["recovery"]})
    baseline = torch.cuda.mem_get_info(ordinal)[0]
    recovery_floor = baseline - STEADY_OVERHEAD
    receipt["gpu_recovery"] = {"initial_free": initial_free, "initial_floor": initial_floor,
                               "initial_allowance": INIT_OVERHEAD, "baseline_free": baseline,
                               "fixed_floor": recovery_floor, "steady_allowance": STEADY_OVERHEAD}
    receipt["stage"], receipt["rows"] = "measurements", []
    for rep, destination, reader, cold in schedule():
        receipt["active_cell"] = {"rep": rep, "destination": destination, "reader": reader, "cold": cold}
        if cold:
            fixture.drop_cache(reader)
            warmup = None
        else:
            warmup = fixture.once(destination, reader, torch, args.device, ordinal, recovery_floor)["recovery"]
        state = memory_state()
        free, _ = torch.cuda.mem_get_info(ordinal)
        # Page cache is reclaimable; the original admission guard accounted for both it and outputs.
        require(state["available"] >= fixture.payload + 2 * fixture.largest + 2 * GIB,
                "insufficient host memory before full measurement")
        require(free >= fixture.payload + max(2 * GIB, fixture.payload // 20), "full GPU capacity no longer available")
        row = fixture.once(destination, reader, torch, args.device, ordinal, recovery_floor)
        row.update(rep=rep, destination=destination, reader=reader, cold=cold,
                   host_available=state["available"], gpu_free=free,
                   warmup_recovery=warmup,
                   cache_control="POSIX_FADV_DONTNEED requested" if cold else "untimed full read")
        receipt["rows"].append(row)
        receipt["summary"] = summarize(receipt["rows"], fixture.payload)
        Path(args.result).write_bytes(canonical(receipt))
        print(f"rep {rep + 1} {destination} {reader} {'cold' if cold else 'warm'}: {row['seconds']:.6f}s", flush=True)
    receipt["measurement_complete"] = receipt["summary"]["complete"]
    receipt["parity_pass"] = receipt["summary"]["parity_pass"]
    qualified = receipt["measurement_complete"] and receipt["summary"]["observations_complete"]
    receipt["status"] = ("complete_parity_pass" if receipt["parity_pass"] else "complete_slower") if qualified else "incomplete_observations"
    receipt["stage"] = "complete"
    receipt.pop("active_cell")
    # A slower complete result is valid benchmark data; parity remains a separate review gate.
    return qualified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "work", "result", "wheel", "wheel-sha256", "native-source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-version", default="2.14.0+cu130")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    receipt = {"format": "tensorfs.h3-load/1", "status": "incomplete", "stage": "starting",
               "measurement_complete": False, "parity_pass": False,
               "scope": "all stored tensor parts; no decode/inference/offload/window",
               "cold_is_advisory": True, "r4_link_review_required": True, "r4_lifecycle_review_required": True}
    with Path(args.result).open("xb") as stream:
        stream.write(canonical(receipt))
    try:
        return 0 if run(args, receipt) else 1
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "detail": str(error)})
        raise
    finally:
        Path(args.result).write_bytes(canonical(receipt))


if __name__ == "__main__":
    sys.exit(main())
