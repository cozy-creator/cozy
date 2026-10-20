"""Pure receipt rules for the frozen full-H3 benchmark, with no native imports."""
import hashlib
import json
import math
import re
import statistics

HARNESS_SOURCE = "5ea7092ce12d51630c4114539c4137b8ae467826"
HARNESS_SHA = "4b65bacfcdcfb7e5bde13d51875dcd3a01a847e6b522eb048335a6c785b2b8fc"
CLIENT_SOURCE = "c7accfc6af5e6775fa12d2d7f6a8977db3c64cd4"
CLIENT_PRODUCT = "25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2"
CLIENT_SHA = "b0712ca8c21ce695e52e412a135778c648f0dac8b8f4c9e991788a685a31ebe9"
METADATA_SHA = "ceafd89e2e5c61b73f2153663b3ea9de9c45d5b7c0121554758715eaba9da25a"
HEADER = "0f03804af2280bac6c81c633cd058f4dfbb5fd71f5db0a3c5829907e4ef21f2e"
MANIFEST = "bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff"
PAYLOAD = 75_528_893_476
PHYSICAL = 75_535_069_920
CHUNKS_SHA = "daa42aadf133b548d0dd61a7fc789b784546ffc7b7381ceb1bd7966d341e3403"
HOST = 156_316_919_880
GPU = 79_305_338_149
DISK_BASE = 228_760_140_584
HEADROOM = 8 << 30
MAX_WALL = 1200
READERS = {"cpu": ("tensorfs", "safetensors-copy"),
           "cuda": ("tensorfs", "fastsafetensors", "safetensors-cuda", "safetensors-cpu-cuda")}
COUNTS = {"audio_vae": 1087, "text_encoder": 902, "transformer": 583, "transformer_ref": 583, "vae": 703}


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(exact(left[k], v) for k, v in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def uuid(value):
    value = str(value).lower().removeprefix("gpu-").replace("-", "")
    require(re.fullmatch("[0-9a-f]{32}", value), "invalid GPU UUID")
    return value


def schedule():
    # Literal reader order and rotation from frozen5ea schedule(); no shortened cells.
    return [{"rep": rep, "destination": dst, "reader": reader, "cold": cold}
            for rep in range(5) for dst, readers in READERS.items()
            for reader in readers[rep % len(readers):] + readers[:rep % len(readers)]
            for cold in (True, False)]


def deadline_budget(now, deadline):
    require(type(deadline) is int and finite(now) and 13 < deadline - now <= MAX_WALL,
            "deadline must leave cleanup time and fit the fixed wall budget")
    return deadline - now - 13  # Existing owned-child helper's3s TERM +10s KILL bound.


def check_spec(spec):
    require(exact(spec.get("schema"), 1), "wrong H3 proof schema")
    require(spec.get("harness") == {"source": HARNESS_SOURCE, "sha256": HARNESS_SHA}, "harness drift")
    require(spec.get("client") == {"source": CLIENT_SOURCE, "sha256": CLIENT_SHA, "bytes": 6281488}, "client drift")
    require(spec.get("metadata") == {"sha256": METADATA_SHA, "bytes": 316402, "header": HEADER}, "metadata drift")
    require(exact(spec.get("limits"), {"host": HOST, "gpu": GPU, "disk_base": DISK_BASE,
                                       "headroom": HEADROOM, "max_wall_seconds": MAX_WALL}), "budget drift")
    require(exact(spec.get("schedule"), schedule()), "schedule drift")
    require(re.fullmatch("[0-9a-f]{40}", str(spec.get("proof_source"))), "proof source unresolved")
    require(set(spec.get("sources", {})) == {"contract.py", "probe.py", "owned_child.py", "h3_load.py", "requirements.txt"},
            "source inventory differs")
    require(isinstance(spec.get("metadata_files"), dict) and len(spec["metadata_files"]) == 23,
            "metadata manifest missing")
    require(re.fullmatch("[0-9a-f]{64}", str(spec.get("pair_receipt_sha256"))), "unresolved candidate review receipt")
    for name in ("runtime", "native"):
        item = spec.get(name)
        require(isinstance(item, dict), f"unresolved {name} candidate")
        require(re.fullmatch("[0-9a-f]{40}", str(item.get("source"))), f"unresolved {name} source")
        require(re.fullmatch("[0-9a-f]{64}", str(item.get("sha256"))), f"unresolved {name} wheel")
        require(type(item.get("bytes")) is int and item["bytes"] > 0 and item.get("version"), f"unresolved {name} metadata")
        require(isinstance(item.get("files"), dict) and item["files"], f"unresolved {name} file hashes")
    versions = spec.get("dependencies", {}).get("versions", {})
    require(re.fullmatch(r"[0-9.]+\+cu130", str(versions.get("torch"))), "exact cu130 Torch required")
    require(versions.get("safetensors") == "0.8.0" and versions.get("fastsafetensors") == "0.5.0",
            "frozen baseline dependencies differ")
    require("numpy" in versions, "numpy dependency missing")


def template(metadata_files):
    return {"schema": 1, "harness": {"source": HARNESS_SOURCE, "sha256": HARNESS_SHA},
            "client": {"source": CLIENT_SOURCE, "sha256": CLIENT_SHA, "bytes": 6281488},
            "metadata": {"sha256": METADATA_SHA, "bytes": 316402, "header": HEADER},
            "metadata_files": metadata_files,
            "limits": {"host": HOST, "gpu": GPU, "disk_base": DISK_BASE, "headroom": HEADROOM,
                       "max_wall_seconds": MAX_WALL},
            "schedule": schedule(), "runtime": None, "native": None, "dependencies": None,
            "pair_receipt_sha256": None, "proof_source": None, "sources": {}}


def check_capacity(value):
    require(value.get("visible_count") == 1 and type(value.get("visible_count")) is int, "not one visible GPU")
    require(value.get("device") == "cuda:0" and "h100" in value.get("name", "").lower(), "assigned GPU is not H100")
    require(exact(value.get("compute_capability"), [9, 0]), "wrong H100 architecture")
    uuid(value.get("gpu_uuid"))
    visible = value.get("cuda_visible_devices")
    require(isinstance(visible, str) and bool(visible) and "," not in visible, "unsealed GPU namespace")
    require(value.get("same_work_filesystem") is True, "source/fixtures require one guarded filesystem")
    for key, needed in (("host_available", HOST), ("gpu_available", GPU), ("disk_available", DISK_BASE + HEADROOM)):
        require(type(value.get(key)) is int and value[key] >= needed, f"insufficient {key}")


def check_stage(value):
    for key, wanted in {"status": "staged_verified", "client_source": CLIENT_SOURCE,
                        "product_source": CLIENT_PRODUCT, "benchmark_run": False, "gpu_used": False,
                        "header": {"sha256": HEADER, "length": 433449},
                        "old_manifest": {"sha256": MANIFEST, "length": 743},
                        "chunk_count": 1676, "verified_chunk_count": 1676,
                        "physical_chunk_bytes": PHYSICAL, "verified_chunk_bytes": PHYSICAL,
                        "closure_verified": True, "download_requested": True, "payload_transfer_called": True,
                        "missing": [], "corrupt": [], "sync_error": None, "transfer_error": None}.items():
        require(exact(value.get(key), wanted), "staging failed: " + key)


def check_benchmark(value, spec, gpu_uuid):
    require(value.get("measurement_complete") is True, "measurement incomplete")
    require(value.get("cold_is_advisory") is True, "false guaranteed-cold claim")
    env = value.get("environment", {})
    require(env.get("harness_sha256") == HARNESS_SHA, "benchmark source drift")
    require(env.get("actual_visible_gpus") == 1 and env.get("expected_visible_gpus") == 1 and env.get("device") == "cuda:0",
            "benchmark device scope drift")
    require(uuid(env.get("device_uuid")) == uuid(gpu_uuid), "benchmark UUID differs")
    cohort = value.get("cohort", {})
    require(cohort.get("declared_native_source") == spec["native"]["source"] and
            cohort.get("wheel_sha256") == spec["native"]["sha256"], "native cohort drift")
    require(cohort.get("installed_file_sha256") == spec["native"]["files"], "installed native bytes drift")
    require(value.get("versions") == {k: spec["dependencies"]["versions"][k]
                                      for k in ("torch", "safetensors", "fastsafetensors")}, "dependency drift")
    source = value.get("source", {})
    require(source.get("header_sha256") == HEADER and source.get("legacy_manifest_sha256") == MANIFEST,
            "benchmark source identity drift")
    capacity = value.get("capacity", {})
    require(capacity.get("host_required") == HOST and capacity.get("gpu_required") == GPU and
            capacity.get("available", 0) >= HOST and capacity.get("gpu_free", 0) >= GPU, "benchmark capacity differs")
    fixture = value.get("fixture", {})
    require(exact(fixture.get("component_tensors"), COUNTS) and fixture.get("tensor_count") == 3858 and
            fixture.get("part_count") == 4834 and fixture.get("stored_bytes") == PAYLOAD, "partial H3 fixture")
    parts = fixture.get("parts", [])
    require(len(parts) == 4834 and len({(p["component"], p["key"], p["role"]) for p in parts}) == 4834,
            "partial/duplicate stored parts")
    require(all(type(p.get("length")) is int and p["length"] > 0 and re.fullmatch("[0-9a-f]{64}", str(p.get("sha256")))
                for p in parts) and sum(p["length"] for p in parts) == PAYLOAD, "part bytes/hash missing")
    chunks = [{"sha256": p["sha256"], "length": p["length"]} for p in fixture.get("source_files", [])]
    require(len(chunks) == 1676 and hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == CHUNKS_SHA, "physical source chunks drift")
    rows = value.get("rows", [])
    wanted = schedule()
    require(len(rows) == len(wanted), "missing measured row")
    recovery = value.get("gpu_recovery", {})
    require(recovery.get("initial_allowance") == 64 << 20 and recovery.get("steady_allowance") == 2 << 20 and
            recovery.get("initial_floor") == recovery.get("initial_free", 0) - (64 << 20) and
            recovery.get("fixed_floor") == recovery.get("baseline_free", 0) - (2 << 20), "recovery floor drift")
    def settled(record, floor):
        require(record.get("floor") == floor and record.get("settled_free", -1) >= floor and
                record.get("before_free", -1) >= floor and record.get("deadline_seconds") == 10 and
                finite(record.get("seconds")) and record["seconds"] <= 10, "unsettled native ownership")
    initial = value.get("initialization", [])
    require([(r.get("destination"), r.get("reader")) for r in initial] ==
            [(dst, reader) for dst, readers in READERS.items() for reader in readers], "initialization inventory differs")
    for item in initial:
        settled(item.get("recovery", {}), recovery["initial_floor"])
    for row, expected in zip(rows, wanted):
        require(all(exact(row.get(k), v) for k, v in expected.items()), "missing/reordered rep or cache cell")
        require(row.get("verified") is True and exact(row.get("bytes"), PAYLOAD) and finite(row.get("seconds"), True),
                "partial/unverified full read")
        mode = "POSIX_FADV_DONTNEED requested" if row["cold"] else "untimed full read"
        require(row.get("cache_control") == mode, "false cold/warm claim")
        require((row.get("warmup_recovery") is None) == row["cold"], "missing full warmup")
        settled(row.get("recovery", {}), recovery["fixed_floor"])
        if not row["cold"]:
            settled(row["warmup_recovery"], recovery["fixed_floor"])
        if row["destination"] == "cuda" and row["reader"] == "tensorfs":
            metrics = row.get("metrics", {})
            require(metrics.get("kind") == "h2d" and metrics.get("measured") is True and
                    metrics.get("bytes", 0) >= PAYLOAD, "partial H2D telemetry")
    summary = value.get("summary", {})
    require(summary.get("complete") is True and summary.get("observations_complete") is True, "incomplete summary")
    require(summary.get("r4_link_review_required") is True and summary.get("r4_lifecycle_review_required") is True,
            "full R4 acceptance is not automatic")
    parity = True
    for dst, readers in READERS.items():
        for cold in (True, False):
            medians = {reader: statistics.median(r["seconds"] for r in rows
                       if (r["destination"], r["reader"], r["cold"]) == (dst, reader, cold)) for reader in readers}
            best = min(readers[1:], key=medians.__getitem__)
            passed = medians["tensorfs"] <= medians[best]
            cell = summary.get("cells", {}).get(f"{dst} {'cold' if cold else 'warm'}", {})
            require(cell.get("complete") is True and cell.get("medians_seconds") == medians and
                    cell.get("best_control") == best and exact(cell.get("parity_pass"), passed), "baseline summary drift")
            parity &= passed
    require(exact(value.get("parity_pass"), parity) and exact(summary.get("parity_pass"), parity), "parity drift")
    require(value.get("status") == ("complete_parity_pass" if parity else "complete_slower"), "wrong completion status")
    return {"measurement_complete": True, "parity_pass": parity, "rows": len(rows)}
