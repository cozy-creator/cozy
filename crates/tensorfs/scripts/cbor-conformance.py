#!/usr/bin/env python3
"""The CozyTensors CBOR conformance gate: an independent Python seat, and
optionally the production Rust codec, over the same frozen green and red vectors."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "conformance" / "cbor"
PYTHON = CONF / "python"
GREEN = CONF / "vectors" / "green"
RED = CONF / "vectors" / "red"
sys.path.insert(0, str(PYTHON))

import cbor2
import codec


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, capture_output=True, check=check)


def python_codec(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return run(sys.executable, str(PYTHON / "codec.py"), *args, check=check)


def rust_codec(
    tfs: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return run(str(tfs), "cbor", *args, check=check)


def canonical_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fence_sources() -> None:
    tracked = (
        run("git", "-C", str(ROOT), "ls-files", "conformance/cbor")
        .stdout.decode()
        .splitlines()
    )
    bad = [
        p
        for p in tracked
        if "/.venv/" in p
        or "/__pycache__/" in p
        or p.endswith(".pyc")
    ]
    if bad:
        raise RuntimeError("generated conformance cache is tracked: " + ", ".join(bad))


def green_outputs(source: Path, work: Path, tfs: Path | None) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    out = work / f"{source.stem}.python.cbor"
    result = python_codec("encode", str(source), "--out", str(out), check=False)
    if result.returncode:
        raise RuntimeError(f"python encode {source.name}: {result.stderr.decode().strip()}")
    outputs.append(("python", out))
    if tfs:
        out = work / f"{source.stem}.rust.cbor"
        result = rust_codec(tfs, "encode", str(source), "--out", str(out), check=False)
        if result.returncode:
            raise RuntimeError(
                f"rust encode {source.name}: {result.stderr.decode().strip()}"
            )
        outputs.append(("rust", out))
    return outputs


def red_vectors() -> dict[str, tuple[bytes, str]]:
    minimal = codec.encode_diagnostic((GREEN / "minimal-inline.json").read_bytes())
    root = codec.normalize_diagnostic(canonical_json(GREEN / "minimal-inline.json"))
    # Generic-library encoding is intentional here: reds demonstrate why the strict
    # schema wrapper and exact re-encode check are load-bearing.
    malformed_digest = list(root)
    malformed_digest[3] = [[b"\x11" * 31, 470]]
    long_digest = list(root)
    long_digest[3] = [[b"\x11" * 33, 470]]
    unsorted = list(root)
    unsorted[1] = [
        ["z", b"{}"],
        ["a", b"{}"],
    ]
    rank = list(root)
    rank[4] = [
        ["transformer", [["x", 3, [1] * 33, 0, [["value", 3, [1] * 33, b"\0" * 2]]]]]
    ]
    empty_segments = copy.deepcopy(root)
    empty_segments[4][0][1][0][4][0][3] = []
    uncited = copy.deepcopy(root)
    uncited[3].append([b"\x22" * 32, 512])
    empty_config = copy.deepcopy(root)
    empty_config[1] = [["a", b""]]
    zero_encoding = copy.deepcopy(root)
    zero_encoding[3][0][1] = 0
    bool_bits = copy.deepcopy(root)
    bool_part = bool_bits[4][0][1][0][4][0]
    bool_part[1:4] = [11, [4], b"\x00\x01\x02\xff"]
    zero_extent = copy.deepcopy(root)
    zero_extent[4][0][1][0][2] = [0]
    zero_extent[4][0][1][0][4][0][2] = [0]
    zero_extent[4][0][1][0][4][0][3] = b""
    duplicate_component = copy.deepcopy(root)
    duplicate_component[4].append(copy.deepcopy(duplicate_component[4][0]))
    duplicate_tensor = copy.deepcopy(root)
    duplicate_tensor[4][0][1].append(copy.deepcopy(duplicate_tensor[4][0][1][0]))
    duplicate_role = copy.deepcopy(root)
    duplicate_role[4][0][1][0][4].append(
        copy.deepcopy(duplicate_role[4][0][1][0][4][0])
    )
    inline_over = copy.deepcopy(root)
    inline_over[4][0][1][0][2] = [129]
    inline_over[4][0][1][0][4][0][2] = [129]
    inline_over[4][0][1][0][4][0][3] = b"\0" * 258
    segmented_under = copy.deepcopy(root)
    segmented_under[4][0][1][0][4][0][3] = [[b"\xaa" * 32, 4]]
    overflow = copy.deepcopy(root)
    overflow[2] = [
        ["a", b"\x33" * 32, (1 << 64) - 1, "application/octet-stream", [[b"\x33" * 32, (1 << 64) - 1]]],
        ["b", b"\x44" * 32, (1 << 64) - 1, "application/octet-stream", [[b"\x44" * 32, (1 << 64) - 1]]],
    ]
    huge_encoding_index = copy.deepcopy(root)
    huge_encoding_index[4][0][1][0][3] = (1 << 64) - 1
    huge_dtype = copy.deepcopy(root)
    huge_dtype[4][0][1][0][1] = (1 << 64) - 1

    def bad_name(field: str) -> list[Any]:
        value = copy.deepcopy(root)
        if field == "config":
            value[1] = [["bad name", b"{}"]]
        elif field == "component":
            value[4][0][0] = "bad name"
        elif field == "tensor":
            value[4][0][1][0][0] = "bad name"
        else:
            value[4][0][1][0][4][0][0] = "bad name"
        return value

    asset_source = codec.normalize_diagnostic(
        canonical_json(GREEN / "segmented-assets.json")
    )
    zero_tensor_segment = copy.deepcopy(asset_source)
    tensor_segments = zero_tensor_segment[4][0][1][0][4][0][3]
    tensor_segments[0][1] = 0
    tensor_segments[1][1] = 258
    zero_asset_segment = copy.deepcopy(asset_source)
    asset_segments = zero_asset_segment[2][1][4]
    asset_segments[0][1] = 0
    asset_segments[1][1] = 12

    def bad_asset(name: str) -> list[Any]:
        value = copy.deepcopy(asset_source)
        value[2][0][0] = name
        value[2].sort(key=lambda row: row[0])
        return value

    return {
        "old-json": (
            (GREEN / "minimal-inline.json").read_bytes(),
            "old JSON is not a CBOR header",
        ),
        "nonpreferred-root-length": (
            b"\x98\x05" + minimal[1:],
            "root length is not shortest",
        ),
        "nonpreferred-text-length": (
            minimal[:1] + b"\x78\x0d" + minimal[2:],
            "text length is not shortest",
        ),
        "indefinite-root": (
            b"\x9f" + minimal[1:] + b"\xff",
            "indefinite arrays are forbidden",
        ),
        "trailing-item": (minimal + b"\x00", "a second top-level item is forbidden"),
        "forbidden-map": (b"\xa0", "maps are outside the stored schema"),
        "forbidden-tag": (b"\xd9\xd9\xf7" + minimal, "tags are forbidden"),
        "forbidden-float": (b"\xf9\x3e\x00", "floats are forbidden"),
        "forbidden-null": (b"\xf6", "null is forbidden"),
        "forbidden-undefined": (b"\xf7", "undefined is forbidden"),
        "forbidden-simple": (b"\xf8\x20", "arbitrary simple values are forbidden"),
        "wrong-root-arity": (b"\x80", "the root has exactly five fields"),
        "malformed-digest": (
            cbor2.dumps(malformed_digest, canonical=True),
            "digests are exactly 32 bytes",
        ),
        "long-digest": (
            cbor2.dumps(long_digest, canonical=True),
            "digests are exactly 32 bytes",
        ),
        "unsorted-config-set": (
            cbor2.dumps(unsorted, canonical=True),
            "config rows are sorted unique",
        ),
        "rank-cap": (cbor2.dumps(rank, canonical=True), "rank is bounded"),
        "empty-segmented-body": (
            cbor2.dumps(empty_segments, canonical=True),
            "segmented bodies are non-empty",
        ),
        "uncited-encoding": (
            cbor2.dumps(uncited, canonical=True),
            "every encoding row is cited",
        ),
        "empty-config": (
            cbor2.dumps(empty_config, canonical=True),
            "inline configs are JSON objects",
        ),
        "zero-encoding-length": (
            cbor2.dumps(zero_encoding, canonical=True),
            "encoding references are non-empty",
        ),
        "zero-tensor-segment-length": (
            cbor2.dumps(zero_tensor_segment, canonical=True),
            "tensor segment references are non-empty",
        ),
        "zero-asset-segment-length": (
            cbor2.dumps(zero_asset_segment, canonical=True),
            "asset segment references are non-empty",
        ),
        "illegal-bool-bits": (
            cbor2.dumps(bool_bits, canonical=True),
            "inline bool bytes are only zero or one",
        ),
        "zero-shape-extent": (
            cbor2.dumps(zero_extent, canonical=True),
            "shape extents are positive",
        ),
        "duplicate-component": (
            cbor2.dumps(duplicate_component, canonical=True),
            "component names are unique",
        ),
        "duplicate-tensor": (
            cbor2.dumps(duplicate_tensor, canonical=True),
            "tensor names are unique within a component",
        ),
        "duplicate-role": (
            cbor2.dumps(duplicate_role, canonical=True),
            "part roles are unique within a tensor",
        ),
        "inline-over-threshold": (
            cbor2.dumps(inline_over, canonical=True),
            "large parts are segmented",
        ),
        "segmented-under-threshold": (
            cbor2.dumps(segmented_under, canonical=True),
            "small parts are inline",
        ),
        "aggregate-byte-overflow": (
            cbor2.dumps(overflow, canonical=True),
            "aggregate byte arithmetic refuses overflow",
        ),
        "huge-encoding-index": (
            cbor2.dumps(huge_encoding_index, canonical=True),
            "encoding indexes never truncate to the platform usize width",
        ),
        "huge-dtype-code": (
            cbor2.dumps(huge_dtype, canonical=True),
            "dtype codes never truncate to the platform usize width",
        ),
        "bad-config-name": (
            cbor2.dumps(bad_name("config"), canonical=True),
            "config names use the closed grammar",
        ),
        "bad-component-name": (
            cbor2.dumps(bad_name("component"), canonical=True),
            "component names use the closed grammar",
        ),
        "bad-tensor-name": (
            cbor2.dumps(bad_name("tensor"), canonical=True),
            "tensor names use the closed grammar",
        ),
        "bad-role-name": (
            cbor2.dumps(bad_name("role"), canonical=True),
            "part roles use the closed grammar",
        ),
        "asset-dotdot": (
            cbor2.dumps(bad_asset("../escape"), canonical=True),
            "asset names are relative safe paths",
        ),
        "asset-absolute": (
            cbor2.dumps(bad_asset("/escape"), canonical=True),
            "asset names are relative safe paths",
        ),
        "asset-empty-component": (
            cbor2.dumps(bad_asset("a//b"), canonical=True),
            "asset names have no empty component",
        ),
        "asset-reserved-character": (
            cbor2.dumps(bad_asset("a\\b"), canonical=True),
            "asset names exclude reserved characters",
        ),
        "truncated": (minimal[:-1], "truncated input is forbidden"),
    }


def update_vectors(tfs: Path | None) -> None:
    RED.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cozy-cbor-") as tmp:
        work = Path(tmp)
        for source in sorted(GREEN.glob("*.json")):
            outputs = green_outputs(source, work, tfs)
            first = outputs[0][1].read_bytes()
            for name, path in outputs[1:]:
                if path.read_bytes() != first:
                    raise RuntimeError(
                        f"{source.name}: {name} disagrees with {outputs[0][0]}"
                    )
            (GREEN / f"{source.stem}.cbor").write_bytes(first)
            (GREEN / f"{source.stem}.sha256").write_text(
                hashlib.sha256(first).hexdigest() + "\n"
            )
        for name, (raw, why) in red_vectors().items():
            (RED / f"{name}.cbor").write_bytes(raw)
            write_json(RED / f"{name}.verdict.json", {"expect": "refuse", "why": why})


def check_vectors(tfs: Path | None) -> None:
    fence_sources()
    seats: list[tuple[str, Any]] = [("python", python_codec)]
    if tfs:
        seats.append(("rust", lambda *a, check=True: rust_codec(tfs, *a, check=check)))
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cozy-cbor-") as tmp:
        work = Path(tmp)
        for source in sorted(GREEN.glob("*.json")):
            expected = source.with_suffix(".cbor")
            if not expected.is_file():
                raise RuntimeError(f"missing {expected}; run --update")
            outputs = green_outputs(source, work, tfs)
            for name, output in outputs:
                if output.read_bytes() != expected.read_bytes():
                    raise RuntimeError(
                        f"{source.name}: {name} differs from frozen bytes"
                    )
            want_hash = source.with_suffix(".sha256").read_text().strip()
            if hashlib.sha256(expected.read_bytes()).hexdigest() != want_hash:
                raise RuntimeError(f"{source.name}: frozen sha256 is stale")
            projection = canonical_json(source)
            for name, command in seats:
                check = command("check", str(expected), check=False)
                if check.returncode:
                    raise RuntimeError(
                        f"{source.name}: {name} check refused: {check.stderr.decode().strip()}"
                    )
                out = work / f"{source.stem}.{name}.json"
                decoded = command(
                    "decode", str(expected), "--out", str(out), check=False
                )
                if decoded.returncode or canonical_json(out) != projection:
                    raise RuntimeError(
                        f"{source.name}: {name} diagnostic projection differs"
                    )
            passed += 1
            print(
                f"PASS green/{source.stem:<28} {len(expected.read_bytes()):>6} B  sha256:{want_hash[:16]}"
            )
        for vector in sorted(RED.glob("*.cbor")):
            for name, command in seats:
                result = command("check", str(vector), check=False)
                if result.returncode == 0:
                    raise RuntimeError(f"{vector.name}: {name} accepted a red vector")
                stderr = result.stderr.lower()
                if (
                    b"panicked at" in stderr
                    or b"traceback (most recent call last)" in stderr
                ):
                    raise RuntimeError(
                        f"{vector.name}: {name} crashed instead of refusing: "
                        f"{result.stderr.decode(errors='replace').splitlines()[0:3]}"
                    )
            passed += 1
            print(f"PASS red/{vector.stem:<30} refused by {len(seats)} seats")

        def identity_flip(label: str, source_name: str, mutate: Any) -> None:
            nonlocal passed
            source = GREEN / f"{source_name}.json"
            changed = copy.deepcopy(canonical_json(source))
            mutate(changed)
            diagnostic = work / f"{label}.json"
            write_json(diagnostic, changed)
            outputs = green_outputs(diagnostic, work, tfs)
            alternate = outputs[0][1].read_bytes()
            if any(path.read_bytes() != alternate for _, path in outputs[1:]):
                raise RuntimeError(f"{label}: encoders disagree on reordered header")
            if alternate == source.with_suffix(".cbor").read_bytes():
                raise RuntimeError(f"{label}: reordered header retained its identity")
            passed += 1
            print(f"PASS identity/{label:<25} reordered bytes changed sha256")

        identity_flip(
            "tensor-order",
            "ordered-layers",
            lambda d: d["components"][0][1].reverse(),
        )
        identity_flip(
            "part-order",
            "ordered-parts",
            lambda d: d["components"][0][1][0][4].reverse(),
        )
        identity_flip(
            "asset-segment-order",
            "segmented-assets",
            lambda d: d["assets"][1][4].reverse(),
        )
        identity_flip(
            "part-segment-order",
            "segmented-assets",
            lambda d: d["components"][0][1][0][4][0][3]["segments"].reverse(),
        )
    print(
        f"\nCBOR conformance: {passed}/{passed} vectors passed across {len(seats)} seats"
    )


def benchmark(diagnostic: Path, tfs: Path | None, reps: int) -> None:
    with tempfile.TemporaryDirectory(prefix="cozy-cbor-bench-") as tmp:
        work = Path(tmp)
        commands: list[tuple[str, Any]] = [("python", python_codec)]
        if tfs:
            commands.append(
                ("rust", lambda *a, check=True: rust_codec(tfs, *a, check=check))
            )
        for name, command in commands:
            out = work / f"{name}.cbor"
            samples: list[float] = []
            for _ in range(reps):
                started = time.perf_counter_ns()
                result = command(
                    "encode", str(diagnostic.resolve()), "--out", str(out), check=False
                )
                if result.returncode:
                    raise RuntimeError(result.stderr.decode())
                samples.append((time.perf_counter_ns() - started) / 1e6)
            raw = out.read_bytes()
            zstd = (
                run("zstd", "-q", "-3", "-c", str(out)).stdout
                if shutil.which("zstd")
                else b""
            )
            samples.sort()
            print(
                json.dumps(
                    {
                        "seat": name,
                        "reps": reps,
                        "median_encode_process_ms": samples[len(samples) // 2],
                        "raw_bytes": len(raw),
                        "gzip_6_bytes": len(
                            gzip.compress(raw, compresslevel=6, mtime=0)
                        ),
                        "zstd_3_bytes": len(zstd) if zstd else None,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "note": "process-start-inclusive scaffold; production Rust benchmark remains tfs-027",
                    },
                    sort_keys=True,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", type=Path, help="optional Rust tfs binary")
    sub = parser.add_subparsers(dest="command", required=True)
    conform = sub.add_parser("conform")
    conform.add_argument("--update", action="store_true")
    bench = sub.add_parser("benchmark")
    bench.add_argument("diagnostic", type=Path)
    bench.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    if args.command == "conform":
        if args.update:
            update_vectors(args.tfs)
        check_vectors(args.tfs)
    else:
        benchmark(args.diagnostic, args.tfs, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
