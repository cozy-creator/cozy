"""Generated native versus real Runtime broker write profile; no Hub or rental calls.

Run with the chosen TensorFS and Runtime source on PYTHONPATH.
Requires the ordinary Runtime Python dependencies; no model or CUDA fixture is used. CPU fixture bytes
are generated here; the bounded parent spool and admission path are production code.
This diagnoses machinery and does not qualify ordinary CLI execution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import tensorfs
from cozy_runtime.internal.weights_writer import StoreProxy, WriterBroker


def run(mode: str, keys: int, fanout: int, writes: int, size: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="cozy-derived-profile-") as folder:
        root = Path(folder)
        store = tensorfs.Store.init(root / "store")
        spool = root / "spool"
        spool.mkdir()
        plain = dict(tensorfs.seed_digests())["plain/1"]
        targets = {
            "model": {
                "drop": [],
                "add": {
                    f"weight_{index:04}": {
                        "logical_dtype": "u8",
                        "shape": [size],
                        "encoding": plain,
                        "parts": {"value": {"dtype": "u8", "shape": [size]}},
                    }
                    for index in range(keys)
                },
            }
        }
        order = [("model", key) for key in targets["model"]["add"]]
        maximum = keys * size
        fingerprint = tensorfs.object_id(b"generated-profile")
        declaration = store.derived_declaration(
            {}, targets, {}, order, maximum, work_fingerprint=fingerprint
        )
        counts = {}
        spans = {}
        broker = WriterBroker(lambda: store, store_root=root / "store")
        attempt = SimpleNamespace(
            request_id="profile",
            attempt=1,
            state="running",
            canceling="",
            spool=spool,
            weights_work_fingerprint=fingerprint,
            spec={
                "inputs": [],
                "outputs": [
                    {"output_id": f"out{i}", "max_bytes": maximum}
                    for i in range(fanout)
                ],
            },
        )

        def exchange(kind, frame):
            started = time.perf_counter()
            result = broker.handle(attempt, frame)
            op = frame["operation"]
            counts[op] = counts.get(op, 0) + 1
            spans[op] = spans.get(op, 0.0) + time.perf_counter() - started
            return result

        selected = StoreProxy(store, spool, exchange) if mode == "broker" else store
        writers = []
        started = time.perf_counter()
        for index in range(fanout):
            transaction = tensorfs.object_id(f"out{index}".encode())
            if mode == "broker":
                broker.authorize(
                    attempt,
                    transaction,
                    1,
                    hashlib.sha256(declaration).digest(),
                    f"out{index}",
                )
            writers.append(
                selected.begin_derived(
                    transaction,
                    1,
                    {},
                    targets,
                    {},
                    order,
                    maximum,
                    work_fingerprint=fingerprint,
                )
            )
        opening = time.perf_counter() - started
        payload = bytes([49]) * size
        part_s = checkpoint_s = 0.0
        before = Path("/proc/self/io").read_text()
        for index in range(writes):
            for number, writer in enumerate(writers):
                started = time.perf_counter()
                writer.add_part(
                    "model", f"weight_{index:04}", "value", io.BytesIO(payload)
                )
                part_s += time.perf_counter() - started
                started = time.perf_counter()
                writer.checkpoint("profile", f"out{number}")
                checkpoint_s += time.perf_counter() - started
        after = Path("/proc/self/io").read_text()
        for writer in writers:
            writer.fence()
        broker.close_attempt(attempt)

        def parse(raw):
            return {
                key: int(value)
                for key, value in (line.split(":") for line in raw.splitlines())
            }

        before, after = parse(before), parse(after)
        return {
            "mode": mode,
            "keys": keys,
            "fanout": fanout,
            "writes": writes,
            "part_bytes": size,
            "open_s": opening,
            "part_s": part_s,
            "checkpoint_s": checkpoint_s,
            "exchange_counts": counts,
            "exchange_s": spans,
            "io": {key: after[key] - value for key, value in before.items()},
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["native", "broker"], required=True)
    parser.add_argument("--keys", type=int, default=219)
    parser.add_argument("--fanout", type=int, default=1)
    parser.add_argument("--writes", type=int, default=10)
    parser.add_argument("--bytes", type=int, default=16384)
    args = parser.parse_args()
    if not (
        1 <= args.fanout <= 4
        and 1 <= args.writes <= args.keys <= 8192
        and 1 <= args.bytes <= 128 << 20
        and args.writes * args.bytes * args.fanout <= 256 << 20
    ):
        parser.error("profile exceeds bounded generated-data fixture limits")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    print(
        json.dumps(run(args.mode, args.keys, args.fanout, args.writes, args.bytes)),
        flush=True,
    )


if __name__ == "__main__":
    main()
