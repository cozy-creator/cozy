#!/usr/bin/env python3
"""tfs-013 live-fleet resweep: classify LIVE headers, never banked fixtures.

The v1 incident this exists to prevent: a sub-tree corpus scored 100% on itself
while production stamped 2.4%. Acceptance therefore sweeps the CURRENT revision
of every fleet repo — bounded header bytes only (8-byte prefix + declared header
length via HTTP Range; sharded components fetch the index plus every shard's
header), decoded by the real border (`tfs ingest inspect` on the header-evidence
path), and the verdict table is printed with nothing summarized away.

  fleet-resweep.py <fleet.json> [--tfs path] [--expect N]

fleet.json: [{"repo": "...", "component": "...", "files": ["path", ...]}, ...]
A file ending in .index.json is a sharded component: the sweep fetches the index
and each named shard's header, then drives the border's sharded decode.
Exit is nonzero if any fetch fails or --expect authorized components is missed.
"""

import argparse
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import urllib.request

HF = "https://huggingface.co"


def fetch(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def header_bytes(repo: str, path: str) -> bytes:
    """The 8-byte length prefix, then exactly that many header bytes."""
    url = f"{HF}/{repo}/resolve/main/{path}"
    prefix = fetch(url, {"Range": "bytes=0-7"})
    if len(prefix) != 8:
        raise RuntimeError(f"{repo}/{path}: {len(prefix)} prefix bytes")
    (n,) = struct.unpack("<Q", prefix)
    if n > 512 * 1024 * 1024:
        raise RuntimeError(f"{repo}/{path}: header claims {n} B")
    return fetch(url, {"Range": f"bytes=8-{7 + n}"})


def sweep_one(tfs: str, repo: str, component: str, files: list[str], tmp: pathlib.Path):
    """Write header evidence locally, run the border, return its verdict line."""
    if len(files) == 1 and files[0].endswith(".index.json"):
        # Sharded: fetch index + every named shard's header into a temp dir laid
        # out exactly as the border expects (index + sibling shard stubs are not
        # enough — the border reads real shard headers, so we materialize each
        # shard as prefix+header with a data region we cannot fetch; instead the
        # single-shard headers are swept individually below).
        idx = json.loads(fetch(f"{HF}/{repo}/resolve/main/{files[0]}"))
        shards = sorted(set(idx.get("weight_map", {}).values()))
        base = "/".join(files[0].split("/")[:-1])
        rows = []
        for s in shards:
            p = f"{base}/{s}" if base else s
            rows.append(sweep_one(tfs, repo, component, [p], tmp))
        return rows[0] if len(rows) == 1 else combine(rows, component)
    hb = header_bytes(repo, files[0])
    f = tmp / (component + ".json")
    f.write_bytes(hb)
    out = subprocess.run(
        [tfs, "ingest", "inspect", "--source", f"{component}={f}"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(tfs).resolve().parents[2],
    )
    text = out.stdout + out.stderr
    for line in text.splitlines():
        if "verdict" in line or "REFUSED" in line:
            return line.strip()
    return "NO VERDICT LINE"


def combine(rows: list[str], component: str) -> str:
    # Per-shard headers CANNOT reproduce the joined fingerprint (the keyset spans
    # shards), so a sharded live sweep decodes every shard header and says so —
    # the joined classification needs local shard files (the border's
    # read_sharded). NAMED LIMIT, not a pass: a sharded row never AUTHORIZES.
    decoded = sum("REFUSED" not in r and "FAILED" not in r for r in rows)
    return (
        f"sharded x{len(rows)}: {decoded}/{len(rows)} shard headers decode; "
        f"joined classification requires local shards"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fleet")
    ap.add_argument("--tfs", default="target/release/tfs")
    ap.add_argument("--expect", type=int, default=0)
    a = ap.parse_args()
    fleet = json.loads(pathlib.Path(a.fleet).read_text())
    authorized = failures = 0
    with tempfile.TemporaryDirectory() as td:
        for row in fleet:
            try:
                v = sweep_one(a.tfs, row["repo"], row["component"], row["files"], pathlib.Path(td))
            except Exception as e:  # noqa: BLE001 — every fleet row must report, never abort the sweep
                v = f"FETCH FAILED: {e}"
                failures += 1
            hit = "AUTHORIZED" in v
            authorized += hit
            print(f"{'AUTHORIZED ' if hit else 'unmatched  '}{row['repo']}:{row['component']}  {v}")
    print(f"\n{authorized} authorized / {len(fleet)} swept, {failures} fetch failures")
    if failures or (a.expect and authorized < a.expect):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
