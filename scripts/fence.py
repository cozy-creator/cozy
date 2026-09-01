#!/usr/bin/env python3
"""The two architecture fences boundaries.md actually mandates.

1. forbidden dependencies — core is torch/CUDA/HTTP/TLS/server-DB/hub-free
2. no upward imports    — core < py; a crate may name only lower ranks

Everything else that lived here was a word blacklist over source text: it could not tell
an identifier from English prose, so it needed a hand-written Rust lexer in Python to stop
tripping on refusal messages, and it banned ordinary vocabulary (`dict.update`, `strides`)
outright. Real invariants belong in `cargo test` and the frozen vector corpus.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DENY = ["torch", "cudarc", "cust", "tch", "reqwest", "hyper", "ureq", "curl", "openssl",
        "rustls", "native-tls", "sqlx", "postgres", "tensorhub-v2",
        "cozy-creator-v2", "cozy-runtime-v2", "worker-protocol-v2"]
SCAN = ["crates/*/Cargo.toml", "crates/*/src/**/*.rs"]
RANK = {"tensorfs-core": 0, "tensorfs-py": 1}


def rel(p):
    return p.relative_to(ROOT)


def lines(path):
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        s = line.strip()
        if not s.startswith(("#", "//", "//!")):
            yield i, s


def fence_dependencies():
    bad = []
    for glob in SCAN:
        for p in sorted(ROOT.glob(glob)):
            if p.is_file():
                for i, s in lines(p):
                    for d in DENY:
                        if re.search(r"(?<![\w.-])" + re.escape(d) + r"(?![\w-])", s, re.I):
                            bad.append(f"{rel(p)}:{i}: forbidden dependency '{d}': {s}")
    return bad, f"{len(DENY)} forbidden tokens over {len(SCAN)} globs"


def fence_layering():
    bad = []
    for crate, rank in sorted(RANK.items()):
        d = ROOT / "crates" / crate
        higher = {c: r for c, r in RANK.items() if r >= rank and c != crate}
        for p in sorted([d / "Cargo.toml"] + list(d.glob("src/**/*.rs"))):
            if not p.is_file():
                continue
            for i, s in lines(p):
                for other in higher:
                    for spell in (other, other.replace("-", "_")):
                        if re.search(r"(?<![\w-])" + re.escape(spell) + r"(?![\w-])", s):
                            bad.append(
                                f"{rel(p)}:{i}: upward import — {crate} (rank {rank}) "
                                f"names {other} (rank {RANK[other]}): {s}")
    return bad, "strict rank order " + " < ".join(sorted(RANK, key=RANK.get))


red = 0
for title, fn in [("forbidden dependencies", fence_dependencies),
                  ("no upward imports", fence_layering)]:
    bad, note = fn()
    if bad:
        red += 1
        print(f"FENCE RED — {title}:", file=sys.stderr)
        for v in bad:
            print("  " + str(v), file=sys.stderr)
    else:
        print(f"fence green — {title} ({note})")
sys.exit(1 if red else 0)
