#!/usr/bin/env python3
"""Architecture fences (boundaries.md, tensorfs.md §10). Enforcement, not tests.

1. forbidden dependencies — core is torch/CUDA/HTTP/TLS/SQL/hub-free
2. no upward imports    — core < layout < py; a crate may name only lower ranks
3. marshal-only facade  — zero hashing/chunking/parsing/carrier logic in Python
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 1. dependencies

DENY = ["torch", "cudarc", "cust", "tch", "reqwest", "hyper", "ureq", "curl", "openssl",
        "rustls", "native-tls", "sqlx", "rusqlite", "postgres", "tensorhub-v2",
        "cozy-creator-v2", "cozy-runtime-v2", "worker-protocol-v2"]
SCAN = ["crates/*/Cargo.toml", "crates/*/src/**/*.rs"]


def fence_dependencies():
    bad = []
    for glob in SCAN:
        for p in sorted(ROOT.glob(glob)):
            if not p.is_file():
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                s = line.strip()
                if s.startswith(("#", "//")):
                    continue
                for d in DENY:
                    if re.search(r"(?<![\w.-])" + re.escape(d) + r"(?![\w-])", s, re.I):
                        bad.append(f"{rel(p)}:{i}: forbidden dependency '{d}': {s}")
    return bad, f"{len(DENY)} forbidden tokens over {len(SCAN)} globs"


# ---------------------------------------------------------------- 2. no upward imports

RANK = {"tensorfs-core": 0, "tensorfs-layout": 1, "tensorfs-py": 2}


def fence_layering():
    bad = []
    for crate, rank in sorted(RANK.items()):
        d = ROOT / "crates" / crate
        higher = {c: r for c, r in RANK.items() if r >= rank and c != crate}
        for p in sorted([d / "Cargo.toml"] + list(d.glob("src/**/*.rs"))):
            if not p.is_file():
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                s = line.strip()
                if s.startswith(("#", "//", "//!")):
                    continue
                for other in higher:
                    for spell in (other, other.replace("-", "_")):
                        if re.search(r"(?<![\w-])" + re.escape(spell) + r"(?![\w-])", s):
                            bad.append(
                                f"{rel(p)}:{i}: upward import — {crate} (rank {rank}) "
                                f"names {other} (rank {RANK[other]}): {s}")
    return bad, f"{len(RANK)} crates, strict rank order " + " < ".join(
        sorted(RANK, key=RANK.get))


# ---------------------------------------------------------------- 3. marshal-only facade

FACADE = ROOT / "python"
BANNED_MODULES = {"hashlib", "hmac", "struct", "json", "base64", "binascii", "zlib",
                  "gzip", "lzma", "bz2", "pickle", "marshal", "ctypes", "mmap",
                  "array", "torch", "numpy"}
BANNED_NAMES = {"sha256", "sha1", "md5", "blake2b", "digest", "hexdigest", "update",
                "from_bytes", "to_bytes", "unpack", "unpack_from", "pack", "frombuffer",
                "b64encode", "b64decode", "loads", "dumps", "compress", "decompress",
                "crc32"}
# A second implementation needs constants: chunk sizes, thresholds, caps.
LITERAL_CAP = 1024


def fence_facade():
    bad, files = [], sorted(FACADE.glob("**/*.py"))
    for p in files:
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in BANNED_MODULES:
                    bad.append(f"{rel(p)}:{node.lineno}: facade imports '{m}' — "
                               "marshal-only means the compiled core owns this")
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
            if name in BANNED_NAMES:
                bad.append(f"{rel(p)}:{node.lineno}: facade uses '{name}' — "
                           "hashing/packing/parsing logic belongs in the core")
            if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                    and not isinstance(node.value, bool) and abs(node.value) >= LITERAL_CAP:
                bad.append(f"{rel(p)}:{node.lineno}: facade holds constant "
                           f"{node.value} >= {LITERAL_CAP} — sizes/thresholds are core facts")
    return bad, (f"{len(files)} facade files, {len(BANNED_MODULES)} modules / "
                 f"{len(BANNED_NAMES)} names banned, literal cap {LITERAL_CAP}")


# ----------------------------------------------------------------------------

def rel(p):
    return p.relative_to(ROOT)


FENCES = [("forbidden dependencies", fence_dependencies),
          ("no upward imports", fence_layering),
          ("marshal-only facade", fence_facade)]

red = 0
for title, fn in FENCES:
    bad, note = fn()
    if bad:
        red += 1
        print(f"FENCE RED — {title}:", file=sys.stderr)
        for v in bad:
            print("  " + str(v), file=sys.stderr)
    else:
        print(f"fence green — {title} ({note})")
sys.exit(1 if red else 0)
