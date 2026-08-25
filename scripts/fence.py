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


# ------------------------------------------------- 4. one codec, one evaluator

# A SECOND implementation of an identity-bearing codec is the defect class this repo is
# built against: two readers that disagree produce a green suite and a wrong artifact.
#
# STRUCTURAL OWNERSHIP (audit fix, tfs-003). The previous rule banned every .go/.java/.rb/
# .ts/.js file in the repository outright — a proxy so coarse it would reject a build script
# while a rival codec written in Rust or Python walked past. What actually matters is who
# OWNS each identity-bearing format: a file that names a format mark is either that format's
# one implementation or a rival to it, whatever language it is written in.
CODEC_OWNERS = {
    "tensorfs.snapshot/1": {"crates/tensorfs-core/src/manifest.rs"},
    "cozytensors/1": {"crates/tensorfs-core/src/header.rs"},
    "cozytensors.topology/1": {"crates/tensorfs-core/src/header.rs"},
    "tensorfs.encoding_spec/1": {"crates/tensorfs-core/src/spec.rs"},
    "tensorfs.encoding_vectors/1": {"crates/tensorfs-core/src/vectors.rs"},
}
CODEC_MARKS = list(CODEC_OWNERS)
GRAMMAR_TAGS = ["ceil_block", "elements_div", "ceil_div"]
# Foreign containers belong to tfs-003's isolated ingest border, nowhere else. Matched
# against CODE with string literals stripped: a dialect NAME in a string is data the border
# legitimately carries ("pickle" is a refused dialect); an identifier is a parser.
FOREIGN = ["safetensors", "gguf", "pickle", "unpickle", "zipfile", "ZipArchive", "npz"]
# The border is a MODULE, not a file: carrier decode, classification, converters and the
# quarantine transaction each own one stage, and all four are inside the same fence.
INGEST_OK_PREFIX = "crates/tensorfs-core/src/ingest"
# The fence must be free to NAME what it forbids.
SELF = {"scripts/fence.py"}
GRAMMAR_HOME = {"crates/tensorfs-core/src/relation.rs"}
# The design's deleted vocabulary: a rule VM, arrangements, morphisms, source layouts.
EXCLUDED = ["arrangement", "morphism", "quant_rule", "source_layout", "expression_language",
            "rule_vm", "strides"]


def code_only(line):
    """The line with comments and string literals removed — what the code SAYS, not what
    it carries as data."""
    s = line.strip()
    if s.startswith(("//", "//!", "#")):
        return ""
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    return s


def fence_one_codec():
    bad = []
    # (a) each identity-bearing format has ONE owning module. Two tiers, because naming a
    #     format and IMPLEMENTING it are different acts:
    #       outside the owning crate — naming a mark at all is a rival, in any language;
    #       inside it — the CLI legitimately prints marks and PLANTS them into byte streams
    #       to arm the one implementation's refusals, so the rival signal is naming a mark
    #       while also defining a canonical codec for it (`impl Doc for`).
    for p in sorted(ROOT.glob("**/*")):
        if not p.is_file() or ".git" in p.parts or "target" in p.parts:
            continue
        r = str(rel(p))
        if r in SELF or r.startswith("vectors/"):   # frozen DATA carries its own marks
            continue
        # Languages a codec could actually be HOSTED in. Shell orchestration is excluded on
        # purpose: `scripts/*.sh` plant format strings into byte streams to arm the one
        # implementation's refusals, which is driving the codec, not rivalling it.
        if p.suffix not in (".rs", ".py", ".go", ".java", ".rb", ".ts", ".js", ".c", ".cpp",
                            ".cs", ".swift", ".kt"):
            continue
        text = p.read_text(errors="ignore")
        inside = r.startswith("crates/")
        implements = "impl Doc for" in text
        for m, owners in CODEC_OWNERS.items():
            if m not in text or r in owners:
                continue
            if inside and not implements:
                continue    # a consumer naming or planting the mark, not a second codec
            bad.append(f"{r}: names format '{m}' owned by {sorted(owners)} and "
                       + ("defines its own Doc codec" if inside else "sits outside the owning crate")
                       + " — a rival implementation of an identity-bearing codec")
        # (b) grammar tags live in exactly one file
        if r not in GRAMMAR_HOME:
            for t in GRAMMAR_TAGS:
                if re.search(r'"' + t + r'"', text):
                    bad.append(f"{r}: emits grammar tag '{t}' — a SECOND shape evaluator")
        # (c) foreign container parsing lives only at the ingest border
        if not r.startswith(INGEST_OK_PREFIX) and r.startswith("crates/"):
            for f in FOREIGN:
                for i, line in enumerate(text.splitlines(), 1):
                    s = code_only(line)
                    if not s:
                        continue
                    if re.search(r"(?<![\w])" + re.escape(f) + r"(?![\w])", s):
                        bad.append(f"{r}:{i}: foreign container '{f}' outside tfs-003's border")
        # (d) deleted vocabulary stays deleted
        if r.startswith("crates/"):
            for w in EXCLUDED:
                for i, line in enumerate(text.splitlines(), 1):
                    s = code_only(line)
                    if not s:
                        continue
                    if re.search(r"(?<![\w])" + w + r"(?![\w])", s):
                        bad.append(f"{r}:{i}: deleted vocabulary '{w}': {s}")
    return bad, (f"{len(CODEC_MARKS)} owned formats, {len(GRAMMAR_TAGS)} grammar tags in "
                 f"{len(GRAMMAR_HOME)} file, {len(FOREIGN)} foreign parsers confined to "
                 f"{INGEST_OK_PREFIX}, {len(EXCLUDED)} deleted words")


# ----------------------------------------------------------------------------

def rel(p):
    return p.relative_to(ROOT)


FENCES = [("forbidden dependencies", fence_dependencies),
          ("no upward imports", fence_layering),
          ("marshal-only facade", fence_facade),
          ("one codec, one evaluator", fence_one_codec)]

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
