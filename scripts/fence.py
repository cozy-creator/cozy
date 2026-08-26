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

RANK = {"tensorfs-core": 0, "tensorfs-py": 1}


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
    "tensorfs.meta/1": {"crates/tensorfs-core/src/meta.rs"},
    "tensorfs.operation_receipt/1": {"crates/tensorfs-core/src/receipt.rs"},
    # the layout plane (tfs-009)
    "tensorfs.stamp/1": {"crates/tensorfs-core/src/ingest/stamp.rs"},
    "cozytensors.code_topology/1": {"crates/tensorfs-core/src/compat.rs"},
    "tensorfs.compatibility_report/1": {"crates/tensorfs-core/src/compat.rs"},
    "tensorfs.conversion_registration/1": {"crates/tensorfs-core/src/grade.rs"},
    "tensorfs.grade_decision/1": {"crates/tensorfs-core/src/grade.rs"},
    "tensorfs.verify_request/1": {"crates/tensorfs-core/src/verifier.rs"},
    "tensorfs.verify_response/1": {"crates/tensorfs-core/src/verifier.rs"},
    # the cloud publish protocol (tfs-017)
    "tensorfs.publish_closure/1": {"crates/tensorfs-core/src/cloud.rs"},
    "tensorfs.publish_grant/1": {"crates/tensorfs-core/src/cloud.rs"},
    "tensorfs.verified_object/1": {"crates/tensorfs-core/src/cloud.rs"},
    "tensorfs.catalog_root/1": {"crates/tensorfs-core/src/cloud.rs"},
    # the pull plane (tfs-003 row 220) — the same rule in the other direction
    "tensorfs.fetch_plan/1": {"crates/tensorfs-core/src/fetch.rs"},
    "tensorfs.delivery_grant/1": {"crates/tensorfs-core/src/fetch.rs"},
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
    """One line with comments and string literals removed — what the code SAYS, not what
    it carries as data. Line-local, so it cannot see a literal that spans lines; use
    `code_lines` for anything scanning a whole file."""
    s = line.strip()
    if s.startswith(("//", "//!", "#")):
        return ""
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    return s


def code_lines(text):
    """(lineno, code) over a WHOLE file, with comments and string literals blanked by a
    single pass that carries state across newlines.

    The line-local version was blind to Rust's `"..." \\` continuations, so a refusal message
    wrapped over two lines read as code and fired every prose-matching fence below. A fence
    that cannot tell a sentence from an identifier is a fence people learn to route around.
    """
    out, buf = [], []
    i, n = 0, len(text)
    state = "code"       # code | line_comment | block_comment | string | raw_string
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state, i = "line_comment", i + 2
                continue
            if c == "/" and nxt == "*":
                state, i = "block_comment", i + 2
                continue
            if c == '"':
                state, i = "string", i + 1
                buf.append('""')
                continue
            buf.append(c)
        elif state == "line_comment":
            if c == "\n":
                state = "code"
                buf.append(c)
                i += 1
                continue
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state, i = "code", i + 2
                continue
            if c == "\n":
                buf.append(c)
        elif state == "string":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                state = "code"
            elif c == "\n":
                buf.append(c)
        i += 1
    for no, line in enumerate("".join(buf).splitlines(), 1):
        out.append((no, line.strip()))
    return out


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
        # The exemption below is for RUST files inside the Rust crate — the CLI is a
        # consumer of the one implementation. A Go, Python or TS file sitting in the Rust
        # source tree and naming a format is unambiguously a second implementation, whatever
        # else it does or does not define.
        inside = r.startswith("crates/") and p.suffix == ".rs"
        implements = "impl Doc for" in text
        for m, owners in CODEC_OWNERS.items():
            if m not in text or r in owners:
                continue
            if inside and not implements:
                continue    # a consumer naming or planting the mark, not a second codec
            if inside:
                why = "defines its own Doc codec"
            elif r.startswith("crates/"):
                why = f"is a {p.suffix} file inside the Rust crate"
            else:
                why = "sits outside the owning crate"
            bad.append(f"{r}: names format '{m}' owned by {sorted(owners)} and {why}"
                       " — a rival implementation of an identity-bearing codec")
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


# ------------------------------------------------- 5. the serve path stays a serve path

# tfs-005's fence, in two halves.
#
# (a) SERVE-PATH PURITY. The read/fill/projection/metadata modules are what a warm worker
#     runs. A carrier parser, a fingerprinter or a classifier reachable from there means an
#     artifact's LAYOUT is being re-decided at serve time, from bytes, after the border
#     already decided it once — the v1 two-deciders defect with the expensive half moved to
#     the latency path. `tfs-003` confines foreign CONTAINERS to the border already; this
#     confines the border's own decision vocabulary.
# `compat.rs` joins the list at tfs-009: check_compatibility is what a WORKER runs at bind,
# before any allocation, and a marker taxonomy or a fingerprinter reachable from there would
# be exactly the border's decision made a second time on the latency path. The marker
# vocabulary it needs arrives as DATA from the caller.
SERVE_PATH = ["crates/tensorfs-core/src/read.rs", "crates/tensorfs-core/src/fill.rs",
              "crates/tensorfs-core/src/project.rs", "crates/tensorfs-core/src/meta.rs",
              "crates/tensorfs-core/src/gc.rs", "crates/tensorfs-core/src/receipt.rs",
              "crates/tensorfs-core/src/compat.rs"]
# Module PATHS and type names, not the English nouns: tier 3 legitimately streams "a
# retained carrier blob" in its refusal prose, and a fence that cannot tell a noun from an
# import is a fence people route around.
BORDER_VOCAB = ["ingest::carrier", "ingest::convert", "ingest::fingerprint", "carrier::",
                "fingerprint::", "convert::", "SourceHeader", "IngestProfile", "Converter",
                "classify", "topology_fingerprint"]
# `ingest::transaction` is deliberately NOT on that list. GC must mark the border's expiring
# quarantine roots — they are pin root class 4 (tensorfs.md §2) — so the ROOT plane is a real
# seam between the two. What stays banned is the DECIDING half: carrier decode, conversion
# planning and classification. A fence that cannot tell a root from a parser would force the
# GC to re-implement the root reader, which is the duplication it exists to prevent.

# (b) PINNED BUFFERS ARE NOT OURS. boundaries.md gives TensorFS bytes and leases and gives
#     the runtime GPU memory and pinned staging. Exactly ONE budgeted pinned pool per
#     executor lives with the CALLER (the v1 FillSink lesson); the moment core names a pinned
#     pool it has taken a policy it cannot budget and an allocation it cannot free.
# The MEMORY sense only. Bare "pinned" was over-broad: pinned-by-a-root is this file's
# central concept and a fence that cannot tell it from pinned host memory just teaches people
# to spell it `pinned_by`. What is actually forbidden is owning a host staging allocation.
PINNED_VOCAB = ["pinned_pool", "pinned_buffer", "pinned_buffers", "pinned_staging",
                "pinned_memory", "pin_memory", "cudaHostAlloc", "page_locked", "PinnedPool",
                "StagingPool", "staging_pool", "host_register", "mlock"]

# (c) ONE TRANSFORM. The fill transform is defined once; everyone else re-exports it.
XFORM_HOME = "crates/tensorfs-core/src/fill.rs"


def fence_serve_path():
    bad = []
    for p in sorted((ROOT / "crates").glob("**/*.rs")):
        r = str(rel(p))
        text = p.read_text(errors="ignore")
        lines = list(enumerate(text.splitlines(), 1))
        if r in SERVE_PATH:
            for i, line in lines:
                s = code_only(line)
                if not s:
                    continue
                for w in BORDER_VOCAB:
                    hit = w in s if "::" in w else re.search(
                        r"(?<![\w])" + re.escape(w) + r"(?![\w])", s)
                    if hit:
                        bad.append(f"{r}:{i}: serve path names border vocabulary '{w}' — "
                                   "layout is decided once, at the border, never at read time")
        for i, line in lines:
            s = code_only(line)
            if not s:
                continue
            for w in PINNED_VOCAB:
                if re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", s):
                    bad.append(f"{r}:{i}: core names '{w}' — pinned staging belongs to the "
                               "caller's one budgeted pool, never to TensorFS")
        if r != XFORM_HOME and re.search(r"(enum|impl)\s+Xform\b", text):
            bad.append(f"{r}: defines Xform outside {XFORM_HOME} — the fill transform has "
                       "ONE implementation; every other site re-exports it")
    return bad, (f"{len(SERVE_PATH)} serve-path files, {len(BORDER_VOCAB)} border words, "
                 f"{len(PINNED_VOCAB)} pinned-buffer words, one Xform in {XFORM_HOME}")


# ------------------------------------------------- 6. no grading below the hub

# tfs-009's fence. There is ONE grading layer and it is the hub's. `check_compatibility`
# returns the ONE runtime-satisfaction vocabulary — `Satisfied | Mismatch(details)` — and a
# `verdict`, `grade` or `remedy` field anywhere in it would be a third vocabulary for one
# decision chain, which is precisely the earlier mandatory-`derivable` design the audit
# removed. The four-verdict ladder lives in ONE module; the CLI may print it.
GRADE_VOCAB = ["COMPATIBLE", "CONVERTIBLE", "PRODUCIBLE", "INCOMPATIBLE",
               "Compatible", "Convertible", "Producible", "Incompatible", "remedy"]
GRADE_HOME = {"crates/tensorfs-core/src/grade.rs",
              "crates/tensorfs-core/src/bin/tfs/layout.rs"}


def fence_grading():
    bad = []
    for p in sorted((ROOT / "crates").glob("**/*.rs")):
        r = str(rel(p))
        if r in GRADE_HOME:
            continue
        for i, s in code_lines(p.read_text(errors="ignore")):
            if not s:
                continue
            for w in GRADE_VOCAB:
                if re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", s):
                    bad.append(f"{r}:{i}: names grading vocabulary '{w}' outside "
                               f"{sorted(GRADE_HOME)} — grading a mismatch is the HUB's job")
    return bad, (f"{len(GRADE_VOCAB)} grade words confined to {len(GRADE_HOME)} files")


# ------------------------------------------------- 7. the staged fallback stays unbuilt

# tfs-017's fence. The gate came back GREEN against live R2, so publish writes CONDITIONAL
# CHECKSUMMED PUTs at final keys and the 5-part staged protocol — session-scoped staging
# keys, a staged-verifier pass, a promotion copy — is the fallback that was NOT built. A
# staging namespace appearing later would be a second write path with a second set of
# races, arriving without the qualification that would justify it.
#
# Re-arming is a code change here plus a RED gate to point at, which is the point.
STAGING_VOCAB = ["staging_key", "staging_keys", "staging_namespace", "staged_verifier",
                 "promotion_copy", "promote_staged", "stage_upload", "StagingKey",
                 "StagedObject", "PromotionCopy"]


def fence_staging():
    bad = []
    for p in sorted((ROOT / "crates").glob("**/*.rs")):
        r = str(rel(p))
        for i, s in code_lines(p.read_text(errors="ignore")):
            if not s:
                continue
            for w in STAGING_VOCAB:
                if re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", s):
                    bad.append(f"{r}:{i}: names staged-protocol vocabulary '{w}' — the "
                               "qualification was GREEN, so the staged fallback is not built")
    return bad, f"{len(STAGING_VOCAB)} staged-protocol words, none built"


# ----------------------------------------------------------------------------

def rel(p):
    return p.relative_to(ROOT)


FENCES = [("forbidden dependencies", fence_dependencies),
          ("no upward imports", fence_layering),
          ("marshal-only facade", fence_facade),
          ("one codec, one evaluator", fence_one_codec),
          ("serve path, pinned buffers, one transform", fence_serve_path),
          ("no grading below the hub", fence_grading),
          ("the staged fallback stays unbuilt", fence_staging)]

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
