#!/usr/bin/env python3
"""Boundary fence (boundaries.md). Architecture enforcement, not a test.

Eight families:
  deps      forbidden dependencies (raw lines, incl. import paths and go.mod)
  impl      storage/chunk/loader/residency/tensor implementation vocabulary — cozy-creator
            renders and coordinates, it never implements the byte plane (TensorFS owns it)
  prompt    interactive prompts: no cozy command may ever ask a question (AXI)
  matrix    internal/exit/exit.go must equal docs/exit-matrix.md row for row
  manifest  a reclaiming/removing verb must DECLARE its gate: Destructive (exit 7 without
            --yes) or PlanFirst (a read without --yes) — exactly one, and it must
            advertise --yes. Nothing removes bytes on a bare invocation.
  env       (cl-001) the environment is read in internal/config/config.go and NOWHERE else:
            one entrypoint reader, a frozen typed value thereafter.
  store     (cl-001) no second lifecycle store: the ONE local libSQL database is the
            authority, so a state.json/pidfile-class sidecar name is a violation wherever
            it appears — those sidecars outlive their launcher and lie.
  cloud     (cl-001) no Tensorhub implementation and no cloud policy here: cozy-creator
            shares schemas and the client contract, and emulates nothing. Local grants are
            a CAS root plus an output dir; a minted bearer/JWT token would be a fake.

Identifier families scan Go source with comments and string literals removed, so a
word inside help text or a doc comment is never a violation. Doors, both greppable:
  //cozy:allow        this line is exempt from the impl family (state the reason)
  //cozy:stdin-value  this line reads stdin as a VALUE (e.g. --token-stdin), never a prompt
"""
import pathlib, re, sys

SCAN = ["go.mod", "cmd/**/*.go", "internal/**/*.go"]

DENY_DEPS = ["tensorhub-v2", "varena"]

DENY_IMPL = [
    "safetensors", "cozytensor", "tensorbytes", "tensorchunk", "chunk", "residency",
    "mmap", "dtype", "quantiz", "gguf", "loadtensor", "weightbytes", "pagein",
]

# Prompt-library qualifiers and password readers, as bare identifiers. Import paths
# are string literals and are stripped, so a prompt library shows up as its qualifier.
DENY_PROMPT = {"ReadPassword", "readline", "promptui", "survey"}

# fmt's scanners, QUALIFIED: `Scan` alone is also database/sql and bufio, neither of
# which reads a terminal. A scanner pointed at stdin is caught by the Stdin rule.
DENY_PROMPT_CALLS = [
    "fmt.Scan", "fmt.Scanln", "fmt.Scanf", "fmt.Fscan", "fmt.Fscanln", "fmt.Fscanf",
]

# The ONE environment reader. Every other package takes the frozen typed value.
ENV_READER = "internal/config/config.go"
DENY_ENV_CALLS = ["os.Getenv", "os.LookupEnv", "os.Environ", "syscall.Getenv", "syscall.Environ"]

# Lifecycle-sidecar names, matched on RAW lines (a comment naming one is still a plan to
# write one). The libSQL database is the sole lifecycle authority.
DENY_STORE = ["state.json", "status.json", "workers.json", "sessions.json", "pidfile", ".pidfile"]

# Cloud emulation and fabricated credentials, as bare identifiers.
DENY_CLOUD = {"tensorhub", "jwt", "Bearer", "SignedString", "mintToken", "ServiceClass"}

ALLOW_DOOR = "//cozy:allow"
STDIN_DOOR = "//cozy:stdin-value"

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def strip_go(src: str) -> str:
    """Blank out comments, string and rune literals; keep offsets and newlines."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j])); i = j
        elif c in ('"', "'", "`"):
            j = i + 1
            while j < n:
                if src[j] == "\\" and c != "`":
                    j += 2; continue
                if src[j] == c:
                    j += 1; break
                j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j])); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def files():
    for glob in SCAN:
        for p in sorted(pathlib.Path(".").glob(glob)):
            if p.is_file():
                yield p


def check_sources():
    bad = []
    for p in files():
        raw = p.read_text(errors="ignore")
        raw_lines = raw.splitlines()
        for i, line in enumerate(raw_lines, 1):
            s = line.strip()
            if s.startswith(("#", "//")):
                continue
            for d in DENY_DEPS:
                if re.search(r"(?<![\w.-])" + re.escape(d) + r"(?![\w-])", s, re.I):
                    bad.append(f"{p}:{i}: [deps] forbidden dependency '{d}': {s}")
            for d in DENY_STORE:
                if d in s.lower():
                    bad.append(f"{p}:{i}: [store] lifecycle sidecar '{d}' — the one libSQL "
                               f"database is the authority: {s}")
        if p.suffix != ".go":
            continue
        for i, line in enumerate(strip_go(raw).splitlines(), 1):
            src_line = raw_lines[i - 1] if i <= len(raw_lines) else ""
            idents = IDENT.findall(line)
            low = line.lower()
            if ALLOW_DOOR not in src_line:
                for d in DENY_IMPL:
                    if d in low:
                        bad.append(f"{p}:{i}: [impl] byte-plane vocabulary '{d}' — TensorFS owns it: {line.strip()}")
            for call in DENY_PROMPT_CALLS:
                if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", line):
                    bad.append(f"{p}:{i}: [prompt] '{call}' reads input — no cozy command may prompt: {line.strip()}")
            if str(p).replace("\\", "/") != ENV_READER:
                for call in DENY_ENV_CALLS:
                    if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", line):
                        bad.append(f"{p}:{i}: [env] '{call}' outside {ENV_READER} — the "
                                   f"environment is read once, at the entrypoint: {line.strip()}")
            for ident in idents:
                if ident in DENY_CLOUD:
                    bad.append(f"{p}:{i}: [cloud] '{ident}' — cozy-creator emulates no cloud "
                               f"policy and mints no credential: {line.strip()}")
                if ident in DENY_PROMPT:
                    bad.append(f"{p}:{i}: [prompt] '{ident}' reads input — no cozy command may prompt: {line.strip()}")
                if ident == "Stdin" and STDIN_DOOR not in src_line:
                    bad.append(f"{p}:{i}: [prompt] stdin read without the {STDIN_DOOR} door: {line.strip()}")
    return bad


# A verb whose name reclaims or removes must declare how it is gated.
RECLAIM_VERB = re.compile(r"\b(rm|gc|purge|delete|destroy|prune|reset|clean)\b")


def check_manifest():
    src_path = pathlib.Path("internal/manifest/commands.go")
    if not src_path.exists():
        return ["[manifest] missing internal/manifest/commands.go"]
    blocks = src_path.read_text().split("\n\t{\n")[1:]
    bad, rows = [], 0
    for block in blocks:
        body = block.split("\n\t},")[0]
        m = re.search(r"Path:\s*\[\]string\{([^}]*)\}", body)
        if not m:
            continue
        rows += 1
        name = " ".join(re.findall(r'"([^"]+)"', m.group(1)))
        destructive = "Destructive: true" in body
        plan_first = "PlanFirst: true" in body
        has_yes = "yesFlag" in body or '"--yes"' in body
        if RECLAIM_VERB.search(name) and not (destructive or plan_first):
            bad.append(f"[manifest] '{name}' reclaims or removes but declares neither "
                       "Destructive nor PlanFirst — a bare invocation would mutate silently")
        if has_yes and destructive == plan_first:
            joined = "both Destructive and PlanFirst" if destructive else "neither Destructive nor PlanFirst"
            bad.append(f"[manifest] '{name}' takes --yes but is {joined} — exactly one")
        if (destructive or plan_first) and not has_yes:
            bad.append(f"[manifest] '{name}' is gated on --yes but advertises no --yes flag")
    if not rows:
        return ["[manifest] no command rows parsed out of commands.go"]
    return bad


def parse_doc_matrix(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|$", line.strip())
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3)))
    return rows


def parse_go_matrix(path: pathlib.Path):
    src = path.read_text()
    consts = dict(re.findall(r"^\t([A-Za-z]+)\s+Code = (\d+)$", src, re.M))
    rows = []
    for name, label, meaning in re.findall(r'^\t\{([A-Za-z]+), "([a-z_]+)", "(.*)"\},$', src, re.M):
        if name not in consts:
            return None, f"matrix entry {label} uses undeclared constant {name}"
        rows.append((int(consts[name]), label, meaning.replace('\\"', '"')))
    return rows, None


def check_matrix():
    doc, go = pathlib.Path("docs/exit-matrix.md"), pathlib.Path("internal/exit/exit.go")
    if not doc.exists() or not go.exists():
        return [f"[matrix] missing {doc if not doc.exists() else go}"]
    expected = parse_doc_matrix(doc)
    got, err = parse_go_matrix(go)
    if err:
        return [f"[matrix] {err}"]
    if not expected:
        return ["[matrix] docs/exit-matrix.md has no table rows"]
    bad = []
    if len(expected) != len(got):
        bad.append(f"[matrix] {len(expected)} rows frozen in {doc}, {len(got)} in {go}")
    for e, g in zip(expected, got):
        if e != g:
            bad.append(f"[matrix] frozen {e} != code {g}")
    return bad


violations = check_sources() + check_matrix() + check_manifest()
if violations:
    print("FENCE RED (boundaries.md):", file=sys.stderr)
    for v in violations:
        print("  " + v, file=sys.stderr)
    sys.exit(1)
print(
    f"fence green — deps({len(DENY_DEPS)}) impl({len(DENY_IMPL)}) "
    f"prompt({len(DENY_PROMPT) + len(DENY_PROMPT_CALLS) + 1}) matrix(15 rows) "
    f"env({len(DENY_ENV_CALLS)}) store({len(DENY_STORE)}) cloud({len(DENY_CLOUD)}) "
    f"manifest({RECLAIM_VERB.pattern})"
)
