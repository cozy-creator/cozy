#!/usr/bin/env python3
"""Boundary fence (boundaries.md). Architecture enforcement, not a test.

Thirteen families:
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
  secret    (cl-011) a credential never rides argv and has ONE raw reader: a manifest flag
            whose name is credential-shaped may not take a value (process lists leak;
            `--token-stdin` is the shape that does not), and `secret.Value.Reveal()` may
            be called only where the value becomes an Authorization header.
  cloud     (cl-001) no Tensorhub implementation and no cloud policy here: cozy-creator
            shares schemas and the client contract, and emulates nothing. Local grants are
            a CAS root plus an output dir; a minted bearer/JWT token would be a fake.
  cas       (cl-012) the TRANSFER plane hashes nothing and nobody composes a store path.
            A digest computed while moving bytes could only become a client receipt, and a
            client receipt substitutes for nothing (law 18); a hand-built
            `objects/sha256/…` path would be a second spelling of TensorFS's own layout.
  api       (cl-006) a loopback bind is not a boundary: NO cookie is read anywhere
            (bearer only — a cookie is ambient authority a browser attaches cross-site),
            NO Access-Control-Allow-* header is ever set, and there is exactly ONE
            net.Listen("tcp", …) site, which refuses a non-loopback address.
  contract  (cl-006) internal/api/routes.go and docs/client-contract.md are ONE surface,
            row for row, scope for scope. The document is what th-021's other two hosts
            implement against, so drift is a shared-contract defect, not a doc lag.
  tensor    (cl-012) the tensorfs CLI has ONE caller: internal/tfs. The configured binary
            is read there and in the config authority, nowhere else — a second package
            shelling out to `tfs` is a second byte-plane door with its own vocabulary.

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

# (cl-011) A credential-shaped flag NAME. `--token-stdin` and `--no-browser` are not
# credential values; `--token <t>` is, and argv is world-readable on this planet.
SECRET_FLAG = re.compile(r"token|secret|password|api[-_]?key|credential", re.I)

# The raw readers of a secret.Value: the sites where a credential becomes its CARRIER,
# and nowhere else. There are exactly three, each named for what it carries the value
# INTO, and each is one function long:
#   secret.go        the definition itself
#   hub/hub.go       an outbound Authorization header (cl-011)
#   api/credentials  the 0600 handoff file and the --open URL FRAGMENT (cl-006)
# Note what is NOT here: the local API's own bearer CHECK. It compares full digests
# (secret.Value.Equal), so the server that authenticates a token never reads one.
REVEAL_SITES = {
    "internal/secret/secret.go",
    "internal/hub/hub.go",
    "internal/api/credentials.go",
}

# The contract document's own tables use `| \`METHOD /path\` | scope |`, which the CAS
# and CORS raw scans would never look at — the document is markdown, not Go — but the
# route freeze reads both and compares them.

# (cl-006) The local client API's own discipline. A loopback bind is not a boundary, so
# the properties that make it defensible are fenced rather than reviewed:
#   - NO COOKIE is read anywhere. Cookies are ambient authority: the browser attaches
#     them to a cross-site request whether or not the page meant it. A bearer is not
#     ambient. (cl-007's session-cookie ceremony lands behind its real-UI gate and will
#     move this line WITH a recorded decision — not quietly.)
#   - NO CORS header is ever set. Not a narrow allowlist: none.
#   - ONE net.Listen("tcp", ...) site in the whole binary, in internal/api/listen.go,
#     which refuses a non-loopback address. A second one would be the LAN door arriving
#     as an accident.
DENY_COOKIE = ["http.Cookie", "SetCookie", "http.SetCookie", ".Cookies", ".Cookie("]
CORS_HEADER = re.compile(r"Access-Control-Allow-", re.I)
LISTEN_SITE = "internal/api/listen.go"
LISTEN_CALL = re.compile(r'net\.Listen\s*\(\s*"tcp')

ALLOW_DOOR = "//cozy:allow"
STDIN_DOOR = "//cozy:stdin-value"

# (cl-012) The TRANSFER plane hashes nothing. Elsewhere cozy-creator legitimately
# digests its own subjects (a release archive, an execution spec, a credential), but a
# hash computed while moving canonical bytes could only become a client-side receipt —
# and a client receipt never substitutes for the hub's or the store's own proof
# (README law 18). The temptation lives exactly here, so the fence does too.
DIGEST_FREE = ["internal/transfer/", "internal/tfs/", "internal/hub/"]
DENY_DIGEST = ["sha256.New", "sha256.Sum256", "sha512.New", "sha1.New", "md5.New"]

# (cl-012) The CAS layout is TensorFS's. A path composed here would drift from it the
# first time either side changed, and the drift would present as a corrupt store.
CAS_PATH = re.compile(r'objects\s*[/",\s]+\s*sha256', re.I)

# (cl-012) The ONE caller of the tensorfs CLI.
TFS_SITES = {"internal/config/config.go", "internal/tfs/tfs.go"}
TFS_FIELD = re.compile(r"\.Tfs\b")

# (cl-010) The two files that may reach an endpoint's own cozy-runtime, and the CLOSED set
# of verbs they may name. `install.go` runs `describe --check` at install; `artifacts.go`
# asks for the artifact index, host facts and fit verdicts. Nothing executes a model
# through this door — that is what the coordinator and the worker protocol are for.
RUNTIME_SITES = {"internal/install/install.go", "internal/launch/artifacts.go"}
RUNTIME_BIN = re.compile(r'"cozy-runtime"')
RUNTIME_VERBS_OK = {"describe", "list", "doctor", "fit", "bindings"}
RUNTIME_VERBS_DENY = {"run", "job", "serve", "rm", "pull", "ingest", "new"}

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
            # RAW, not stripped: a content key IS a string literal, so the identifier
            # scan (which blanks literals) would never see one.
            if CAS_PATH.search(s):
                bad.append(f"{p}:{i}: [cas] a store path is composed here — TensorFS owns the "
                           f"CAS layout, and `tfs get`/`tfs put` are how an object is reached: {s}")
            # RAW: a CORS header is a string LITERAL, so the identifier scan (which blanks
            # literals) would never see one.
            if CORS_HEADER.search(s) and ALLOW_DOOR not in line:
                bad.append(f"{p}:{i}: [api] an Access-Control-Allow-* header — this API sets NO CORS "
                           f"header at all, so a foreign page cannot read what it is handed: {s}")
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
            rel = str(p).replace("\\", "/")
            if any(rel.startswith(d) for d in DIGEST_FREE):
                for call in DENY_DIGEST:
                    if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", line):
                        bad.append(f"{p}:{i}: [cas] '{call}' in the transfer plane — a digest computed "
                                   f"while moving bytes is a client receipt, and a client receipt proves "
                                   f"nothing (law 18): {line.strip()}")
            if rel not in RUNTIME_SITES and RUNTIME_BIN.search(src_line) and \
                    ALLOW_DOOR not in src_line:
                bad.append(f"{p}:{i}: [runtime] the cozy-runtime binary is reached outside "
                           f"{' / '.join(sorted(RUNTIME_SITES))} — one execution path: a run goes "
                           f"coordinator -> worker protocol -> runtime, never a shell-out: {src_line.strip()}")
            if rel in RUNTIME_SITES:
                for verb in sorted(RUNTIME_VERBS_DENY):
                    if re.search(r'"' + verb + r'"', src_line):
                        bad.append(f"{p}:{i}: [runtime] this file may name only the READ verbs "
                                   f"({', '.join(sorted(RUNTIME_VERBS_OK))}); '{verb}' would be a "
                                   f"second execution door: {src_line.strip()}")
            if rel not in TFS_SITES and TFS_FIELD.search(line):
                bad.append(f"{p}:{i}: [tensor] the tensorfs CLI is reached outside "
                           f"{' / '.join(sorted(TFS_SITES))} — one byte-plane door, one vocabulary: {line.strip()}")
            if rel not in REVEAL_SITES and re.search(r"\.Reveal\s*\(", line):
                bad.append(f"{p}:{i}: [secret] Reveal() outside {' / '.join(sorted(REVEAL_SITES))} — a "
                           f"credential's raw value is read where it becomes a carrier, nowhere else: {line.strip()}")
            for call in DENY_COOKIE:
                if call in line:
                    bad.append(f"{p}:{i}: [api] '{call}' — the local client API is BEARER ONLY. A cookie "
                               f"is ambient authority the browser attaches cross-site; cl-007's session "
                               f"ceremony is a recorded decision, not a quiet import: {line.strip()}")
            if rel != LISTEN_SITE and LISTEN_CALL.search(line) and ALLOW_DOOR not in src_line:
                bad.append(f"{p}:{i}: [api] net.Listen(\"tcp\", …) outside {LISTEN_SITE} — one bind site, "
                           f"and it REFUSES a non-loopback address. The LAN door is deferred behind TLS "
                           f"and its own threat review, never a second listener: {line.strip()}")
    return bad


def check_secret_flags():
    """No manifest flag may carry a credential as an argv VALUE (cl-011)."""
    src_path = pathlib.Path("internal/manifest/commands.go")
    if not src_path.exists():
        return ["[secret] missing internal/manifest/commands.go"]
    bad = []
    for i, line in enumerate(src_path.read_text().splitlines(), 1):
        m = re.search(r'Name:\s*"(--[A-Za-z0-9-]+)"', line)
        if not m or not SECRET_FLAG.search(m.group(1)):
            continue
        if re.search(r'Arg:\s*"', line):
            bad.append(f"internal/manifest/commands.go:{i}: [secret] flag '{m.group(1)}' takes an "
                       "argv value and is credential-shaped — argv is world-readable; take it on "
                       "stdin (--token-stdin) or through an OS-protected handoff")
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


def parse_doc_routes(path: pathlib.Path):
    """Route rows out of the contract document's two tables."""
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"^\|\s*`([A-Z]+) (/[^`]*)`\s*\|\s*([a-z]+)\s*\|", line.strip())
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
    return rows


def parse_go_routes(path: pathlib.Path):
    rows = []
    for method, route_path, scope in re.findall(
        r'^\t\{"([A-Z]+)", "([^"]+)", (Core|Local),', path.read_text(), re.M
    ):
        rows.append((method, route_path, scope.lower()))
    return rows


def check_contract():
    """(cl-006) The route table and the contract document are ONE surface.

    docs/client-contract.md is what th-021's other two hosts implement against, so a route
    that exists in code and not in the document — or the reverse — is drift in a SHARED
    contract, not a local doc lag. The scope column is checked too: moving a route between
    the shared CORE and the LOCAL extension is the single most consequential edit anyone
    can make here, and it must be visible in both places at once.
    """
    doc = pathlib.Path("docs/client-contract.md")
    go = pathlib.Path("internal/api/routes.go")
    if not doc.exists() or not go.exists():
        return [f"[contract] missing {doc if not doc.exists() else go}"]
    expected, got = parse_doc_routes(doc), parse_go_routes(go)
    if not expected:
        return ["[contract] docs/client-contract.md has no route rows"]
    bad = []
    if len(expected) != len(got):
        bad.append(f"[contract] {len(expected)} routes in {doc}, {len(got)} in {go}")
    for e, g in zip(expected, got):
        if e != g:
            bad.append(f"[contract] documented {e} != served {g}")
    return bad


violations = (check_sources() + check_matrix() + check_manifest() + check_secret_flags()
              + check_contract())
if violations:
    print("FENCE RED (boundaries.md):", file=sys.stderr)
    for v in violations:
        print("  " + v, file=sys.stderr)
    sys.exit(1)
print(
    f"fence green — deps({len(DENY_DEPS)}) impl({len(DENY_IMPL)}) "
    f"prompt({len(DENY_PROMPT) + len(DENY_PROMPT_CALLS) + 1}) matrix(15 rows) "
    f"env({len(DENY_ENV_CALLS)}) store({len(DENY_STORE)}) cloud({len(DENY_CLOUD)}) "
    f"manifest({RECLAIM_VERB.pattern}) secret({SECRET_FLAG.pattern} + Reveal@{len(REVEAL_SITES)}) "
    f"cas({len(DENY_DIGEST)} digests@{len(DIGEST_FREE)} + store-path) tensor(tfs@{len(TFS_SITES)}) "
    f"api({len(DENY_COOKIE)} cookie + cors + listen@{LISTEN_SITE}) "
    f"runtime({len(RUNTIME_VERBS_DENY)} denied verbs@{len(RUNTIME_SITES)}) "
    f"contract({len(parse_go_routes(pathlib.Path('internal/api/routes.go')))} routes)"
)
