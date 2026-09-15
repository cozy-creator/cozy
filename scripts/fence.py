#!/usr/bin/env python3
"""Boundary fence (boundaries.md). Architecture enforcement, not a test.

Enforced families:
  deps      forbidden dependencies (raw lines, incl. import paths and go.mod)
  impl      the CANONICAL FORMATS TensorFS parses. A second reader of safetensors/gguf/
            cozytensors here would drift from the one that produced the published bytes.
            (Narrowed 2026-08-28: the generic systems words — chunk, mmap, dtype,
            residency, quantiz, pagein — were a vocabulary ban with no failure behind
            them; they fired on the worker protocol's own field names.)
  prompt    interactive prompts: no cozy command may ever ask a question (AXI)
  matrix    internal/exit/exit.go must equal docs/exit-matrix.md row for row
  grammar   internal/cli/grammar.go is the sole 19-command Kong tree; the retired manifest
            and command families stay absent, and -v/-V/--version remain a pre-config fast path.
  output    internal/output writes one typed document to stdout; progress alone uses stderr.
  env       (cl-001) the environment is read in internal/config/config.go and NOWHERE else:
            one entrypoint reader, a frozen typed value thereafter.
  store     (cl-001) no second lifecycle store: the ONE local SQLite database is the
            authority, so a state.json/pidfile-class sidecar name is a violation in CODE
            — those sidecars outlive their launcher and lie. A whole-line comment naming
            one is NOT scanned (check_sources skips lines starting with `//`), and the
            docstring said otherwise until 2026-08-28; a word in prose was never the
            hazard, writing the sidecar is.
  secret    (cl-011) a credential never rides argv and has ONE raw reader: a Kong field
            whose name is credential-shaped may not take a value (process lists leak;
            `--token-stdin` is the shape that does not), and `secret.Value.Reveal()` may
            be called only where the value becomes an Authorization header.
  cas       (cl-012) nobody composes a store path: a hand-built `objects/sha256/…` path
            would be a second spelling of TensorFS's own layout, and would silently read
            the wrong object the first time either side moved.
  api       (cl-006) a loopback bind is not a boundary: NO cookie is read anywhere
            (bearer only — a cookie is ambient authority a browser attaches cross-site),
            an Access-Control-Allow-* header needs a //cozy:allow door stating where its
            origins come from, and inside internal/api the rule is ABSOLUTE — no door
            (#628): a CORS header on a loopback bind is what makes it readable by any
            page the browser loads. And there is exactly ONE net.Listen(…) site per
            program, each with its own stated rule.
  contract  internal/api/routes.go and docs/client-contract.md have the same route
            method, path, scope, and order. Payload and behavior are tested separately.
  tensor    (cl-012) the tensorfs CLI has ONE caller: internal/tfs. The configured binary
            is read there and in the config authority, nowhere else — a second package
            shelling out to `tfs` is a second byte-plane door with its own vocabulary.
  resources package and model are the product nouns. Generic repo/create/show routes and
            untyped model-transfer verbs are refused, not retained as aliases.

Identifier families scan Go source with comments and string literals removed, so a
word inside help text or a doc comment is never a violation. Doors, both greppable:
  //cozy:allow        this line is exempt from the impl family (state the reason)
  //cozy:stdin-value  this line reads stdin as a VALUE (e.g. --token-stdin), never a prompt
"""
import os, pathlib, re, sys

SCAN = ["go.mod", "main.go", "cmd/**/*.go", "internal/**/*.go", "tests/**/*.go"]

DENY_DEPS = ["tensorhub-v2", "varena"]
YAML_IMPORT = "go.yaml.in/yaml/v3"

# The canonical carriers TensorFS parses. A second reader here would drift from the one
# that produced the published bytes. The generic systems words that used to sit beside
# them (chunk, mmap, dtype, residency, quantiz, pagein) were deleted 2026-08-28: they
# banned ordinary Go vocabulary — including the worker protocol's own ComputeDtype and
# Placement field names — and named no failure the code would otherwise have had.
DENY_IMPL = [
    "safetensors", "cozytensor", "tensorbytes", "tensorchunk", "loadtensor",
    "weightbytes", "gguf",
]

# Prompt-library qualifiers and password readers, as bare identifiers. Import paths
# are string literals and are stripped, so a prompt library shows up as its qualifier.
DENY_PROMPT = {"ReadPassword", "readline", "promptui", "survey"}

# fmt's scanners, QUALIFIED: `Scan` alone is also database/sql and bufio, neither of
# which reads a terminal. A scanner pointed at stdin is caught by the Stdin rule.
DENY_PROMPT_CALLS = [
    "fmt.Scan", "fmt.Scanln", "fmt.Scanf", "fmt.Fscan", "fmt.Fscanln", "fmt.Fscanf",
]

# The `cozy` process has one environment reader. Its inherited allowlist is
# class A of tracker-v2/spawn-allowlists.md (#616.d); every other package takes
# the frozen typed value.
ENV_READERS = {
    "internal/config/config.go": "the cozy CLI's entrypoint reader",
}
ENV_READER = " / ".join(sorted(ENV_READERS))
DENY_ENV_CALLS = ["os.Getenv", "os.LookupEnv", "os.Environ", "syscall.Getenv", "syscall.Environ"]

# Lifecycle-sidecar names, matched on RAW code lines — raw because a sidecar name is a
# string literal, which the identifier scan blanks. Whole-line comments are skipped by
# check_sources, so this catches the sidecar being WRITTEN, not the word being said.
# The SQLite database is the sole lifecycle authority.
DENY_STORE = ["state.json", "status.json", "workers.json", "sessions.json", "pidfile", ".pidfile"]

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
    # The flip's one addition (#436/#463): the owner PRESENTS the credential to the
    # worker as `Claim.proof` — the exact dual of the old metadata echo, and the only
    # place the value leaves this process (over the worker's own channel).
    "internal/orchestrator/owner.go",
    # The byte plane's (#506b): the rental's media bearer becomes the Authorization
    # header for the POD's media server. Same rule as hub/hub.go one plane over — one
    # request builder, and the raw value is read exactly where it becomes a carrier.
    "internal/media/media.go",
    # cl-005: the local model-source client puts the selected provider token
    # directly into that provider's Authorization header.
    "internal/modelsource/provider.go",
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
# (#628, owner ruling 2026-08-27) Inside the owner's loopback API the no-CORS
# rule is absolute: that surface is the DNS-rebinding target.
CORS_ABSOLUTE = "internal/api/"
# The Cozy binary binds once, on loopback, for its local client API.
LISTEN_SITES = {
    "internal/api/listen.go": "the owner's local client API — loopback only",
}
LISTEN_SITE = " / ".join(sorted(LISTEN_SITES))
# (cl-028 tightening) ANY net.Listen* call, not only the literal-"tcp" spelling: a bind
# whose network rides a variable was invisible to the old regex, and a copycat unix or
# ListenTCP bind is the same second door.
LISTEN_CALL = re.compile(r"net\.Listen\w*\s*\(")

# (cl-031) Cozy has one declaration of the media contract it expects.
# Tensorhub's independently shipped server declares its own revision; the live
# health handshake refuses skew before this client moves bytes.
MEDIA_CONTRACT_HOME = "internal/mediawire/wire.go"
MEDIA_CONTRACT_FIELDS = ["contract_rev"]
# Package distribution is the worker's signed-intent Tensorhub lane. The invocation
# media plane may never grow back the retired owner-push special case for binding plans.
MEDIA_DISTRIBUTION_DIRS = ("internal/media/",)
DENY_MEDIA_DISTRIBUTION = ("PutPlan", "/v1/plans/")

ALLOW_DOOR = "//cozy:allow"
STDIN_DOOR = "//cozy:stdin-value"

# (cl-012) The CAS layout is TensorFS's. A path composed here would drift from it the
# first time either side changed, and the drift would present as a corrupt store.
CAS_PATH = re.compile(r'objects\s*[/",\s]+\s*sha256', re.I)

# (cl-012) The ONE caller of the tensorfs CLI.
TFS_SITES = {"internal/config/config.go", "internal/tfs/tfs.go"}
TFS_FIELD = re.compile(r"\.Tfs\b")

# (cl-010) The files that may reach a package's own cozy-runtime, and the CLOSED set
# of verbs they may name. Publication and install derive the package interface once; launch
# may ask for host facts and fit verdicts. Nothing executes a model
# through these doors — that is what the orchestrator and worker protocol are for.
RUNTIME_SITES = {
    "internal/install/install.go",
    "internal/install/published.go",
    "internal/launch/artifacts.go",
    "internal/hostruntime/hostruntime.go",
    "internal/packagepublish/package.go",
}
RUNTIME_BIN = re.compile(r'"cozy-runtime"')
# (cl-028) The INDIRECTIONS to the same binary, which the literal above cannot see:
# `launch.Binary()` resolves the install venv's cozy-runtime and `launch.RuntimeCLI{}`
# is its invoker. Product code outside internal/launch reaching either is the same second
# execution door the literal rule refuses; only the verification driver may (with a door).
RUNTIME_INDIRECT = re.compile(r"launch\.Binary\s*\(|launch\.RuntimeCLI\s*\{")
RUNTIME_INDIRECT_HOME = "internal/launch/"
RUNTIME_VERBS_OK = {"describe", "doctor", "fit", "bindings", "version"}
RUNTIME_VERBS_DENY = {"run", "job", "serve", "rm", "pull", "ingest", "new"}

# (cl-028) EMBEDDED SCRIPTS ARE SOURCE TOO. The impl family scans Go with string literals
# blanked, so a Python script inside a raw string could author byte-plane knowledge the
# fence never saw — `tensorfs.parse_header` and a hand-summed tensor table already did.
# Every multi-line Go string literal that looks like a script (it imports something) is
# scanned for the byte-plane vocabulary below; the door is //cozy:allow on the line the
# literal starts on. scripts/*.py are scanned with the same vocabulary.
DENY_EMBED = [
    "parse_header", "safetensors", "cozytensor", "tensorbytes", "tensorchunk",
    "loadtensor", "weightbytes", "gguf",
]
PY_SCAN = "scripts/*.py"
# fence.py names the vocabulary in order to deny it. No package implementation is kept
# under scripts/, so it is the only Python source exempt from its own vocabulary scan.
PY_ALLOW = {
    "scripts/fence.py": "the fence itself: it spells the vocabulary to deny it",
}

# (cl-028) The PRODUCT-TEST HOME is the only place an //cozy:allow door may exempt a listen
# or a runtime indirection. `tests/product` drives the product and `tests/support/fakeworker`
# is the independent protocol peer it spawns; both bind sockets by design. Product code gets no
# door for either — a doored non-loopback bind in the product would be the LAN door arriving as a comment.
DRIVER_DIRS = ("tests/product/", "tests/support/fakeworker/")
DRIVER_DIR = " / ".join(DRIVER_DIRS)

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


def go_raw_literals(src: str):
    """Yield (start_line, text) for every backtick raw-string literal in Go source."""
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j); i = j
        elif c in ('"', "'"):
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src[j] == c:
                    j += 1; break
                j += 1
            line += src.count("\n", i, j); i = j
        elif c == "`":
            j = src.find("`", i + 1)
            j = n if j < 0 else j
            yield line, src[i + 1:j]
            line += src.count("\n", i, j); i = j + 1
        else:
            if c == "\n":
                line += 1
            i += 1


def check_embedded():
    """(cl-028) Multi-line script literals in Go source scan like source, not like prose."""
    bad = []
    for p in files():
        if p.suffix != ".go":
            continue
        raw = p.read_text(errors="ignore")
        raw_lines = raw.splitlines()
        for start, text in go_raw_literals(raw):
            if "\n" not in text or "import " not in text:
                continue  # not a script — a format string or a fixture blob
            opener = raw_lines[start - 1] if start <= len(raw_lines) else ""
            if ALLOW_DOOR in opener:
                continue
            for offset, line in enumerate(text.splitlines()):
                body = line.split("#", 1)[0].lower()
                for d in DENY_EMBED:
                    if d in body:
                        bad.append(
                            f"{p}:{start + offset}: [embed] byte-plane vocabulary '{d}' inside an "
                            f"embedded script — TensorFS/Runtime own it; delegate to a runtime "
                            f"verb instead of authoring tensor knowledge here: {line.strip()}")
    return bad


def check_scripts():
    """(cl-028) scripts/*.py are in the SCAN set, with the explicit fixture allows."""
    bad = []
    for p in sorted(pathlib.Path(".").glob(PY_SCAN)):
        rel = str(p).replace("\\", "/")
        if rel in PY_ALLOW:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            body = line.split("#", 1)[0].lower()
            for d in DENY_EMBED:
                if d in body:
                    bad.append(f"{p}:{i}: [embed] byte-plane vocabulary '{d}' in a repo script — "
                               f"only the named fixture allows carry it: {line.strip()}")
    return bad


def files():
    for glob in SCAN:
        for p in sorted(pathlib.Path(".").glob(glob)):
            if p.is_file():
                yield p


def check_test_boundary():
    """(#661) Go verification has one home: the product suite in tests/product."""
    bad = []
    for p in sorted(pathlib.Path(".").rglob("*_test.go")):
        rel = p.as_posix()
        if not rel.startswith("tests/product/"):
            bad.append(f"{rel}: [test] *_test.go outside tests/product — #661 keeps one "
                       "real-system verification home and no package-local mocked/unit layer")
    return bad


DOCUMENT_KINDS = {}
# HMAC domain-separation tags are security protocol constants, not document formats. A peer repo
# reproduces these exact bytes, so they remain single-owner fenced without inflating the document
# count.
HMAC_DOMAINS = {
    "cozy.rental_request/1": "internal/cli/rentals.go",
}
# Kinds another repo authors and this one only reads: the owner's fence polices the name.
FOREIGN_KIND_PREFIXES = ("cozy.worker.v1.", "cozy.package.", "cozy.runtime.", "tensorhub.",
                         "tensorfs.", "cozytensors")
# Runtime owns this deterministic wheel generator ABI. Creator checks returned
# metadata; it neither authors a new document nor defines the generator format.
FOREIGN_ABI_TAGS = {"cozy.interface-generator/6"}
KIND_READERS: dict[str, set[str]] = {}
# No trailing quote: a domain-separation tag is a PREFIX inside a longer literal — it ends
# in `\x00` or `\n`, and requiring the close quote made both of this repo's tags invisible
# to the registry that exists to hold exactly this class of cross-repo agreed name.
KIND_LITERAL = re.compile(r'"((?:cozy|cozytensors|tensorhub|tensorfs)\.[A-Za-z0-9_.-]+/\d+)')


def check_document_kinds():
    bad = []
    for p in files():
        if p.suffix != ".go":
            continue
        rel = p.as_posix()
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            for kind in KIND_LITERAL.findall(line):
                if kind.startswith(FOREIGN_KIND_PREFIXES) or kind in FOREIGN_ABI_TAGS:
                    continue
                owner = DOCUMENT_KINDS.get(kind) or HMAC_DOMAINS.get(kind)
                if owner is None:
                    bad.append(f"{rel}:{i}: [kinds] names '{kind}', a document kind outside "
                               "DOCUMENT_KINDS — a new kind needs a decision row (#616.a) and a "
                               "boundary + storage/digest party (proto-007)")
                elif rel != owner and rel not in KIND_READERS.get(kind, set()):
                    bad.append(f"{rel}:{i}: [kinds] names '{kind}', declared in {owner} — a "
                               "second declaration of one kind")
    return bad


def check_media_contract():
    """(cl-031) The media plane's contract fields are DECLARED once, not spelled twice."""
    home = pathlib.Path(MEDIA_CONTRACT_HOME)
    if not home.exists():
        return [f"[media] {MEDIA_CONTRACT_HOME} is missing — the media plane's contract has "
                f"no single home, so the pod's pinned binary and the floating owner client "
                f"have nothing to agree on"]
    bad = []
    for p in files():
        rel = p.as_posix()
        if p.suffix != ".go" or rel == MEDIA_CONTRACT_HOME:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.strip().startswith("//"):
                continue
            for field in MEDIA_CONTRACT_FIELDS:
                if field in line:
                    bad.append(f"{p}:{i}: [media] spells the media contract field "
                               f"'{field}' outside {MEDIA_CONTRACT_HOME} — both ends of this "
                               f"plane ship separately, so its wire shape is declared once "
                               f"and imported, never restated: {line.strip()}")
    return bad


def check_unpublished_vocabulary():
    """Publication state is not a privacy or access-control mode."""
    bad = []
    retired = re.compile(r"\bprivate[ -](?:package|job|script|wheel|revision|callable)\b", re.I)
    for path in pathlib.Path("internal").rglob("*.go"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            # Existing on-disk markers and stable machine error codes are not prose.
            prose = re.sub(r'"private[-_][A-Za-z0-9_-]+"', '""', line)
            if retired.search(prose):
                bad.append(f"{path}:{number}: [terminology] use unpublished package or local script")
    return bad


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
            if p.suffix == ".go" and YAML_IMPORT in s and str(p) != "internal/config/config.go":
                bad.append(f"{p}:{i}: [deps] YAML parsing belongs only to centralized config: {s}")
            for d in DENY_STORE:
                if d in s.lower():
                    bad.append(f"{p}:{i}: [store] lifecycle sidecar '{d}' — the one SQLite "
                               f"database is the authority: {s}")
            # RAW, not stripped: a content key IS a string literal, so the identifier
            # scan (which blanks literals) would never see one.
            if CAS_PATH.search(s):
                bad.append(f"{p}:{i}: [cas] a store path is composed here — TensorFS owns the "
                           f"CAS layout, and `tfs get`/`tfs put` are how an object is reached: {s}")
            # RAW: a CORS header is a string LITERAL, so the identifier scan (which blanks
            # literals) would never see one. SCOPED to the owner's loopback API (narrowed
            # 2026-08-28): a CORS header there lets any page the user's browser loads read
            # 127.0.0.1 — the Ollama DNS-rebinding class.
            if CORS_HEADER.search(s) and not (ALLOW_DOOR in line and not p.as_posix().startswith(CORS_ABSOLUTE)):
                where = ("the OWNER'S loopback API, where the rule is ABSOLUTE and the door is "
                         "not honored") if p.as_posix().startswith(CORS_ABSOLUTE) else \
                        "this repo, and it carries no //cozy:allow door stating its origin source"
                bad.append(f"{p}:{i}: [api] an Access-Control-Allow-* header in {where} — a CORS "
                           f"header on a loopback, bearer-authenticated bind is what lets any page "
                           f"the browser loads read 127.0.0.1 (#628): {s}")
            rel = p.as_posix()
            if rel.startswith(MEDIA_DISTRIBUTION_DIRS):
                for retired in DENY_MEDIA_DISTRIBUTION:
                    if retired in s:
                        bad.append(f"{p}:{i}: [media] retired package-distribution surface "
                                   f"'{retired}' — the worker resolves signed package intent "
                                   f"directly, and this plane carries invocation inputs/outputs only: {s}")
        if p.suffix != ".go":
            continue
        for i, line in enumerate(strip_go(raw).splitlines(), 1):
            src_line = raw_lines[i - 1] if i <= len(raw_lines) else ""
            s_raw_line = src_line
            idents = IDENT.findall(line)
            low = line.lower()
            if ALLOW_DOOR not in src_line:
                for d in DENY_IMPL:
                    if d in low:
                        bad.append(f"{p}:{i}: [impl] byte-plane vocabulary '{d}' — TensorFS owns it: {line.strip()}")
            for call in DENY_PROMPT_CALLS:
                if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", line):
                    bad.append(f"{p}:{i}: [prompt] '{call}' reads input — no cozy command may prompt: {line.strip()}")
            if str(p).replace("\\", "/") not in ENV_READERS:
                for call in DENY_ENV_CALLS:
                    if re.search(r"(?<![\w.])" + re.escape(call) + r"\s*\(", line):
                        bad.append(f"{p}:{i}: [env] '{call}' outside {ENV_READER} — the "
                                   f"environment is read once, at the entrypoint: {line.strip()}")
            for ident in idents:
                if ident in DENY_PROMPT:
                    bad.append(f"{p}:{i}: [prompt] '{ident}' reads input — no cozy command may prompt: {line.strip()}")
                if ident == "Stdin" and STDIN_DOOR not in src_line:
                    bad.append(f"{p}:{i}: [prompt] stdin read without the {STDIN_DOOR} door: {line.strip()}")
            rel = str(p).replace("\\", "/")
            if rel not in RUNTIME_SITES and RUNTIME_BIN.search(src_line) and \
                    ALLOW_DOOR not in src_line:
                bad.append(f"{p}:{i}: [runtime] the cozy-runtime binary is reached outside "
                           f"{' / '.join(sorted(RUNTIME_SITES))} — one execution path: a run goes "
                           f"orchestrator -> worker protocol -> runtime, never a shell-out: {src_line.strip()}")
            if rel in RUNTIME_SITES:
                for verb in sorted(RUNTIME_VERBS_DENY):
                    if re.search(r'"' + verb + r'"', src_line):
                        if rel == "internal/packagepublish/package.go" and verb == "run" and '"uv"' in src_line:
                            continue  # uv run selects the locked Runtime; it is not Runtime's run verb
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
            # The door is honored ONLY in the verification driver (cl-028): a doored bind
            # in product code would be the LAN door arriving as a comment.
            listen_doored = ALLOW_DOOR in src_line and rel.startswith(DRIVER_DIRS)
            if rel not in LISTEN_SITES and LISTEN_CALL.search(line) and not listen_doored:
                bad.append(f"{p}:{i}: [api] net.Listen outside {LISTEN_SITE} — one bind site "
                           f"per program, each with its own stated rule. The owner's LAN door is "
                           f"deferred behind TLS and its own threat review, never a second listener "
                           f"(only {DRIVER_DIR} may door one): {line.strip()}")
            if not rel.startswith(RUNTIME_INDIRECT_HOME) and RUNTIME_INDIRECT.search(line) and \
                    not (ALLOW_DOOR in src_line and rel.startswith(DRIVER_DIRS)):
                bad.append(f"{p}:{i}: [runtime] launch.Binary()/launch.RuntimeCLI outside "
                           f"{RUNTIME_INDIRECT_HOME} — the same second execution door as the "
                           f"\"cozy-runtime\" literal, reached by indirection; only "
                           f"{DRIVER_DIR} may door it: {line.strip()}")
    return bad


def check_secret_flags():
    """No Kong field may turn a credential into an argv value (cl-011)."""
    src_path = pathlib.Path("internal/cli/grammar.go")
    if not src_path.exists():
        return ["[secret] missing internal/cli/grammar.go"]
    bad = []
    for i, line in enumerate(src_path.read_text().splitlines(), 1):
        field = re.search(r'^\s*(\w+)\s+(?:\[\])?string\s+`', line)
        if not field or not SECRET_FLAG.search("--" + field.group(1).lower()):
            continue
        bad.append(f"internal/cli/grammar.go:{i}: [secret] credential-shaped string field "
                   f"{field.group(1)!r} would take an argv value — use a boolean stdin door or "
                   "OS-protected configuration")
    return bad


# AXI 10's version spellings are answered before Kong or configuration loads.
VERSION_SPELLINGS = ("--version", "-v", "-V")


def check_manifest():
    """Kong is the sole grammar and carries exactly the launch command tree."""
    src_path = pathlib.Path("internal/cli/grammar.go")
    if not src_path.exists():
        return ["[grammar] missing internal/cli/grammar.go"]
    source = src_path.read_text()
    fields = re.sub(r"\s+", " ", source)
    required = (
        "Package PackageCmd", "Model ModelCmd", "Run RunCmd", "Rental RentalCmd",
        "Up UpCmd", "Down DownCmd", "Unload UnloadCmd",
        "Search PackageSearchCmd", "Install PackageInstallCmd", "Remove PackageRemoveCmd",
        "List PackageListCmd", "Publish PackagePublishCmd", "UpdateAll PackageUpdateAllCmd",
        "Search ModelSearchCmd", "Download ModelDownloadCmd", "Remove ModelRemoveCmd",
        "List ModelListCmd", "Upload ModelUploadCmd", "Publish ModelPublishCmd", "Yank ModelYankCmd",
        "Execute RunExecuteCmd", "Cancel RunCancelCmd", "List RunListCmd", "Watch RunWatchCmd",
        "List RentalListCmd", "New RentalNewCmd", "End RentalEndCmd",
    )
    bad = [f"[grammar] missing Kong command field {item!r}" for item in required if item not in fields]
    for retired in ("StackCmd", "ExitCmd", "WorkflowCmd", "VideoCmd", "JobCmd", "CommandsCmd", "StatusCmd"):
        if retired in source:
            bad.append(f"[grammar] retired command family remains: {retired}")
    if re.search(r"^\s*Reason\s+string\s+`", source, re.M) or '"--reason"' in source:
        bad.append("[grammar] public Cozy command retained caller-authored audit prose; decision #667 derives every audit reason")
    if (
        "type PackagePublishCmd struct{}" not in source
        or "Package source tree." in source
    ):
        bad.append("[grammar] package publish identity must come only from project metadata")
    derived_audits = {
        "internal/cli/package_releases.go": '"cozy package publish " + ref.String() + "@" + release',
        "internal/cli/rentals.go": '"cozy rental new " + skuName',
    }
    for path, spelling in derived_audits.items():
        if spelling not in pathlib.Path(path).read_text():
            bad.append(f"[grammar] {path} no longer derives its internal audit reason from exact operation facts")
    transfer_plan = pathlib.Path("internal/modeltransfer/plan.go")
    if not transfer_plan.is_file() or '"cozy-model-transfer-instruction/1\\x00"' not in transfer_plan.read_text():
        bad.append("[grammar] model transfer dry-run identity lost its domain separation")
    for retired in (
        "internal/api/model_production.go",
        "internal/cli/model_production_manager.go",
        "internal/cli/model_production_run.go",
        "internal/records/model_productions.go",
    ):
        if pathlib.Path(retired).exists():
            bad.append(f"[grammar] retired special model-production lifecycle remains: {retired}")
    app = pathlib.Path("internal/cli/app.go").read_text()
    for spelling in VERSION_SPELLINGS:
        if spelling not in app:
            bad.append(f"[grammar] version fast path omits {spelling}")
    if pathlib.Path("internal/manifest").exists():
        bad.append("[grammar] parallel internal/manifest still exists")
    return bad


def check_web_boundary():
    """cl-045: one embedded stub, bounded pathless uploads; cl-096: the daemon's ONE log.

    cl-045 refused any persistent daemon log. cl-096 (owner, 2026-09-02) reverses that with
    the smallest honest surface: the daemon writes its own words to $COZY_HOME/daemon.log,
    bounded by rotation on observed bytes (never a timer), and `cozy daemon log` reads it.
    stdout stays /dev/null — the log is the surface, not an attached stream."""
    bad = []
    for required in ("web/index.html", "web/app.css", "web/app.js", "web/embed.go"):
        if not pathlib.Path(required).is_file():
            bad.append(f"[web] missing {required}")
    daemon = pathlib.Path("internal/cli/daemon.go").read_text()
    for required in ("os.DevNull", "command.StderrPipe()", "maxDaemonStartupDiagnostic",
                     "readBoundedDiagnostic", "daemonStartupFailure"):
        if required not in daemon:
            bad.append(f"[web] daemon startup boundary missing {required!r}")
    if "daemon.OpenLog(layout.Log)" not in daemon:
        bad.append("[web] the daemon does not write its own bounded log (daemon.OpenLog)")
    daemon_log = pathlib.Path("internal/daemon/log.go")
    if not daemon_log.is_file() or "const LogBytes = 32 << 20" not in daemon_log.read_text():
        bad.append("[web] the daemon log is not bounded by rotation on observed bytes (daemon.LogBytes)")
    if "time.After" in (daemon_log.read_text() if daemon_log.is_file() else ""):
        bad.append("[web] the daemon log rotates on a timer; the bound is bytes")
    if "func handleDaemonLog" not in pathlib.Path("internal/cli/daemon_log.go").read_text():
        bad.append("[web] `cozy daemon log` is missing")
    for forbidden in ("os.FindProcess(", 'exec.Command("ps"', 'exec.Command("pgrep"', '"/proc/'):
        if forbidden in daemon:
            bad.append(f"[web] daemon singleton uses process-list evidence {forbidden!r}")
    if pathlib.Path("internal/service").exists():
        bad.append("[web] retired internal/service package remains")
    retired_names = (
        "cozy-controller", "Cozy controller", "local controller", "controller.log",
        "controller_startup_", "/v1/local/service/", "service.lock",
    )
    for source in pathlib.Path(".").rglob("*"):
        if not source.is_file() or source == pathlib.Path("scripts/fence.py"):
            continue
        if any(part in {".git", "dist", "vendor"} for part in source.parts):
            continue
        if source.suffix not in {".go", ".md", ".html", ".sh", ".py", ".yaml", ".yml"}:
            continue
        text = source.read_text(errors="replace")
        for retired in retired_names:
            if retired in text:
                bad.append(f"[web] {source} retains whole-process alias {retired!r}")
    # cl-116 deleted the standalone upload store: a second content-addressed blob plane
    # under the home must not come back. Future upload admission becomes a request input.
    if pathlib.Path("internal/upload").exists():
        bad.append("[web] retired internal/upload package remains (cl-116)")
    return bad


RENDER_SRC = "internal/output/output.go"
RENDER_STDERR = re.compile(r"Fprint(?:f|ln)?\(\s*(?:[A-Za-z_.]*\.)?[Ss]tderr\b")


def check_render_streams():
    """(AXI 6) The one output layer writes one stream. Errors are structured output the
    agent consumes, so they leave on stdout beside the data; a stderr writer here is the
    regression where `cozy ls nonexistent` gave a stdout-capturing agent an empty buffer."""
    src_path = pathlib.Path(RENDER_SRC)
    if not src_path.exists():
        return [f"[render] missing {RENDER_SRC}"]
    bad = []
    for i, line in enumerate(src_path.read_text().splitlines(), 1):
        if RENDER_STDERR.search(line):
            bad.append(f"{RENDER_SRC}:{i}: [render] the render layer writes to stderr — an error "
                       "is structured output and belongs on stdout with the data (AXI 6); stderr "
                       f"carries progress, which this layer does not emit: {line.strip()}")
    return bad


def check_output_shape():
    """cl-046: success is domain data, never generic renderer metadata."""
    output = pathlib.Path("internal/output/output.go").read_text()
    views = pathlib.Path("internal/output/views.go").read_text()
    bad = []
    for forbidden in ('json:"ok"', 'json:"kind"', 'json:"data"', "type Result struct", "type ListData struct"):
        if forbidden in output or forbidden in views:
            bad.append(f"[output] generic success scaffolding remains: {forbidden!r}")
    for required in ('document[l.Name] = values', 'document[l.Name] = rows'):
        if required not in views:
            bad.append(f"[output] domain list projection missing {required!r}")
    cli = "\n".join(path.read_text() for path in pathlib.Path("internal/cli").glob("*.go"))
    if re.search(r"output\.(?:Record|List)\s*\{[^}]*\bKind\s*:", cli, re.S):
        bad.append("[output] command success still carries a generic kind discriminator")
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
    """The route table and Cozy's contract document expose the same route inventory.

    This fence checks method, path, scope, and order. Payload and behavior conformance are
    separate tests; passing this check makes no claim about unimplemented external hosts.
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


def check_video_boundary():
    """The retired pre-launch workflow/video plane does not survive the CLI hardcut."""
    bad = []
    for retired in ("internal/video", "internal/workflow", "internal/api/videos.go",
                    "internal/api/workflows.go", "internal/records/video.go",
                    "internal/records/workflows.go"):
        if pathlib.Path(retired).exists():
            bad.append(f"[video] retired plane remains: {retired}")
    return bad


def check_typed_resources():
    """The product surface names package and model directly; no generic compatibility door."""
    bad = []
    product_files = [
        pathlib.Path("internal/cli/grammar.go"),
        pathlib.Path("internal/cli/catalog.go"),
        pathlib.Path("internal/cli/transfer.go"),
        pathlib.Path("internal/hub/hub.go"),
        pathlib.Path("internal/hub/publish.go"),
    ]
    forbidden = (
        "/v1/repos",
        "/v1/resolve",
        "/v1/checkpoints",
        "/publishes",
        "cozy repo ",
        "cozy search",
        "cozy push ",
        "cozy pull ",
        "cozy promote ",
        "cozy packages",
        "cmd.repo.",
        "cmd.push",
        "cmd.pull",
        'Handler: "repo.',
        'Handler: "push"',
        'Handler: "pull"',
    )
    for path in product_files:
        text = path.read_text()
        for old in forbidden:
            if old in text:
                bad.append(f"{path}: [resources] retired product surface remains: {old!r}")

    retired_package_noun = "end" + "point"
    for path in (
        pathlib.Path("docs/package-publication.md"),
        pathlib.Path("scripts/accept.sh"),
        pathlib.Path("internal/orchestrator/dispatch.go"),
        pathlib.Path("internal/records/records.go"),
    ):
        if retired_package_noun in path.read_text().lower():
            bad.append(f"{path}: [resources] retired package-domain vocabulary remains")

    manifest = pathlib.Path("internal/cli/grammar.go").read_text()
    if "--kind" in manifest:
        bad.append("internal/cli/grammar.go: [resources] retired --kind discriminator remains")

    hub_sources = (pathlib.Path("internal/hub/hub.go").read_text() +
                   pathlib.Path("internal/hub/publish.go").read_text() +
                   pathlib.Path("internal/hub/package_releases.go").read_text())
    for route in ('resourceSearchPath("packages"', 'resourceSearchPath("models"', '"/v1/models/"',
                  '"/publications"', '"/publish/"', '"/finalize"', '"/releases/"', '"/download"'):
        if route not in hub_sources:
            bad.append(f"internal/hub: [resources] missing typed route prefix {route}")
    wheel_build = pathlib.Path("internal/wheel/build.go").read_text()
    if "cmd.Env = config.Frozen().Tool()" not in wheel_build:
        bad.append("internal/wheel/build.go: [env] uv build inherits the parent environment; "
                   "the PEP 517 backend executes project code and must receive only Tool()")
    retired_debug = (pathlib.Path("internal/transfer/fetch.go").read_text() +
                     pathlib.Path("internal/transfer/upload.go").read_text() +
                     pathlib.Path("internal/cli/transfer.go").read_text())
    for retired in ("FailAfter", "--crash-after", "devKill", '"/seal"'):
        if retired in retired_debug:
            bad.append(f"internal transfer path: [resources] unreachable development kill surface remains: {retired}")
    return bad


def check_python_seat():
    """Published local execution uses the release's own uv-selected Python environment."""
    path = pathlib.Path("internal/install/published.go")
    text = path.read_text()
    bad = []
    publication = pathlib.Path("internal/packagepublish/package.go").read_text()
    for forbidden in ("PYTHONPATH=", "PYTHONHOME="):
        if forbidden in publication:
            bad.append(
                f"internal/packagepublish/package.go: [python] publication injects {forbidden[:-1]}"
            )
    for required in (
        '"prepare-package"',
        '"--environment-python"',
        "MaterializePublishedEnvironment",
    ):
        if required not in text:
            bad.append(f"{path}: [python] missing full uv environment handoff {required!r}")
    install = pathlib.Path("internal/install/install.go").read_text()
    retired_inventory = text + "\n" + "\n".join(
        pathlib.Path(name).read_text()
        for name in (
            "internal/home/home.go",
            "internal/launch/artifacts.go",
            "internal/orchestrator/worker.go",
        )
    )
    for deleted in (
        '"local-base"',
        '"--base-manifest"',
        "RefreshGenerationBase",
        "LocalBase string",
        "BaseManifest string",
    ):
        if deleted in retired_inventory:
            bad.append(
                "internal local execution: [python] retired installed-environment "
                f"inventory remains: {deleted}"
            )
    for deleted in ('"--environment-root"', 'Python: "CPython 3.12"', "LinkMode", "pythonForABI"):
        if deleted in text + install:
            bad.append(f"internal/install: [python] retired shared-base install path remains: {deleted}")

    fixtures = (pathlib.Path("tests/product/testdata/build-weightless.py"),)
    for fixture in fixtures:
        body = fixture.read_text()
        if ">=3.12,<3.13" not in body:
            bad.append(f"{fixture}: [python] package fixture is not pinned to CPython 3.12")
        for deleted in ("cp314", ">=3.14", "<3.15"):
            if deleted in body:
                bad.append(f"{fixture}: [python] deleted Python seat remains: {deleted}")

    package_client = pathlib.Path("internal/hub/package_releases.go").read_text()
    for retired in ("PackageInstallTarget", "qualification_state", 'json:"profile"'):
        if retired in package_client:
            bad.append(f"internal/hub/package_releases.go: [python] retired profile selection remains: {retired}")
    launcher = pathlib.Path("internal/orchestrator/worker.go").read_text()
    if "--instance-id" in launcher:
        bad.append("internal/orchestrator/worker.go: [python] Creator still passes Runtime's deleted --instance-id")
    return bad


if sys.argv[1:]:
    if sys.argv[1:] == ["--help"] or sys.argv[1:] == ["-h"]:
        print(__doc__.strip())
        sys.exit(0)
    print("usage: scripts/fence.py [--help]", file=sys.stderr)
    sys.exit(2)
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

violations = (check_unpublished_vocabulary() + check_sources() + check_matrix()
              + check_manifest() + check_secret_flags()
              + check_contract() + check_video_boundary() + check_embedded() + check_scripts()
              + check_document_kinds() + check_render_streams() + check_output_shape()
              + check_media_contract() + check_typed_resources() + check_web_boundary()
              + check_test_boundary() + check_python_seat())
if violations:
    print("FENCE RED (boundaries.md):", file=sys.stderr)
    for v in violations:
        print("  " + v, file=sys.stderr)
    sys.exit(1)
print(
    f"fence green — deps({len(DENY_DEPS)}) impl({len(DENY_IMPL)}) "
    f"prompt({len(DENY_PROMPT) + len(DENY_PROMPT_CALLS) + 1}) matrix(15 rows) "
    f"env({len(DENY_ENV_CALLS)} calls@{len(ENV_READERS)} programs) store({len(DENY_STORE)}) "
    f"grammar(kong@internal/cli/grammar.go + version {'/'.join(VERSION_SPELLINGS)}) "
    f"output(domain-shaped, one document@{RENDER_SRC}) secret({SECRET_FLAG.pattern} + Reveal@{len(REVEAL_SITES)}) "
    f"cas(store-path) tensor(tfs@{len(TFS_SITES)}) "
    f"api({len(DENY_COOKIE)} cookie + cors(absolute@{CORS_ABSOLUTE}) + listen@{len(LISTEN_SITES)} programs) "
    f"media-client({len(MEDIA_CONTRACT_FIELDS)} contract fields@{MEDIA_CONTRACT_HOME}) "
    f"runtime({len(RUNTIME_VERBS_DENY)} denied verbs@{len(RUNTIME_SITES)} + indirection) "
    f"embed({len(DENY_EMBED)} words, scripts allow@{len(PY_ALLOW)}) "
    f"contract({len(parse_go_routes(pathlib.Path('internal/api/routes.go')))} routes) "
    f"web(stub+bounded-upload+bounded-daemon-log) retired-planes(absent) "
    f"resources(typed package/model, no aliases) python(independent uv venvs) test(tests/product only)"
)
