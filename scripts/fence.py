#!/usr/bin/env python3
"""Boundary fence (boundaries.md). Architecture enforcement, not a test.

Fifteen families:
  deps      forbidden dependencies (raw lines, incl. import paths and go.mod)
  impl      the CANONICAL FORMATS TensorFS parses. A second reader of safetensors/gguf/
            cozytensors here would drift from the one that produced the published bytes.
            (Narrowed 2026-08-28: the generic systems words — chunk, mmap, dtype,
            residency, quantiz, pagein — were a vocabulary ban with no failure behind
            them; they fired on the worker protocol's own field names.)
  prompt    interactive prompts: no cozy command may ever ask a question (AXI)
  matrix    internal/exit/exit.go must equal docs/exit-matrix.md row for row
  manifest  a reclaiming/removing verb must DECLARE its gate: Destructive (exit 7 without
            --yes) or PlanFirst (a read without --yes) — exactly one, and it must
            advertise --yes. Nothing removes bytes on a bare invocation. GlobalFlags
            never names --version/-v/-V: AXI 10's version probe is a pre-parse fast path
            in app.Run, and a global row would shadow `cozy pack --version <x.y.z>`. And
            every IMPLEMENTED row carries AXI 9's disclosure and AXI 10's examples: a
            `Next:` default (or `SelfContained: true`, the detail-view exemption, whose
            handler computes a state-dependent one) and at least one `Examples:` line.
  render    (AXI 6) internal/render writes ONE stream. An error is structured output the
            agent must read, so it leaves on stdout with the data; stderr is progress and
            diagnostics, which this layer does not emit.
  env       (cl-001) the environment is read in internal/config/config.go and NOWHERE else:
            one entrypoint reader, a frozen typed value thereafter.
  store     (cl-001) no second lifecycle store: the ONE local SQLite database is the
            authority, so a state.json/pidfile-class sidecar name is a violation in CODE
            — those sidecars outlive their launcher and lie. A whole-line comment naming
            one is NOT scanned (check_sources skips lines starting with `//`), and the
            docstring said otherwise until 2026-08-28; a word in prose was never the
            hazard, writing the sidecar is.
  secret    (cl-011) a credential never rides argv and has ONE raw reader: a manifest flag
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
  contract  (cl-006) internal/api/routes.go and docs/client-contract.md are ONE surface,
            row for row, scope for scope. The document is what th-021's other two hosts
            implement against, so drift is a shared-contract defect, not a doc lag.
  media     (cl-014/cl-031) the pod's byte plane has NO outbound capability of any kind and
            cannot import the worker protocol, and its wire contract — the service name, the
            revision both ends compare before a byte moves, and the bounds it publishes — is
            declared in internal/mediawire and restated nowhere.
  bootstrap (xs-004/cl-036) the POD SUPERVISOR has NO outbound capability of any kind —
            the same absolute rule as the media plane beside it, since cl-036 deleted the
            two provision-document fetches that were its whole licence to egress. It still
            legitimately EXECS its two children, and it still may not import the worker
            protocol — a supervisor that could talk to the worker is the second control
            plane cl-014 forbids one directory over.
  tensor    (cl-012) the tensorfs CLI has ONE caller: internal/tfs. The configured binary
            is read there and in the config authority, nowhere else — a second package
            shelling out to `tfs` is a second byte-plane door with its own vocabulary.

Identifier families scan Go source with comments and string literals removed, so a
word inside help text or a doc comment is never a violation. Doors, both greppable:
  //cozy:allow        this line is exempt from the impl family (state the reason)
  //cozy:stdin-value  this line reads stdin as a VALUE (e.g. --token-stdin), never a prompt
"""
import pathlib, re, sys

SCAN = ["go.mod", "main.go", "cmd/**/*.go", "internal/**/*.go"]

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

# ONE environment reader PER PROGRAM — the same shape as LISTEN_SITES below, and for the
# same reason: this repo builds three binaries, and collapsing the rule to "one reader
# somewhere" would let either program's reader go missing. Every other package in each
# program takes the frozen typed value.
#   config/config.go   the `cozy` CLI. Its Inherited allowlist is class A of
#                      tracker-v2/spawn-allowlists.md (#616.d).
#   cozy-bootstrap     the POD SUPERVISOR, whose entire launch surface IS the injected
#                      environment: it takes no arguments and reads no configuration file,
#                      so its one reader is also its one ALLOWLIST — an unrecognized COZY_*
#                      name is refused rather than ignored, which is what makes a renamed
#                      grant a boot failure instead of a silent default.
ENV_READERS = {
    "internal/config/config.go": "the cozy CLI's entrypoint reader",
    "cmd/cozy-bootstrap/config.go": "the pod supervisor's entrypoint reader and allowlist",
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
    # The byte plane's (#506b): the rental's owner token becomes the Authorization
    # header for the POD's media server. Same rule as hub/hub.go one plane over — one
    # request builder, and the raw value is read exactly where it becomes a carrier.
    "internal/media/media.go",
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
# (#628, owner ruling 2026-08-27) The CORS family stands, with the `//cozy:allow` door as
# the escape a pod-side route takes when it echoes the RentalProvisionSpec's origins. Inside
# the owner's loopback API the rule is ABSOLUTE and the door is NOT honored: that surface is
# the DNS-rebinding target, and a doored CORS header there would be the hole arriving as a
# comment.
CORS_ABSOLUTE = "internal/api/"
#   Two programs, two bind rules, and they are NOT the same rule (#506b). The owner's
#   `cozy` binary binds once, on loopback, for its local client API. The POD's media
#   server (`cmd/cozy-media`) binds once, off-loopback on purpose — it is the leg an
#   off-machine owner reaches — and its own rule is TLS-or-loopback, enforced in the same
#   file. Naming both here keeps "one bind site" true PER PROGRAM instead of collapsing
#   into "one bind site somewhere in the repo", which would let either rule go missing.
LISTEN_SITES = {
    "internal/api/listen.go": "the owner's local client API — loopback only",
    "cmd/cozy-media/listen.go": "the POD's media server — off-loopback requires TLS",
}
LISTEN_SITE = " / ".join(sorted(LISTEN_SITES))
# (cl-028 tightening) ANY net.Listen* call, not only the literal-"tcp" spelling: a bind
# whose network rides a variable was invisible to the old regex, and a copycat unix or
# ListenTCP bind is the same second door.
LISTEN_CALL = re.compile(r"net\.Listen\w*\s*\(")

# (cl-014 / #506b) THE MEDIA SERVER HAS NO OUTBOUND NETWORK. It joins the liability fence
# by having nothing to egress WITH: a pod-side process that can be talked into fetching a
# URL is a request-content-to-hub channel with an extra step, and one that can call the
# worker is the live RPC cl-014's filesystem-handoff coupling exists to forbid. This is the
# structural half of "prove no request-path call crosses to the worker process".
MEDIA_DIR = "cmd/cozy-media/"
DENY_MEDIA_EGRESS = [
    "http.Get", "http.Post", "http.PostForm", "http.Head", "http.NewRequest",
    "http.DefaultClient", "http.Client", "net.Dial", "net.DialTimeout", "grpc.NewClient",
    "grpc.Dial", "exec.Command", "exec.CommandContext",
]
# …and it may not import the protocol or the orchestrator: a media server that could speak
# `cozy.worker.v1` would be a second control plane inside the pod.
DENY_MEDIA_IMPORT = ["protocol/cozy/worker", "internal/orchestrator", "internal/api"]
# An import LINE, so a doc comment naming the owner's bind site is prose and not a door.
MEDIA_IMPORT_LINE = re.compile(
    r'^\s*(?:[A-Za-z_]\w*\s+)?"github\.com/cozy-creator/cozy-creator/([^"]+)"\s*$')

# (cl-031) ONE DECLARATION OF THE MEDIA PLANE'S CONTRACT. The pod's media server ships
# inside an image pinned by commit and the owner's client floats with master, so the pair
# both ends compare — the service name, the revision, and the bounds the plane publishes —
# has exactly one home. A second spelling of a contract field is how the two ends drift
# back apart in silence, which is the whole defect cl-031 closed.
MEDIA_CONTRACT_HOME = "internal/mediawire/wire.go"
MEDIA_CONTRACT_FIELDS = ["contract_rev", "max_receipt_bytes"]
# …and the ceiling itself, as a NUMBER. The field-name rule above catches a second JSON
# spelling; it did not catch what actually happened, which is that the pod supervisor
# declared `maxReceiptEnvelopeSize = 64 << 10` of its own while it lived in another repo —
# a second copy of one bound, agreeing by luck. Scoped to the two POD binaries, because
# 64 KiB is an ordinary buffer size everywhere else in this tree.
MEDIA_CEILING_LITERAL = re.compile(r"64\s*<<\s*10")

# (xs-004, tightened by cl-036) THE POD SUPERVISOR DIALS NOTHING. `cmd/cozy-bootstrap` is
# PID 1 of a rented pod. It used to hold a BOUNDED egress licence — one file, two
# exact-grant provision-document fetches — because the pod could not boot without them.
# cr-048/th-067 retired those documents (a pod boots ready-but-empty and its closure
# arrives on the rev-2 placement lane after the RecordOwner connects), so the licence has
# no subject left and the rule is now ABSOLUTE, exactly like DENY_MEDIA_EGRESS one
# directory over. A supervisor's whole launch surface is the injected environment; the
# first `http.Get` back — a health poll, a telemetry ping, a "just fetch the config"
# convenience — is an exfiltration channel with an extra step and no exact-grant checks
# around it. `exec.Command` is deliberately NOT here and IS in the media list: execing the
# two children is the one capability this process exists to have, and it is the only reason
# these stay two families rather than one. With this rule absolute, cl-036's merge into one
# `cozy-pod` binary can be fenced WHOLE — its precondition, "no carve-out for a file that
# still egresses", is now met.
BOOTSTRAP_DIR = "cmd/cozy-bootstrap/"
DENY_BOOTSTRAP_EGRESS = [
    "http.Get", "http.Post", "http.PostForm", "http.Head", "http.NewRequest",
    "http.DefaultClient", "http.Client", "http.Transport", "net.Dial", "net.DialTimeout",
    "grpc.NewClient", "grpc.Dial",
]
# It holds the receipt HMAC key and the renter token hashes and now has nowhere to send
# them. Keep fencing the header anyway: the day this process presents a credential is the
# day a pod supervisor becomes a capability an attacker can aim, and this rule is what
# makes that arrive as a red fence rather than as a diff nobody read.
BOOTSTRAP_CREDENTIAL_HEADER = re.compile(r"Authorization", re.I)
# It holds token HASHES and mints a self-signed leaf; it never speaks a control plane. An
# import of the worker protocol, the orchestrator, the owner's API or its media/hub clients
# would each be a second control plane inside the pod (cl-014's rule, one directory over).
# `internal/mediawire` is deliberately NOT here: importing the contract is the point.
DENY_BOOTSTRAP_IMPORT = [
    "protocol/cozy/worker", "internal/orchestrator", "internal/api", "internal/hub",
    "internal/media",
]

ALLOW_DOOR = "//cozy:allow"
STDIN_DOOR = "//cozy:stdin-value"

# (cl-012) The CAS layout is TensorFS's. A path composed here would drift from it the
# first time either side changed, and the drift would present as a corrupt store.
CAS_PATH = re.compile(r'objects\s*[/",\s]+\s*sha256', re.I)

# (cl-012) The ONE caller of the tensorfs CLI.
TFS_SITES = {"internal/config/config.go", "internal/tfs/tfs.go"}
TFS_FIELD = re.compile(r"\.Tfs\b")

# (cl-010) The two files that may reach an endpoint's own cozy-runtime, and the CLOSED set
# of verbs they may name. `install.go` runs `describe --check` at install; `artifacts.go`
# asks for the artifact index, host facts and fit verdicts. Nothing executes a model
# through this door — that is what the orchestrator and the worker protocol are for.
RUNTIME_SITES = {"internal/install/install.go", "internal/launch/artifacts.go"}
RUNTIME_BIN = re.compile(r'"cozy-runtime"')
# (cl-028) The INDIRECTIONS to the same binary, which the literal above cannot see:
# `launch.Binary()` resolves the generation venv's cozy-runtime and `launch.RuntimeCLI{}`
# is its invoker. Product code outside internal/launch reaching either is the same second
# execution door the literal rule refuses; only the verification driver may (with a door).
RUNTIME_INDIRECT = re.compile(r"launch\.Binary\s*\(|launch\.RuntimeCLI\s*\{")
RUNTIME_INDIRECT_HOME = "internal/launch/"
RUNTIME_VERBS_OK = {"describe", "list", "doctor", "fit", "bindings"}
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
# fence.py names the vocabulary in order to deny it; sdxl_proof.py is FIXTURE SOURCE for
# the non-cooperative-cancel arm (cl-003/M4) — copied into the proof release by
# sdxl-release.sh and executed only inside that released endpoint's own venv, never by
# Creator. The allow is explicit so a second file of UNet math cannot ride in unseen.
PY_ALLOW = {
    "scripts/fence.py": "the fence itself: it spells the vocabulary to deny it",
    "scripts/sdxl_proof.py": "fixture source for the M4 non-cooperative-cancel arm; "
                             "runs only inside the released endpoint's own venv",
    "scripts/xfer-live.py": "cl-012's live transfer driver: its corruption arm flips one "
                            "byte in a store filename — an adversary plant, no tensor is "
                            "parsed or composed",
}

# (cl-028) The VERIFICATION HOME is the only place an //cozy:allow door may exempt a listen
# or a runtime indirection: `internal/live` is the go-test suite and it contains the adversary
# peer (`fakeworker`) it spawns; both bind sockets by design. Product code gets no door for
# either — a doored non-loopback bind in the product would be the LAN door arriving as a comment.
DRIVER_DIRS = ("internal/live/",)
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


DOCUMENT_KINDS = {
    # proto-007 (#616.a): the `cozy.<name>/<N>` names this repo AUTHORS, each with the one
    # file that declares it. A format name exists only for a document that crosses a
    # repo/process boundary AND is stored or digested. A literal outside this table is a new
    # name without its decision row; a literal in a second file is a second declaration.
    #
    # Two rows are HMAC DOMAIN-SEPARATION TAGS rather than stored documents
    # (`cozy.pod-readiness/1`, `cozy.rental_request/1`). They are registered for the same
    # reason and are if anything stricter: a peer repo reproduces those exact bytes to
    # verify a MAC, so a silent edit does not misparse — it fails authentication at a
    # rental boundary, which is the worst place to discover a renamed constant.
    "cozy.client.JobSubmission/1": "internal/api/jobs.go",
    "cozy.client.Submission/1": "internal/api/requests.go",
    "cozy.local.EntrypointBindingRecord/2": "internal/plan/plan.go",
    "cozy.local.EvaluatedConfig/1": "internal/app/identity.go",
    "cozy.local.ExecutionEnvironment/1": "internal/app/identity.go",
    "cozy.pod-readiness/1": "cmd/cozy-bootstrap/run.go",
    "cozy.rental_request/1": "internal/app/rentals.go",
    "cozy.video/1": "internal/video/source.go",
    "cozy.video.CreativePlan/1": "internal/video/composition.go",
    "cozy.workflow.ChildIdentity/1": "internal/workflow/materialize.go",
    "cozy.workflow.ExecutionIdentity/1": "internal/workflow/engine.go",
    "cozy.workflow.MaterializedSubmission/1": "internal/workflow/materialize.go",
    "cozy.workflow.Plan/1": "internal/workflow/plan.go",
}
# Kinds another repo authors and this one only reads: the owner's fence polices the name.
FOREIGN_KIND_PREFIXES = ("cozy.worker.v1.", "cozy.endpoint.", "cozy.runtime.", "tensorhub.",
                         "tensorfs.", "cozytensors")
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
                if kind.startswith(FOREIGN_KIND_PREFIXES):
                    continue
                owner = DOCUMENT_KINDS.get(kind)
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
            if p.suffix == ".go" and YAML_IMPORT in s and not str(p).startswith("internal/video/"):
                bad.append(f"{p}:{i}: [deps] YAML parsing belongs only to internal/video: {s}")
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
            # 127.0.0.1 — the Ollama DNS-rebinding class. Off-loopback surfaces the browser
            # is MEANT to reach cross-origin (cmd/cozy-media) open their own door at the route.
            if CORS_HEADER.search(s) and not (ALLOW_DOOR in line and not p.as_posix().startswith(CORS_ABSOLUTE)):
                where = ("the OWNER'S loopback API, where the rule is ABSOLUTE and the door is "
                         "not honored") if p.as_posix().startswith(CORS_ABSOLUTE) else \
                        "this repo, and it carries no //cozy:allow door stating its origin source"
                bad.append(f"{p}:{i}: [api] an Access-Control-Allow-* header in {where} — a CORS "
                           f"header on a loopback, bearer-authenticated bind is what lets any page "
                           f"the browser loads read 127.0.0.1 (#628): {s}")
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
            if rel.startswith(BOOTSTRAP_DIR):
                for call in DENY_BOOTSTRAP_EGRESS:
                    if call in line:
                        bad.append(f"{p}:{i}: [bootstrap] '{call}' in the pod supervisor — it "
                                   f"has NO outbound capability of any kind. Its launch "
                                   f"surface is the injected environment and nothing else; "
                                   f"the two exact-grant fetches that once licensed one file "
                                   f"went out with the provision documents (cl-036), so there "
                                   f"is no door to reopen: {line.strip()}")
                # RAW, because a header name is a string literal the identifier scan blanks;
                # a whole-line comment is prose (same convention as check_media_contract).
                if not s_raw_line.strip().startswith("//") and \
                        BOOTSTRAP_CREDENTIAL_HEADER.search(s_raw_line):
                    bad.append(f"{p}:{i}: [bootstrap] the pod supervisor sets an Authorization "
                               f"header — it holds the receipt key and the renter token "
                               f"hashes and has nowhere to send them; a supervisor that "
                               f"presents a credential is a capability an attacker can aim: "
                               f"{s_raw_line.strip()}")
                imported = MEDIA_IMPORT_LINE.match(s_raw_line)
                for dep in DENY_BOOTSTRAP_IMPORT if imported else ():
                    path = imported.group(1)
                    if path == dep or path.startswith(dep + "/"):
                        bad.append(f"{p}:{i}: [bootstrap] the pod supervisor imports '{dep}' — it "
                                   f"mints credentials and execs two children; a control-plane "
                                   f"client here is the second control plane inside the pod that "
                                   f"cl-014 forbids one directory over: {line.strip()}")
            if rel.startswith(BOOTSTRAP_DIR) or rel.startswith(MEDIA_DIR):
                if MEDIA_CEILING_LITERAL.search(line):
                    bad.append(f"{p}:{i}: [media] the receipt ceiling is spelled as a literal in a "
                               f"pod binary — one end writes that envelope and the other refuses "
                               f"it, so the number lives in {MEDIA_CONTRACT_HOME} and is imported "
                               f"(cl-031's defect was two copies agreeing by luck): {line.strip()}")
            if rel.startswith(MEDIA_DIR):
                for call in DENY_MEDIA_EGRESS:
                    if call in line:
                        bad.append(f"{p}:{i}: [media] '{call}' in the pod's media server — it has NO "
                                   f"outbound capability of any kind (cl-014): no fetch, no worker "
                                   f"call, no child process. It is coupled to the worker BY THE "
                                   f"FILESYSTEM ONLY: {line.strip()}")
                imported = MEDIA_IMPORT_LINE.match(s_raw_line)
                for dep in DENY_MEDIA_IMPORT if imported else ():
                    if dep in imported.group(1):
                        bad.append(f"{p}:{i}: [media] the media server imports '{dep}' — a byte plane "
                                   f"that could speak the worker protocol would be a second control "
                                   f"plane inside the pod: {line.strip()}")
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

# AXI 10's three version spellings. They are answered before the manifest is even read;
# as GlobalFlags rows they would instead be resolved by parse.findFlag, which checks
# globals FIRST and would hand `cozy pack --version 1.2.3` to the wrong flag.
VERSION_SPELLINGS = ("--version", "-v", "-V")
FLAG_SPELLING = re.compile(r'(?:Name|Short):\s*"(-[-A-Za-z0-9]*)"')


def check_global_flags():
    src_path = pathlib.Path("internal/manifest/manifest.go")
    if not src_path.exists():
        return ["[manifest] missing internal/manifest/manifest.go"]
    bad, seen = [], False
    inside = False
    for i, line in enumerate(src_path.read_text().splitlines(), 1):
        if line.startswith("var GlobalFlags"):
            inside, seen = True, True
            continue
        if not inside:
            continue
        if line.startswith("}"):
            break
        for m in FLAG_SPELLING.finditer(line):
            if m.group(1) in VERSION_SPELLINGS:
                bad.append(f"internal/manifest/manifest.go:{i}: [manifest] GlobalFlags declares "
                           f"'{m.group(1)}' — the version probe is a pre-parse fast path in "
                           "app.Run, and a global row shadows `cozy pack --version <x.y.z>` "
                           "because parse.findFlag resolves globals before command flags")
    if not seen:
        bad.append("[manifest] internal/manifest/manifest.go declares no GlobalFlags block")
    return bad


# AXI 9/10, per IMPLEMENTED row. `Next` is the default disclosure the ONE emit seam
# attaches; `SelfContained: true` is AXI 9's "omit when self-contained" exemption, taken
# only by a detail view of one thing the caller already named. `Examples` is AXI 10's
# 2-3 worked invocations, which is what `cozy help <cmd>` renders.
ROW_NEXT = re.compile(r'\bNext:\s*\[\]string\{\s*"')
ROW_EXAMPLES = re.compile(r'\bExamples:\s*\[\]string\{\s*"')
ROW_SELF_CONTAINED = "SelfContained: true"


def check_manifest():
    src_path = pathlib.Path("internal/manifest/commands.go")
    if not src_path.exists():
        return ["[manifest] missing internal/manifest/commands.go"]
    blocks = src_path.read_text().split("\n\t{\n")[1:]
    bad, rows = check_global_flags(), 0
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
        if "Status: Implemented" not in body:
            continue
        if not ROW_NEXT.search(body) and ROW_SELF_CONTAINED not in body:
            bad.append(f"[manifest] '{name}' is implemented but declares no 'Next:' — AXI 9 wants "
                       "a next step after normal output, and the emit seam can only attach what "
                       "the row declares. A detail view whose handler computes its own says so "
                       "with 'SelfContained: true'")
        if ROW_NEXT.search(body) and ROW_SELF_CONTAINED in body:
            bad.append(f"[manifest] '{name}' declares both 'Next:' and 'SelfContained: true' — "
                       "the exemption means there is no default to declare")
        if not ROW_EXAMPLES.search(body):
            bad.append(f"[manifest] '{name}' is implemented but declares no 'Examples:' — AXI 10 "
                       "wants 2-3 worked, parameterized invocations in `cozy help " + name + "`")
    if not rows:
        bad.append("[manifest] no command rows parsed out of commands.go")
    return bad


RENDER_SRC = "internal/render/render.go"
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


def check_video_boundary():
    """Creative source and video CLI never acquire placement/model policy."""
    commands = pathlib.Path("internal/manifest/commands.go").read_text()
    start = commands.find("// ---- editable Cozy Video sources")
    stop = commands.find("// ---- jobs", start)
    if start < 0 or stop < 0:
        return ["[video] manifest has no bounded Cozy Video command section"]
    surface = commands[start:stop]
    bad = []
    for forbidden in ("--worker", "--lane", "--model", "--provider", "--datacenter",
                      "--region", "--accelerator", "--snapshot-root", ".artifacts"):
        if forbidden in surface:
            bad.append(f"[video] video command surface contains placement/model input {forbidden}")
    source = pathlib.Path("internal/video/source.go").read_text()
    forbidden_fields = re.compile(
        r'yaml:"(?:lane|model|provider|datacenter|region|accelerator|gpu|snapshot|model_root|artifact)'
    )
    if match := forbidden_fields.search(source):
        bad.append(f"[video] cozy.video/1 acquired execution field {match.group(0)}")
    proof = pathlib.Path("proofs/h3-long-form/accept.sh")
    if proof.exists():
        text = proof.read_text().lower()
        # The lane spellings that used to sit here died with #621's adaln-pruned hardcut;
        # a fence that polices retired vocabulary polices nothing.
        for forbidden in ("runpod", ".artifacts"):
            if forbidden in text:
                bad.append(f"[video] paid acceptance script hardcodes {forbidden}")
    return bad


violations = (check_sources() + check_matrix() + check_manifest() + check_secret_flags()
              + check_contract() + check_video_boundary() + check_embedded() + check_scripts()
              + check_document_kinds() + check_render_streams()
              + check_media_contract())
if violations:
    print("FENCE RED (boundaries.md):", file=sys.stderr)
    for v in violations:
        print("  " + v, file=sys.stderr)
    sys.exit(1)
print(
    f"fence green — deps({len(DENY_DEPS)}) impl({len(DENY_IMPL)}) "
    f"prompt({len(DENY_PROMPT) + len(DENY_PROMPT_CALLS) + 1}) matrix(15 rows) "
    f"env({len(DENY_ENV_CALLS)} calls@{len(ENV_READERS)} programs) store({len(DENY_STORE)}) "
    f"manifest({RECLAIM_VERB.pattern} + globals!{'/'.join(VERSION_SPELLINGS)} "
    f"+ next/examples per implemented row) "
    f"render(one stream@{RENDER_SRC}) secret({SECRET_FLAG.pattern} + Reveal@{len(REVEAL_SITES)}) "
    f"cas(store-path) tensor(tfs@{len(TFS_SITES)}) "
    f"api({len(DENY_COOKIE)} cookie + cors(absolute@{CORS_ABSOLUTE}) + listen@{len(LISTEN_SITES)} programs) "
    f"media({len(DENY_MEDIA_EGRESS)} egress + {len(DENY_MEDIA_IMPORT)} imports@{MEDIA_DIR} "
    f"+ {len(MEDIA_CONTRACT_FIELDS)} contract fields + ceiling literal@{MEDIA_CONTRACT_HOME}) "
    f"bootstrap({len(DENY_BOOTSTRAP_EGRESS)} egress(absolute@{BOOTSTRAP_DIR}) + no-credential "
    f"+ {len(DENY_BOOTSTRAP_IMPORT)} imports@{BOOTSTRAP_DIR}) "
    f"runtime({len(RUNTIME_VERBS_DENY)} denied verbs@{len(RUNTIME_SITES)} + indirection) "
    f"embed({len(DENY_EMBED)} words, scripts allow@{len(PY_ALLOW)}) "
    f"contract({len(parse_go_routes(pathlib.Path('internal/api/routes.go')))} routes) video-boundary"
)
