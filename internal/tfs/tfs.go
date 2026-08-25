// Package tfs is the ONE door to a byte-plane fact in this binary (cl-012).
//
// cozy-creator owns no byte plane: TensorFS does (README law 3, boundaries.md). So
// every question about canonical bytes — what a snapshot reaches, what an object
// hashes to, whether the store holds it, whether a whole checkpoint verifies — is
// asked of the `tfs` binary and consumed as a document it produced or an identifier
// it printed. Nothing here decodes a manifest, a header, or a tensor.
//
// The one document this package reads is `tensorfs.publish_closure/1`, and it reads
// exactly two fields per entry (the object id and its length) because that list IS
// the declaration the hub answers — the same list tensorhub's reference publisher
// builds. The closure BYTES ride the declaration verbatim; nothing re-encodes them.
//
// Two fences hold this (scripts/fence.py, family `tensor`): no Go decoder for a
// canonical tensor document, and no second CAS key derivation — a path under
// `objects/sha256/…` composed in Go would be a second spelling of TensorFS's own
// layout, and the two would drift.
package tfs

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// Tool is one resolved tensorfs CLI plus the store it operates on.
type Tool struct {
	Bin    string
	Root   string // the local CAS
	Source string // where the binary path came from, for the remedy text
	env    []string
}

// Open resolves the binary and the store. A missing `tfs` is a structural refusal
// (exit 6) naming what to install: this binary cannot substitute for it, and a
// cozy-creator-side reimplementation is exactly what boundaries.md forbids.
func Open(cfg config.Config, layout home.Layout) (*Tool, *exit.Error) {
	bin, err := exec.LookPath(cfg.Tfs)
	if err != nil {
		return nil, exit.Named(exit.Structural, "tfs_missing",
			"the tensorfs CLI %q is not on PATH: %s", cfg.Tfs, err).
			WithRemedy("build it from the tensorfs repo (cargo build --release -p tensorfs-core --bin tfs) and set COZY_TFS to the binary").
			WithNext("COZY_TFS=/path/to/tfs cozy hub status")
	}
	t := &Tool{Bin: bin, Root: layout.CAS, Source: cfg.TfsSource, env: cfg.Tool()}
	if err := os.MkdirAll(t.Root, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the local store at %s: %s", t.Root, err)
	}
	if _, e := t.run("store", "init", t.Root); e != nil {
		return nil, e
	}
	return t, nil
}

// Version identifies the binary for a receipt line. It is the path plus its size and
// modification time: tfs prints no version of its own, and inventing one would be a
// claim nobody produced.
func (t *Tool) Version() string {
	st, err := os.Stat(t.Bin)
	if err != nil {
		return t.Bin
	}
	return fmt.Sprintf("%s (%d B, %s)", t.Bin, st.Size(), st.ModTime().UTC().Format(time.RFC3339))
}

func (t *Tool) run(args ...string) (string, *exit.Error) {
	cmd := exec.Command(t.Bin, args...)
	cmd.Env = t.env
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		// tfs refuses in its own vocabulary (REFUSED <CODE>: …). That vocabulary is
		// the byte plane's, so it reaches the user unchanged; only the exit code is ours.
		said := strings.TrimSpace(errb.String())
		if said == "" {
			said = strings.TrimSpace(out.String())
		}
		if said == "" {
			said = err.Error()
		}
		return out.String(), exit.Named(exit.Validation, "tfs_refused",
			"tfs %s: %s", strings.Join(redact(args), " "), firstLine(said)).
			WithRemedy("the byte plane refused in its own words; nothing above it may overrule that")
	}
	return out.String(), nil
}

// redact keeps an absolute store path out of a refusal line's head, where it buries
// the reason under a path nobody is debugging.
func redact(args []string) []string {
	out := make([]string, 0, len(args))
	for _, a := range args {
		if filepath.IsAbs(a) {
			out = append(out, filepath.Base(a))
			continue
		}
		out = append(out, a)
	}
	return out
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// ---------------------------------------------------------------- identifiers

// Object is one entry of a publish closure: an identity and a length. This is the
// whole vocabulary cozy-creator needs about a byte — everything else is TensorFS's.
type Object struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
}

type closureDoc struct {
	Objects []struct {
		SHA256 string `json:"sha256"`
		Length int64  `json:"length"`
	} `json:"objects"`
}

// Closure declares the exact object set a publish of this snapshot MOVES. The
// returned bytes are the document verbatim — they ride the declaration and the hub
// reproduces them at completion, so re-encoding them here would be the drift.
func (t *Tool) Closure(snapshot, session, outPath string) ([]byte, []Object, *exit.Error) {
	if _, e := t.run("cloud", "closure", t.Root, hex(snapshot), session, "--out", outPath); e != nil {
		return nil, nil, e
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		return nil, nil, exit.Internalf("the closure tfs wrote is unreadable: %s", err)
	}
	var doc closureDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, nil, exit.Internalf("the closure document is not readable as an object list: %s", err)
	}
	objs := make([]Object, 0, len(doc.Objects))
	for _, o := range doc.Objects {
		objs = append(objs, Object{ID: "sha256:" + o.SHA256, Length: o.Length})
	}
	return raw, objs, nil
}

var attachment = regexp.MustCompile(`attachment\s+sha256:([0-9a-f]{64})`)

// Header is the snapshot's TYPED header attachment. tfs names it; no pathname
// convention is consulted here or anywhere.
func (t *Tool) Header(snapshot string) (string, *exit.Error) {
	out, e := t.run("snapshot", "show", t.Root, hex(snapshot))
	if e != nil {
		return "", e
	}
	m := attachment.FindStringSubmatch(out)
	if m == nil {
		return "", exit.Internalf("tfs snapshot show named no header attachment for %s", snapshot)
	}
	return m[1], nil
}

// Topology is the header's own code topology, as the declaration carries it.
func (t *Tool) Topology(headerHex, outPath string) ([]byte, *exit.Error) {
	if _, e := t.run("compat", "topology", t.Root, headerHex, "--out", outPath); e != nil {
		return nil, e
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		return nil, exit.Internalf("the topology tfs wrote is unreadable: %s", err)
	}
	return raw, nil
}

var entry = regexp.MustCompile(`file\s+\S+\s+sha256:([0-9a-f]{64}) \((\d+) B\)`)

// Entries is what a snapshot manifest names DIRECTLY — documents and configs. It is
// the first round of a fetch: with these in the store, the closure walk below can
// compute the transitive set without any byte of the tensors having arrived.
func (t *Tool) Entries(snapshot string) ([]Object, *exit.Error) {
	out, e := t.run("snapshot", "show", t.Root, hex(snapshot))
	if e != nil {
		return nil, e
	}
	var objs []Object
	for _, m := range entry.FindAllStringSubmatch(out, -1) {
		n, _ := strconv.ParseInt(m[2], 10, 64)
		objs = append(objs, Object{ID: "sha256:" + m[1], Length: n})
	}
	if len(objs) == 0 {
		return nil, exit.Internalf("tfs snapshot show listed no file entries for %s", snapshot)
	}
	return objs, nil
}

var present = regexp.MustCompile(`contains:\s+(true|false)`)

// Held answers whether the store already holds an object. It is a HINT and tfs says
// so on every line it prints: presence is never a resume predicate, which is why the
// resume path below is `Fill` (verification records) and not this.
func (t *Tool) Held(id string) (bool, *exit.Error) {
	out, e := t.run("contains", t.Root, hex(id))
	if e != nil {
		return false, e
	}
	m := present.FindStringSubmatch(out)
	return m != nil && m[1] == "true", nil
}

// ---------------------------------------------------------------- bytes

// Extract streams one verified object OUT of the store, for upload. The read is
// verified by tfs against the identity asked for, so a publisher cannot upload bytes
// its own store has silently corrupted.
func (t *Tool) Extract(id, outPath string) *exit.Error {
	_, e := t.run("get", t.Root, hex(id), "--out", outPath)
	return e
}

// Admit installs one object under a declared identity: a digest-checked, no-clobber
// write. This is how a fetched manifest enters the store — it proves itself.
func (t *Tool) Admit(path, id string, length int64) *exit.Error {
	_, e := t.run("put", t.Root, path,
		"--expect", hex(id), "--expect-length", strconv.FormatInt(length, 10))
	return e
}

// FillResult is what one resumable install pass did.
type FillResult struct {
	Put     int
	Skipped int
	Refused int
}

var fillLine = regexp.MustCompile(`fill: put (\d+), skipped (\d+), refused (\d+)`)

// Fill is the transactional install of a fetched set, and the whole resume story
// (cl-009's sibling on the CAS side): tfs skips only objects with a VALID
// verification record, rehashes a present-but-unverified one, quarantines a corrupt
// squatter out of the way, and admits every other object under its declared
// identity. The store's own records are the journal — this client keeps none.
func (t *Tool) Fill(plan []byte, planPath string) (FillResult, *exit.Error) {
	if err := os.WriteFile(planPath, plan, 0o644); err != nil {
		return FillResult{}, exit.Internalf("cannot write the fill plan: %s", err)
	}
	out, e := t.run("fill", t.Root, planPath)
	if e != nil {
		return FillResult{}, e
	}
	m := fillLine.FindStringSubmatch(out)
	if m == nil {
		return FillResult{}, exit.Internalf("tfs fill printed no result line")
	}
	put, _ := strconv.Atoi(m[1])
	skipped, _ := strconv.Atoi(m[2])
	refused, _ := strconv.Atoi(m[3])
	return FillResult{Put: put, Skipped: skipped, Refused: refused}, nil
}

var verified = regexp.MustCompile(`(\d+) tensors, (\d+) parts, (\d+) blobs, (\d+) distinct objects`)

// Verify is the whole-checkpoint proof: every byte the snapshot declares, hashed.
// A fetch is not finished until this passes — and the root below is registered only
// after it does, which is what "no partial visibility" means locally.
func (t *Tool) Verify(snapshot string) (tensors, parts, objects int, e *exit.Error) {
	out, e := t.run("snapshot", "verify", t.Root, hex(snapshot))
	if e != nil {
		return 0, 0, 0, e
	}
	m := verified.FindStringSubmatch(out)
	if m == nil {
		return 0, 0, 0, exit.Internalf("tfs snapshot verify printed no summary")
	}
	tensors, _ = strconv.Atoi(m[1])
	parts, _ = strconv.Atoi(m[2])
	objects, _ = strconv.Atoi(m[4])
	return tensors, parts, objects, nil
}

// Register names a verified snapshot in the local store's root authority, and notes
// that the hub holds it durably. `--hub-published` is durability WITHOUT pinning:
// the note says the bytes survive elsewhere, it does not promise to keep them here.
func (t *Tool) Register(name, snapshot string) *exit.Error {
	if _, e := t.run("root", "register", t.Root, name, hex(snapshot)); e != nil {
		return e
	}
	_, e := t.run("root", "note", t.Root, hex(snapshot), "--hub-published")
	return e
}

// Note records that the hub holds this snapshot, without claiming a local root. A
// publisher's own store already had the bytes; what publishing added is durability.
func (t *Tool) Note(snapshot string) *exit.Error {
	_, e := t.run("root", "note", t.Root, hex(snapshot), "--hub-published")
	return e
}

// hex strips the `sha256:` spelling for the argument position tfs takes bare hex in.
func hex(id string) string { return strings.TrimPrefix(id, "sha256:") }

// Snapshot normalizes a snapshot reference to the `sha256:<64 hex>` spelling, or
// refuses. It is the one place that decides what a local ref looks like.
func Snapshot(ref string) (string, *exit.Error) {
	h := hex(strings.TrimSpace(ref))
	if len(h) != 64 {
		return "", exit.Usagef("%q is not a snapshot id", ref).
			WithRemedy("a snapshot id is sha256:<64 hex> — `tfs ingest install` prints one, and so does `cozy pull`")
	}
	for _, c := range h {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return "", exit.Usagef("%q is not a snapshot id: %q is not a hex digit", ref, c).
				WithRemedy("a snapshot id is sha256:<64 hex>, lowercase")
		}
	}
	return "sha256:" + h, nil
}
