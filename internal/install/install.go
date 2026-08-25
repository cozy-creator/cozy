// Package install is the staged install transaction (cozy-creator.md function 2):
// stage → verify source → build venv → verify descriptor → activate the pin in ONE
// database transaction. An install is an IMMUTABLE GENERATION plus a pin. The active
// generation is never extracted over, rebuilt in place, or mutated; a kill at any
// pre-activation stage leaves the previous pin runnable.
package install

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

// Stages, in order. `--crash-after <stage>` kills the process after the named one;
// that is how the crash matrix is observed on the real binary.
var Stages = []string{"stage", "verify", "venv", "descriptor", "activate"}

// DescriptorPending is what the environment record carries until the runtime can
// derive a descriptor. cr-003 defines the descriptor; cr-016 lands the verb.
const DescriptorPending = "pending-cr-003"

type Request struct {
	Ref           Ref
	Archive       string // --from: the local release archive (pre-hub door; cl-011 resolves it instead)
	ExpectDigest  string // --digest: the source digest the publisher declared
	Dir           string // --dir: an editable local tree, the development trust path
	AllowUnsigned bool
	Force         bool
	CrashAfter    string
}

type Timing struct {
	Stage string
	Took  time.Duration
}

type Result struct {
	Gen        records.Generation
	Superseded string
	Idempotent bool
	Timings    []Timing
	Warnings   []string
	Files      int
	Bytes      int64
	Compressed int64
}

// Run executes the whole transaction. Every refusal before Activate leaves the
// records untouched, so the previously pinned generation stays runnable.
func Run(l home.Layout, st *records.Store, req Request) (*Result, *exit.Error) {
	if (req.Archive == "") == (req.Dir == "") {
		return nil, exit.Usagef("`cozy install` needs exactly one source: --from <archive.tar.gz> or --dir <tree>").
			WithRemedy("--from installs a published release archive; --dir installs an editable local tree").
			WithNext("cozy help install")
	}
	if req.CrashAfter != "" && !containsStage(req.CrashAfter) {
		return nil, exit.Usagef("--crash-after %q is not an install stage", req.CrashAfter).
			WithRemedy("stages: %s", strings.Join(Stages, ", "))
	}

	id, e := newGenerationID()
	if e != nil {
		return nil, e
	}
	genDir := l.GenerationDir(id)
	res := &Result{}
	clock := time.Now()
	mark := func(stage string) *exit.Error {
		res.Timings = append(res.Timings, Timing{stage, time.Since(clock)})
		clock = time.Now()
		if req.CrashAfter == stage {
			// A real SIGKILL, not an orderly exit: the crash matrix is only
			// evidence if the process actually dies where a crash would.
			_ = syscall.Kill(os.Getpid(), syscall.SIGKILL)
		}
		return nil
	}
	fail := func(err *exit.Error) (*Result, *exit.Error) {
		_ = os.RemoveAll(genDir)
		return nil, err
	}

	gen := records.Generation{ID: id, Dir: genDir}

	// ---- stage: bytes land under bounds; no code from the release has run ----
	var sourceDir string
	switch {
	case req.Archive != "":
		staged, err := StageArchive(req.Archive, filepath.Join(genDir, "source"))
		if err != nil {
			return fail(err)
		}
		sourceDir = staged.Root
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "archive", req.Archive, staged.Digest
		gen.Endpoint, gen.Version = staged.Decl.Endpoint, staged.Decl.Version
		res.Files, res.Bytes, res.Compressed = staged.Files, staged.Bytes, staged.Compressed
	default:
		abs, err := filepath.Abs(req.Dir)
		if err != nil {
			return fail(exit.Usagef("--dir %q is not resolvable: %s", req.Dir, err))
		}
		digest, files, bytes, e := SnapshotDigest(abs)
		if e != nil {
			return fail(e)
		}
		// Editable by design: the venv is built against the live tree, never a copy.
		sourceDir = abs
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "dir", abs, digest
		gen.Endpoint = req.Ref.Endpoint
		gen.Version = "dev+" + strings.TrimPrefix(digest, "sha256:")[:12]
		res.Files, res.Bytes = files, bytes
	}
	if err := mark("stage"); err != nil {
		return fail(err)
	}

	// ---- verify: source identity settles BEFORE anything can execute ----
	if e := verifySource(&gen, req, &res.Warnings); e != nil {
		return fail(e)
	}
	if e := resolveTarget(&gen, req.Ref); e != nil {
		return fail(e)
	}
	if err := mark("verify"); err != nil {
		return fail(err)
	}

	// The pin decision is made before the expensive step, never after it.
	prior, priorGen, e := st.ActivePin(gen.Endpoint, gen.Major)
	if e != nil {
		return fail(e)
	}
	if prior != nil && !req.Force {
		if priorGen != nil && priorGen.SourceDigest == gen.SourceDigest {
			res.Idempotent = true
			res.Gen = *priorGen
			_ = os.RemoveAll(genDir)
			return res, nil
		}
		_ = os.RemoveAll(genDir)
		return nil, exit.New(exit.Conflict,
			"%s is already installed and pinned to generation %s", gen.Endpoint+majorSuffix(gen.Major), short12(prior.Generation)).
			WithRemedy("install never silently upgrades; --force builds a new generation and swaps the pin").
			WithNext("cozy install " + req.Ref.String() + " --force")
	}
	// A --force that fails must keep the working install: exit 13, nothing mutated.
	guard := func(err *exit.Error) (*Result, *exit.Error) {
		if prior == nil {
			return fail(err)
		}
		_ = os.RemoveAll(genDir)
		return nil, exit.New(exit.Conflict,
			"the replacement generation for %s failed; the working install (generation %s) is untouched and still runnable",
			gen.Endpoint+majorSuffix(gen.Major), short12(prior.Generation)).
			WithRemedy("cause: %s — %s", err.ErrName(), err.Message).
			WithNext("cozy ls")
	}

	if e := checkCapacity(l.Generations, res.Bytes); e != nil {
		return guard(e)
	}

	// ---- venv: the first code-executing step, on verified source only ----
	env, err := BuildVenv(sourceDir, filepath.Join(genDir, "venv"))
	if err != nil {
		return guard(err)
	}
	gen.Python, gen.UV, gen.LockDigest = env.Python, env.UV, env.LockDigest
	gen.Platform, gen.Extra, gen.LinkMode = env.Platform, env.Extra, env.LinkMode
	gen.Packages, gen.Closure = env.Packages, env.Closure
	res.Warnings = append(res.Warnings, env.Warnings...)
	if err := mark("venv"); err != nil {
		return guard(err)
	}

	// ---- descriptor: a typed, labelled pending stage (see deriveDescriptor) ----
	gen.Descriptor = deriveDescriptor()
	if err := mark("descriptor"); err != nil {
		return guard(err)
	}

	// Disk is measured once, here, and read back from the record forever after.
	gen.BytesExcl, gen.BytesShared = Disk(genDir)

	// ---- activate: the generation row and the pin swap commit together ----
	superseded, e := st.Activate(gen)
	if e != nil {
		return guard(e)
	}
	res.Gen, res.Superseded = gen, superseded
	if err := mark("activate"); err != nil {
		return nil, err
	}
	return res, nil
}

// deriveDescriptor is the descriptor stage's seam, deliberately pending.
//
// When cr-016 lands, this stage runs `cozy-runtime describe --write-descriptor` in
// the generation's own venv and compares the derived descriptor against the
// release's declared descriptor evidence, refusing typed on disagreement. Until
// cr-003 defines that descriptor there is nothing to derive and nothing to compare,
// so the stage records the pending marker and refuses nothing. A fabricated
// descriptor would be worse than an honest gap.
func deriveDescriptor() string { return DescriptorPending }

// verifySource settles source identity before any build backend or import can run.
func verifySource(gen *records.Generation, req Request, warn *[]string) *exit.Error {
	if gen.SourceKind == "dir" {
		// --dir IS the development trust path: an explicit local tree the operator
		// already controls. It carries a snapshot identity, never publisher evidence.
		gen.Verified = false
		*warn = append(*warn, "editable --dir install: source identity is a local snapshot digest, not publisher evidence")
		return nil
	}
	if req.ExpectDigest == "" {
		if !req.AllowUnsigned {
			return exit.Named(exit.Validation, "source_unverified",
				"the release archive carries no verified source digest").
				WithRemedy("pass --digest <sha256:…> from the release record, or --allow-unsigned for a development install").
				WithNext("cozy help install")
		}
		gen.Verified = false
		*warn = append(*warn, "--allow-unsigned: this generation was installed WITHOUT source verification (development door)")
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(req.ExpectDigest), gen.SourceDigest) {
		return exit.Named(exit.Validation, "source_digest_mismatch",
			"the release archive does not match the declared source digest").
			WithRemedy("declared %s, archive %s", short(req.ExpectDigest), short(gen.SourceDigest)).
			WithNext("cozy help install")
	}
	gen.Verified = true
	return nil
}

// resolveTarget reconciles what the release says with what the caller asked for.
func resolveTarget(gen *records.Generation, ref Ref) *exit.Error {
	if gen.SourceKind == "dir" {
		if !ref.HasMajor {
			return exit.Usagef("an editable --dir install needs an explicit major").
				WithRemedy("a local tree carries no release record, so the major is declared: cozy install %s@v1 --dir …", ref.Endpoint).
				WithNext("cozy help install")
		}
		gen.Major = ref.Major
		return nil
	}
	if gen.Endpoint == "" || gen.Version == "" {
		return exit.Named(exit.Validation, "release_undeclared",
			"%s declares no endpoint and version", DeclName)
	}
	if gen.Endpoint != ref.Endpoint {
		return exit.Named(exit.Validation, "release_mismatch",
			"the archive publishes %q but %q was requested", gen.Endpoint, ref.Endpoint).
			WithRemedy("install the ref the release declares, or point --from at the right archive")
	}
	major, e := MajorOf(gen.Version)
	if e != nil {
		return e
	}
	if ref.HasMajor && ref.Major != major {
		return exit.Named(exit.Validation, "release_mismatch",
			"%s was requested but the archive publishes version %s (major %d)", ref.String(), gen.Version, major).
			WithRemedy("the pin is per (endpoint, major); majors never substitute for one another")
	}
	gen.Major = major
	return nil
}

// checkCapacity refuses BELOW the physical floor with a quantified shortfall
// rather than filling the disk halfway through a venv build.
func checkCapacity(dir string, staged int64) *exit.Error {
	var fs syscall.Statfs_t
	if err := syscall.Statfs(dir, &fs); err != nil {
		return nil
	}
	free := int64(fs.Bavail) * int64(fs.Bsize)
	need := staged*2 + (256 << 20)
	if free >= need {
		return nil
	}
	return exit.New(exit.Capacity,
		"not enough disk to build this generation: needed %s, had %s, short by %s on %s",
		bytesText(need), bytesText(free), bytesText(need-free), dir).
		WithRemedy("free space or move COZY_HOME to a larger filesystem").
		WithNext("cozy gc")
}

// Remove drops one install: the pin and the generation directory. The generation
// row survives as unreferenced until `cozy gc` reclaims it, and shared-CAS weights
// are never touched here.
func Remove(st *records.Store, endpoint string, major int) (int64, *exit.Error) {
	pin, gen, e := st.ActivePin(endpoint, major)
	if e != nil || pin == nil {
		return 0, e
	}
	if e := st.Unpin(endpoint, major); e != nil {
		return 0, e
	}
	var freed int64
	if gen != nil {
		freed = gen.BytesExcl
		_ = os.RemoveAll(gen.Dir)
	}
	return freed, nil
}

func newGenerationID() (string, *exit.Error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", exit.Internalf("cannot mint a generation id: %s", err)
	}
	return hex.EncodeToString(b), nil
}

func majorSuffix(major int) string { return fmt.Sprintf("@v%d", major) }

func short12(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func containsStage(s string) bool {
	for _, st := range Stages {
		if st == s {
			return true
		}
	}
	return false
}

