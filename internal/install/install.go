// Package install is the staged install transaction (cozy-creator.md function 2):
// stage → verify source → build venv → verify descriptor → activate the pin in ONE
// database transaction. An install is an IMMUTABLE GENERATION plus a pin. The active
// generation is never extracted over, rebuilt in place, or mutated; a kill at any
// pre-activation stage leaves the previous pin runnable.
package install

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type Request struct {
	Ref           Ref
	Archive       string // --from: the local release archive (pre-hub door; cl-011 resolves it instead)
	ExpectDigest  string // --digest: the source digest the publisher declared
	Dir           string // --dir: an editable local tree, the development trust path
	AllowUnsigned bool
	Force         bool
	Published     *PublishedSource
}

type PublishedSource struct {
	Artifacts    map[string]string
	Bytes        int64
	Files        int
	Package      string
	ProjectWheel string
	Release      string
	SourceDigest string
	SourceDir    string
	Wheels       []string
	Selection    Selection
}

type ExactDocument struct {
	Bytes  []byte
	Digest string
	Length int64
}

type Selection struct {
	Profile            string
	PlacementSet       ExactDocument
	PackageRelease     ExactDocument
	PackageDescriptor  ExactDocument
	Qualification      ExactDocument
	EnvironmentReceipt ExactDocument
	WheelhouseManifest ExactDocument
}

func persistSelection(genDir string, gen *records.PackageInstall, selection Selection) *exit.Error {
	if selection.Profile == "" {
		return exit.Named(exit.Structural, "package_selection_missing",
			"published install has no selected compatibility profile")
	}
	documents := []struct {
		name   string
		format string
		exact  ExactDocument
	}{
		{"placement_set", "cozy.worker.v1.PlacementSet/3", selection.PlacementSet},
		{"package_release", "cozy.package.release/1", selection.PackageRelease},
		{"package_descriptor", "cozy.package.descriptor/1", selection.PackageDescriptor},
		{"qualification", "cozy.runtime.Qualification/1", selection.Qualification},
		{"environment_receipt", "cozy.runtime.EnvironmentReceipt/1", selection.EnvironmentReceipt},
		{"wheelhouse_manifest", "WheelhouseManifest/3", selection.WheelhouseManifest},
	}
	cache := filepath.Join(genDir, "artifact-cache")
	if err := os.MkdirAll(cache, 0o700); err != nil {
		return exit.Internalf("cannot create the package artifact cache: %s", err)
	}
	for _, document := range documents {
		digest, err := canonical.Raw(document.exact.Digest)
		if err != nil || document.exact.Length != int64(len(document.exact.Bytes)) ||
			!bytes.Equal(canonical.Digest(document.exact.Bytes), digest) {
			return exit.Named(exit.Conflict, "package_selection_identity_mismatch",
				"selected %s bytes do not match their digest and length", document.name)
		}
		object, err := canonical.ReadObject(document.exact.Bytes)
		if err != nil || object.Str("format") != document.format {
			return exit.Named(exit.Conflict, "package_selection_document_invalid",
				"selected %s is not exact canonical %s bytes", document.name, document.format)
		}
		path := filepath.Join(cache, strings.TrimPrefix(document.exact.Digest, "sha256:"))
		if err := os.WriteFile(path, document.exact.Bytes, 0o600); err != nil {
			return exit.Internalf("cannot persist selected %s: %s", document.name, err)
		}
	}
	set, err := canonical.Read(selection.PlacementSet.Bytes, &pb.PlacementSet{})
	if err != nil || len(set.List("placements")) != 1 {
		return exit.Named(exit.Conflict, "package_selection_placement_invalid",
			"selected PlacementSet must carry exactly one placement")
	}
	placement := set.List("placements")[0]
	for name, exact := range map[string]ExactDocument{
		"package_release":    selection.PackageRelease,
		"package_descriptor": selection.PackageDescriptor,
		"qualification":      selection.Qualification,
	} {
		ref := placement.Sub(name)
		if ref.Str("digest") != exact.Digest || ref.Int("length") != exact.Length {
			return exit.Named(exit.Conflict, "package_selection_reference_mismatch",
				"PlacementSet %s reference does not name the carried exact bytes", name)
		}
	}
	gen.PlacementSetDigest = selection.PlacementSet.Digest
	gen.SelectionProfile = selection.Profile
	return nil
}

type Timing struct {
	Stage string
	Took  time.Duration
}

type Result struct {
	Gen        records.PackageInstall
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
	sources := 0
	if req.Archive != "" {
		sources++
	}
	if req.Dir != "" {
		sources++
	}
	if req.Published != nil {
		sources++
	}
	if sources != 1 {
		return nil, exit.Usagef("`cozy package install` needs exactly one source: --from <archive.tar.gz> or --dir <tree>").
			WithRemedy("--from installs a published release archive; --dir installs an editable local tree").
			WithNext("cozy help package install")
	}
	id, e := newGenerationID()
	if e != nil {
		return nil, e
	}
	genDir := l.GenerationDir(id)
	res := &Result{}
	clock := time.Now()
	mark := func(stage string) {
		res.Timings = append(res.Timings, Timing{stage, time.Since(clock)})
		clock = time.Now()
	}
	fail := func(err *exit.Error) (*Result, *exit.Error) {
		_ = os.RemoveAll(genDir)
		return nil, err
	}

	gen := records.PackageInstall{ID: id, Dir: genDir}

	// ---- stage: bytes land under bounds; no code from the release has run ----
	var sourceDir string
	switch {
	case req.Published != nil:
		if req.Published.Package == "" || req.Published.Release == "" || req.Published.SourceDigest == "" ||
			req.Published.ProjectWheel == "" {
			return fail(exit.Internalf("published package source is incomplete"))
		}
		sourceDir = genDir
		if err := os.MkdirAll(genDir, 0o700); err != nil {
			return fail(exit.Internalf("cannot create package generation: %s", err))
		}
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "tensorhub", req.Published.Package+"@"+req.Published.Release, req.Published.SourceDigest
		gen.Package, gen.Version, gen.ProjectDir = req.Published.Package, req.Published.Release, genDir
		if e := persistSelection(genDir, &gen, req.Published.Selection); e != nil {
			return fail(e)
		}
		res.Files, res.Bytes = req.Published.Files, req.Published.Bytes
	case req.Archive != "":
		staged, err := StageArchive(req.Archive, filepath.Join(genDir, "source"))
		if err != nil {
			return fail(err)
		}
		sourceDir = staged.Root
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "archive", req.Archive, staged.Digest
		gen.Package, gen.Version = staged.Decl.Package, staged.Decl.Version
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
		gen.Package = req.Ref.Package
		gen.Version = "dev+" + strings.TrimPrefix(digest, "sha256:")[:12]
		res.Files, res.Bytes = files, bytes
	}
	mark("stage")

	// ---- verify: source identity settles BEFORE anything can execute ----
	if e := verifySource(&gen, req, &res.Warnings); e != nil {
		return fail(e)
	}
	if e := resolveTarget(&gen, req.Ref); e != nil {
		return fail(e)
	}
	mark("verify")

	// The pin decision is made before the expensive step, never after it.
	prior, priorGen, e := st.ActivePackage(gen.Package)
	if e != nil {
		return fail(e)
	}
	if prior != nil && priorGen != nil && priorGen.SourceDigest == gen.SourceDigest {
		res.Idempotent = true
		res.Gen = *priorGen
		_ = os.RemoveAll(genDir)
		return res, nil
	}
	if prior != nil && !req.Force {
		_ = os.RemoveAll(genDir)
		return nil, exit.New(exit.Conflict,
			"%s is already installed and pinned to generation %s", gen.Package+majorSuffix(gen.Major), short12(prior.InstallID)).
			WithRemedy("install never silently upgrades; rerun the same source command with --force to build and swap a new generation").
			WithNext("cozy package list")
	}
	// A --force that fails must keep the working install: exit 13, nothing mutated.
	guard := func(err *exit.Error) (*Result, *exit.Error) {
		if prior == nil {
			return fail(err)
		}
		_ = os.RemoveAll(genDir)
		return nil, exit.New(exit.Conflict,
			"the replacement generation for %s failed; the working install (generation %s) is untouched and still runnable",
			gen.Package+majorSuffix(gen.Major), short12(prior.InstallID)).
			WithRemedy("cause: %s — %s", err.ErrName(), err.Message).
			WithNext("cozy package list")
	}

	if e := checkCapacity(l.Generations, res.Bytes); e != nil {
		return guard(e)
	}

	// ---- venv: the first code-executing step, on verified source only ----
	venvDir := filepath.Join(genDir, "venv")
	var env *EnvironmentReceipt
	var err *exit.Error
	if req.Published != nil {
		wheels := append([]string{req.Published.ProjectWheel}, req.Published.Wheels...)
		manifest, manifestErr := canonical.ReadObject(req.Published.Selection.WheelhouseManifest.Bytes)
		if manifestErr != nil {
			return guard(exit.Named(exit.Structural, "published_wheelhouse_invalid",
				"selected base worker image inventory is unreadable: %s", manifestErr))
		}
		env, err = MaterializePublishedEnvironment(
			manifest.Sub("compatibility_profile").Str("python_abi"), venvDir, wheels)
	} else {
		env, err = MaterializeEnvironment(sourceDir, venvDir)
	}
	if err != nil {
		return guard(err)
	}
	if req.Published != nil {
		if e := persistPublishedArtifacts(genDir, req.Published.Artifacts); e != nil {
			return guard(e)
		}
	}
	gen.Python, gen.UV, gen.LockDigest = env.Python, env.UV, env.LockDigest
	gen.Platform, gen.Extra, gen.LinkMode = env.Platform, env.Extra, env.LinkMode
	gen.Packages, gen.Closure = env.Packages, env.Closure
	res.Warnings = append(res.Warnings, env.Warnings...)
	mark("venv")

	// ---- descriptor: published installs consume the exact qualified descriptor;
	// development installs still derive their local, non-runnable surface.
	var descriptor *launch.PackageDescriptor
	if req.Published != nil {
		descriptor, e = launch.DecodeDescriptor(req.Published.Selection.PackageDescriptor.Bytes)
	} else {
		descriptor, e = deriveDescriptor(venvDir, sourceDir)
	}
	if e != nil {
		return guard(e)
	}
	descriptorPath := launch.DescriptorPath(genDir)
	if err := os.MkdirAll(filepath.Dir(descriptorPath), 0o700); err != nil {
		return guard(exit.Internalf("cannot create private descriptor root: %s", err))
	}
	if err := os.WriteFile(descriptorPath, descriptor.Raw, 0o600); err != nil {
		return guard(exit.Internalf("cannot store private descriptor: %s", err))
	}
	gen.PackageDescriptor = descriptor.Digest
	if req.Published != nil && descriptor.Digest != req.Published.Selection.PackageDescriptor.Digest {
		return guard(exit.Named(exit.Conflict, "descriptor_identity_mismatch",
			"the installed package derived descriptor %s, selected release requires %s",
			descriptor.Digest, req.Published.Selection.PackageDescriptor.Digest))
	}
	mark("package_descriptor")

	// Disk is measured once, here, and read back from the record forever after.
	gen.BytesExcl, gen.BytesShared = Disk(genDir)

	// ---- activate: the generation row and the pin swap commit together ----
	superseded, e := st.Activate(gen)
	if e != nil {
		return guard(e)
	}
	res.Gen, res.Superseded = gen, superseded
	mark("activate")
	return res, nil
}

func persistPublishedArtifacts(genDir string, artifacts map[string]string) *exit.Error {
	cache := filepath.Join(genDir, "artifact-cache")
	for digest, source := range artifacts {
		if _, err := canonical.Raw(digest); err != nil {
			return exit.Internalf("downloaded package artifact has invalid digest %q", digest)
		}
		target := filepath.Join(cache, strings.TrimPrefix(digest, "sha256:"))
		if err := os.Link(source, target); err == nil {
			continue
		}
		input, err := os.Open(source)
		if err != nil {
			return exit.Internalf("cannot reopen downloaded package artifact: %s", err)
		}
		output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			_, err = io.Copy(output, input)
		}
		closeInput, closeOutput := input.Close(), error(nil)
		if output != nil {
			closeOutput = output.Close()
		}
		if err != nil || closeInput != nil || closeOutput != nil {
			_ = os.Remove(target)
			return exit.Internalf("cannot retain downloaded package artifact")
		}
	}
	return nil
}

// deriveDescriptor runs the generation's own Runtime over its source. Runtime emits the
// complete descriptor without writing the source tree; Cozy validates the closed grammar
// and stores the canonical bytes under the immutable generation root.
func deriveDescriptor(venvDir, sourceDir string) (*launch.PackageDescriptor, *exit.Error) {
	bin := home.VenvTool(venvDir, "cozy-runtime")
	if _, err := os.Stat(bin); err != nil {
		return nil, exit.Named(exit.Structural, "runtime_missing",
			"this generation's venv provides no cozy-runtime at %s", bin).
			WithRemedy("a package depends on cozy-runtime; its surface is described by the runtime the release itself pinned, never this host's").
			WithNext("cozy help package install")
	}
	cmd := exec.Command(bin, "--json", "--dir", sourceDir, "describe")
	cmd.Env = config.Frozen().Tool()
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, exit.Internalf("cannot run %s: %s", bin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, describeRefusal(code, stderr.String())
	}
	descriptor, problem := launch.DecodeDescriptor([]byte(stdout.String()))
	if problem != nil {
		return nil, exit.Named(exit.Validation, "descriptor_invalid",
			"cozy-runtime describe returned an invalid descriptor: %s", problem.Message)
	}
	return descriptor, nil
}

// describeRefusal renders the runtime's typed refusal under the shared exit vocabulary.
func describeRefusal(code int, stderr string) *exit.Error {
	var doc struct {
		Error struct{ Name, Message, Remedy string } `json:"error"`
	}
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	name := "descriptor_refused"
	if json.Unmarshal([]byte(stderr), &doc) != nil || doc.Error.Message == "" {
		return exit.Named(c, name,
			"`cozy-runtime describe` refused this generation (exit %d)", code).
			WithRemedy("%s", condense(stderr))
	}
	return exit.Named(c, name, "%s", doc.Error.Message).
		WithRemedy("%s", doc.Error.Remedy).
		WithNext("cozy help package install")
}

// verifySource settles source identity before any build backend or import can run.
func verifySource(gen *records.PackageInstall, req Request, warn *[]string) *exit.Error {
	if gen.SourceKind == "tensorhub" {
		gen.Verified = true
		return nil
	}
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
				WithNext("cozy help package install")
		}
		gen.Verified = false
		*warn = append(*warn, "--allow-unsigned: this generation was installed WITHOUT source verification (development door)")
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(req.ExpectDigest), gen.SourceDigest) {
		return exit.Named(exit.Validation, "source_digest_mismatch",
			"the release archive does not match the declared source digest").
			WithRemedy("declared %s, archive %s", short(req.ExpectDigest), short(gen.SourceDigest)).
			WithNext("cozy help package install")
	}
	gen.Verified = true
	return nil
}

// resolveTarget reconciles what the release says with what the caller asked for.
func resolveTarget(gen *records.PackageInstall, ref Ref) *exit.Error {
	if gen.SourceKind == "dir" {
		if !ref.HasMajor {
			return exit.Usagef("an editable --dir install needs an explicit major").
				WithRemedy("a local tree carries no release record, so the major is declared: cozy package install %s@v1 --dir …", ref.Package).
				WithNext("cozy help package install")
		}
		gen.Major = ref.Major
		return nil
	}
	if gen.Package == "" || gen.Version == "" {
		return exit.Named(exit.Validation, "release_undeclared",
			"%s declares no package and version", DeclName)
	}
	if gen.Package != ref.Package {
		return exit.Named(exit.Validation, "release_mismatch",
			"the archive publishes %q but %q was requested", gen.Package, ref.Package).
			WithRemedy("install the ref the release declares, or point --from at the right archive")
	}
	major, e := MajorOf(gen.Version)
	if e != nil {
		return e
	}
	if ref.HasMajor && ref.Major != major {
		return exit.Named(exit.Validation, "release_mismatch",
			"%s was requested but the archive publishes version %s (major %d)", ref.String(), gen.Version, major).
			WithRemedy("install the requested release; one package has only one active version")
	}
	gen.Major = major
	return nil
}

// checkCapacity refuses BELOW the physical floor with a quantified shortfall
// rather than filling the disk halfway through a venv build.
func checkCapacity(dir string, staged int64) *exit.Error {
	free, ok := freeBytes(dir)
	if !ok {
		return nil
	}
	need := staged*2 + (256 << 20)
	if free >= need {
		return nil
	}
	return exit.New(exit.Capacity,
		"not enough disk to build this generation: needed %s, had %s, short by %s on %s",
		bytesText(need), bytesText(free), bytesText(need-free), dir).
		WithRemedy("free space, remove an installed package, or move COZY_HOME to a larger filesystem").
		WithNext("cozy package list")
}

// Remove drops one install: its pin, generation row, and exclusive directory.
// Shared TensorFS bytes are never touched here.
func Remove(st *records.Store, pkg string, major int) (int64, *exit.Error) {
	pin, gen, e := st.ActivePin(pkg, major)
	if e != nil || pin == nil {
		return 0, e
	}
	if e := st.Unpin(pkg, major); e != nil {
		return 0, e
	}
	if gen == nil {
		return 0, nil
	}
	return Reclaim(st, gen.ID)
}

// Reclaim removes an unpinned generation immediately. The database claim wins
// before filesystem deletion, so a newly pinned generation is never removed.
func Reclaim(st *records.Store, id string) (int64, *exit.Error) {
	gen, problem := st.Install(id)
	if problem != nil || gen == nil {
		return 0, problem
	}
	claimed, problem := st.ForgetIfUnreferenced(id)
	if problem != nil || !claimed {
		return 0, problem
	}
	if err := os.RemoveAll(gen.Dir); err != nil {
		return 0, exit.Internalf("cannot remove generation directory %s: %s", gen.Dir, err)
	}
	return gen.BytesExcl, nil
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
