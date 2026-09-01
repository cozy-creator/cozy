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
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/units"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type Request struct {
	Ref       Ref
	Force     bool
	Local     *LocalSource
	Published *PublishedSource
}

// LocalSource is one author-controlled directory after Creator's bounded source
// scan. It is deliberately neither a wheel nor a Hub release/qualification: the
// live source digest is its local-only identity.
type LocalSource struct {
	SourceDigest string
	Bytes        int64
	Files        int
	Package      string
	Release      string
	Tree         string
}

type PublishedSource struct {
	Bytes         int64
	Files         int
	Package       string
	PackageConfig ExactDocument
	Pyproject     ExactDocument
	ProjectWheel  PublishedWheel
	LocalWheels   []PublishedWheel
	Release       string
	SourceDigest  string
	UVLock        ExactDocument
	Wheels        []PublishedWheel
	Models        []PublishedModel
	Selection     Selection
}

type PublishedModel struct {
	Lane           string `json:"lane"`
	Manifest       string `json:"manifest"`
	ManifestLength int64  `json:"manifest_length"`
	Model          string `json:"model"`
	Package        string `json:"package"`
	Release        string `json:"release"`
	Slot           string `json:"slot"`
	Reused         bool   `json:"-"`
}

// ModelPrefetchStatus is the concise user-facing result of optional post-install
// acquisition. Exact identities remain in the package/Manifest authorities, not output.
func ModelPrefetchStatus(models []PublishedModel) string {
	if len(models) == 0 {
		return "none"
	}
	reused := 0
	for _, model := range models {
		if model.Reused {
			reused++
		}
	}
	switch reused {
	case 0:
		return "downloaded"
	case len(models):
		return "reused locally"
	default:
		return "downloaded + local reuse"
	}
}

type PublishedWheel struct {
	Digest       string
	Distribution string
	Filename     string
	ImportRoots  []string
	Length       int64
	Path         string
	Tags         []string
	Version      string
}

type ExactDocument struct {
	Bytes  []byte `json:"canonical_bytes_base64"`
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type Selection struct {
	PackageDescriptor ExactDocument
}

func validatePublished(gen records.PackageInstall, published *PublishedSource) *exit.Error {
	descriptor := published.Selection.PackageDescriptor
	digest, err := canonical.Raw(descriptor.Digest)
	if err != nil || descriptor.Length != int64(len(descriptor.Bytes)) ||
		!bytes.Equal(canonical.Digest(descriptor.Bytes), digest) {
		return exit.Named(exit.Conflict, "package_descriptor_identity_mismatch",
			"published descriptor bytes do not match their digest and length")
	}
	if published.ProjectWheel.Distribution != gen.Package[strings.LastIndex(gen.Package, "/")+1:] ||
		published.ProjectWheel.Version != gen.Version ||
		published.ProjectWheel.Digest == "" || published.ProjectWheel.Path == "" {
		return exit.Named(exit.Conflict, "package_wheel_release_mismatch",
			"published project wheel does not name %s@%s", gen.Package, gen.Version)
	}
	config := published.PackageConfig
	configDigest, err := canonical.Raw(config.Digest)
	if err != nil || config.Length <= 0 || config.Length != int64(len(config.Bytes)) ||
		!bytes.Equal(canonical.Digest(config.Bytes), configDigest) {
		return exit.Named(exit.Conflict, "package_config_identity_mismatch",
			"published package.toml bytes do not match their digest and length")
	}
	for _, item := range []struct {
		name     string
		document ExactDocument
	}{{"pyproject.toml", published.Pyproject}, {"uv.lock", published.UVLock}} {
		name, document := item.name, item.document
		digest, err := canonical.Raw(document.Digest)
		if err != nil || document.Length <= 0 || document.Length != int64(len(document.Bytes)) ||
			!bytes.Equal(canonical.Digest(document.Bytes), digest) {
			return exit.Named(exit.Conflict, "package_environment_document_identity_mismatch",
				"published %s bytes do not match their digest and length", name)
		}
	}
	seenDependencies := map[string]bool{}
	for _, wheel := range published.Wheels {
		if wheel.Distribution == "" || seenDependencies[wheel.Distribution] {
			return exit.Named(exit.Conflict, "package_dependency_wheel_duplicate",
				"published package repeats dependency distribution %q", wheel.Distribution)
		}
		seenDependencies[wheel.Distribution] = true
	}
	if len(published.LocalWheels) > 2 {
		return exit.Named(exit.Conflict, "package_local_wheel_count_invalid",
			"published package carries more than two local materialization wheels")
	}
	seenLocal := map[string]bool{}
	for _, wheel := range published.LocalWheels {
		allowed := wheel.Distribution == "cozy-runtime" || wheel.Distribution == "tensorfs"
		if !allowed || seenLocal[wheel.Distribution] || seenDependencies[wheel.Distribution] {
			return exit.Named(exit.Conflict, "package_local_wheel_invalid",
				"published local materialization wheels must be unique Runtime/TensorFS distributions outside package dependencies")
		}
		seenLocal[wheel.Distribution] = true
	}
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
	// ModelStatus is post-commit Creator UX, not part of the install transaction.
	// A failed optional prefetch therefore cannot roll this successful result back.
	ModelStatus string
	ModelError  string
}

// Run executes the whole transaction. Every refusal before Activate leaves the
// records untouched, so the previously pinned generation stays runnable.
func Run(l home.Layout, st *records.Store, req Request) (*Result, *exit.Error) {
	if (req.Published == nil) == (req.Local == nil) {
		return nil, exit.Usagef("`cozy package install` needs exactly one package source").
			WithRemedy("pass org/package for Tensorhub, or an explicit directory such as . or ./project").
			WithNext("cozy help package install")
	}
	id, e := newGenerationID()
	if e != nil {
		return nil, e
	}
	genDir := l.InstallDir(id)
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
			req.Published.ProjectWheel.Path == "" {
			return fail(exit.Internalf("published package source is incomplete"))
		}
		sourceDir = genDir
		if err := os.MkdirAll(genDir, 0o700); err != nil {
			return fail(exit.Internalf("cannot create package generation: %s", err))
		}
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "tensorhub", req.Published.Package+"@"+req.Published.Release, req.Published.SourceDigest
		gen.Package, gen.Version, gen.ProjectDir = req.Published.Package, req.Published.Release,
			filepath.Join(genDir, "source")
		if e := validatePublished(gen, req.Published); e != nil {
			return fail(e)
		}
		res.Files, res.Bytes = req.Published.Files, req.Published.Bytes
	case req.Local != nil:
		local := req.Local
		if local.Package == "" || local.Release == "" || local.Tree == "" || local.SourceDigest == "" {
			return fail(exit.Internalf("local package build input is incomplete"))
		}
		abs, err := filepath.Abs(local.Tree)
		if err != nil {
			return fail(exit.Usagef("local package directory %q is not resolvable: %s", local.Tree, err))
		}
		sourceDir = abs
		gen.SourceKind, gen.SourceRef, gen.SourceDigest = "local", abs, local.SourceDigest
		gen.Package, gen.Version = local.Package, local.Release
		res.Files, res.Bytes = local.Files, local.Bytes
	}
	mark("stage")

	// ---- verify: source identity settles BEFORE anything can execute ----
	if e := verifySource(&gen, &res.Warnings); e != nil {
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

	if e := checkCapacity(l.Installs, res.Bytes); e != nil {
		return guard(e)
	}

	// ---- environment: the first code-executing step, on verified source only ----
	venvDir := filepath.Join(genDir, "venv")
	var env *EnvironmentReceipt
	var descriptor *launch.PackageDescriptor
	var placement ExactDocument
	var err *exit.Error
	if req.Published != nil {
		descriptor, placement, gen.Runtime, env, err = preparePublished(l, genDir, req.Published)
	} else {
		env, err = MaterializeEnvironment(sourceDir, venvDir)
	}
	if err != nil {
		return guard(err)
	}
	gen.Python, gen.UV, gen.LockDigest = env.Python, env.UV, env.LockDigest
	gen.Platform, gen.Extra = env.Platform, env.Extra
	gen.Packages, gen.Closure = env.Packages, env.Closure
	res.Warnings = append(res.Warnings, env.Warnings...)
	mark("environment")

	// ---- descriptor: Runtime authored both the imported published surface and its
	// resident placement. Editable source retains its existing development path.
	if req.Local != nil {
		descriptor, placement, e = deriveDevelopmentPlacement(
			venvDir, sourceDir, l.CAS, *req.Local)
		if e != nil {
			return guard(e)
		}
	}
	descriptorPath := launch.DescriptorPath(genDir)
	if err := os.MkdirAll(filepath.Dir(descriptorPath), 0o700); err != nil {
		return guard(exit.Internalf("cannot create private descriptor root: %s", err))
	}
	if err := os.WriteFile(descriptorPath, descriptor.Raw, 0o600); err != nil {
		return guard(exit.Internalf("cannot store private descriptor: %s", err))
	}
	gen.PackageDescriptor = descriptor.Digest
	if req.Local != nil {
		cache := filepath.Join(genDir, "artifact-cache")
		if err := os.MkdirAll(cache, 0o700); err != nil {
			return guard(exit.Internalf("cannot create editable placement cache: %s", err))
		}
		if err := os.WriteFile(filepath.Join(cache,
			strings.TrimPrefix(placement.Digest, "sha256:")), placement.Bytes, 0o600); err != nil {
			return guard(exit.Internalf("cannot store editable PlacementSet: %s", err))
		}
	}
	gen.PlacementSetDigest = placement.Digest
	mark("package_descriptor")
	if req.Local != nil {
		current, problem := packagepublish.PrepareLocalFrom(sourceDir)
		if problem != nil {
			return guard(exit.Named(exit.Conflict, "editable_source_changed",
				"the editable package changed while its replacement was being prepared").
				WithRemedy("source recheck failed: %s", problem.Message))
		}
		defer current.Close()
		digest, _, _, problem := current.SourceIdentity()
		if problem != nil || digest != req.Local.SourceDigest ||
			"local/"+current.Name != req.Local.Package || current.Release != req.Local.Release {
			return guard(exit.Named(exit.Conflict, "editable_source_changed",
				"the editable package changed while its replacement was being prepared").
				WithRemedy("retry after the source tree is stable"))
		}
	}

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

// deriveDescriptor runs the generation's own Runtime over its source. Runtime emits the
// complete descriptor without writing the source tree; Cozy validates the closed grammar
// and stores the canonical bytes under the immutable generation root.
func deriveDevelopmentPlacement(venvDir, sourceDir, artifactStore string, local LocalSource) (
	*launch.PackageDescriptor, ExactDocument, *exit.Error,
) {
	var empty ExactDocument
	bin := home.VenvTool(venvDir, "cozy-runtime")
	if _, err := os.Stat(bin); err != nil {
		return nil, empty, exit.Named(exit.Structural, "runtime_missing",
			"this generation's venv provides no cozy-runtime at %s", bin).
			WithRemedy("a package depends on cozy-runtime; its surface is described by the runtime the release itself pinned, never this host's").
			WithNext("cozy help package install")
	}
	cmd := exec.Command(bin, "--json", "--dir", sourceDir, "development-placement",
		"--package", local.Package, "--release", local.Release,
		"--source-digest", local.SourceDigest, "--artifact-store", artifactStore)
	cmd.Env = config.Frozen().Tool()
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, empty, exit.Internalf("cannot run %s: %s", bin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, empty, metadataRefusal(code, "development-placement", stderr.String())
	}
	type exact struct {
		Bytes  []byte `json:"canonical_bytes_base64"`
		Digest string `json:"digest"`
		Length int64  `json:"length"`
	}
	var answer struct {
		Package           string `json:"package"`
		Release           string `json:"release"`
		SourceDigest      string `json:"source_digest"`
		PlacementSet      exact  `json:"placement_set"`
		PackageDescriptor exact  `json:"package_descriptor"`
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	decoder.DisallowUnknownFields()
	decodeErr := decoder.Decode(&answer)
	var trailing any
	if decodeErr == nil {
		decodeErr = decoder.Decode(&trailing)
	}
	if decodeErr != io.EOF || answer.Package != local.Package ||
		answer.Release != local.Release || answer.SourceDigest != local.SourceDigest {
		return nil, empty, exit.Named(exit.Structural, "editable_placement_invalid",
			"cozy-runtime development-placement returned a mismatched answer")
	}
	validExact := func(value exact) bool {
		digest, err := canonical.Raw(value.Digest)
		return err == nil && value.Length == int64(len(value.Bytes)) &&
			bytes.Equal(canonical.Digest(value.Bytes), digest)
	}
	if !validExact(answer.PlacementSet) || !validExact(answer.PackageDescriptor) {
		return nil, empty, exit.Named(exit.Structural, "editable_placement_identity_mismatch",
			"cozy-runtime development-placement returned bytes that do not match their digest and length")
	}
	descriptor, problem := launch.DecodeDescriptor(answer.PackageDescriptor.Bytes)
	if problem != nil || descriptor.Digest != answer.PackageDescriptor.Digest {
		return nil, empty, exit.Named(exit.Validation, "descriptor_invalid",
			"cozy-runtime development-placement returned an invalid package descriptor")
	}
	set, readErr := canonical.Read(answer.PlacementSet.Bytes, &pb.PlacementSet{})
	if readErr != nil || len(set.List("placements")) != 1 {
		return nil, empty, exit.Named(exit.Structural, "editable_placement_invalid",
			"cozy-runtime returned an invalid development PlacementSet")
	}
	placement := set.List("placements")[0]
	development := placement.Sub("development")
	if development.Str("package") != local.Package || development.Str("release") != local.Release ||
		development.Str("source_digest") != local.SourceDigest ||
		placement.Sub("package_descriptor").Str("digest") != answer.PackageDescriptor.Digest ||
		placement.Sub("package_descriptor").Int("length") != answer.PackageDescriptor.Length ||
		placement.Str("environment_digest") != "" {
		return nil, empty, exit.Named(exit.Structural, "editable_placement_invalid",
			"cozy-runtime development PlacementSet mixes local source with published selection facts")
	}
	return descriptor, ExactDocument{Bytes: answer.PlacementSet.Bytes,
		Digest: answer.PlacementSet.Digest, Length: answer.PlacementSet.Length}, nil
}

// metadataRefusal preserves Runtime's typed metadata refusal under the shared exit vocabulary.
func metadataRefusal(code int, verb, stderr string) *exit.Error {
	var doc struct {
		Error struct{ Name, Message, Remedy string } `json:"error"`
	}
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	if json.Unmarshal([]byte(stderr), &doc) != nil || doc.Error.Message == "" {
		return exit.Named(c, "runtime_metadata_refused",
			"`cozy-runtime %s` refused this generation (exit %d)", verb, code).
			WithRemedy("%s", condense(stderr))
	}
	name := doc.Error.Name
	if name == "" {
		name = "runtime_metadata_refused"
	}
	return exit.Named(c, name, "%s", doc.Error.Message).
		WithRemedy("%s", doc.Error.Remedy).
		WithNext("cozy help package install")
}

// verifySource settles source identity before any build backend or import can run.
func verifySource(gen *records.PackageInstall, warn *[]string) *exit.Error {
	if gen.SourceKind == "tensorhub" {
		gen.Verified = true
		return nil
	}
	if gen.SourceKind == "local" {
		gen.Verified = false
		*warn = append(*warn,
			"local directory install: source bytes are pinned locally but are not a published Tensorhub release")
		return nil
	}
	return exit.Internalf("unknown package source kind %q", gen.SourceKind)
}

// resolveTarget reconciles what the release says with what the caller asked for.
func resolveTarget(gen *records.PackageInstall, ref Ref) *exit.Error {
	if gen.SourceKind == "local" {
		major, e := MajorOf(gen.Version)
		if e != nil {
			return e
		}
		gen.Major = major
		return nil
	}
	if gen.Package == "" || gen.Version == "" {
		return exit.Named(exit.Validation, "release_undeclared", "the package source declares no package and version")
	}
	if gen.Package != ref.Package {
		return exit.Named(exit.Validation, "release_mismatch",
			"the package source names %q but %q was requested", gen.Package, ref.Package).
			WithRemedy("install the package named by its project metadata")
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
		units.Bytes(need), units.Bytes(free), units.Bytes(need-free), dir).
		WithRemedy("free space, remove an installed package, or move COZY_HOME to a larger filesystem").
		WithNext("cozy package list")
}

// Remove drops one install: its pin, generation row, and exclusive directory.
// Shared TensorFS bytes are never touched here.
func Remove(l home.Layout, st *records.Store, pkg string, major int) (int64, *exit.Error) {
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
	return Reclaim(l, st, gen.ID)
}

// Reclaim removes an unpinned generation immediately. The database claim wins
// before filesystem deletion, so a newly pinned generation is never removed. Only
// after that claim may the read-only published tree be made removable.
func Reclaim(l home.Layout, st *records.Store, id string) (int64, *exit.Error) {
	gen, problem := st.Install(id)
	if problem != nil || gen == nil {
		return 0, problem
	}
	target, problem := generationRemovalTarget(l, *gen)
	if problem != nil {
		return 0, problem
	}
	claimed, problem := st.ForgetIfUnreferenced(id)
	if problem != nil || !claimed {
		return 0, problem
	}
	if problem := removeInstallTree(target); problem != nil {
		return 0, problem
	}
	return gen.BytesExcl, nil
}

// Swept is what one sweep of the install root did. Bytes is EXCLUSIVE bytes: an install
// tree is hardlinked against the shared uv cache, so removing it frees only the bytes
// whose last link went away — never the tree's apparent size.
type Swept struct {
	Scanned int
	Removed int
	Bytes   int64
}

// Sweep reclaims every install directory no record references. Reclaim is reached only
// through a row, so a records.db that was lost or rebuilt strands every directory on
// disk: unreferenced, and unreachable by any verb (cl-076). The sweep walks the
// DIRECTORY instead and lets the refcounted database claim decide, so an install that
// is still pinned, still serving a request, or still held by a live worker survives.
// Callers hold the single-writer lock: an install stages its directory before it commits
// the row that names it.
func Sweep(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	entries, err := os.ReadDir(l.Installs)
	if err != nil {
		return Swept{}, exit.Internalf("cannot scan the install root %s: %s", l.Installs, err)
	}
	var swept Swept
	var first *exit.Error
	for _, entry := range entries {
		// A DirEntry's type is its own lstat, so a symlink is skipped as a symlink: the
		// sweep removes trees this root owns and nothing it merely names.
		if !entry.IsDir() {
			continue
		}
		swept.Scanned++
		freed, removed, problem := sweepInstall(l, st, entry.Name())
		if problem != nil {
			// Each entry is independent, so one directory that will not go is not worth
			// abandoning the rest of the sweep for. The first refusal is returned once
			// the pass finishes; the directory itself waits for the next sweep.
			if first == nil {
				first = problem
			}
			continue
		}
		if removed {
			swept.Removed++
			swept.Bytes += freed
		}
	}
	return swept, first
}

// sweepInstall reclaims one directory under the install root. A recorded install goes
// through Reclaim so the claim still decides; a directory with NO record has no claim
// left to win and nothing that can reference it, and its bytes are measured here because
// the row that would have remembered them is exactly what went missing.
func sweepInstall(l home.Layout, st *records.Store, id string) (int64, bool, *exit.Error) {
	target, problem := installRemovalTarget(l, id)
	if problem != nil {
		return 0, false, problem
	}
	gen, problem := st.Install(id)
	if problem != nil {
		return 0, false, problem
	}
	if gen != nil {
		freed, problem := Reclaim(l, st, id)
		if problem != nil {
			return 0, false, problem
		}
		// Reclaim answers bytes, not verdict, and a generation may record zero exclusive
		// bytes. The directory itself is the one honest count of what the sweep removed.
		_, err := os.Lstat(target)
		return freed, os.IsNotExist(err), nil
	}
	freed, _ := Disk(target)
	if problem := removeInstallTree(target); problem != nil {
		return 0, false, problem
	}
	return freed, true, nil
}

// installRemovalTarget resolves one install id under the install root. An id that is
// anything but a single path element, or that resolves anywhere else, is refused.
func installRemovalTarget(l home.Layout, id string) (string, *exit.Error) {
	root, err := filepath.Abs(l.Installs)
	if err != nil {
		return "", exit.Internalf("cannot resolve the install root %s: %s", l.Installs, err)
	}
	if id == "" || id == "." || id == ".." || filepath.Base(id) != id {
		return "", exit.Internalf("refusing to remove generation with unsafe id %q", id)
	}
	target := filepath.Join(root, id)
	if filepath.Dir(target) != root {
		return "", exit.Internalf("refusing to remove generation %s outside %s", id, root)
	}
	return target, nil
}

func generationRemovalTarget(l home.Layout, gen records.PackageInstall) (string, *exit.Error) {
	target, problem := installRemovalTarget(l, gen.ID)
	if problem != nil {
		return "", problem
	}
	recorded, err := filepath.Abs(gen.Dir)
	if err != nil || filepath.Clean(recorded) != target {
		return "", exit.Internalf("refusing to remove generation %s outside %s", gen.ID, filepath.Dir(target))
	}
	return target, nil
}

// removeInstallTree deletes one resolved install directory. Only after the caller has
// settled that nothing references it may the read-only published tree be made removable.
func removeInstallTree(target string) *exit.Error {
	if err := makeGenerationRemovable(target); err != nil {
		return exit.Internalf("cannot prepare retired generation directory %s for removal: %s", target, err)
	}
	if err := os.RemoveAll(target); err != nil {
		return exit.Internalf("cannot remove generation directory %s: %s", target, err)
	}
	return nil
}

func makeGenerationRemovable(root string) error {
	return filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			if os.IsNotExist(walkErr) {
				return nil
			}
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		mode := removalMode(info)
		if mode == info.Mode().Perm() {
			return nil
		}
		return os.Chmod(path, mode)
	})
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
