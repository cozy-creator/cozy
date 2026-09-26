// Package install is the staged install transaction (cozy-creator.md function 2):
// stage → verify source → build venv → verify package interface → activate the pin in ONE
// database transaction. An install is an IMMUTABLE TREE plus a pin. The active install is
// never extracted over, rebuilt in place, or mutated; a kill at any pre-activation stage
// leaves the previous pin runnable.
package install

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/units"
)

type Request struct {
	Ref   Ref
	Force bool
	// Snapshot freezes local source into the install and leaves its active pin alone.
	// It is the same installation path, owned by one or more durable invocations.
	Snapshot  bool
	Local     *LocalSource
	Published *PublishedSource
}

// LocalSource is one author-controlled directory after Creator's bounded source
// scan. It is deliberately neither a wheel nor a Hub release/qualification: the
// source is installed through normal uv editable semantics.
type LocalSource struct {
	Bytes   int64
	Files   int
	Package string
	Release string
	Tree    string
}

type PublishedSource struct {
	PythonVersion string
	Bytes         int64
	Files         int
	Package       string
	PackageConfig ExactDocument
	Pyproject     ExactDocument
	ProjectWheel  PublishedWheel
	LocalWheels   []PublishedWheel
	Release       string
	UVLock        ExactDocument
	Wheels        []PublishedWheel
	// IndexURL is the org's public PEP 503 index (th-113): the second index the
	// locked-requirements export names, serving the release's own wheels.
	IndexURL  string
	Models    []PublishedModel
	Selection Selection
	// ReportDefect relays a package-interface falsification observed during LOCAL
	// preparation (cl-078). Best-effort: the hub's sound authorization wants a
	// rental chain, which a local install does not hold, so only an
	// admin-credentialed daemon's report lands; everyone else still refuses the
	// install locally, which is the load-bearing half.
	ReportDefect func(code, detail string)
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

// PublishedWheel is one exact release wheel FACT (wire 30): identity only, never a
// local file. The bytes come from the indexes under the release's own hashes.
type PublishedWheel struct {
	Digest       string
	Distribution string
	Filename     string
	ImportRoots  []string
	Length       int64
	Tags         []string
	Version      string
}

type ExactDocument struct {
	Bytes  []byte `json:"canonical_bytes_base64"`
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type Selection struct {
	PackageInterface ExactDocument
}

func validatePublished(inst records.PackageInstall, published *PublishedSource) *exit.Error {
	packageInterface := published.Selection.PackageInterface
	digest, err := canonical.Raw(packageInterface.Digest)
	if err != nil || packageInterface.Length != int64(len(packageInterface.Bytes)) ||
		!bytes.Equal(canonical.Digest(packageInterface.Bytes), digest) {
		return exit.Named(exit.Conflict, "package_interface_identity_mismatch",
			"published package interface bytes do not match their digest and length")
	}
	if published.ProjectWheel.Distribution != inst.Package[strings.LastIndex(inst.Package, "/")+1:] ||
		published.ProjectWheel.Version != inst.Version ||
		published.ProjectWheel.Digest == "" || published.ProjectWheel.Filename == "" {
		return exit.Named(exit.Conflict, "package_wheel_release_mismatch",
			"published project wheel does not name %s@%s", inst.Package, inst.Version)
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
	Install records.PackageInstall
	// CapturedProjectWheel is the metadata-sealed executable of a captured App
	// wheel. Its original archive remains under this install's original/ directory.
	CapturedProjectWheel string
	Superseded           string
	Idempotent           bool
	Timings              []Timing
	Warnings             []string
	Files                int
	Bytes                int64
}

// Run executes the whole transaction. Every refusal before Activate leaves the
// records untouched, so the previously pinned install stays runnable.
func Run(l home.Layout, st *records.Store, req Request) (*Result, *exit.Error) {
	if req.Snapshot && req.Local == nil {
		return nil, exit.Internalf("an invocation snapshot requires local package source")
	}
	if (req.Published == nil) == (req.Local == nil) {
		return nil, exit.Usagef("`cozy package install` needs exactly one package source").
			WithRemedy("pass org/package for Tensorhub, or an explicit directory such as . or ./project").
			WithNext("cozy help package install")
	}
	id, e := newInstallID()
	if e != nil {
		return nil, e
	}
	installDir := l.InstallDir(id)
	res := &Result{}
	clock := time.Now()
	mark := func(stage string) {
		res.Timings = append(res.Timings, Timing{stage, time.Since(clock)})
		clock = time.Now()
	}
	fail := func(err *exit.Error) (*Result, *exit.Error) {
		_ = os.RemoveAll(installDir)
		return nil, err
	}

	inst := records.PackageInstall{ID: id, Dir: installDir}

	// ---- stage: bytes land under bounds; no code from the release has run ----
	var sourceDir string
	var sourceStats map[string]packagepublish.SourceStamp
	switch {
	case req.Published != nil:
		if req.Published.Package == "" || req.Published.Release == "" ||
			req.Published.ProjectWheel.Digest == "" {
			return fail(exit.Internalf("published package source is incomplete"))
		}
		sourceDir = installDir
		if err := os.MkdirAll(installDir, 0o700); err != nil {
			return fail(exit.Internalf("cannot create the package install directory: %s", err))
		}
		inst.SourceKind, inst.SourceRef = "tensorhub", req.Published.Package+"@"+req.Published.Release
		inst.Package, inst.Version, inst.ProjectDir = req.Published.Package, req.Published.Release,
			filepath.Join(installDir, "source")
		if e := validatePublished(inst, req.Published); e != nil {
			return fail(e)
		}
		res.Files, res.Bytes = req.Published.Files, req.Published.Bytes
	case req.Local != nil:
		local := req.Local
		if local.Package == "" || local.Release == "" || local.Tree == "" {
			return fail(exit.Internalf("local package build input is incomplete"))
		}
		abs, err := filepath.Abs(local.Tree)
		if err != nil {
			return fail(exit.Usagef("local package directory %q is not resolvable: %s", local.Tree, err))
		}
		sourceDir = abs
		observed, problem := packagepublish.PrepareLocalFrom(abs)
		if problem != nil {
			return fail(problem)
		}
		sourceStats, _, problem = observed.SourceStats()
		observed.Close()
		if problem != nil {
			return fail(problem)
		}
		inst.SourceKind, inst.SourceRef = "local", abs
		inst.Package, inst.Version = local.Package, local.Release
		res.Files, res.Bytes = local.Files, local.Bytes
		// Keep the authored tree as the watch target, while the installation and
		// its sibling dependencies share one owned root for imports and describe.
		sourceDir, e = snapshotSource(installDir, local)
		if e != nil {
			return fail(e)
		}
		inst.ProjectDir = sourceDir
		if req.Snapshot {
			inst.SourceRef = sourceDir
		}
	}
	mark("stage")

	// ---- verify: source identity settles BEFORE anything can execute ----
	if e := verifySource(&inst, &res.Warnings); e != nil {
		return fail(e)
	}
	if e := resolveTarget(&inst, req.Ref); e != nil {
		return fail(e)
	}
	mark("verify")

	// The pin decision is made before the expensive step, never after it.
	prior, priorInstall, e := st.ActivePackage(inst.Package)
	if e != nil {
		return fail(e)
	}
	if !req.Snapshot && prior != nil && priorInstall != nil && priorInstall.SourceKind == inst.SourceKind &&
		priorInstall.Package == inst.Package && priorInstall.Version == inst.Version &&
		inst.SourceKind == "tensorhub" {
		res.Idempotent = true
		res.Install = *priorInstall
		_ = os.RemoveAll(installDir)
		// The same source is the same self-callable surface. Capture is idempotent, so
		// this also gives an install made before self bindings existed its rows back.
		if e := CaptureSelfBindings(st, *priorInstall, nil); e != nil {
			return nil, e
		}
		return res, nil
	}
	if !req.Snapshot && prior != nil && !req.Force {
		_ = os.RemoveAll(installDir)
		return nil, exit.New(exit.Conflict,
			"%s is already installed and pinned to install %s", inst.Package+majorSuffix(inst.Major), short12(prior.InstallID)).
			WithRemedy("install never silently upgrades; rerun the same source command with --force to build and swap a new install").
			WithNext("cozy package list")
	}
	// A --force that fails must keep the working install: exit 13, nothing mutated.
	guard := func(err *exit.Error) (*Result, *exit.Error) {
		if prior == nil {
			return fail(err)
		}
		_ = os.RemoveAll(installDir)
		cause := err.ErrName() + " — " + err.Message
		if err.Remedy != "" {
			cause += "; try: " + err.Remedy
		}
		return nil, exit.New(exit.Conflict,
			"the replacement install for %s failed; the working install (%s) is untouched and still runnable",
			inst.Package+majorSuffix(inst.Major), short12(prior.InstallID)).
			WithRemedy("cause: %s", cause).
			WithNext("cozy package list")
	}

	if e := checkCapacity(l.Installs, res.Bytes); e != nil {
		return guard(e)
	}

	// ---- environment: the first code-executing step, on verified source only ----
	venvDir := filepath.Join(installDir, "venv")
	var env *EnvironmentReceipt
	var packageInterface *launch.PackageInterface
	var placement ExactDocument
	var err *exit.Error
	if req.Published != nil {
		packageInterface, placement, inst.Runtime, env, err = preparePublished(l, installDir, req.Published)
	} else {
		env, err = materializeEnvironment(sourceDir, venvDir, !req.Snapshot)
	}
	if err != nil {
		return guard(err)
	}
	inst.Python, inst.UV = env.Python, env.UV
	inst.Platform, inst.Extra = env.Platform, env.Extra
	inst.Packages, inst.Closure = env.Packages, env.Closure
	res.Warnings = append(res.Warnings, env.Warnings...)
	mark("environment")

	// ---- package interface: retain the published document or statically read the
	// editable source. Model placement is prepared only when a worker is requested.
	if req.Local != nil {
		packageInterface, e = readDevelopmentInterface(venvDir, sourceDir)
		if e != nil {
			return guard(e)
		}
	}
	packageInterfacePath := launch.PackageInterfacePath(installDir)
	if err := os.MkdirAll(filepath.Dir(packageInterfacePath), 0o700); err != nil {
		return guard(exit.Internalf("cannot create unpublished package interface root: %s", err))
	}
	if err := os.WriteFile(packageInterfacePath, packageInterface.Raw, 0o600); err != nil {
		return guard(exit.Internalf("cannot store unpublished package interface: %s", err))
	}
	if req.Local != nil {
		cache := filepath.Join(installDir, "artifact-cache")
		if err := os.MkdirAll(cache, 0o700); err != nil {
			return guard(exit.Internalf("cannot create editable placement cache: %s", err))
		}
		if placement.Digest != "" {
			if err := os.WriteFile(filepath.Join(cache, strings.TrimPrefix(placement.Digest, "sha256:")), placement.Bytes, 0o600); err != nil {
				return guard(exit.Internalf("cannot store editable PlacementSet: %s", err))
			}
		}
	}
	inst.PlacementSetDigest = placement.Digest
	mark("package_interface")
	if req.Local != nil {
		current, problem := packagepublish.PrepareLocalFrom(sourceDir)
		if problem != nil {
			return guard(exit.Named(exit.Conflict, "editable_source_changed",
				"the editable package changed while its replacement was being prepared").
				WithRemedy("source recheck failed: %s", problem.Message))
		}
		defer current.Close()
		if problem := packagepublish.RecordSourceStats(installDir, sourceStats); problem != nil {
			return guard(problem)
		}
		if "local/"+current.Name != req.Local.Package || current.Release != req.Local.Release {
			return guard(exit.Named(exit.Conflict, "editable_source_changed",
				"the editable package changed while its replacement was being prepared").
				WithRemedy("retry after the source tree is stable"))
		}
	}

	// Disk is measured once, here, and read back from the record forever after.
	inst.BytesExcl, inst.BytesShared = Disk(installDir)

	// ---- activate: the install row and the pin swap commit together ----
	var superseded string
	if req.Snapshot {
		e = st.RecordInstall(inst)
	} else {
		superseded, e = st.Activate(inst)
	}
	if e != nil {
		return guard(e)
	}
	// The install row is committed, so its directory stays: reporting the failure
	// without deleting a pinned environment is the safer half of an impossible path.
	if e := CaptureSelfBindings(st, inst, packageInterface); e != nil {
		return nil, e
	}
	res.Install, res.Superseded = inst, superseded
	mark("activate")
	return res, nil
}

// readDevelopmentInterface reads only the source interface. Model construction
// belongs to an admitted worker; a source install retains no serving PlacementSet.
func readDevelopmentInterface(venvDir, sourceDir string) (*launch.PackageInterface, *exit.Error) {
	env := config.Frozen().Tool("COZY_HOME=" + runtimeScratchHome())
	runtimeBin, problem := hostruntime.Path(env)
	if problem != nil {
		return nil, problem
	}
	cmd := exec.Command(runtimeBin, "--json", "--dir", sourceDir, "describe",
		"--environment-python", home.VenvPython(venvDir))
	cmd.Env = env
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, exit.Internalf("cannot run %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, hostruntime.RuntimeExit(code, "describe", "runtime_preparation_failed", stdout.String(), stderr.String())
	}
	decoded, problem := launch.DecodePackageInterface([]byte(stdout.String()))
	if problem != nil {
		return nil, problem
	}
	return decoded, nil
}

// verifySource settles source identity before any build backend or import can run.
func verifySource(inst *records.PackageInstall, warn *[]string) *exit.Error {
	if inst.SourceKind == "tensorhub" {
		inst.Verified = true
		return nil
	}
	if inst.SourceKind == "local" {
		inst.Verified = false
		*warn = append(*warn,
			"local directory install: source bytes are pinned locally but are not a published Tensorhub release")
		return nil
	}
	return exit.Internalf("unknown package source kind %q", inst.SourceKind)
}

// resolveTarget reconciles what the release says with what the caller asked for.
func resolveTarget(inst *records.PackageInstall, ref Ref) *exit.Error {
	if inst.SourceKind == "local" {
		major, e := MajorOf(inst.Version)
		if e != nil {
			return e
		}
		inst.Major = major
		return nil
	}
	if inst.Package == "" || inst.Version == "" {
		return exit.Named(exit.Validation, "release_undeclared", "the package source declares no package and version")
	}
	if inst.Package != ref.Package {
		return exit.Named(exit.Validation, "release_mismatch",
			"the package source names %q but %q was requested", inst.Package, ref.Package).
			WithRemedy("install the package named by its project metadata")
	}
	major, e := MajorOf(inst.Version)
	if e != nil {
		return e
	}
	if ref.HasMajor && ref.Major != major {
		return exit.Named(exit.Validation, "release_mismatch",
			"%s was requested but the archive publishes version %s (major %d)", ref.String(), inst.Version, major).
			WithRemedy("install the requested release; one package has only one active version")
	}
	inst.Major = major
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
		"not enough disk to build this install: needed %s, had %s, short by %s on %s",
		units.Bytes(need), units.Bytes(free), units.Bytes(need-free), dir).
		WithRemedy("free space, remove an installed package, or move COZY_HOME to a larger filesystem").
		WithNext("cozy package list")
}

// Remove drops one install: its pin, install row, and exclusive directory.
// Shared TensorFS bytes are never touched here.
func Remove(l home.Layout, st *records.Store, pkg string, major int) (int64, *exit.Error) {
	pin, inst, e := st.ActivePin(pkg, major)
	if e != nil || pin == nil {
		return 0, e
	}
	if e := st.Unpin(pkg, major); e != nil {
		return 0, e
	}
	if inst == nil {
		return 0, nil
	}
	return Reclaim(l, st, inst.ID)
}

// Reclaim removes an unpinned install immediately. The database claim wins
// before filesystem deletion, so a newly pinned install is never removed. Only
// after that claim may the read-only published tree be made removable.
func Reclaim(l home.Layout, st *records.Store, id string) (int64, *exit.Error) {
	inst, problem := st.Install(id)
	if problem != nil || inst == nil {
		return 0, problem
	}
	target, problem := recordedRemovalTarget(l, *inst)
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
	return inst.BytesExcl, nil
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
	inst, problem := st.Install(id)
	if problem != nil {
		return 0, false, problem
	}
	if inst != nil {
		freed, problem := Reclaim(l, st, id)
		if problem != nil {
			return 0, false, problem
		}
		// Reclaim answers bytes, not verdict, and an install may record zero exclusive
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
		return "", exit.Internalf("refusing to remove an install with unsafe id %q", id)
	}
	target := filepath.Join(root, id)
	if filepath.Dir(target) != root {
		return "", exit.Internalf("refusing to remove install %s outside %s", id, root)
	}
	return target, nil
}

func recordedRemovalTarget(l home.Layout, inst records.PackageInstall) (string, *exit.Error) {
	target, problem := installRemovalTarget(l, inst.ID)
	if problem != nil {
		return "", problem
	}
	recorded, err := filepath.Abs(inst.Dir)
	if err != nil || filepath.Clean(recorded) != target {
		return "", exit.Internalf("refusing to remove install %s outside %s", inst.ID, filepath.Dir(target))
	}
	return target, nil
}

// removeInstallTree deletes one resolved install directory. Only after the caller has
// settled that nothing references it may the read-only published tree be made removable.
func removeInstallTree(target string) *exit.Error {
	if err := makeInstallRemovable(target); err != nil {
		return exit.Internalf("cannot prepare the retired install directory %s for removal: %s", target, err)
	}
	if err := os.RemoveAll(target); err != nil {
		return exit.Internalf("cannot remove the install directory %s: %s", target, err)
	}
	return nil
}

func makeInstallRemovable(root string) error {
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

func newInstallID() (string, *exit.Error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", exit.Internalf("cannot mint an install id: %s", err)
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
