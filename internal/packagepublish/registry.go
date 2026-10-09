package packagepublish

// cl-078: the client no longer proxies PyPI. `uv export --locked` still authors
// the pylock and every row still passes the exact discipline the download had —
// the pinned index, the files.pythonhosted.org origin shape, the bounded
// sha256/size identity, the platform-target wheel selection (th-107), the
// full selected dependency closure — but the bytes never move through this machine:
// publish sends the rows and Tensorhub fetches, verifies with its own hash,
// and stores content-addressed.

import (
	"context"
	"encoding/hex"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

type registryLock struct {
	LockVersion string            `toml:"lock-version"`
	Packages    []registryPackage `toml:"packages"`
}

type registryPackage struct {
	Marker  string          `toml:"marker"`
	Index   string          `toml:"index"`
	Name    string          `toml:"name"`
	Sdist   map[string]any  `toml:"sdist"`
	VCS     map[string]any  `toml:"vcs"`
	Version string          `toml:"version"`
	Wheels  []registryWheel `toml:"wheels"`
}

type registryWheel struct {
	Hashes map[string]string `toml:"hashes"`
	Size   int64             `toml:"size"`
	URL    string            `toml:"url"`
}

// RegistryRow is one locked registry dependency for Tensorhub to fetch itself:
// exactly the pylock facts, nothing derived.
type RegistryRow struct {
	// Captured Hub bytes travel with private revisions, never as worker-local URLs.
	captureLocally bool
	packageRef     string // original Hub index identity for private captured wheels
	Name           string `json:"name"`
	SHA256         string `json:"sha256"`
	Size           int64  `json:"size"`
	URL            string `json:"url"`
	Version        string `json:"version"`
}

type exactDependency struct {
	digest  string
	version string
}

// collectRegistryRows exports the locked registry closure of a project whose account index,
// if any, is already bound to the publishing account.
func collectRegistryRows(ctx context.Context, project, stage, account string, existing []DependencyWheel, selected hostruntime.PythonInterpreter) ([]RegistryRow, *exit.Error) {
	raw, problem := exportLockedRegistry(ctx, project, stage, selected, nil)
	if problem != nil {
		return nil, problem
	}
	return RegistryRowsFromLock(raw, existing, account, selected.Version)
}

func exportLockedRegistry(ctx context.Context, project, stage string, selected hostruntime.PythonInterpreter, extras []string) ([]byte, *exit.Error) {
	lockPath := filepath.Join(stage, "pylock.registry.toml")
	args := []string{"export", "--locked", "--no-dev", "--no-default-groups", "--no-emit-project", "--no-emit-local",
		"--format", "pylock.toml", "--output-file", lockPath, "--no-progress", "--directory", project,
		"--python", selected.Executable, "--no-python-downloads"}
	for _, extra := range extras {
		args = append(args, "--extra", extra)
	}
	command := exec.CommandContext(ctx, "uv", args...)
	command.Env = config.Frozen().Tool("UV_PYTHON_DOWNLOADS=never")
	output, err := command.CombinedOutput()
	if err != nil {
		detail := strings.Join(strings.Fields(string(output)), " ")
		if len(detail) > 4096 {
			detail = detail[:4096]
		}
		name := "registry_dependency_export_failed"
		lower := strings.ToLower(detail)
		if strings.Contains(lower, "needs to be updated") && strings.Contains(lower, "lock") {
			name = "registry_dependency_lock_drift"
		}
		return nil, exit.Named(exit.Validation, name, "uv export --locked refused: %s", detail).
			WithRemedy("run `cozy package lock`, review uv.lock, and publish again")
	}
	raw, err := os.ReadFile(lockPath)
	if err != nil || len(raw) == 0 || int64(len(raw)) > maxLockBytes {
		return nil, exit.Named(exit.Structural, "registry_dependency_export_invalid",
			"uv export did not produce a non-empty pylock.toml at or below %d B", maxLockBytes)
	}
	return raw, nil
}

// orgIndexNamespace answers the org namespace when raw is one hub org index —
// http(s), path exactly /v1/index/<org>/simple/ (th-113) — and "" otherwise.
func orgIndexNamespace(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return ""
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(parts) != 4 || parts[0] != "v1" || parts[1] != "index" || parts[3] != "simple" ||
		parts[2] == "" {
		return ""
	}
	return parts[2]
}

// RegistryRowsFromLock retains every selected dependency, including framework
// and accelerator distributions. Tensorhub fetches public references directly. Rows
// from a Tensorhub index must come from the publishing account's own index.
func RegistryRowsFromLock(raw []byte, existing []DependencyWheel, account string, targetPython ...string) ([]RegistryRow, *exit.Error) {
	var lock registryLock
	// PEP 751 readers accept any lock-version of the major they implement.
	if err := toml.Unmarshal(raw, &lock); err != nil || lock.LockVersion != "1" && !strings.HasPrefix(lock.LockVersion, "1.") {
		return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid",
			"uv export produced an invalid PEP 751 pylock.toml")
	}
	var selected []bool
	markers := make([]string, len(lock.Packages))
	marked := false
	for i, pkg := range lock.Packages {
		markers[i] = pkg.Marker
		marked = marked || pkg.Marker != ""
	}
	if marked {
		selection, problem := readActiveRequirements(context.Background(), map[string]any{"markers": markers, "python": selectedPythonVersion(targetPython)})
		if problem != nil {
			return nil, problem
		}
		if len(selection.Markers) != len(lock.Packages) {
			return nil, exit.New(exit.Validation, "registry marker selection is incomplete")
		}
		selected = selection.Markers
	}
	seen := make(map[string]exactDependency, len(existing)+len(lock.Packages))
	for _, dependency := range existing {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		if problem != nil {
			return nil, problem
		}
		digest, problem := dependencyDigest(dependency.Path)
		if problem != nil {
			return nil, problem
		}
		seen[identity.Distribution] = exactDependency{digest: digest, version: identity.Version}
	}
	out := []RegistryRow{}
	count := len(existing)
	for i, pkg := range lock.Packages {
		if selected != nil && !selected[i] {
			continue
		}
		name := normalizedProjectName(pkg.Name)
		if pkg.VCS != nil {
			// A pinned git source rides as the wheel built at its commit, never as a row.
			if _, carried := seen[name]; carried {
				continue
			}
			return nil, exit.Named(exit.Validation, "registry_dependency_git_undeclared",
				"%s is locked to a git source this package does not declare", name).
				WithRemedy("add `%s @ git+https://<host>/<repository>@<40-hex commit>` to [project].dependencies", name)
		}
		if name == "" || strings.TrimSpace(pkg.Version) == "" {
			return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid",
				"pylock.toml contains a package without an exact name and version")
		}
		pytorchRow := strings.HasPrefix(pkg.Index, "https://download.pytorch.org/whl/")
		orgRow := pkg.Index != "https://pypi.org/simple" && !pytorchRow
		if orgRow && (account == "" || orgIndexNamespace(pkg.Index) != account) {
			return nil, exit.Named(exit.Validation, "registry_dependency_index_refused",
				"%s==%s is locked to neither a supported public index nor the publishing account's index",
				name, pkg.Version).
				WithRemedy("resolve your own packages with `%s = { index = %q }` in [tool.uv.sources]; other indexes cannot publish",
					name, AccountIndexName)
		}
		candidate, problem := selectRegistryWheel(name, pkg, targetPython...)
		if problem != nil {
			return nil, problem
		}
		// th-113: a same-org index row references a wheel already in the hub's
		// own custody. Publish declares the row and ships nothing; the hub
		// custody-shares the committed org-index claim into the release, and
		// install serves it from the plan exactly like any registry wheel.
		var digest string
		if pytorchRow {
			candidate, problem = pytorchRegistryWheel(name, pkg.Version, pkg.Index, pkg.Wheels, targetPython...)
			digest = candidate.Hashes["sha256"]
		} else if orgRow {
			digest, problem = orgIndexWheelIdentity(name, pkg.Version, account, candidate, targetPython...)
		} else {
			digest, problem = registryWheelIdentity(name, pkg.Version, candidate, targetPython...)
		}
		if problem != nil {
			return nil, problem
		}
		if prior, ok := seen[name]; ok {
			// The same exact bytes may ride as an already-built local wheel; a
			// DIFFERENT resolution of one normalized name is still a refusal.
			if prior.version == pkg.Version && prior.digest == "sha256:"+digest {
				continue
			}
			return nil, exit.Named(exit.Validation, "dependency_wheel_duplicate",
				"normalized dependency %s resolves to more than one exact wheel", name)
		}
		if count >= MaxDependencyWheels {
			return nil, tooManyDependencies()
		}
		count++
		seen[name] = exactDependency{digest: "sha256:" + digest, version: pkg.Version}
		out = append(out, RegistryRow{Name: name, SHA256: digest, Size: candidate.Size,
			URL: candidate.URL, Version: pkg.Version})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// registryWheelIdentity is the whole per-row discipline, minus the transfer:
// bounded sha256/size identity and the exact files.pythonhosted.org origin
// shape. Tensorhub re-runs the same checks and then hashes what it fetched.
func registryWheelIdentity(name, version string, selected registryWheel, targetPython ...string) (string, *exit.Error) {
	return registryWheelIdentityBound(name, version, selected, MaxRegistryWheelBytes, targetPython...)
}

func registryWheelIdentityBound(name, version string, selected registryWheel, maxBytes int64, targetPython ...string) (string, *exit.Error) {
	digest := selected.Hashes["sha256"]
	if selected.Size <= 0 || selected.Size > maxBytes || len(digest) != 64 {
		return "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has no bounded SHA-256 wheel identity", name, version)
	}
	if _, err := hex.DecodeString(digest); err != nil || strings.ToLower(digest) != digest {
		return "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has an invalid SHA-256", name, version)
	}
	parsed, err := url.Parse(selected.URL)
	if err != nil || parsed.Scheme != "https" || parsed.Hostname() != "files.pythonhosted.org" ||
		parsed.Port() != "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", exit.Named(exit.Validation, "registry_dependency_origin_refused",
			"%s==%s wheel is not an exact files.pythonhosted.org HTTPS object", name, version)
	}
	filename, err := url.PathUnescape(filepath.Base(parsed.Path))
	if err != nil || filepath.Base(filename) != filename {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one safe wheel basename", name, version)
	}
	if pure, score := classifyWheelFilename(filename, targetPython...); !pure && score < 0 {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one wheel installable on the platform target", name, version)
	}
	return digest, nil
}

// orgIndexWheelIdentity is the identity discipline for a row locked to the
// publisher's own org index (th-113): the hub's stable file URL shape
// /v1/index/<org>/files/<sha256hex>/<filename>, whose path digest IS the
// wheel's sha256. The index page advertises no size, so the lock's 0 is legal
// here and the hub's committed claim supplies the length at declare.
func orgIndexWheelIdentity(name, version, account string, selected registryWheel, targetPython ...string) (string, *exit.Error) {
	digest := selected.Hashes["sha256"]
	if len(digest) != 64 || strings.ToLower(digest) != digest || selected.Size < 0 || selected.Size > MaxRegistryWheelBytes {
		return "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has no exact SHA-256 wheel identity", name, version)
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has an invalid SHA-256", name, version)
	}
	parsed, err := url.Parse(selected.URL)
	if err != nil || parsed.Scheme != "http" && parsed.Scheme != "https" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", exit.Named(exit.Validation, "registry_dependency_origin_refused",
			"%s==%s wheel is not an exact org index file object", name, version)
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(parts) != 6 || parts[0] != "v1" || parts[1] != "index" || parts[2] != account ||
		parts[3] != "files" || parts[4] != digest {
		return "", exit.Named(exit.Validation, "registry_dependency_origin_refused",
			"%s==%s wheel is not this package's own org index file for its locked sha256",
			name, version)
	}
	filename, err := url.PathUnescape(parts[5])
	if err != nil || filepath.Base(filename) != filename {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one safe wheel basename", name, version)
	}
	if pure, score := classifyWheelFilename(filename, targetPython...); !pure && score < 0 {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one wheel installable on the platform target", name, version)
	}
	return digest, nil
}

// Artifact selection uses the captured CPython minor on Linux x86_64 and
// caps manylinux at the oldest fleet libc, glibc 2.36. Pure wheels remain the
// first choice; native wheels must match the selected executor's ABI.
const targetGlibcMinor = 36

// selectRegistryWheel picks the lock row wheel to publish: the pure wheel when
// one exists (URL tie-break for determinism), otherwise the best admissible
// native wheel under standard PEP 425 preference — more specific python tag
// wins, newest manylinux wins.
func selectRegistryWheel(name string, pkg registryPackage, targetPython ...string) (registryWheel, *exit.Error) {
	var pure []registryWheel
	best, bestScore := registryWheel{}, -1
	for _, candidate := range pkg.Wheels {
		parsed, err := url.Parse(candidate.URL)
		if err != nil {
			continue
		}
		base, err := url.PathUnescape(filepath.Base(parsed.Path))
		if err != nil {
			continue
		}
		isPure, score := classifyWheelFilename(base, targetPython...)
		if isPure {
			pure = append(pure, candidate)
		}
		if score > bestScore || score == bestScore && candidate.URL > best.URL {
			best, bestScore = candidate, score
		}
	}
	if len(pure) > 0 {
		sort.Slice(pure, func(i, j int) bool { return pure[i].URL < pure[j].URL })
		return pure[len(pure)-1], nil
	}
	if bestScore >= 0 {
		return best, nil
	}
	refusal, detail := "registry_dependency_platform_mismatch",
		"has no manylinux x86_64 wheel for Python "+selectedPythonVersion(targetPython)
	if len(pkg.Wheels) == 0 && pkg.Sdist != nil {
		refusal, detail = "registry_dependency_source_only", "is available only as source"
	}
	return registryWheel{}, exit.Named(exit.Validation, refusal,
		"%s==%s %s", name, pkg.Version, detail).
		WithRemedy("use a PyPI dependency that ships a wheel for the platform target or publish an explicit local custom wheel")
}

// classifyWheelFilename reads a PEP 427 wheel filename's compressed tag sets.
// pure reports a universal py3-none-any wheel; score is the best admissible
// native triple's rank (-1 when none): python/abi specificity dominates
// (exact CPython ABI, stable ABI, interpreter-specific then generic tags), the
// manylinux glibc floor breaks ties.
func classifyWheelFilename(filename string, targetPython ...string) (pure bool, score int) {
	targetMinor := selectedPythonMinor(targetPython)
	score = -1
	stem, found := strings.CutSuffix(strings.ToLower(filename), ".whl")
	parts := strings.Split(stem, "-")
	if !found || len(parts) < 5 {
		return false, -1
	}
	for _, python := range strings.Split(parts[len(parts)-3], ".") {
		for _, abi := range strings.Split(parts[len(parts)-2], ".") {
			for _, platform := range strings.Split(parts[len(parts)-1], ".") {
				if platform == "any" {
					pure = pure || abi == "none" && universalPythonTag(python, targetMinor)
					continue
				}
				glibc, ok := manylinuxGlibcMinor(platform)
				if !ok {
					continue
				}
				if rank, ok := pythonABIRank(python, abi, targetMinor); ok && rank*1000+glibc > score {
					score = rank*1000 + glibc
				}
			}
		}
	}
	return pure, score
}

func pythonABIRank(python, abi string, targetPythonMinor int) (int, bool) {
	targetABI := "cp3" + strconv.Itoa(targetPythonMinor)
	switch abi {
	case targetABI:
		if python == targetABI {
			return 400, true
		}
	case "abi3":
		if minor, ok := pythonTagMinor(python, "cp"); ok && minor <= targetPythonMinor {
			return 300 + minor, true
		}
	case "none":
		switch {
		case python == targetABI:
			return 200, true
		case python == "py3":
			return 112, true
		default:
			if minor, ok := pythonTagMinor(python, "py"); ok && minor <= targetPythonMinor {
				if minor == targetPythonMinor {
					return 113, true
				}
				return 100 + minor, true
			}
		}
	}
	return 0, false
}

// universalPythonTag recognizes the python tags a pure any-platform wheel may
// carry for the selected interpreter: py3, older py3N, or the exact CPython tag.
func universalPythonTag(python string, targetPythonMinor int) bool {
	targetABI := "cp3" + strconv.Itoa(targetPythonMinor)
	if python == "py3" || python == targetABI {
		return true
	}
	minor, ok := pythonTagMinor(python, "py")
	return ok && minor <= targetPythonMinor
}

// pythonTagMinor reads the 3.N minor out of a prefix3N python tag ("cp38" -> 8).
func pythonTagMinor(python, prefix string) (int, bool) {
	digits, found := strings.CutPrefix(python, prefix+"3")
	if !found || digits == "" {
		return 0, false
	}
	minor, err := strconv.Atoi(digits)
	if err != nil || minor < 0 {
		return 0, false
	}
	return minor, true
}

// manylinuxGlibcMinor reads the glibc 2.N floor a manylinux x86_64 platform
// tag demands, refusing tags above the fleet cap. The legacy aliases spell
// exact glibc floors: manylinux1 is 2.5, manylinux2010 is 2.12, manylinux2014
// is 2.17 (PEP 600). musllinux, plain linux, and foreign arches never match.
func manylinuxGlibcMinor(platform string) (int, bool) {
	switch platform {
	case "manylinux1_x86_64":
		return 5, true
	case "manylinux2010_x86_64":
		return 12, true
	case "manylinux2014_x86_64":
		return 17, true
	}
	body, found := strings.CutPrefix(platform, "manylinux_2_")
	if !found {
		return 0, false
	}
	digits, found := strings.CutSuffix(body, "_x86_64")
	if !found {
		return 0, false
	}
	minor, err := strconv.Atoi(digits)
	if err != nil || minor < 0 || minor > targetGlibcMinor {
		return 0, false
	}
	return minor, true
}

// Legacy callers default to 3.12; production capture always supplies its observed
// interpreter. This is an artifact target, not the Runtime supported policy.
func selectedPythonVersion(values []string) string {
	if len(values) > 0 && values[0] != "" {
		return values[0]
	}
	return "3.12.0"
}
func selectedPythonMinor(values []string) int {
	parts := strings.Split(selectedPythonVersion(values), ".")
	if len(parts) < 2 || parts[0] != "3" {
		return -1
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return -1
	}
	return minor
}
