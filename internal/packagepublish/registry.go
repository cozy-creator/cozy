package packagepublish

// cl-078: the client no longer proxies PyPI. `uv export --locked` still authors
// the pylock and every row still passes the exact discipline the download had —
// the pinned index, the files.pythonhosted.org origin shape, the bounded
// sha256/size identity, the platform-target wheel selection (th-107), the
// platform-root refusal — but the bytes never move through this machine:
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
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

type registryLock struct {
	LockVersion string            `toml:"lock-version"`
	Packages    []registryPackage `toml:"packages"`
}

type registryPackage struct {
	Index   string          `toml:"index"`
	Name    string          `toml:"name"`
	Sdist   map[string]any  `toml:"sdist"`
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
	Name    string `json:"name"`
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size"`
	URL     string `json:"url"`
	Version string `json:"version"`
}

type exactDependency struct {
	digest  string
	version string
}

func collectRegistryRows(ctx context.Context, project, stage, organization string, existing []DependencyWheel) ([]RegistryRow, *exit.Error) {
	lockPath := filepath.Join(stage, "pylock.registry.toml")
	args := []string{"export", "--locked", "--no-dev", "--no-default-groups", "--no-emit-project", "--no-emit-local",
		"--format", "pylock.toml", "--output-file", lockPath, "--no-progress", "--directory", project}
	for _, name := range PrunedDistributions() {
		args = append(args, "--prune", name)
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
			WithRemedy("run `uv lock`, review uv.lock, and publish again")
	}
	raw, err := os.ReadFile(lockPath)
	if err != nil || len(raw) == 0 || int64(len(raw)) > maxLockBytes {
		return nil, exit.Named(exit.Structural, "registry_dependency_export_invalid",
			"uv export did not produce a non-empty pylock.toml at or below %d B", maxLockBytes)
	}
	return registryRowsFromLock(raw, existing, organization, true)
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

func RegistryRowsFromLock(raw []byte, existing []DependencyWheel, organization string) ([]RegistryRow, *exit.Error) {
	return registryRowsFromLock(raw, existing, organization, false)
}

// A locally exported lock may retain image-owned prefix families as direct
// dependencies of captured interface wheels even after `uv --prune torch`.
// Omit every canonical image family before selecting/counting its wheel. External
// declarations must already be pruned and still refuse such rows.
func registryRowsFromLock(raw []byte, existing []DependencyWheel, organization string, pruneBase bool) ([]RegistryRow, *exit.Error) {
	var lock registryLock
	if err := toml.Unmarshal(raw, &lock); err != nil || lock.LockVersion != "1.0" {
		return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid",
			"uv export produced an invalid PEP 751 pylock.toml")
	}
	seen := make(map[string]exactDependency, len(existing)+len(lock.Packages))
	var total int64
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
		total += identity.Length
	}
	out := []RegistryRow{}
	count := len(existing)
	for _, pkg := range lock.Packages {
		name := normalizedProjectName(pkg.Name)
		if name == "" || strings.TrimSpace(pkg.Version) == "" {
			return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid",
				"pylock.toml contains a package without an exact name and version")
		}
		if ImageOwnedDistribution(name) {
			if pruneBase {
				continue
			}
			return nil, exit.Named(exit.Validation, "registry_dependency_platform_root_present",
				"uv export retained platform-owned root %s", name)
		}
		orgRow := pkg.Index != "https://pypi.org/simple"
		if orgRow && (organization == "" || orgIndexNamespace(pkg.Index) != organization) {
			return nil, exit.Named(exit.Validation, "registry_dependency_index_refused",
				"%s==%s is locked to neither the public PyPI index nor this package's own org index",
				name, pkg.Version)
		}
		candidate, problem := selectRegistryWheel(name, pkg)
		if problem != nil {
			return nil, problem
		}
		// th-113: a same-org index row references a wheel already in the hub's
		// own custody. Publish declares the row and ships nothing; the hub
		// custody-shares the committed org-index claim into the release, and
		// install serves it from the plan exactly like any registry wheel.
		var digest string
		if orgRow {
			digest, problem = orgIndexWheelIdentity(name, pkg.Version, organization, candidate)
		} else {
			digest, problem = registryWheelIdentity(name, pkg.Version, candidate)
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
		if candidate.Size > MaxDependencyWheelBytes-total {
			return nil, exit.Named(exit.Validation, "dependency_wheels_too_large",
				"dependency wheels exceed %d B combined", MaxDependencyWheelBytes)
		}
		total += candidate.Size
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
func registryWheelIdentity(name, version string, selected registryWheel) (string, *exit.Error) {
	digest := selected.Hashes["sha256"]
	if selected.Size <= 0 || selected.Size > MaxDependencyWheelBytes || len(digest) != 64 {
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
	if pure, score := classifyWheelFilename(filename); !pure && score < 0 {
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
func orgIndexWheelIdentity(name, version, organization string, selected registryWheel) (string, *exit.Error) {
	digest := selected.Hashes["sha256"]
	if len(digest) != 64 || strings.ToLower(digest) != digest || selected.Size < 0 {
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
	if len(parts) != 6 || parts[0] != "v1" || parts[1] != "index" || parts[2] != organization ||
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
	if pure, score := classifyWheelFilename(filename); !pure && score < 0 {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one wheel installable on the platform target", name, version)
	}
	return digest, nil
}

// The fleet's ONE platform target (th-107): CPython 3.12 on linux x86_64,
// manylinux capped at the oldest fleet libc (glibc 2.36, python:3.12-slim-
// bookworm; the CUDA bases and dev machines carry 2.39). A pure py3-none-any
// wheel is universal and stays first choice; a native wheel is selected only
// when no pure wheel exists. Tensorhub re-runs the same admission on the
// declared rows and on the fetched wheel's exact WHEEL metadata.
const (
	targetPythonMinor = 12
	targetGlibcMinor  = 36
)

// selectRegistryWheel picks the lock row wheel to publish: the pure wheel when
// one exists (URL tie-break for determinism), otherwise the best admissible
// native wheel under standard PEP 425 preference — more specific python tag
// wins, newest manylinux wins.
func selectRegistryWheel(name string, pkg registryPackage) (registryWheel, *exit.Error) {
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
		isPure, score := classifyWheelFilename(base)
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
		"has no py3-none-any or cp312 manylinux x86_64 wheel"
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
// (cp312-cp312 > cp3N-abi3 > cp312-none > py312 > py3 > older py3N), the
// manylinux glibc floor breaks ties.
func classifyWheelFilename(filename string) (pure bool, score int) {
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
					pure = pure || abi == "none" && universalPythonTag(python)
					continue
				}
				glibc, ok := manylinuxGlibcMinor(platform)
				if !ok {
					continue
				}
				if rank, ok := pythonABIRank(python, abi); ok && rank*1000+glibc > score {
					score = rank*1000 + glibc
				}
			}
		}
	}
	return pure, score
}

func pythonABIRank(python, abi string) (int, bool) {
	switch abi {
	case "cp312":
		if python == "cp312" {
			return 400, true
		}
	case "abi3":
		if minor, ok := pythonTagMinor(python, "cp"); ok && minor <= targetPythonMinor {
			return 300 + minor, true
		}
	case "none":
		switch {
		case python == "cp312":
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
// carry for CPython 3.12: py3, py3N at or below the target minor, or cp312.
func universalPythonTag(python string) bool {
	if python == "py3" || python == "cp312" {
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
