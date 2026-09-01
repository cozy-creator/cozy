package packagepublish

// cl-078: the client no longer proxies PyPI. `uv export --locked` still authors
// the pylock and every row still passes the exact discipline the download had —
// the pinned index, the files.pythonhosted.org origin shape, the bounded
// sha256/size identity, the pure-wheel selection, the platform-root refusal —
// but the bytes never move through this machine: publish sends the rows and
// Tensorhub fetches, verifies with its own hash, and stores content-addressed.

import (
	"context"
	"encoding/hex"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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

func collectRegistryRows(ctx context.Context, project, stage string, existing []DependencyWheel) ([]RegistryRow, *exit.Error) {
	lockPath := filepath.Join(stage, "pylock.registry.toml")
	args := []string{"export", "--locked", "--no-dev", "--no-emit-project", "--no-emit-local",
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
	return RegistryRowsFromLock(raw, existing)
}

func RegistryRowsFromLock(raw []byte, existing []DependencyWheel) ([]RegistryRow, *exit.Error) {
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
		if remoteBaseRoots[name] {
			return nil, exit.Named(exit.Validation, "registry_dependency_platform_root_present",
				"uv export retained platform-owned root %s", name)
		}
		if pkg.Index != "https://pypi.org/simple" {
			return nil, exit.Named(exit.Validation, "registry_dependency_index_refused",
				"%s==%s is not locked to the public PyPI index", name, pkg.Version)
		}
		candidate, problem := pureRegistryWheel(name, pkg)
		if problem != nil {
			return nil, problem
		}
		digest, problem := registryWheelIdentity(name, pkg.Version, candidate)
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
	if err != nil || filepath.Base(filename) != filename || !strings.HasSuffix(strings.ToLower(filename), "-py3-none-any.whl") {
		return "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one pure py3-none-any wheel", name, version)
	}
	return digest, nil
}

func pureRegistryWheel(name string, pkg registryPackage) (registryWheel, *exit.Error) {
	candidates := make([]registryWheel, 0, 1)
	for _, candidate := range pkg.Wheels {
		parsed, err := url.Parse(candidate.URL)
		if err != nil {
			continue
		}
		base, err := url.PathUnescape(filepath.Base(parsed.Path))
		if err != nil {
			continue
		}
		if strings.HasSuffix(strings.ToLower(base), "-py3-none-any.whl") {
			candidates = append(candidates, candidate)
		}
	}
	if len(candidates) == 0 {
		refusal, detail := "registry_dependency_native_only", "has no pure py3-none-any wheel"
		if len(pkg.Wheels) == 0 && pkg.Sdist != nil {
			refusal, detail = "registry_dependency_source_only", "is available only as source"
		}
		return registryWheel{}, exit.Named(exit.Validation, refusal,
			"%s==%s %s", name, pkg.Version, detail).
			WithRemedy("use a pure-Python PyPI dependency or publish an explicit local custom wheel")
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].URL < candidates[j].URL })
	return candidates[len(candidates)-1], nil
}
