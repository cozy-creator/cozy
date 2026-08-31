package packagepublish

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

// A rental cannot replace these image-owned platform families. Ordinary libraries are
// package-owned even when one image happens to carry a copy; Runtime compares this small
// declared family against the actual selected base before any remote installation.
var remoteBaseRoots = map[string]bool{
	"cozy-runtime": true, //cozy:allow base distribution identity, not executable access
	"msgspec":      true,
	"tensorfs":     true,
	"torch":        true,
	"torchaudio":   true,
	"torchvision":  true,
	"triton":       true,
}

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

type exactDependency struct {
	digest  string
	version string
}

func collectRegistryDependencies(ctx context.Context, project, stage string, existing []DependencyWheel) ([]DependencyWheel, *exit.Error) {
	lockPath := filepath.Join(stage, "pylock.registry.toml")
	args := []string{"export", "--locked", "--no-dev", "--no-emit-project", "--no-emit-local",
		"--format", "pylock.toml", "--output-file", lockPath, "--no-progress", "--directory", project}
	roots := make([]string, 0, len(remoteBaseRoots))
	for name := range remoteBaseRoots {
		roots = append(roots, name)
	}
	sort.Strings(roots)
	for _, name := range roots {
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
	return registryDependenciesFromLock(ctx, raw, filepath.Join(stage, "dependencies", "registry"), existing, registryClient())
}

func registryDependenciesFromLock(ctx context.Context, raw []byte, stage string, existing []DependencyWheel, client *http.Client) ([]DependencyWheel, *exit.Error) {
	var lock registryLock
	if err := toml.Unmarshal(raw, &lock); err != nil || lock.LockVersion != "1.0" {
		return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid",
			"uv export produced an invalid PEP 751 pylock.toml")
	}
	out := append([]DependencyWheel(nil), existing...)
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
		path, digest, problem := downloadRegistryWheel(ctx, client, stage, name, pkg.Version, candidate)
		if problem != nil {
			return nil, problem
		}
		identity, problem := wheel.InspectIdentity(path)
		if problem != nil {
			return nil, problem
		}
		if identity.Distribution != name || identity.Version != pkg.Version {
			return nil, exit.Named(exit.Validation, "registry_dependency_identity_mismatch",
				"lock declares %s==%s but %s contains %s==%s",
				name, pkg.Version, identity.Filename, identity.Distribution, identity.Version)
		}
		if prior, ok := seen[name]; ok {
			if prior.version == identity.Version && prior.digest == digest {
				_ = os.Remove(path)
				continue
			}
			return nil, exit.Named(exit.Validation, "dependency_wheel_duplicate",
				"normalized dependency %s resolves to more than one exact wheel", name)
		}
		if len(out) >= MaxDependencyWheels {
			return nil, tooManyDependencies()
		}
		if identity.Length > MaxDependencyWheelBytes-total {
			return nil, exit.Named(exit.Validation, "dependency_wheels_too_large",
				"dependency wheels exceed %d B combined", MaxDependencyWheelBytes)
		}
		total += identity.Length
		seen[name] = exactDependency{digest: digest, version: identity.Version}
		out = append(out, DependencyWheel{Filename: identity.Filename, Path: path})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Filename < out[j].Filename })
	return out, nil
}

func pureRegistryWheel(name string, pkg registryPackage) (registryWheel, *exit.Error) {
	candidates := make([]registryWheel, 0, 1)
	for _, candidate := range pkg.Wheels {
		parsed, err := url.Parse(candidate.URL)
		if err != nil {
			continue
		}
		filename, err := url.PathUnescape(filepath.Base(parsed.Path))
		if err == nil && strings.HasSuffix(strings.ToLower(filename), "-py3-none-any.whl") {
			candidates = append(candidates, candidate)
		}
	}
	if len(candidates) == 0 {
		code, message := "registry_dependency_native_only", "has no pure py3-none-any wheel"
		if len(pkg.Wheels) == 0 && pkg.Sdist != nil {
			code, message = "registry_dependency_source_only", "is available only as source"
		}
		return registryWheel{}, exit.Named(exit.Validation, code, "%s==%s %s", name, pkg.Version, message).
			WithRemedy("use a pure-Python PyPI dependency or publish an explicit local custom wheel")
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].URL < candidates[j].URL })
	return candidates[len(candidates)-1], nil
}

func downloadRegistryWheel(ctx context.Context, client *http.Client, stage, name, version string, selected registryWheel) (string, string, *exit.Error) {
	digest := selected.Hashes["sha256"]
	if selected.Size <= 0 || selected.Size > MaxDependencyWheelBytes || len(digest) != 64 {
		return "", "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has no bounded SHA-256 wheel identity", name, version)
	}
	if _, err := hex.DecodeString(digest); err != nil || strings.ToLower(digest) != digest {
		return "", "", exit.Named(exit.Validation, "registry_dependency_identity_invalid",
			"%s==%s has an invalid SHA-256", name, version)
	}
	parsed, err := url.Parse(selected.URL)
	if err != nil || parsed.Scheme != "https" || parsed.Hostname() != "files.pythonhosted.org" ||
		parsed.Port() != "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", "", exit.Named(exit.Validation, "registry_dependency_origin_refused",
			"%s==%s wheel is not an exact files.pythonhosted.org HTTPS object", name, version)
	}
	filename, err := url.PathUnescape(filepath.Base(parsed.Path))
	if err != nil || filepath.Base(filename) != filename || !strings.HasSuffix(strings.ToLower(filename), "-py3-none-any.whl") {
		return "", "", exit.Named(exit.Validation, "registry_dependency_wheel_invalid",
			"%s==%s does not name one pure py3-none-any wheel", name, version)
	}
	if err := os.MkdirAll(stage, 0o700); err != nil {
		return "", "", exit.Internalf("cannot create registry dependency staging: %s", err)
	}
	partial := filepath.Join(stage, filename+".partial")
	path := filepath.Join(stage, filename)
	_ = os.Remove(partial)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, selected.URL, nil)
	if err != nil {
		return "", "", exit.Internalf("cannot request registry dependency: %s", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return "", "", exit.Named(exit.Structural, "registry_dependency_download_failed",
			"cannot download %s==%s: %s", name, version, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || (response.ContentLength >= 0 && response.ContentLength != selected.Size) {
		return "", "", exit.Named(exit.Structural, "registry_dependency_download_failed",
			"%s==%s returned HTTP %d with length %d", name, version, response.StatusCode, response.ContentLength)
	}
	file, err := os.OpenFile(partial, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", "", exit.Internalf("cannot stage registry dependency: %s", err)
	}
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, selected.Size+1))
	closeErr := file.Close()
	actual := hex.EncodeToString(hash.Sum(nil))
	if copyErr != nil || closeErr != nil || written != selected.Size || actual != digest {
		_ = os.Remove(partial)
		return "", "", exit.Named(exit.Validation, "registry_dependency_identity_mismatch",
			"downloaded %s==%s does not match its locked size and SHA-256", name, version)
	}
	if err := os.Rename(partial, path); err != nil {
		_ = os.Remove(partial)
		return "", "", exit.Internalf("cannot commit registry dependency staging: %s", err)
	}
	return path, "sha256:" + digest, nil
}

func dependencyDigest(path string) (string, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return "", exit.Named(exit.Structural, "dependency_wheel_unreadable", "%s: %v", path, err)
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		file.Close()
		return "", exit.Named(exit.Structural, "dependency_wheel_unreadable", "%s: %v", path, err)
	}
	if err := file.Close(); err != nil {
		return "", exit.Named(exit.Structural, "dependency_wheel_unreadable", "%s: %v", path, err)
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil)), nil
}

func registryClient() *http.Client {
	return &http.Client{
		Timeout: 2 * time.Minute,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return fmt.Errorf("registry wheel redirect refused")
		},
	}
}
