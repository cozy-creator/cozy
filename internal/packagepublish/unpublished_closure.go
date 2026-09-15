package packagepublish

// Unpublished package revisions carry the selected installed closure, including extras. Public
// publication retains its existing registry-custody path.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

type capturedLock struct {
	Version  int `toml:"version"`
	Packages []struct {
		Name    string `toml:"name"`
		Version string `toml:"version"`
		Source  struct {
			Registry string `toml:"registry"`
		} `toml:"source"`
		Wheels []struct {
			URL  string `toml:"url"`
			Hash string `toml:"hash"`
			Size int64  `toml:"size"`
		} `toml:"wheels"`
	} `toml:"package"`
}

func capturedPins(closure string) (map[string]string, *exit.Error) {
	pins := map[string]string{}
	for _, row := range strings.Split(strings.TrimSpace(closure), "\n") {
		name, version, ok := strings.Cut(row, "==")
		if !ok || name == "" || version == "" || strings.ContainsAny(version, " \t\r\n;<>!=") || normalizedProjectName(name) != name || pins[name] != "" {
			return nil, exit.Named(exit.Validation, "private_dependency_closure_invalid", "captured environment requires a unique exact installed name/version roster")
		}
		pins[name] = version
	}
	if len(pins) > MaxDependencyWheels {
		return nil, tooManyDependencies()
	}
	return pins, nil
}

// CapturedRegistryRows selects only the installed closure from the captured uv.lock.
// It does not guess selected extras from a second resolution or export all extras.
func CapturedRegistryRows(raw []byte, closure, project, version string, existing []DependencyWheel) ([]RegistryRow, []string, *exit.Error) {
	pins, problem := capturedPins(closure)
	if problem != nil {
		return nil, nil, problem
	}
	if pins[project] != version {
		return nil, nil, exit.Named(exit.Conflict, "private_dependency_project_changed", "captured environment does not contain this exact project")
	}
	expected := map[string]string{}
	for name, version := range pins {
		if name != project {
			expected[name] = version
		}
	}
	for _, supplied := range existing {
		fact, problem := wheel.InspectIdentity(supplied.Path)
		if problem != nil {
			return nil, nil, problem
		}
		if expected[fact.Distribution] != fact.Version {
			return nil, nil, exit.Named(exit.Conflict, "private_dependency_local_changed", "local dependency differs from its installed version: %s", fact.Distribution)
		}
		delete(expected, fact.Distribution)
	}
	var lock capturedLock
	if len(raw) == 0 || int64(len(raw)) > maxLockBytes || toml.Unmarshal(raw, &lock) != nil || lock.Version != 1 {
		return nil, nil, exit.Named(exit.Validation, "private_dependency_lock_invalid", "unpublished package revision requires its captured uv.lock")
	}
	var rows []RegistryRow
	matched := map[string]bool{}
	for _, entry := range lock.Packages {
		name := normalizedProjectName(entry.Name)
		if pins[name] != entry.Version {
			continue
		}
		if matched[name] {
			return nil, nil, exit.Named(exit.Conflict, "private_dependency_lock_ambiguous", "captured uv.lock has multiple sources for installed dependency %s", name)
		}
		matched[name] = true
		if expected[name] != entry.Version {
			continue
		}
		row := registryPackage{Name: name, Version: entry.Version, Index: entry.Source.Registry}
		for _, candidate := range entry.Wheels {
			hash, ok := strings.CutPrefix(candidate.Hash, "sha256:")
			if !ok {
				continue
			}
			row.Wheels = append(row.Wheels, registryWheel{URL: candidate.URL, Size: candidate.Size, Hashes: map[string]string{"sha256": hash}})
		}
		candidate, problem := selectRegistryWheel(name, row)
		if problem != nil {
			return nil, nil, problem
		}
		if entry.Source.Registry == "https://pypi.org/simple" {
			// Registry artifacts are fetched by Runtime's bounded storage; the
			// private client upload byte bound applies only to local wheels.
			if _, problem := registryWheelIdentityBound(name, entry.Version, candidate, 2<<30); problem != nil {
				return nil, nil, problem
			}
		} else if _, problem := pytorchBaseRequirement(name, entry.Version, entry.Source.Registry, row.Wheels); problem != nil {
			return nil, nil, problem
		}
		rows = append(rows, RegistryRow{Name: name, Version: entry.Version, URL: candidate.URL, SHA256: candidate.Hashes["sha256"], Size: candidate.Size})
		delete(expected, name)
	}
	for name := range pins {
		if !matched[name] {
			return nil, nil, exit.Named(exit.Conflict, "private_dependency_lock_drift", "captured uv.lock does not match installed dependency %s", name)
		}
	}
	if len(expected) != 0 {
		names := make([]string, 0, len(expected))
		for name := range expected {
			names = append(names, name)
		}
		sort.Strings(names)
		return nil, nil, exit.Named(exit.Conflict, "private_dependency_lock_drift", "captured uv.lock has no exact wheel for installed dependencies: %s", strings.Join(names, ", "))
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })
	requirements := make([]string, 0, len(pins)-1)
	for name, version := range pins {
		if name != project {
			requirements = append(requirements, name+"=="+version)
		}
	}
	sort.Strings(requirements)
	return rows, requirements, nil
}

// CaptureUnpublishedClosure freezes public registry references while local wheel bytes stay private.
func (p *Package) CaptureUnpublishedClosure(ctx context.Context, closure string, extras []string, python string) *exit.Error {
	raw, err := os.ReadFile(p.Files["uv.lock"])
	if err != nil {
		return exit.Named(exit.Validation, "private_dependency_lock_invalid", "captured uv.lock is unavailable")
	}
	rows, requirements, problem := CapturedRegistryRows(raw, closure, p.Name, p.Release, p.DependencyWheels)
	if problem != nil {
		return problem
	}
	p.DependencyRequirements = RegistryRequirements(rows)
	requirements = append(requirements, "cozy-runtime>="+hostruntime.Floor)
	sort.Strings(requirements)
	sealed := filepath.Join(p.Root, "private-project", filepath.Base(p.Wheel))
	if problem := wheel.PinDependencies(p.Wheel, sealed, requirements); problem != nil {
		return problem
	}
	p.Wheel = sealed
	return nil
}

func fetchCapturedWheel(ctx context.Context, client *http.Client, row RegistryRow, path string) *exit.Error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, row.URL, nil)
	if err != nil {
		return exit.New(exit.Validation, "captured dependency download request is invalid")
	}
	response, err := client.Do(request)
	if err != nil {
		return exit.Unavailablef("captured dependency %s download was interrupted", row.Name)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.ContentLength > 0 && response.ContentLength != row.Size {
		return exit.Named(exit.Conflict, "private_dependency_download_changed", "captured dependency %s download differs from the captured wheel", row.Name)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage captured dependency %s", row.Name)
	}
	hash := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, row.Size+1))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || n != row.Size || hex.EncodeToString(hash.Sum(nil)) != row.SHA256 {
		return exit.Named(exit.Conflict, "private_dependency_download_changed", "captured dependency %s bytes differ from the captured wheel", row.Name)
	}
	fact, problem := wheel.InspectIdentity(path)
	if problem != nil {
		return problem
	}
	if fact.Distribution != row.Name || fact.Version != row.Version {
		return exit.New(exit.Conflict, "captured dependency metadata differs from its locked identity")
	}
	return nil
}

// RegistryRequirements preserves the selected public wheel URL and hash without
// resolving again or uploading registry artifacts through the unpublished wheel lane.
func RegistryRequirements(rows []RegistryRow) []byte {
	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		lines = append(lines, row.Name+" @ "+row.URL+" --hash=sha256:"+row.SHA256)
	}
	sort.Strings(lines)
	if len(lines) == 0 {
		return nil
	}
	return []byte(strings.Join(lines, "\n") + "\n")
}
