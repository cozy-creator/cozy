package packagepublish

// Private revisions carry the selected installed closure, including extras. Public
// publication keeps its registry references and image-owned version policy.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

type privateLock struct {
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

func privatePins(closure string) (map[string]string, *exit.Error) {
	pins := map[string]string{}
	for _, row := range strings.Split(strings.TrimSpace(closure), "\n") {
		name, version, ok := strings.Cut(row, "==")
		if !ok || name == "" || version == "" || strings.ContainsAny(version, " \t\r\n;<>!=") || normalizedProjectName(name) != name || pins[name] != "" {
			return nil, exit.Named(exit.Validation, "private_dependency_closure_invalid", "private environment requires a unique exact installed name/version roster")
		}
		pins[name] = version
	}
	if len(pins) > MaxDependencyWheels {
		return nil, tooManyDependencies()
	}
	return pins, nil
}

// PrivateRegistryRows selects only the installed closure from the captured uv.lock.
// It does not guess selected extras from a second resolution or export all extras.
func PrivateRegistryRows(raw []byte, closure, project, version string, existing []DependencyWheel) ([]RegistryRow, []string, *exit.Error) {
	pins, problem := privatePins(closure)
	if problem != nil {
		return nil, nil, problem
	}
	if pins[project] != version {
		return nil, nil, exit.Named(exit.Conflict, "private_dependency_project_changed", "captured environment does not contain this exact project")
	}
	expected := map[string]string{}
	for name, version := range pins {
		if name != project && !ImageOwnedDistribution(name) {
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
	var lock privateLock
	if len(raw) == 0 || int64(len(raw)) > maxLockBytes || toml.Unmarshal(raw, &lock) != nil || lock.Version != 1 {
		return nil, nil, exit.Named(exit.Validation, "private_dependency_lock_invalid", "private revision requires its captured uv.lock")
	}
	selected := registryLock{LockVersion: "1.0"}
	for _, entry := range lock.Packages {
		name := normalizedProjectName(entry.Name)
		if expected[name] != entry.Version {
			continue
		}
		if entry.Source.Registry != "https://pypi.org/simple" {
			return nil, nil, exit.Named(exit.Validation, "private_dependency_origin_unsupported", "private dependency %s is not a captured local wheel or public PyPI wheel", name)
		}
		row := registryPackage{Name: name, Version: entry.Version, Index: entry.Source.Registry}
		for _, candidate := range entry.Wheels {
			hash, ok := strings.CutPrefix(candidate.Hash, "sha256:")
			if !ok {
				continue
			}
			row.Wheels = append(row.Wheels, registryWheel{URL: candidate.URL, Size: candidate.Size, Hashes: map[string]string{"sha256": hash}})
		}
		selected.Packages = append(selected.Packages, row)
		delete(expected, name)
	}
	if len(expected) != 0 {
		names := make([]string, 0, len(expected))
		for name := range expected {
			names = append(names, name)
		}
		sort.Strings(names)
		return nil, nil, exit.Named(exit.Conflict, "private_dependency_lock_drift", "captured uv.lock has no exact wheel for installed dependencies: %s", strings.Join(names, ", "))
	}
	encoded, err := toml.Marshal(selected)
	if err != nil {
		return nil, nil, exit.Internalf("cannot encode private dependency selection")
	}
	rows, problem := RegistryRowsFromLock(encoded, existing, "")
	if problem != nil {
		return nil, nil, problem
	}
	requirements := make([]string, 0, len(pins)-1)
	for name, version := range pins {
		if name != project {
			requirements = append(requirements, name+"=="+version)
		}
	}
	sort.Strings(requirements)
	return rows, requirements, nil
}

// CapturePrivateClosure keeps wheel bytes entirely on the client-to-worker path.
func (p *Package) CapturePrivateClosure(ctx context.Context, closure string) *exit.Error {
	raw, err := os.ReadFile(p.Files["uv.lock"])
	if err != nil {
		return exit.Named(exit.Validation, "private_dependency_lock_invalid", "captured private uv.lock is unavailable")
	}
	rows, requirements, problem := PrivateRegistryRows(raw, closure, p.Name, p.Release, p.DependencyWheels)
	if problem != nil {
		return problem
	}
	directory := filepath.Join(p.Root, "private-registry")
	if err := os.Mkdir(directory, 0o700); err != nil {
		return exit.Internalf("cannot stage private registry wheels: %s", err)
	}
	client := &http.Client{CheckRedirect: func(request *http.Request, via []*http.Request) error {
		if len(via) > 5 || request.URL.Scheme != "https" || request.URL.Host != "files.pythonhosted.org" {
			return exit.New(exit.Validation, "private registry download changed origin")
		}
		return nil
	}}
	for _, row := range rows {
		address, _ := url.Parse(row.URL)
		path := filepath.Join(directory, filepath.Base(address.Path))
		if problem := fetchPrivateWheel(ctx, client, row, path); problem != nil {
			return problem
		}
		p.DependencyWheels = append(p.DependencyWheels, DependencyWheel{Filename: filepath.Base(path), Path: path})
	}
	sealed := filepath.Join(p.Root, "private-project", filepath.Base(p.Wheel))
	if problem := wheel.PinDependencies(p.Wheel, sealed, requirements); problem != nil {
		return problem
	}
	p.Wheel = sealed
	return nil
}

func fetchPrivateWheel(ctx context.Context, client *http.Client, row RegistryRow, path string) *exit.Error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, row.URL, nil)
	if err != nil {
		return exit.New(exit.Validation, "private dependency download request is invalid")
	}
	response, err := client.Do(request)
	if err != nil {
		return exit.Unavailablef("private dependency %s download was interrupted", row.Name)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.ContentLength > 0 && response.ContentLength != row.Size {
		return exit.Named(exit.Conflict, "private_dependency_download_changed", "private dependency %s download differs from the captured wheel", row.Name)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage private dependency %s", row.Name)
	}
	hash := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, row.Size+1))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || n != row.Size || hex.EncodeToString(hash.Sum(nil)) != row.SHA256 {
		return exit.Named(exit.Conflict, "private_dependency_download_changed", "private dependency %s bytes differ from the captured wheel", row.Name)
	}
	fact, problem := wheel.InspectIdentity(path)
	if problem != nil {
		return problem
	}
	if fact.Distribution != row.Name || fact.Version != row.Version {
		return exit.New(exit.Conflict, "private dependency metadata differs from its locked identity")
	}
	return nil
}
