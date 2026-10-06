package packagepublish

// Unpublished package revisions carry the selected installed closure, including extras. Public
// publication retains its existing registry-custody path.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"slices"
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
func CapturedRegistryRows(raw []byte, closure, project, version string, existing []DependencyWheel, targetPython ...string) ([]RegistryRow, []string, *exit.Error) {
	pins, problem := capturedPins(closure)
	if problem != nil {
		return nil, nil, problem
	}
	if !wheel.SameVersion(pins[project], version) {
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
		candidate, problem := selectRegistryWheel(name, row, targetPython...)
		if problem != nil {
			return nil, nil, problem
		}
		captureLocally := false
		packageRef := ""
		if entry.Source.Registry == "https://pypi.org/simple" {
			// Registry artifacts are fetched by Runtime's bounded storage; the
			// private client upload byte bound applies only to local wheels.
			if _, problem := registryWheelIdentityBound(name, entry.Version, candidate, MaxRegistryWheelBytes, targetPython...); problem != nil {
				return nil, nil, problem
			}
		} else if organization := orgIndexNamespace(entry.Source.Registry); organization != "" {
			packageRef = organization + "/" + name
			index, _ := url.Parse(entry.Source.Registry)
			object, err := url.Parse(candidate.URL)
			if err != nil || index.Host == "" || object.Scheme != index.Scheme || object.Host != index.Host {
				return nil, nil, exit.Named(exit.Validation, "registry_dependency_origin_refused", "captured Hub wheel differs from its locked index origin")
			}
			if _, problem := orgIndexWheelIdentity(name, entry.Version, organization, candidate, targetPython...); problem != nil {
				return nil, nil, problem
			}
			captureLocally = true
		} else {
			candidate, problem = pytorchRegistryWheel(name, entry.Version, entry.Source.Registry, row.Wheels, targetPython...)
			if problem != nil {
				return nil, nil, problem
			}
		}
		rows = append(rows, RegistryRow{captureLocally: captureLocally, packageRef: packageRef, Name: name, Version: entry.Version, URL: candidate.URL, SHA256: candidate.Hashes["sha256"], Size: candidate.Size})
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
	rows, requirements, problem := CapturedRegistryRows(raw, closure, p.Name, p.Release, p.DependencyWheels, python)
	if problem != nil {
		return problem
	}
	public := rows[:0]
	p.DependencyPackages = map[string]string{}
	for _, row := range rows {
		if row.packageRef != "" {
			p.DependencyPackages[row.Name] = row.packageRef
		}
		if !row.captureLocally {
			public = append(public, row)
			continue
		}
		address, _ := url.Parse(row.URL)
		directory := filepath.Join(p.Root, "captured-hub-wheels")
		if err := os.MkdirAll(directory, 0700); err != nil {
			return exit.Internalf("cannot stage captured Hub wheels: %s", err)
		}
		path := filepath.Join(directory, filepath.Base(address.Path))
		if problem := fetchCapturedWheel(ctx, capturedWheelClient(), row, path); problem != nil {
			return problem
		}
		p.DependencyWheels = append(p.DependencyWheels, DependencyWheel{Filename: filepath.Base(path), Path: path})
	}
	p.DependencyRequirements = RegistryRequirements(public)
	declared, problem := sdkRequirements(p.Wheel)
	if problem != nil {
		return problem
	}
	requirements = append(slices.DeleteFunc(requirements, func(pin string) bool {
		name, _, _ := strings.Cut(pin, "==")
		return sdkPair[name]
	}), declared...)
	requirements = append(requirements, "cozy-runtime>="+hostruntime.PackageFloor)
	sort.Strings(requirements)
	sealed := filepath.Join(p.Root, "private-project", filepath.Base(p.Wheel))
	if problem := wheel.PinDependencies(p.Wheel, sealed, requirements); problem != nil {
		return problem
	}
	p.Wheel = sealed
	return nil
}

// sdkPair is what every machine supplies itself: an exact pin on either would refuse each
// machine whose Runtime differs from the capturing one. The author's own bounds still hold.
var sdkPair = map[string]bool{"cozy-runtime": true, "tensorfs": true}

func sdkRequirements(projectWheel string) ([]string, *exit.Error) {
	metadata, problem := wheel.Metadata(projectWheel)
	if problem != nil {
		return nil, problem
	}
	headers, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(metadata))).ReadMIMEHeader()
	if err != nil && err != io.EOF {
		return nil, exit.New(exit.Validation, "captured project wheel metadata is malformed")
	}
	var declared []string
	for _, raw := range headers.Values("Requires-Dist") {
		req, problem := parseRequirement(raw)
		if problem != nil {
			return nil, problem
		}
		if sdkPair[req.name] {
			declared = append(declared, raw)
		}
	}
	return declared, nil
}

func fetchCapturedWheel(ctx context.Context, client *http.Client, row RegistryRow, path string) *exit.Error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, row.URL, nil)
	if err != nil {
		return exit.New(exit.Validation, "captured dependency download request is invalid")
	}
	if row.captureLocally {
		local := *client
		// Hub file custody redirects to a presigned storage object. Only the
		// redirect transport changes; the locked hash and bounded bytes below
		// still decide whether the artifact is accepted.
		local.CheckRedirect = func(next *http.Request, via []*http.Request) error {
			if len(via) > 5 || next.URL.Scheme != "https" || next.URL.Host == "" || next.URL.User != nil {
				return fmt.Errorf("captured Hub wheel storage redirect is invalid")
			}
			return nil
		}
		client = &local
	}
	response, err := client.Do(request)
	if err != nil {
		return exit.Named(exit.Unavailable, "private_dependency_download_failed", "captured dependency %s download or storage redirect failed", row.Name)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return exit.Named(exit.Unavailable, "private_dependency_download_failed", "captured dependency %s download returned HTTP %d", row.Name, response.StatusCode)
	}
	bound := row.Size
	if row.captureLocally && bound == 0 {
		bound = MaxDependencyWheelBytes
	}
	if bound <= 0 || bound > MaxDependencyWheelBytes || response.ContentLength > bound || row.Size > 0 && response.ContentLength > 0 && response.ContentLength != row.Size {
		return exit.Named(exit.Conflict, "private_dependency_download_changed", "captured dependency %s download differs from the captured wheel", row.Name)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage captured dependency %s", row.Name)
	}
	hash := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, bound+1))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil {
		return exit.Named(exit.Unavailable, "private_dependency_download_failed", "captured dependency %s download was interrupted", row.Name)
	}
	if syncErr != nil || closeErr != nil {
		return exit.Internalf("cannot retain captured dependency %s", row.Name)
	}
	if n > bound || row.Size > 0 && n != row.Size || hex.EncodeToString(hash.Sum(nil)) != row.SHA256 {
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
