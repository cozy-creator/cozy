package packagepublish

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// CapturedDependency retains an immutable wheel and/or an exact public registry
// requirement. Public dependency bytes are fetched independently by each Runtime.
type CapturedDependency struct {
	Name, Version, Path, Digest string
	Requirement                 string
	RegistryRequirement         string
	Package                     string // original Hub package, empty for an unpublished wheel
	Application                 bool
}

// CaptureWheelDependencies retains the selected dependency bytes without
// rebuilding the caller or inventing a source project for any wheel library.
func CaptureWheelDependencies(ctx context.Context, tree, project, installed, stage string, selected map[string]map[string]string, targetPython ...string) (map[string]CapturedDependency, *exit.Error) {
	metadata, problem := readProjectDocument(filepath.Join(tree, "pyproject.toml"))
	if problem != nil {
		return nil, problem
	}
	pins, problem := capturedPins(installed)
	if problem != nil {
		return nil, problem
	}
	wanted := map[string]string{project: pins[project]}
	for _, closure := range selected {
		for name, version := range closure {
			wanted[name] = version
		}
	}
	// Only source projects in a selected callable closure need local wheel
	// capture. Other editable packages may be installed in the parent for its
	// own execution and should not trigger another source build here.
	existing, _, problem := collectLocalDependenciesForClosure(ctx, tree, metadata, stage, wanted, targetPython...)
	if problem != nil {
		return nil, problem
	}
	filtered := existing[:0]
	for _, dependency := range existing {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		if problem != nil {
			return nil, problem
		}
		if wanted[identity.Distribution] != "" {
			filtered = append(filtered, dependency)
		}
	}
	existing = filtered
	raw, err := os.ReadFile(filepath.Join(tree, "uv.lock"))
	if err != nil {
		return nil, exit.New(exit.Validation, "captured wheel dependencies have no uv.lock")
	}
	rows, _, problem := CapturedRegistryRows(raw, PinnedClosure(wanted), project, pins[project], existing, targetPython...)
	if problem != nil {
		return nil, problem
	}
	client := capturedWheelClient()
	out := map[string]CapturedDependency{}
	for _, dependency := range existing {
		captured, problem := CaptureDependency(dependency.Path)
		if problem != nil {
			return nil, problem
		}
		if wanted[captured.Name] != captured.Version || out[captured.Name].Name != "" {
			return nil, exit.New(exit.Conflict, "captured wheel differs from the selected dependency graph")
		}
		out[captured.Name] = captured
	}
	for _, row := range rows {
		requirement := row.Name + " @ " + row.URL + " --hash=sha256:" + row.SHA256
		captured := CapturedDependency{Name: row.Name, Version: row.Version, Requirement: requirement, RegistryRequirement: requirement}
		// Inspect ordinary dependency wheels for callable App exports. Large
		// framework artifacts already have their selected identity in uv.lock.
		if !ImageOwnedDistribution(row.Name) || row.captureLocally {
			address, _ := url.Parse(row.URL)
			path := filepath.Join(stage, filepath.Base(address.Path))
			if row.Size > MaxDependencyWheelBytes {
				return nil, exit.New(exit.Validation, "callable wheel exceeds the unpublished wheel bound")
			}
			if problem := fetchCapturedWheel(ctx, client, row, path); problem != nil {
				return nil, problem
			}
			var problem *exit.Error
			captured, problem = CaptureDependency(path)
			if problem != nil {
				return nil, problem
			}
			if !row.captureLocally {
				captured.RegistryRequirement = requirement
			}
		}
		captured.Package = row.packageRef
		out[row.Name] = captured
	}
	return out, nil
}

func CaptureDependency(path string) (CapturedDependency, *exit.Error) {
	identity, problem := wheel.InspectIdentity(path)
	if problem != nil {
		return CapturedDependency{}, problem
	}
	digest, problem := dependencyDigest(path)
	if problem != nil {
		return CapturedDependency{}, problem
	}
	contents, problem := wheel.InspectContents(path)
	if problem != nil {
		return CapturedDependency{}, problem
	}
	entries := contents.Group(applicationGroup)
	if len(entries) > 1 {
		return CapturedDependency{}, exit.Named(exit.Validation, "private_wheel_application_ambiguous", "dependency %s declares more than one App entry point", identity.Distribution)
	}
	address := (&url.URL{Scheme: "file", Path: path}).String()
	return CapturedDependency{Name: identity.Distribution, Version: identity.Version, Path: path, Digest: digest,
		Requirement: identity.Distribution + " @ " + address + " --hash=" + digest, Application: len(entries) == 1}, nil
}

func capturedWheelClient() *http.Client {
	return &http.Client{CheckRedirect: func(request *http.Request, via []*http.Request) error {
		if len(via) > 5 || request.URL.Scheme != "https" || request.URL.Host != "files.pythonhosted.org" {
			return fmt.Errorf("captured wheel download changed origin")
		}
		return nil
	}}
}
