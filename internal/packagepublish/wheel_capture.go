package packagepublish

import (
	"context"
	"fmt"
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

// CapturedDependency is one immutable original wheel, or an image-owned exact
// requirement for local uv materialization. Base requirements never travel as
// private worker overlays.
type CapturedDependency struct {
	Name, Version, Path, Digest string
	Requirement                 string
	Application                 bool
}

// CaptureWheelDependencies retains the selected dependency bytes without
// rebuilding the caller or inventing a source project for any wheel library.
func CaptureWheelDependencies(ctx context.Context, tree, project, installed, stage string, selected map[string]map[string]string) (map[string]CapturedDependency, *exit.Error) {
	metadata, problem := readProjectDocument(filepath.Join(tree, "pyproject.toml"))
	if problem != nil {
		return nil, problem
	}
	pins, problem := privatePins(installed)
	if problem != nil {
		return nil, problem
	}
	wanted := map[string]string{project: pins[project]}
	for _, closure := range selected {
		for name, version := range closure {
			wanted[name] = version
		}
	}
	existing, _, _, problem := collectLocalDependencies(ctx, tree, metadata, stage, false)
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
	rows, _, problem := PrivateRegistryRows(raw, PinnedClosure(wanted), project, pins[project], existing)
	if problem != nil {
		return nil, problem
	}
	client := &http.Client{CheckRedirect: func(request *http.Request, via []*http.Request) error {
		if len(via) > 5 || request.URL.Scheme != "https" || request.URL.Host != "files.pythonhosted.org" {
			return fmt.Errorf("private wheel download changed origin")
		}
		return nil
	}}
	for _, row := range rows {
		address, _ := url.Parse(row.URL)
		path := filepath.Join(stage, filepath.Base(address.Path))
		if problem := fetchPrivateWheel(ctx, client, row, path); problem != nil {
			return nil, problem
		}
		existing = append(existing, DependencyWheel{Filename: filepath.Base(path), Path: path})
	}
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
	// uv installs these from its captured lock hashes (and existing artifact cache),
	// without resolving versions. A supplied local base wheel remains exact.
	local, problem := LocalDependencySelections(tree)
	if problem != nil {
		return nil, problem
	}
	var lock privateLock
	if toml.Unmarshal(raw, &lock) != nil {
		return nil, exit.New(exit.Validation, "captured dependency lock is invalid")
	}
	for name, version := range wanted {
		if name == project || !ImageOwnedDistribution(name) {
			continue
		}
		if source := local[name]; source.Path != "" {
			path := source.Path
			info, err := os.Stat(path)
			if err != nil {
				return nil, exit.New(exit.Conflict, "captured base dependency disappeared")
			}
			if info.IsDir() {
				built, problem := wheel.Build(wheel.Request{Context: ctx, Tree: path, OutDir: stage})
				if problem != nil {
					return nil, problem
				}
				path = built.Path
			}
			captured, problem := CaptureDependency(path)
			if problem != nil {
				return nil, problem
			}
			if captured.Name != name || captured.Version != version {
				return nil, exit.New(exit.Conflict, "captured base wheel identity changed")
			}
			out[name] = captured
			continue
		}
		var hashes []string
		matched := false
		for _, entry := range lock.Packages {
			if normalizedProjectName(entry.Name) != name || entry.Version != version {
				continue
			}
			if matched || entry.Source.Registry != "https://pypi.org/simple" {
				return nil, exit.New(exit.Conflict, "base dependency has ambiguous or unsupported locked origin")
			}
			matched = true
			for _, candidate := range entry.Wheels {
				hash := strings.TrimPrefix(candidate.Hash, "sha256:")
				if len(hash) == 64 && candidate.Hash == "sha256:"+hash {
					hashes = append(hashes, "--hash="+candidate.Hash)
				}
			}
		}
		if len(hashes) == 0 {
			return nil, exit.New(exit.Conflict, "base dependency has no captured wheel hashes")
		}
		sort.Strings(hashes)
		out[name] = CapturedDependency{Name: name, Version: version, Requirement: name + "==" + version + " " + strings.Join(hashes, " ")}
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
