package packagepublish

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

// HubWheels are the wheels a uv.lock selects from Tensorhub account indexes, and the package
// each such distribution is. A machine may not reach that Hub (one on this computer), so
// unpublished code carries them: each is named by its locked hash and fetched only for a
// machine that does not hold it. Size is 0 where the index stated none.
func HubWheels(lock []byte) ([]RegistryRow, map[string]string, *exit.Error) {
	var document capturedLock
	if len(lock) == 0 {
		return nil, nil, nil
	}
	if int64(len(lock)) > maxLockBytes || toml.Unmarshal(lock, &document) != nil {
		return nil, nil, exit.Named(exit.Validation, "private_dependency_lock_invalid", "the package's uv.lock is unreadable")
	}
	var rows []RegistryRow
	packages := map[string]string{}
	for _, entry := range document.Packages {
		organization := orgIndexNamespace(entry.Source.Registry)
		if organization == "" {
			continue
		}
		name := normalizedProjectName(entry.Name)
		packages[name] = organization + "/" + name
		for _, candidate := range entry.Wheels {
			hash, hashed := strings.CutPrefix(candidate.Hash, "sha256:")
			address, err := url.Parse(candidate.URL)
			if !hashed || err != nil {
				return nil, nil, exit.Named(exit.Validation, "private_dependency_lock_invalid",
					"uv.lock names a Tensorhub wheel of %s without its hash", name)
			}
			rows = append(rows, RegistryRow{captureLocally: true, Name: name, Version: entry.Version,
				URL: candidate.URL, SHA256: hash, Size: candidate.Size, Filename: filepath.Base(address.Path)})
		}
	}
	return rows, packages, nil
}

// KeptHubWheel is where a home keeps row: one file per hash, however many installs lock it.
func KeptHubWheel(store string, row RegistryRow) string {
	return filepath.Join(store, row.SHA256, row.Filename)
}

// KeepHubWheels fetches into store each wheel project's lock selects from a Tensorhub index
// that store lacks, so its machines install it while that Hub is out of reach. A kept wheel's
// time is its last use: the install sweep drops wheels no install used for a month.
func KeepHubWheels(ctx context.Context, project, store string) *exit.Error {
	lock, _ := os.ReadFile(filepath.Join(project, "uv.lock"))
	rows, _, problem := HubWheels(lock)
	for _, row := range rows {
		if problem != nil {
			break
		}
		kept := KeptHubWheel(store, row)
		if capturedWheelMatches(kept, row) {
			now := time.Now()
			_ = os.Chtimes(kept, now, now)
			continue
		}
		problem = keepHubWheel(ctx, row, kept)
	}
	return problem
}

func keepHubWheel(ctx context.Context, row RegistryRow, kept string) *exit.Error {
	if err := os.MkdirAll(filepath.Dir(kept), 0o700); err != nil {
		return exit.Internalf("cannot keep Tensorhub wheels: %s", err)
	}
	stage, err := os.MkdirTemp(filepath.Dir(kept), ".fetch-")
	if err != nil {
		return exit.Internalf("cannot keep Tensorhub wheels: %s", err)
	}
	defer os.RemoveAll(stage)
	staged := filepath.Join(stage, row.Filename)
	if problem := fetchCapturedWheel(ctx, capturedWheelClient(), row, staged); problem != nil {
		return problem
	}
	if err := os.Rename(staged, kept); err != nil {
		return exit.Internalf("cannot keep Tensorhub wheels: %s", err)
	}
	return nil
}

// LockedClosure is a local project's dependency roster, one name==version per line, as its
// uv.lock pins them (every platform's), else the names its pyproject declares: what says
// whether its callables need an accelerator, with no environment built.
func LockedClosure(project string) string {
	var lock capturedLock
	raw, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	if err == nil && toml.Unmarshal(raw, &lock) == nil && len(lock.Packages) > 0 {
		rows := make([]string, 0, len(lock.Packages))
		for _, entry := range lock.Packages {
			rows = append(rows, normalizedProjectName(entry.Name)+"=="+entry.Version)
		}
		return strings.Join(rows, "\n")
	}
	document, problem := readProjectDocument(filepath.Join(project, "pyproject.toml"))
	if problem != nil {
		return ""
	}
	return strings.Join(document.Project.Dependencies, "\n")
}

func capturedWheelClient() *http.Client {
	return &http.Client{CheckRedirect: func(request *http.Request, via []*http.Request) error {
		if len(via) > 5 || request.URL.Scheme != "https" || request.URL.Host != "files.pythonhosted.org" {
			return fmt.Errorf("captured wheel download changed origin")
		}
		return nil
	}}
}
