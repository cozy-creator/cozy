package packagepublish

import (
	"context"
	"net/url"
	"os"
	"path/filepath"
	"strings"

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

// HubWheelDir is where an owned project copy keeps the wheels its lock selects from a Tensorhub
// index, as a slash path under its root.
const HubWheelDir = ".cozy-dependencies/hub"

// HubWheelPath is where an owned project copy keeps row.
func HubWheelPath(row RegistryRow) string { return HubWheelDir + "/" + row.Filename }

// RetainHubWheels links into an owned project copy every wheel its lock selects from a Tensorhub
// index, so the copy runs on any machine while that Hub is out of reach. store holds one file
// per hash: a wheel is fetched once, however many copies hold it.
func RetainHubWheels(ctx context.Context, root, store string) *exit.Error {
	lock, _ := os.ReadFile(filepath.Join(root, "uv.lock"))
	rows, _, problem := HubWheels(lock)
	for _, row := range rows {
		if problem != nil {
			break
		}
		path := filepath.Join(root, filepath.FromSlash(HubWheelPath(row)))
		if capturedWheelMatches(path, row) {
			continue
		}
		kept := filepath.Join(store, row.SHA256, row.Filename)
		if !capturedWheelMatches(kept, row) {
			problem = keepHubWheel(ctx, row, kept)
		}
		if problem == nil {
			_ = os.MkdirAll(filepath.Dir(path), 0o700)
			_ = os.Remove(path)
			if os.Link(kept, path) != nil {
				problem = copySnapshotFile(kept, path, MaxDependencyWheelBytes)
			}
		}
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
