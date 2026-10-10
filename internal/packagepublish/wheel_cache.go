package packagepublish

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// Callable inspection outlives neither its capture nor its scratch directory, but
// immutable dependency bytes can be reused by the next capture. The current lock
// still selects and authorizes every URL before this content-addressed lookup.
func fetchCachedWheel(ctx context.Context, client *http.Client, row RegistryRow, path, cache string) *exit.Error {
	if cache == "" {
		return exit.Internalf("captured dependency cache is unset")
	}
	entry := filepath.Join(cache, row.SHA256)
	if err := os.MkdirAll(entry, 0o700); err != nil {
		return exit.Internalf("cannot create captured dependency cache")
	}
	lock, err := os.OpenFile(filepath.Join(entry, ".lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return exit.Internalf("cannot lock captured dependency cache")
	}
	defer lock.Close()
	if err := flock.Wait(ctx, lock); err != nil {
		return exit.New(exit.Unavailable, "captured dependency cache wait was interrupted")
	}
	defer flock.Release(lock)
	cached := filepath.Join(entry, filepath.Base(path))
	if !capturedWheelMatches(cached, row) {
		// Interrupted downloads never acquire the committed wheel's name. A
		// verified replacement atomically repairs any corrupt prior entry.
		partial, err := os.MkdirTemp(entry, ".partial-")
		if err != nil {
			return exit.Internalf("cannot stage cached dependency")
		}
		defer os.RemoveAll(partial)
		downloaded := filepath.Join(partial, filepath.Base(path))
		if problem := fetchCapturedWheel(ctx, client, row, downloaded); problem != nil {
			return problem
		}
		if err := os.Chmod(downloaded, 0o400); err != nil {
			return exit.Internalf("cannot protect cached dependency")
		}
		if err := os.Rename(downloaded, cached); err != nil {
			return exit.Internalf("cannot retain cached dependency")
		}
	}
	// Capture scratch is an independent copy: sealing or changing it must not
	// mutate an object another capture will reuse.
	if problem := copySnapshotFile(cached, path, MaxDependencyWheelBytes); problem != nil {
		return problem
	}
	if !capturedWheelMatches(path, row) {
		_ = os.Remove(path)
		return exit.New(exit.Conflict, "cached dependency changed during capture")
	}
	return nil
}

func capturedWheelMatches(path string, row RegistryRow) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxDependencyWheelBytes || row.Size > 0 && info.Size() != row.Size {
		return false
	}
	hash := sha256.New()
	n, err := io.Copy(hash, io.LimitReader(file, MaxDependencyWheelBytes+1))
	if err != nil || n != info.Size() || hex.EncodeToString(hash.Sum(nil)) != row.SHA256 {
		return false
	}
	identity, problem := wheel.InspectIdentity(path)
	return problem == nil && identity.Distribution == row.Name && identity.Version == row.Version
}
