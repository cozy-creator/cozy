package localpackage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

var ownership sync.Mutex

// Guard protects staging until an ordinary request takes installation ownership.
func Guard() func() { ownership.Lock(); return ownership.Unlock }

func Drop(layout home.Layout, id string) *exit.Error {
	if !validInstallationID(id) {
		return exit.New(exit.Validation, "invalid installation ID")
	}
	root := filepath.Join(layout.LocalPackages, id)
	info, err := os.Lstat(root)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.New(exit.Validation, "installation directory is unavailable")
	}
	if err := os.RemoveAll(root); err != nil {
		return exit.Internalf("cannot remove installation staging: %s", err)
	}
	return nil
}

func DropUnowned(layout home.Layout, store *records.Store, id string) *exit.Error {
	used, problem := store.LocalInstallationInUse(id)
	if problem != nil || used {
		return problem
	}
	return Drop(layout, id)
}

func Sweep(layout home.Layout, store *records.Store) *exit.Error {
	entries, err := os.ReadDir(layout.LocalPackages)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return exit.Internalf("cannot scan installation staging: %s", err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".stage-") {
			continue
		}
		if !entry.IsDir() || !validInstallationID(entry.Name()) {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(layout.LocalPackages, entry.Name(), installationFile))
		var installation Installation
		if err != nil || len(raw) > 1<<20 || json.Unmarshal(raw, &installation) != nil || installation.ID != entry.Name() {
			continue
		}
		if problem := DropUnowned(layout, store, installation.ID); problem != nil {
			return problem
		}
	}
	return nil
}
