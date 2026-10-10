// Package reclaim removes what nothing references any more under the two roots that
// used to grow without bound: closed workers' roots and tmp/. The process that wrote the
// bytes removes them when they die; each sweep here is the BACKSTOP for a writer that
// crashed, and mirrors install.Sweep: walk the DIRECTORY and let the records authority
// (or a live process's kernel lock) decide what still has a claim. A directory that
// refuses to go is reported and left for the next sweep. There is no attempt root: a
// local attempt writes its result where it lives and stages nothing.
package reclaim

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// Swept is what one sweep did. Bytes counts regular files' sizes, hardlinked ones once
// per link the tree held.
type Swept struct {
	Scanned int
	Removed int
	Bytes   int64
}

func (s *Swept) add(freed int64, removed bool) {
	if removed {
		s.Removed++
		s.Bytes += freed
	}
}

// Request reclaims one request's `tmp/<id>/` once the request has settled. It is the
// settlement-time call — after an ack and after the export settles — so the writer's own
// removal is the rule and this is the backstop, safe to repeat.
func Request(l home.Layout, st *records.Store, requestID string) (Swept, *exit.Error) {
	var swept Swept
	if _, err := os.Lstat(filepath.Join(l.Tmp, requestID)); err != nil {
		return swept, nil
	}
	swept.Scanned++
	freed, removed, problem := tmpEntry(l, st, requestID)
	swept.add(freed, removed)
	return swept, problem
}

// ------------------------------------------------------------------ workers

// Workers reclaims the root of every worker whose process row is closed or absent. It
// runs after boot reconciliation has closed the rows of processes that did not survive.
func Workers(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	entries, err := os.ReadDir(l.Workers)
	if err != nil {
		if os.IsNotExist(err) {
			return Swept{}, nil
		}
		return Swept{}, exit.Internalf("cannot scan the worker root %s: %s", l.Workers, err)
	}
	live, problem := st.LiveWorkers()
	if problem != nil {
		return Swept{}, problem
	}
	held := make(map[string]bool, len(live))
	for _, row := range live {
		held[row.InstanceID] = true
	}
	var swept Swept
	var first *exit.Error
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		swept.Scanned++
		if held[entry.Name()] {
			continue
		}
		freed, problem := Worker(l, entry.Name(), false)
		if problem != nil {
			if first == nil {
				first = problem
			}
			continue
		}
		swept.add(freed, true)
	}
	return swept, first
}

// Worker reclaims one closed worker's root. With keepLog the log stays for the person
// the exit message pointed at it; the next daemon start removes the rest.
func Worker(l home.Layout, instanceID string, keepLog bool) (int64, *exit.Error) {
	dir := l.WorkerDir(instanceID)
	if instanceID == "" || filepath.Base(instanceID) != instanceID || filepath.Dir(dir) != l.Workers {
		return 0, exit.Internalf("refusing to remove a worker root with unsafe id %q", instanceID)
	}
	if !keepLog {
		return removeTree(dir)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, exit.Internalf("cannot scan the worker root %s: %s", dir, err)
	}
	var freed int64
	for _, entry := range entries {
		if entry.Name() == "worker.log" {
			continue
		}
		n, problem := removeTree(filepath.Join(dir, entry.Name()))
		if problem != nil {
			return freed, problem
		}
		freed += n
	}
	return freed, nil
}

// ------------------------------------------------------------------ tmp

// Tmp reclaims every entry under tmp/ that is dead: no live process holds its scratch
// claim, and the request it is named for — if it names one — has settled. `locks/` is
// the model-acquisition lock directory and stays. On a healthy box this reclaims 0: the
// writer removed its own entry when the bytes died.
func Tmp(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	entries, err := os.ReadDir(l.Tmp)
	if err != nil {
		if os.IsNotExist(err) {
			return Swept{}, nil
		}
		return Swept{}, exit.Internalf("cannot scan the tmp root %s: %s", l.Tmp, err)
	}
	protected := map[string]bool{"locks": true}
	if st != nil {
		updates, problem := st.ActiveRuntimeUpdates()
		if problem != nil {
			return Swept{}, problem
		}
		for _, update := range updates {
			// Public and local updates share this stage, including the pre-plan window.
			protected["runtime-updates"] = true
			if len(update.Selection) == 0 {
				continue
			}
			type localWheel struct {
				Path string `json:"path"`
			}
			var selection struct {
				LocalRuntime  *localWheel `json:"local_runtime"`
				LocalTensorFS *localWheel `json:"local_tensorfs"`
			}
			if err := json.Unmarshal(update.Selection, &selection); err != nil {
				return Swept{}, exit.Internalf("cannot read active Runtime update staging: %s", err)
			}
			for _, wheel := range []*localWheel{selection.LocalRuntime, selection.LocalTensorFS} {
				if wheel == nil {
					continue
				}
				relative, err := filepath.Rel(l.Tmp, wheel.Path)
				if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative) {
					protected[strings.Split(relative, string(filepath.Separator))[0]] = true
				}
			}
		}
	}
	var swept Swept
	var first *exit.Error
	for _, entry := range entries {
		if protected[entry.Name()] {
			continue
		}
		swept.Scanned++
		freed, removed, problem := tmpEntry(l, st, entry.Name())
		if problem != nil {
			if first == nil {
				first = problem
			}
			continue
		}
		swept.add(freed, removed)
	}
	return swept, first
}

// tmpEntry removes one `tmp/<name>` unless a live process holds it or it is named for a
// request that has not settled.
func tmpEntry(l home.Layout, st *records.Store, name string) (int64, bool, *exit.Error) {
	path := filepath.Join(l.Tmp, name)
	info, err := os.Lstat(path)
	if err != nil {
		return 0, false, nil
	}
	if info.IsDir() && scratch.Held(path) {
		return 0, false, nil
	}
	if st != nil {
		request, problem := st.RequestRow(name)
		if problem != nil {
			return 0, false, problem
		}
		if request != nil && !records.Settled(request.State) {
			return 0, false, nil
		}
	}
	freed, problem := removeTree(path)
	if problem != nil {
		return 0, false, problem
	}
	return freed, true, nil
}

// ------------------------------------------------------------------ helpers

// removeTree measures then removes one path. A tree a worker left read-only is made
// removable first, the way a retired install is.
func removeTree(path string) (int64, *exit.Error) {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, exit.Internalf("cannot inspect %s: %s", path, err)
	}
	var freed int64
	if info.IsDir() {
		freed = disk(path)
	} else if info.Mode().IsRegular() {
		freed = info.Size()
	}
	if err := os.RemoveAll(path); err != nil {
		if err := makeRemovable(path); err != nil {
			return 0, exit.Internalf("cannot prepare %s for removal: %s", path, err)
		}
		if err := os.RemoveAll(path); err != nil {
			return 0, exit.Internalf("cannot remove %s: %s", path, err)
		}
	}
	return freed, nil
}

func disk(dir string) (total int64) {
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || !d.Type().IsRegular() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}

func makeRemovable(root string) error {
	return filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return nil
		}
		mode := info.Mode().Perm() | 0o200
		if info.IsDir() {
			mode |= 0o700
		}
		if mode != info.Mode().Perm() {
			return os.Chmod(path, mode)
		}
		return nil
	})
}

// ------------------------------------------------------------------ publications

// Publications reclaims the job publication plane's DEBRIS while preserving every
// committed publication: a root a publication row names survives untouched (minus its
// settled `.staging`); a root whose request settled without ever committing one — or
// whose request no longer exists — is removed whole. Live requests are left alone.
// Empty org directories, and finally the empty plane itself, are pruned.
func Publications(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	orgs, err := os.ReadDir(l.Publications)
	if err != nil {
		if os.IsNotExist(err) {
			return Swept{}, nil
		}
		return Swept{}, exit.Internalf("cannot scan the publication plane %s: %s", l.Publications, err)
	}
	var swept Swept
	var first *exit.Error
	note := func(problem *exit.Error) {
		if problem != nil && first == nil {
			first = problem
		}
	}
	for _, org := range orgs {
		if !org.IsDir() {
			continue
		}
		orgDir := filepath.Join(l.Publications, org.Name())
		jobs, err := os.ReadDir(orgDir)
		if err != nil {
			note(exit.Internalf("cannot scan %s: %s", orgDir, err))
			continue
		}
		for _, job := range jobs {
			requestID, isJob := strings.CutPrefix(job.Name(), "_job-")
			if !job.IsDir() || !isJob {
				continue
			}
			swept.Scanned++
			freed, removed, problem := publicationRoot(l, st, filepath.Join(orgDir, job.Name()), requestID)
			note(problem)
			swept.add(freed, removed)
		}
		_ = os.Remove(orgDir) // only when empty
	}
	_ = os.Remove(l.Publications)
	return swept, first
}

// publicationRoot settles one `<org>/_job-<id>` directory against the records authority.
func publicationRoot(l home.Layout, st *records.Store, root, requestID string) (int64, bool, *exit.Error) {
	request, problem := st.RequestRow(requestID)
	if problem != nil {
		return 0, false, problem
	}
	if request != nil && !records.Settled(request.State) {
		return 0, false, nil // an attempt may be writing its stage right now
	}
	publication, problem := st.PublicationOf(requestID)
	if problem != nil {
		return 0, false, problem
	}
	if publication == nil {
		// A run's product store is the daemon's copy of its output log.
		if products, problem := st.Products(requestID); problem != nil || len(products) > 0 {
			return 0, false, problem
		}
		freed, problem := removeTree(root)
		return freed, problem == nil, problem
	}
	// A committed publication keeps its root; only the settled staging is over.
	freed, problem := removeTree(filepath.Join(root, ".staging"))
	if problem != nil {
		return 0, false, problem
	}
	return freed, freed > 0, nil
}

// ------------------------------------------------------------------ rental secrets

// RentalSecrets erases the secret files of every rental the records authority no longer
// holds: an id in a proven-absent terminal state, an id with no row at all, and every
// pending-operation credential whose paid operation is over. Files this package cannot
// attribute are left where they are. The empty directory is removed last.
func RentalSecrets(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	entries, err := os.ReadDir(l.Rentals)
	if err != nil {
		if os.IsNotExist(err) {
			return Swept{}, nil
		}
		return Swept{}, exit.Internalf("cannot scan the rental credential root %s: %s", l.Rentals, err)
	}
	held, problem := st.HeldRentalIDs()
	if problem != nil {
		return Swept{}, problem
	}
	operations, problem := st.RentalOperationsWithSecrets()
	if problem != nil {
		return Swept{}, problem
	}
	pending := map[string]bool{}
	for _, op := range operations {
		sum := sha256.Sum256([]byte(op.Key))
		pending["pending-"+hex.EncodeToString(sum[:])] = true
	}
	var swept Swept
	var first *exit.Error
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		owner, known := secretOwner(entry.Name())
		if !known {
			continue
		}
		swept.Scanned++
		if strings.HasPrefix(owner, "pending-") {
			if pending[owner] {
				continue
			}
		} else if held[owner] {
			continue
		}
		path := filepath.Join(l.Rentals, entry.Name())
		info, err := entry.Info()
		var size int64
		if err == nil {
			size = info.Size()
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			if first == nil {
				first = exit.Internalf("cannot erase the stale rental secret %s: %s", entry.Name(), err)
			}
			continue
		}
		swept.add(size, true)
	}
	_ = os.Remove(l.Rentals) // only when empty
	return swept, first
}

// secretOwner maps one rental credential filename to the id (or pending key hash) that
// owns it. Only this package's own spellings are recognized.
func secretOwner(name string) (string, bool) {
	for _, suffix := range []string{".media-token", ".creator.pem", ".pem"} {
		if owner, ok := strings.CutSuffix(name, suffix); ok && owner != "" {
			return owner, true
		}
	}
	return "", false
}

// RetiredLocalPackages removes the staged copies of unpublished code an older cozy left.
func RetiredLocalPackages(l home.Layout) (Swept, *exit.Error) {
	path := l.RetiredLocalPackages()
	if _, err := os.Lstat(path); err != nil {
		return Swept{}, nil
	}
	freed, problem := removeTree(path)
	return Swept{Scanned: 1, Removed: 1, Bytes: freed}, problem
}

// EmptyRoots prunes on-demand directories whose work is gone. Every one of these is
// recreated by its writer at the moment work exists, so an empty one is debris, not
// structure. Each remove refuses unless the directory is empty, which is the point.
func EmptyRoots(l home.Layout) {
	for _, dir := range []string{
		l.Workers, filepath.Join(l.Tmp, "locks"), l.Tmp,
	} {
		_ = os.Remove(dir)
	}
}
