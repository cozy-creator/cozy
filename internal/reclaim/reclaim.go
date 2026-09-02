// Package reclaim removes what nothing references any more under the three roots that
// used to grow without bound: attempt working directories, closed workers' roots, and
// tmp/. The process that wrote the bytes removes them when they die; each sweep here is
// the BACKSTOP for a writer that crashed, and mirrors install.Sweep: walk the DIRECTORY
// and let the records authority (or a live process's kernel lock) decide what still has
// a claim. A directory that refuses to go is reported and left for the next sweep.
package reclaim

import (
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
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

// ------------------------------------------------------------------ attempts

// Attempts reclaims every attempt directory whose attempt is closed and whose recorded
// outputs have moved out — or that no attempt row references at all.
func Attempts(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	requests, err := os.ReadDir(l.Attempts)
	if err != nil {
		return Swept{}, exit.Internalf("cannot scan the attempt root %s: %s", l.Attempts, err)
	}
	var swept Swept
	var first *exit.Error
	for _, request := range requests {
		if !request.IsDir() {
			continue
		}
		attempts, err := os.ReadDir(filepath.Join(l.Attempts, request.Name()))
		if err != nil {
			continue
		}
		for _, entry := range attempts {
			if !entry.IsDir() {
				continue
			}
			attempt, err := strconv.ParseUint(entry.Name(), 10, 64)
			if err != nil {
				continue
			}
			swept.Scanned++
			freed, removed, problem := Attempt(l, st, request.Name(), attempt)
			if problem != nil && first == nil {
				first = problem
			}
			swept.add(freed, removed)
		}
		_ = os.Remove(l.RequestAttempts(request.Name())) // only when empty
	}
	return swept, first
}

// Attempt reclaims one attempt directory if nothing claims it. It is the settlement-time
// call too, so the same rule runs after an ack and at the next daemon start.
func Attempt(l home.Layout, st *records.Store, requestID string, attempt uint64) (int64, bool, *exit.Error) {
	dir := l.AttemptDir(requestID, attempt)
	if _, err := os.Lstat(dir); os.IsNotExist(err) {
		return 0, false, nil
	}
	row, problem := st.AttemptRow(requestID, int64(attempt))
	if problem != nil {
		return 0, false, problem
	}
	if row != nil {
		if row.State != "closed" && row.State != "dispatch_aborted" {
			return 0, false, nil
		}
		paths, problem := st.AttemptOutputPaths(requestID, int64(attempt))
		if problem != nil {
			return 0, false, problem
		}
		for _, path := range paths {
			if within(dir, path) {
				return 0, false, nil
			}
		}
	}
	freed, problem := removeTree(dir)
	if problem != nil {
		return 0, false, problem
	}
	_ = os.Remove(l.RequestAttempts(requestID))
	return freed, true, nil
}

// Request reclaims every reclaimable attempt directory of one request, and the
// request's `tmp/<id>/` once the request has settled.
func Request(l home.Layout, st *records.Store, requestID string) (Swept, *exit.Error) {
	var swept Swept
	var first *exit.Error
	entries, err := os.ReadDir(l.RequestAttempts(requestID))
	if err == nil {
		for _, entry := range entries {
			attempt, err := strconv.ParseUint(entry.Name(), 10, 64)
			if !entry.IsDir() || err != nil {
				continue
			}
			swept.Scanned++
			freed, removed, problem := Attempt(l, st, requestID, attempt)
			if problem != nil && first == nil {
				first = problem
			}
			swept.add(freed, removed)
		}
	}
	if _, err := os.Lstat(filepath.Join(l.Tmp, requestID)); err == nil {
		swept.Scanned++
		freed, removed, problem := tmpEntry(l, st, requestID)
		if problem != nil && first == nil {
			first = problem
		}
		swept.add(freed, removed)
	}
	return swept, first
}

// ------------------------------------------------------------------ workers

// Workers reclaims the root of every worker whose process row is closed or absent. It
// runs after boot reconciliation has closed the rows of processes that did not survive.
func Workers(l home.Layout, st *records.Store) (Swept, *exit.Error) {
	entries, err := os.ReadDir(l.Workers)
	if err != nil {
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
	var swept Swept
	var first *exit.Error
	for _, entry := range entries {
		if entry.Name() == "locks" {
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

func within(dir, path string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

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
