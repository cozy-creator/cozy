package machines

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
)

// Replaced says what a replacement kept and removed.
type Replaced struct {
	// Kept are the old machine's files this one uses in place: its TensorFS store (no model
	// is downloaded again) and its wheel cache (package environments rebuild from it).
	Kept []string `json:"kept"`
	// FreedBytes is the disk the old machine's own files gave back: its software, package
	// environments and run journal, which this machine cannot read.
	FreedBytes int64 `json:"freed_bytes"`
}

// wheelCache is uv's cache: where the Go agent's Runtime kept it under the root, and where
// the machine keeps it (cozy-machine `published.rs`).
var wheelCache = [2]string{"var/lib/cozy/dependencies/uv-cache", "var/lib/cozy/rust-machine/published/uv-cache"}

// carried are the paths a replacement moves from the old root into the new one, old then
// new: the wheel cache, and the store when it lives under the root (the box's own store,
// outside it, is simply used where it is).
func (h *Host) carried() [][2]string {
	out := [][2]string{wheelCache}
	if h.store == "" {
		return append(out, [2]string{"var/lib/tensorfs", "var/lib/tensorfs"}) // both machines' unnamed store
	}
	if inside, err := filepath.Rel(h.Root(), h.store); err == nil && inside != "." && filepath.IsLocal(inside) {
		out = append(out, [2]string{inside, inside})
	}
	return out
}

// replaceLocked replaces an installed machine that predates MachineAPI (the Go agent), which
// cannot be updated in place. It stops that machine, sets its root aside whole, installs
// this one into a fresh root and proves it over the API; only then is the old root deleted,
// and any failure puts it back. The machine keeps its id, the box's TensorFS store and the
// wheel cache. The old machine's package environments and run journal are not readable by
// this one and go with its root.
func (h *Host) replaceLocked(ctx context.Context, staged *staged, uv string) (*Installed, *exit.Error) {
	status, problem := h.Status()
	if problem != nil {
		return nil, problem
	}
	if problem := h.stopLocked(ctx); problem != nil {
		return nil, problem
	}
	if problem := h.vacated(ctx); problem != nil {
		return nil, problem
	}
	root, aside := h.Root(), h.path("root.replaced")
	record, recordAside := h.path("installed.json"), h.path("installed.replaced.json")
	if err := errors.Join(os.Rename(root, aside), os.Rename(record, recordAside)); err != nil {
		_ = os.Rename(aside, root)
		return nil, exit.Internalf("cannot set the installed machine aside: %s", err)
	}
	var moved [][2]string
	undo := func(cause *exit.Error) (*Installed, *exit.Error) {
		settle := context.WithoutCancel(ctx)
		_ = h.stopLocked(settle)
		_ = h.vacated(settle)
		for _, pair := range moved {
			_ = os.Rename(filepath.Join(root, pair[1]), filepath.Join(aside, pair[0]))
		}
		err := errors.Join(removeTree(root), os.Rename(aside, root), os.Rename(recordAside, record))
		if err != nil {
			return nil, exit.Internalf("%s; and the installed machine could not be put back from %s: %s", cause.Message, aside, err)
		}
		return nil, exit.Named(cause.Code, "machine.replace_failed", "%s; the installed machine was put back, stopped", cause.Message).
			WithRemedy("the cozy that installed it starts it again")
	}
	installed, problem := h.place(staged, uv)
	if problem != nil {
		return undo(problem)
	}
	installed.Replaced = &Replaced{}
	if h.store != "" {
		installed.Replaced.Kept = []string{h.store}
	}
	for _, pair := range h.carried() {
		from, to := filepath.Join(aside, pair[0]), filepath.Join(root, pair[1])
		if _, err := os.Lstat(from); err != nil {
			continue
		}
		if err := errors.Join(os.MkdirAll(filepath.Dir(to), 0o755), os.Rename(from, to)); err != nil {
			return undo(exit.Internalf("cannot carry %s into the new machine: %s", pair[0], err))
		}
		moved = append(moved, pair)
		if to != h.store {
			installed.Replaced.Kept = append(installed.Replaced.Kept, to)
		}
	}
	// Proof: the new machine starts, seals its receipt and answers Status as this machine.
	launch, problem := h.ensureLocked(ctx, nil, true)
	if problem != nil {
		return undo(problem)
	}
	frame, problem := h.ReadStatus(ctx)
	switch {
	case problem != nil:
		return undo(problem)
	case frame == nil || frame.GetPhase() != "ready" || frame.GetWorkerId() != launch.WorkerID:
		return undo(exit.New(exit.Failed, "the new machine did not answer Status as %s, ready", launch.WorkerID))
	}
	if !status.Running {
		if problem := h.stopLocked(ctx); problem != nil {
			return undo(problem)
		}
	}
	// The commit point: from here the old root is only bytes to delete.
	retired := h.path("root.retired")
	if err := errors.Join(os.Rename(aside, retired), os.Remove(recordAside)); err != nil {
		return undo(exit.Internalf("cannot retire the replaced machine: %s", err))
	}
	installed.Replaced.FreedBytes = treeBytes(retired)
	if err := removeTree(retired); err != nil {
		h.note("the replaced machine's files remain at " + retired + ": " + err.Error())
	}
	if err := h.recordInstalled(*installed); err != nil {
		return nil, exit.Internalf("cannot record the machine installation: %s", err)
	}
	return installed, nil
}

// recoverReplace finishes a replacement an earlier command did not: a root set aside and
// never committed is put back over whatever stands in its place; one already committed is
// deleted.
func (h *Host) recoverReplace(ctx context.Context) *exit.Error {
	if err := removeTree(h.path("root.retired")); err != nil {
		return exit.Internalf("cannot delete the replaced machine's files: %s", err)
	}
	aside := h.path("root.replaced")
	if _, err := os.Lstat(aside); errors.Is(err, os.ErrNotExist) {
		_ = os.Remove(h.path("installed.replaced.json")) // a commit this far never removed it
		return nil
	}
	if problem := h.stopLocked(ctx); problem != nil {
		return problem
	}
	if problem := h.vacated(ctx); problem != nil {
		return problem
	}
	for _, pair := range h.carried() {
		if _, err := os.Lstat(filepath.Join(h.Root(), pair[1])); err == nil {
			_ = os.MkdirAll(filepath.Dir(filepath.Join(aside, pair[0])), 0o755)
			_ = os.Rename(filepath.Join(h.Root(), pair[1]), filepath.Join(aside, pair[0]))
		}
	}
	err := errors.Join(removeTree(h.Root()), os.Rename(aside, h.Root()))
	if _, missing := os.Lstat(h.path("installed.replaced.json")); err == nil && missing == nil {
		err = os.Rename(h.path("installed.replaced.json"), h.path("installed.json"))
	}
	if err != nil {
		return exit.Internalf("cannot put back the machine an interrupted install set aside at %s: %s", aside, err)
	}
	return nil
}

// vacated waits until a stopped machine's processes have let the root go: the machine's
// lock and, under the Go agent, its Python Runtime's, which outlives a killed agent.
func (h *Host) vacated(ctx context.Context) *exit.Error {
	for {
		busy := false
		for _, lock := range []string{guardLock, runtimeLock} {
			file, held, err := probe(filepath.Join(h.Root(), lock), false)
			if err != nil {
				return exit.Internalf("cannot inspect machine ownership: %s", err)
			}
			if file != nil {
				_ = flock.Release(file)
				_ = file.Close()
			}
			busy = busy || held
		}
		if !busy {
			return nil
		}
		select {
		case <-ctx.Done():
			return exit.Named(exit.Conflict, "machine.busy", "the stopped machine's Runtime still runs on this root; nothing was changed").
				WithRemedy("`cozy machine install` again once its work has ended")
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// removeTree deletes path and everything under it, read-only directories included; an
// absent path is deleted already.
func removeTree(path string) error {
	if os.RemoveAll(path) == nil {
		return nil
	}
	_ = filepath.WalkDir(path, func(entry string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			_ = os.Chmod(entry, 0o700)
		}
		return nil
	})
	return os.RemoveAll(path)
}

// treeBytes is the disk deleting a tree gives back: the blocks of every file all of whose
// links are inside it (a file the kept wheel cache also links stays).
func treeBytes(path string) int64 {
	var total int64
	shared := map[uint64][2]int64{} // inode: links not yet seen, bytes
	_ = filepath.WalkDir(path, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		inode, links, bytes := diskOf(info)
		if d.IsDir() || links < 2 {
			total += bytes
			return nil
		}
		left := links - 1
		if seen, ok := shared[inode]; ok {
			left = seen[0] - 1
		}
		if shared[inode] = [2]int64{left, bytes}; left == 0 {
			total += bytes
		}
		return nil
	})
	return total
}

// PredatesAPI says whether this computer's installed machine cannot be driven by this
// controller, so that `cozy machine install` replaces it.
func (h *Host) PredatesAPI(ctx context.Context) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.predatesAPI(ctx)
}

func (h *Host) predatesAPI(ctx context.Context) bool {
	_, err := os.Stat(h.binary())
	return err == nil && !h.servesAPI(ctx)
}
