package cli

// THE EDITABLE SOURCE WATCHER (cl-097). Owner: "if I install an editable package (just a
// directory on my machine) and I edit it, are those edits moved onto the machines I'm
// connected to occasionally? ... otherwise they'd get out of date as I evolve my package
// locally." Before this, an edit reached a worker only at the next `cozy run`, which paid
// the rebuild and the cold placement itself. Now the daemon watches every editable
// install's source tree; a change is rebuilt on the path a run would have taken
// (RefreshSnapshot: rebuild, Runtime accepts, the pin swaps) and every worker holding the
// package — the local lane and each attached rental — is re-prepared the way the next run
// would have, so that run is warm.
//
// Nothing here is timed. A filesystem event is the observed fact that wakes a tree; the
// only "debounce" is the reading itself: a tree is read again while events keep arriving
// during a read, and acted on once one whole read passed with none. A rebuild that fails
// (a syntax error mid-edit) is logged typed and leaves the last good install active; no
// run that was not asked for is failed by it.

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/fsnotify/fsnotify"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

type editableSync struct {
	layout   home.Layout
	store    *records.Store
	resolver *Resolver
	owner    *orchestrator.Orchestrator
	log      io.Writer
	watcher  *fsnotify.Watcher
	ctx      context.Context
	quit     <-chan struct{}
	rescan   chan struct{}
	mu       sync.Mutex
	trees    map[string]*editableTree // by source root
}

type editableTree struct {
	pkg, root  string
	watchRoots []string
	events     atomic.Uint64
	kick       chan struct{}
	stop       chan struct{}
}

// startEditableSync watches the records database for pin changes and every editable
// install's source tree for edits, until quit closes.
func startEditableSync(layout home.Layout, store *records.Store, resolver *Resolver,
	owner *orchestrator.Orchestrator, log io.Writer, quit <-chan struct{},
) (*editableSync, *exit.Error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, exit.Internalf("cannot watch editable package sources: %s", err)
	}
	if err := watcher.Add(filepath.Dir(layout.DB)); err != nil {
		watcher.Close()
		return nil, exit.Internalf("cannot watch %s for package pins: %s", filepath.Dir(layout.DB), err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &editableSync{layout: layout, store: store, resolver: resolver, owner: owner, log: log,
		watcher: watcher, ctx: ctx, quit: quit, rescan: make(chan struct{}, 1),
		trees: map[string]*editableTree{}}
	go func() {
		<-quit
		cancel()
		watcher.Close()
	}()
	go s.route()
	go s.rescanLoop()
	s.reconcile()
	return s, nil
}

// route turns filesystem events into two kinds of wake: the records database moved (a pin
// may have changed) and a source tree moved.
func (s *editableSync) route() {
	dbDir := filepath.Dir(s.layout.DB)
	for {
		select {
		case <-s.quit:
			return
		case err, ok := <-s.watcher.Errors:
			if !ok {
				return
			}
			fmt.Fprintf(s.log, "editable watch: %s\n", err)
		case event, ok := <-s.watcher.Events:
			if !ok {
				return
			}
			if filepath.Dir(event.Name) == dbDir {
				if strings.HasPrefix(filepath.Base(event.Name), filepath.Base(s.layout.DB)) {
					wake(s.rescan)
				}
				continue
			}
			s.onSourceEvent(event)
		}
	}
}

func wake(kick chan struct{}) {
	select {
	case kick <- struct{}{}:
	default:
	}
}

func (s *editableSync) onSourceEvent(event fsnotify.Event) {
	s.mu.Lock()
	type touched struct {
		tree *editableTree
		root string
	}
	var changed []touched
	for _, tree := range s.trees {
		for _, root := range tree.watchRoots {
			if event.Name == root || strings.HasPrefix(event.Name, root+string(filepath.Separator)) {
				changed = append(changed, touched{tree, root})
				break
			}
		}
	}
	s.mu.Unlock()
	for _, item := range changed {
		rel, err := filepath.Rel(item.root, event.Name)
		if err != nil || packagepublish.IgnoredSourcePath(rel) {
			continue
		}
		if event.Has(fsnotify.Create) {
			if info, err := os.Lstat(event.Name); err == nil && info.IsDir() {
				s.watchDirs(item.root, event.Name)
			}
		}
		item.tree.events.Add(1)
		wake(item.tree.kick)
	}
}

// watchDirs adds a watch on dir and every directory beneath it the source rules read.
func (s *editableSync) watchDirs(root, dir string) {
	_ = filepath.WalkDir(dir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || !entry.IsDir() {
			return nil
		}
		if rel, relErr := filepath.Rel(root, path); relErr == nil && rel != "." &&
			packagepublish.IgnoredSourcePath(rel) {
			return fs.SkipDir
		}
		if err := s.watcher.Add(path); err != nil {
			fmt.Fprintf(s.log, "editable watch: cannot watch %s: %s\n", path, err)
		}
		return nil
	})
}

func (s *editableSync) rescanLoop() {
	for {
		select {
		case <-s.quit:
			return
		case <-s.rescan:
			s.reconcile()
		}
	}
}

// reconcile makes the watched set equal the set of editable pins. A pin moves under the
// single-writer lock, and the database event that woke this arrives while the writer is
// still inside its transaction, so the set is read only once that writer has let go.
func (s *editableSync) reconcile() {
	// This read barrier briefly owns the same writer claim as a refresh. Join the
	// daemon's mutation lock so a foreground refresh waits for the barrier instead
	// of reporting our own watcher as a competing install process.
	s.resolver.refreshMu.Lock()
	if file, err := os.OpenFile(s.layout.Lock, os.O_CREATE|os.O_RDWR, 0o644); err == nil {
		_ = flock.Block(file)
		_ = flock.Release(file)
		file.Close()
	}
	s.resolver.refreshMu.Unlock()
	rows, problem := s.store.Installed()
	if problem != nil {
		fmt.Fprintf(s.log, "editable watch: cannot read installs: %s\n", problem.Message)
		return
	}
	want := map[string]string{}
	roots := map[string][]string{}
	for _, row := range rows {
		if row.SourceKind != "local" || row.SourceRef == "" {
			continue
		}
		want[row.SourceRef] = row.Package
		roots[row.SourceRef] = []string{row.SourceRef}
		dependencies, problem := packagepublish.LocalDependencyPaths(row.SourceRef)
		if problem != nil {
			fmt.Fprintf(s.log, "editable %s: dependency watch scan refused: %s\n", row.Package, problem.Message)
			// A temporarily missing file or half-written pyproject must not remove
			// the watches that can observe its repair.
			s.mu.Lock()
			if previous := s.trees[row.SourceRef]; previous != nil {
				roots[row.SourceRef] = append([]string(nil), previous.watchRoots...)
			}
			s.mu.Unlock()
			continue
		}
		for _, path := range dependencies {
			roots[row.SourceRef] = append(roots[row.SourceRef], path)
		}
		sort.Strings(roots[row.SourceRef])
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for root, tree := range s.trees {
		if want[root] == tree.pkg && strings.Join(roots[root], "\n") == strings.Join(tree.watchRoots, "\n") {
			continue
		}
		delete(s.trees, root)
		close(tree.stop)
		for _, watched := range s.watcher.WatchList() {
			owned := sourceWatchNeeded(watched, tree.watchRoots)
			needed := watched == filepath.Dir(s.layout.DB)
			for _, sources := range roots {
				if sourceWatchNeeded(watched, sources) {
					needed = true
					break
				}
			}
			if owned && !needed {
				_ = s.watcher.Remove(watched)
			}
		}
		fmt.Fprintf(s.log, "editable %s: no longer watching %s\n", tree.pkg, root)
	}
	for root, pkg := range want {
		if s.trees[root] != nil {
			continue
		}
		tree := &editableTree{pkg: pkg, root: root, watchRoots: roots[root], kick: make(chan struct{}, 1), stop: make(chan struct{})}
		s.trees[root] = tree
		for _, source := range tree.watchRoots {
			if info, err := os.Stat(source); err == nil && !info.IsDir() {
				_ = s.watcher.Add(filepath.Dir(source))
			} else {
				s.watchDirs(source, source)
			}
		}
		fmt.Fprintf(s.log, "editable %s: watching %s\n", pkg, root)
		go s.syncTree(tree)
		// The daemon has not read this tree yet: what changed while nobody watched is
		// found by reading it once now.
		wake(tree.kick)
	}
}

func sourceWatchNeeded(directory string, roots []string) bool {
	for _, root := range roots {
		if directory == root || directory == filepath.Dir(root) || strings.HasPrefix(directory, root+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

func (s *editableSync) syncTree(tree *editableTree) {
	// seen counts the events the last complete read already accounted for; a wake that
	// brings none is the tail of a burst that read consumed, not a new edit.
	var seen uint64
	read := false
	for {
		select {
		case <-s.quit:
			return
		case <-tree.stop:
			return
		case <-tree.kick:
		}
		if read && tree.events.Load() == seen {
			continue
		}
		seen, read = s.settle(tree), true
	}
}

// settle reads the tree until one whole read sees no new event, acts on what it read, and
// answers the event count that read stood on.
func (s *editableSync) settle(tree *editableTree) uint64 {
	for {
		seen := tree.events.Load()
		snapshot, problem := s.resolver.SnapshotEditable(tree.pkg)
		if snapshot == nil {
			fmt.Fprintf(s.log, "editable %s: %s\n", tree.pkg, problem.Message)
			return seen
		}
		if tree.events.Load() != seen {
			snapshot.Close()
			continue
		}
		if problem != nil {
			snapshot.Close()
			s.refused(tree.pkg, snapshot.Current.ID, problem)
			return seen
		}
		fmt.Fprintf(s.log, "editable %s: syncing source (%d files, %d B) after install %s\n",
			tree.pkg, snapshot.Files, snapshot.Bytes, short12(snapshot.Current.ID))
		installID, changed, problem := s.resolver.RefreshSnapshot(snapshot)
		snapshot.Close()
		if problem != nil {
			s.refused(tree.pkg, installID, problem)
			return seen
		}
		if changed {
			s.reprepare(tree.pkg, snapshot, installID)
		}
		return seen
	}
}

func (s *editableSync) refused(pkg, installID string, problem *exit.Error) {
	fmt.Fprintf(s.log, "editable %s: %s — %s; install %s stays active\n",
		pkg, problem.ErrName(), problem.Remedy, short12(installID))
	if e := s.store.AppendPackageEvent(pkg, "package.refresh_failed", map[string]any{
		"install": installID, "error": problem.ErrName(), "cause": problem.Remedy,
	}); e != nil {
		fmt.Fprintf(s.log, "editable %s: cannot record the refusal: %s\n", pkg, e.Message)
	}
}

// reprepare gives every worker that held the package its placement under the new install.
func (s *editableSync) reprepare(pkg string, snapshot *EditableSnapshot, installID string) {
	locals := s.owner.PackageHolders(pkg)
	var workers []string
	if len(locals) > 0 {
		stopped, problem := s.owner.UnloadIdleLocalPackage(pkg, installID)
		if problem != nil {
			fmt.Fprintf(s.log, "editable %s: stale local workers stay: %s\n", pkg, problem.Message)
		}
		for _, w := range stopped {
			fmt.Fprintf(s.log, "editable %s: stopped idle worker %s\n", pkg, w.InstanceID)
		}
		for _, models := range distinctSelections(locals) {
			spec, problem := s.resolver.ResolveInstall(installID, models)
			if problem != nil {
				fmt.Fprintf(s.log, "editable %s: cannot resolve install %s: %s\n", pkg, short12(installID), problem.Message)
				continue
			}
			instance, change, problem := s.owner.EnsureWorker(spec)
			if problem == nil && len(spec.Placement.Entrypoints) > 0 {
				problem = s.owner.EnsurePlacementReady(instance, spec.Placement.Entrypoints[0].Digest, "")
			}
			if problem != nil {
				fmt.Fprintf(s.log, "editable %s: local worker not re-prepared (%s); the next run starts it\n",
					pkg, problem.Message)
				continue
			}
			fmt.Fprintf(s.log, "editable %s: local worker %s %s and dispatchable for install %s\n",
				pkg, instance, change, short12(installID))
			workers = append(workers, "local/"+instance)
		}
	}
	if e := s.store.AppendPackageEvent(pkg, "package.refreshed", map[string]any{
		"install": installID, "files": snapshot.Files,
		"bytes": snapshot.Bytes, "workers": workers,
	}); e != nil {
		fmt.Fprintf(s.log, "editable %s: cannot record the refresh: %s\n", pkg, e.Message)
	}
	fmt.Fprintf(s.log, "editable %s: install %s active; %d worker(s) re-prepared\n",
		pkg, short12(installID), len(workers))
}

// distinctSelections answers each model selection the local holders were resolved with,
// once, so one worker per selection comes back under the new install.
func distinctSelections(locals []orchestrator.LocalHolder) [][]orchestrator.ModelRef {
	var out [][]orchestrator.ModelRef
	seen := map[string]bool{}
	for _, holder := range locals {
		key := selectedInstallKey("", holder.Models)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, holder.Models)
	}
	sort.Slice(out, func(i, j int) bool {
		return selectedInstallKey("", out[i]) < selectedInstallKey("", out[j])
	})
	return out
}
