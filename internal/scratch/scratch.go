// Package scratch is the ONE spelling of "a temporary directory owned by a live
// process". Every entry under tmp/ is claimed through it: the owner holds a kernel lock
// on `<dir>/.claim` for as long as it needs the bytes, and the lock dies with the
// process, SIGKILL included. The owner removes the directory the moment the bytes are
// dead (Release); the daemon's start-time sweep is the backstop for a writer that
// crashed — observed liveness and the request's recorded state, never elapsed time.
package scratch

import (
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
)

const claimName = ".claim"

// Dir is one claimed scratch directory. Release removes it.
type Dir struct {
	Path  string
	claim *os.File
}

// Temp creates a fresh, uniquely named directory under root and claims it — a verb's
// scratch, alive exactly as long as the verb.
func Temp(root, prefix string) (*Dir, *exit.Error) {
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, exit.Internalf("cannot create the scratch root %s: %s", root, err)
	}
	path, err := os.MkdirTemp(root, prefix)
	if err != nil {
		return nil, exit.Internalf("cannot create scratch under %s: %s", root, err)
	}
	dir, problem := claim(path)
	if problem != nil {
		_ = os.RemoveAll(path)
		return nil, problem
	}
	return dir, nil
}

// Named claims `<root>/<id>` — a request's scratch, named by the request so a restarted
// daemon resumes what a crashed one left (a verified `.part` continues; anything else is
// re-derived) and the sweep can read the request's state for it. What a previous holder
// left is kept: the claim proves nothing live holds it, and every reader re-verifies
// bytes before trusting them.
func Named(root, id string) (*Dir, *exit.Error) {
	if id == "" || filepath.Base(id) != id || id == "." || id == ".." {
		return nil, exit.Internalf("refusing a scratch entry with unsafe id %q", id)
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, exit.Internalf("cannot create the scratch root %s: %s", root, err)
	}
	path := filepath.Join(root, id)
	if err := os.Mkdir(path, 0o700); err != nil && !os.IsExist(err) {
		return nil, exit.Internalf("cannot create scratch %s: %s", path, err)
	}
	return claim(path)
}

func claim(path string) (*Dir, *exit.Error) {
	file, err := os.OpenFile(filepath.Join(path, claimName), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the scratch claim in %s: %s", path, err)
	}
	if err := flock.Exclusive(file); err != nil {
		_ = file.Close()
		return nil, exit.Named(exit.Conflict, "scratch_held",
			"scratch %s is held by another live process", path)
	}
	return &Dir{Path: path, claim: file}, nil
}

// Release drops the claim and removes the directory. Safe to call more than once.
func (d *Dir) Release() {
	if d == nil {
		return
	}
	if d.claim != nil {
		_ = flock.Release(d.claim)
		_ = d.claim.Close()
		d.claim = nil
	}
	_ = os.RemoveAll(d.Path)
}

// Held answers whether a live process holds the claim on one directory. A directory
// without a claim file — or a plain file — is held by nobody.
func Held(path string) bool {
	file, err := os.OpenFile(filepath.Join(path, claimName), os.O_RDWR, 0o600)
	if err != nil {
		return false
	}
	defer file.Close()
	if err := flock.Exclusive(file); err != nil {
		return true
	}
	_ = flock.Release(file)
	return false
}
