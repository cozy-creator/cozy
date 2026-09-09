// Package daemon owns ONE spelling of "is the Cozy daemon running", and it is an OS
// fact rather than a file's contents.
//
// The running Cozy daemon holds an exclusive advisory lock on `<home>/daemon.lock` for
// its whole life. The kernel drops that lock when the process dies — including under
// SIGKILL — so a reader that CAN take the lock has proof the daemon is gone, and one
// that cannot has proof of the opposite. There is no pidfile, no heartbeat and no grace
// period, which is exactly the class of bug cl#85/53/83/82 were: a sidecar that outlived
// its launcher and said COLD forever.
//
// The lock file's bytes are written by the holder and are meaningless without the lock:
// the address is data the live owner publishes, never evidence that it lives. The same
// record carries the per-launch CLI token (cl-116): the file is mode 0600, rewritten on
// every launch, and there is no separate client.cred handoff file.
package daemon

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
)

type State struct {
	Addr          string // the local client API address the live owner published
	Socket        string // the worker-protocol unix socket the live owner published
	PID           int
	Since         string
	Up            bool
	Path          string
	Details       string // why the probe answered as it did, for status output
	SchemaVersion int    // local lifecycle-store generation published by the daemon
}

// Probe answers from the lock, never from a file's existence.
func Probe(cfg config.Config) State {
	st := State{Addr: fmt.Sprintf("127.0.0.1:%d", cfg.Port)}
	if cfg.Home == "" {
		st.Details = "no local root"
		return st
	}
	st.Path = cfg.Home + "/daemon.lock"
	f, err := os.OpenFile(st.Path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		st.Details = "the daemon lock is unreadable: " + err.Error()
		return st
	}
	defer f.Close()
	if err := flock.Exclusive(f); err == nil {
		// Taking it IS the proof of absence. Release immediately: probing must never
		// look like holding.
		_ = flock.Release(f)
		st.Details = "the daemon lock is free — no Cozy daemon owns this root"
		return st
	}
	st.Up = true
	data, _ := os.ReadFile(st.Path)
	for _, line := range strings.Split(string(data), "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		switch key {
		case "addr":
			st.Addr = value
		case "socket":
			st.Socket = value
		case "pid":
			st.PID, _ = strconv.Atoi(value)
		case "since":
			st.Since = value
		case "schema":
			st.SchemaVersion, _ = strconv.Atoi(value)
		}
	}
	st.Details = "the daemon lock is held by pid " + strconv.Itoa(st.PID)
	return st
}

// Held is the live Cozy daemon's own claim on this root. It exists only inside the
// daemon process; nothing reads it, and nothing outlives it.
type Held struct{ f *os.File }

// PublishSchema records the store generation only after the daemon has opened and
// migrated the database. A missing value identifies an older daemon to a newer CLI.
func (h *Held) PublishSchema(version int) *exit.Error {
	if h == nil || h.f == nil || version <= 0 {
		return exit.Internalf("cannot publish daemon schema version")
	}
	if _, err := h.f.WriteString(fmt.Sprintf("schema=%d\n", version)); err != nil {
		return exit.Internalf("cannot publish daemon schema version: %s", err)
	}
	return nil
}

// Hold takes the root's exclusive claim and publishes the live addresses under it. A
// second `cozy run list` on one root gets exit 13 here, before it can bind anything.
// The body is rewritten on every launch; api.Mint appends the per-launch CLI token to
// the same record, which is why the file is 0600 and why no client.cred exists.
func Hold(l home.Layout, addr, socket string) (*Held, *exit.Error) {
	f, err := os.OpenFile(l.Daemon, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot establish daemon ownership for %s: %s", l.Root, err)
	}
	// Chmod anyway: O_CREAT's mode is masked by umask, and a prior build's 0644 lock
	// file would otherwise keep its old width under the new token-carrying body.
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return nil, exit.Internalf("cannot protect the daemon record: %s", err)
	}
	if err := flock.Exclusive(f); err != nil {
		f.Close()
		return nil, exit.New(exit.Conflict,
			"another Cozy daemon already owns %s", l.Root).
			WithRemedy("one daemon per local root; stop it with `cozy down`").
			WithNext("cozy run list", "cozy down")
	}
	body := fmt.Sprintf("addr=%s\nsocket=%s\npid=%d\nsince=%s\n",
		addr, socket, os.Getpid(), time.Now().UTC().Format(time.RFC3339))
	if err := f.Truncate(0); err != nil {
		f.Close()
		return nil, exit.Internalf("cannot publish the daemon address: %s", err)
	}
	if _, err := f.WriteAt([]byte(body), 0); err != nil {
		f.Close()
		return nil, exit.Internalf("cannot publish the daemon address: %s", err)
	}
	return &Held{f: f}, nil
}

// Release drops the claim. The kernel does this anyway, on any exit; this is only the
// polite spelling of it. The published body — the address and the CLI token api.Mint
// appended — is truncated first: a credential must not outlive its daemon on disk, and
// a client polling the record must never read a dead launch's token as a live one. A
// SIGKILL skips this, which is why clients also compare the body across a relaunch.
func (h *Held) Release() {
	if h == nil || h.f == nil {
		return
	}
	_ = h.f.Truncate(0)
	_ = flock.Release(h.f)
	_ = h.f.Close()
	h.f = nil
}

// Claimed reports whether the path this daemon published its claim at still names the
// file it holds. The lock file IS the claim (see the package comment): a client reaches
// this daemon only by opening `<home>/daemon.lock` and failing to take the lock. If the
// root was deleted, moved, or wiped out from under the running daemon, that path no
// longer resolves to this file — nothing can find this process, nothing can stop it, and
// nothing it writes will ever be read again.
//
// The comparison is by identity, not existence: a root that was removed and recreated is
// the same absence as one that was removed, and a second daemon that took the recreated
// lock must not be mistaken for this one. Any answer other than "the path is not there"
// — an unreadable directory, a filesystem in trouble — reads as still claimed: this is a
// reason to leave, so it must never fire on an answer the kernel could not give.
func (h *Held) Claimed() bool {
	if h == nil || h.f == nil {
		return false
	}
	mine, err := h.f.Stat()
	if err != nil {
		return true
	}
	published, err := os.Stat(h.f.Name())
	if err != nil {
		return !errors.Is(err, fs.ErrNotExist)
	}
	return os.SameFile(mine, published)
}

// Unavailable is the typed refusal every server-dependent verb shares.
func (s State) Unavailable() *exit.Error {
	return exit.Unavailablef("the Cozy daemon is not running (%s)", s.Addr).
		WithRemedy("start it with `cozy up`; it binds %s, loopback only", s.Addr).
		WithNext("cozy up")
}
