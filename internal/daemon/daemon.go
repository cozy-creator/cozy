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
// the address is data the live owner publishes, never evidence that it lives.
package daemon

import (
	"fmt"
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
	Addr    string // the local client API address the live owner published
	Socket  string // the worker-protocol unix socket the live owner published
	PID     int
	Since   string
	Up      bool
	Path    string
	Details string // why the probe answered as it did, for status output
}

// Probe answers from the lock, never from a file's existence.
func Probe(cfg config.Config) State {
	st := State{Addr: fmt.Sprintf("127.0.0.1:%d", cfg.Port)}
	if cfg.Home == "" {
		st.Details = "no local root"
		return st
	}
	st.Path = cfg.Home + "/daemon.lock"
	f, err := os.OpenFile(st.Path, os.O_RDWR|os.O_CREATE, 0o644)
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
		}
	}
	st.Details = "the daemon lock is held by pid " + strconv.Itoa(st.PID)
	return st
}

// Held is the live Cozy daemon's own claim on this root. It exists only inside the
// daemon process; nothing reads it, and nothing outlives it.
type Held struct{ f *os.File }

// Hold takes the root's exclusive claim and publishes the live addresses under it. A
// second `cozy invoke list` on one root gets exit 13 here, before it can bind anything.
func Hold(l home.Layout, addr, socket string) (*Held, *exit.Error) {
	f, err := os.OpenFile(l.Daemon, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, exit.Internalf("cannot establish daemon ownership for %s: %s", l.Root, err)
	}
	if err := flock.Exclusive(f); err != nil {
		f.Close()
		return nil, exit.New(exit.Conflict,
			"another Cozy daemon already owns %s", l.Root).
			WithRemedy("one daemon per local root; stop it with `cozy down`").
			WithNext("cozy invoke list", "cozy down")
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
// polite spelling of it.
func (h *Held) Release() {
	if h == nil || h.f == nil {
		return
	}
	_ = flock.Release(h.f)
	_ = h.f.Close()
	h.f = nil
}

// Unavailable is the typed refusal every server-dependent verb shares.
func (s State) Unavailable() *exit.Error {
	return exit.Unavailablef("the Cozy daemon is not running (%s)", s.Addr).
		WithRemedy("start it with `cozy up`; it binds %s, loopback only", s.Addr).
		WithNext("cozy up")
}
