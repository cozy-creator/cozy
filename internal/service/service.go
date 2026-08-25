// Package service owns ONE spelling of "is the LocalService running", and it is an OS
// fact rather than a file's contents.
//
// The running LocalService holds an exclusive advisory lock on `<home>/service.lock` for
// its whole life. The kernel drops that lock when the process dies — including under
// SIGKILL — so a reader that CAN take the lock has proof the service is gone, and one
// that cannot has proof of the opposite. There is no pidfile, no heartbeat and no grace
// period, which is exactly the class of bug cl#85/53/83/82 were: a sidecar that outlived
// its launcher and said COLD forever.
//
// The lock file's bytes are written by the holder and are meaningless without the lock:
// the address is data the live owner publishes, never evidence that it lives.
package service

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// DefaultPort is the loopback bind cozy up uses (cozy-creator.md: 2699, loopback only).
const DefaultPort = 2699

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
	st.Path = cfg.Home + "/service.lock"
	f, err := os.OpenFile(st.Path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		st.Details = "the service lock is unreadable: " + err.Error()
		return st
	}
	defer f.Close()
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err == nil {
		// Taking it IS the proof of absence. Release immediately: probing must never
		// look like holding.
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		st.Details = "the service lock is free — no LocalService owns this root"
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
	st.Details = "the service lock is held by pid " + strconv.Itoa(st.PID)
	return st
}

// Held is the live LocalService's own claim on this root. It exists only inside the
// process that is the service; nothing reads it, and nothing outlives it.
type Held struct{ f *os.File }

// Hold takes the root's exclusive claim and publishes the live addresses under it. A
// second `cozy up` on one root gets exit 13 here, before it can bind anything.
func Hold(l home.Layout, addr, socket string) (*Held, *exit.Error) {
	f, err := os.OpenFile(l.Service, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, exit.Internalf("cannot open the service lock %s: %s", l.Service, err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, exit.New(exit.Conflict,
			"another cozy LocalService already owns %s", l.Root).
			WithRemedy("one service per local root; stop it with `cozy down`").
			WithNext("cozy status", "cozy down")
	}
	body := fmt.Sprintf("addr=%s\nsocket=%s\npid=%d\nsince=%s\n",
		addr, socket, os.Getpid(), time.Now().UTC().Format(time.RFC3339))
	if err := f.Truncate(0); err != nil {
		f.Close()
		return nil, exit.Internalf("cannot publish the service address: %s", err)
	}
	if _, err := f.WriteAt([]byte(body), 0); err != nil {
		f.Close()
		return nil, exit.Internalf("cannot publish the service address: %s", err)
	}
	return &Held{f: f}, nil
}

// Release drops the claim. The kernel does this anyway, on any exit; this is only the
// polite spelling of it.
func (h *Held) Release() {
	if h == nil || h.f == nil {
		return
	}
	_ = syscall.Flock(int(h.f.Fd()), syscall.LOCK_UN)
	_ = h.f.Close()
	h.f = nil
}

// Unavailable is the typed refusal every server-dependent verb shares.
func (s State) Unavailable() *exit.Error {
	return exit.Unavailablef("the cozy LocalService is not running (%s)", s.Addr).
		WithRemedy("start it with `cozy up`; it binds %s, loopback only", s.Addr).
		WithNext("cozy up", "cozy status")
}
