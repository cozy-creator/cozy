// Package service is the CLI's ONE seam onto the LocalService (cl-001) and its
// local client API (cl-006). Until those land, Probe answers honestly: nothing is
// listening, so server-dependent verbs refuse typed-unavailable with `next: cozy up`.
// cl-001 replaces address discovery here (COZY_HOME-recorded address); no other
// package learns whether the service is up.
package service

import (
	"net"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// DefaultAddr is the loopback bind cozy up uses (cozy-creator.md: port 2699, loopback only).
const DefaultAddr = "127.0.0.1:2699"

const dialTimeout = 250 * time.Millisecond

type State struct {
	Addr string
	Up   bool
}

func Probe() State {
	st := State{Addr: DefaultAddr}
	conn, err := net.DialTimeout("tcp", st.Addr, dialTimeout)
	if err == nil {
		_ = conn.Close()
		st.Up = true
	}
	return st
}

// Unavailable is the typed refusal every server-dependent verb shares.
func (s State) Unavailable() *exit.Error {
	return exit.Unavailablef("the cozy LocalService is not running (%s)", s.Addr).
		WithRemedy("start it with `cozy up`; it binds %s, loopback only", s.Addr).
		WithNext("cozy up", "cozy status")
}
