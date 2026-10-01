// Package userunit runs a long-lived process as its own systemd user unit where user systemd
// is available, so it never lives in, or ends with, the scope or cgroup of whatever started it
// (a terminal, an agent's session scope).
package userunit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

var available = sync.OnceValue(func() bool {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		return false
	}
	return exec.Command("systemctl", "--user", "show-environment").Run() == nil
})

// Available says whether this user has a systemd user manager to run units under.
func Available() bool { return available() }

// Name is the unit for one path's process: prefix-<hash of path>, or prefix alone when the
// path is the default one.
func Name(prefix, path string, isDefault bool) string {
	if isDefault {
		return prefix
	}
	sum := sha256.Sum256([]byte(path))
	return prefix + "-" + hex.EncodeToString(sum[:4])
}

// Spec is one process to run as a unit. Output names a file each stream appends to ("" is
// nothing); Fresh truncates Stderr at start instead.
type Spec struct {
	Unit, Dir, Stdout, Stderr string
	Argv, Env                 []string
	Fresh                     bool
}

// Start runs spec as its unit and answers whether a unit of that name was already running,
// which is then left as it is. The unit is collected when its process ends.
func Start(spec Spec) (bool, error) {
	if Running(spec.Unit) {
		return true, nil
	}
	_ = exec.Command("systemctl", "--user", "reset-failed", spec.Unit).Run()
	args := []string{"--user", "--unit=" + spec.Unit, "--collect", "--quiet", "-p", "StandardOutput=" + sink(spec.Stdout, false),
		"-p", "StandardError=" + sink(spec.Stderr, spec.Fresh)}
	if spec.Dir != "" {
		args = append(args, "--working-directory="+spec.Dir)
	}
	for _, pair := range spec.Env {
		args = append(args, "--setenv="+pair)
	}
	out, err := exec.Command("systemd-run", append(append(args, "--"), spec.Argv...)...).CombinedOutput()
	if err != nil {
		if Running(spec.Unit) {
			return true, nil // another caller started it first
		}
		return false, fmt.Errorf("systemd-run %s: %v: %s", spec.Unit, err, strings.TrimSpace(string(out)))
	}
	return false, nil
}

// Running says whether the unit is active, starting or reloading.
func Running(unit string) bool {
	state := Property(unit, "ActiveState")
	return state == "active" || state == "activating" || state == "reloading"
}

func sink(path string, fresh bool) string {
	switch {
	case path == "":
		return "null"
	case fresh:
		return "truncate:" + path
	}
	return "append:" + path
}

// Property is one property of a unit, "" when it cannot be read.
func Property(unit, name string) string {
	out, err := exec.Command("systemctl", "--user", "show", "-p", name, "--value", unit).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// MainPID is the unit's process, 0 when none runs.
func MainPID(unit string) int {
	pid, _ := strconv.Atoi(Property(unit, "MainPID"))
	return pid
}

// Stop ends a unit and its processes.
func Stop(unit string) error {
	return exec.Command("systemctl", "--user", "stop", unit).Run()
}
