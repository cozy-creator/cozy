package cli

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/daemon"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/output"
)

const daemonProcessName = "cozy-daemon"

const maxDaemonStartupDiagnostic = 16 << 10

type daemonExit struct {
	err        error
	code       exit.Code
	diagnostic string
}

type daemonChild struct {
	pid              int
	done             <-chan daemonExit
	closeDiagnostics func()
}

// DaemonProcess reports whether argv0 names the binary's private daemon
// entrypoint. It is a process boundary, not a hidden public command.
func DaemonProcess(argv0 string) bool {
	return filepath.Base(argv0) == daemonProcessName
}

// RunDaemon owns the persistent loopback daemon. Public `up` and stateful
// commands start this private process entrypoint.
func RunDaemon(stdout, stderr io.Writer) int {
	cfg, problem := config.Load()
	if problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		if problem.Code.Valid() && problem.Code != exit.OK {
			return int(problem.Code)
		}
		return int(exit.Internal)
	}
	ctx := &Context{
		Inv: &Invocation{Bools: map[string]bool{}, Values: values(
			"--port", intText(cfg.Port), "--yield", cfg.Yield), Mode: output.Mode{}},
		Out: stdout, Err: stderr, Cfg: cfg,
	}
	if problem := serveDaemon(ctx); problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		if problem.Code.Valid() && problem.Code != exit.OK {
			return int(problem.Code)
		}
		return int(exit.Internal)
	}
	return 0
}

func ensureDaemon(ctx *Context) (daemon.State, bool, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return daemon.State{}, false, problem
	}
	staleCredential, _ := os.ReadFile(layout.Client)
	var child *daemonChild
	var childResult *daemonExit
	started := false
	defer func() {
		if child != nil {
			child.closeDiagnostics()
		}
	}()

	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if state := daemon.Probe(ctx.Cfg); state.Up {
			current, _ := os.ReadFile(layout.Client)
			credentialReady := !started || !bytes.Equal(current, staleCredential)
			if credentialReady {
				if _, problem := api.ClientCredential(layout); problem == nil && uiReady(state.Addr) {
					changed := child != nil && state.PID == child.pid
					if child != nil {
						child.closeDiagnostics()
						child = nil
					}
					return state, changed, nil
				}
			}
		} else if childResult != nil {
			return daemon.State{}, false, daemonStartupFailure(*childResult)
		} else if !started {
			spawned, err := startDaemon(ctx)
			if err != nil {
				return daemon.State{}, false, err
			}
			started = true
			child = spawned
		}
		var done <-chan daemonExit
		if child != nil {
			done = child.done
		}
		select {
		case result := <-done:
			// Another concurrent auto-start may be the winner. Keep observing the
			// shared lock/credential readiness until it appears or the startup bound ends.
			childResult, child = &result, nil
		case <-tick.C:
		case <-deadline.C:
			if childResult != nil {
				return daemon.State{}, false, daemonStartupFailure(*childResult)
			}
			return daemon.State{}, false, exit.Named(exit.Conflict,
				"daemon_startup_incomplete",
				"the Cozy daemon did not become healthy within 15s").
				WithRemedy("retry `cozy up`; startup is complete only after the authenticated API and web UI answer")
		}
	}
}

func startDaemon(ctx *Context) (*daemonChild, *exit.Error) {
	self, err := os.Executable()
	if err != nil {
		return nil, exit.Internalf("cannot locate the Cozy executable: %s", err)
	}
	// The daemon is a background product process, not an attached Compose-style
	// log producer. Runtime/request diagnostics are structured state; process chatter
	// has no persistent user-facing log surface.
	discard, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return nil, exit.Internalf("cannot open the null diagnostics sink: %s", err)
	}
	command := exec.Command(self)
	command.Args[0] = daemonProcessName
	command.Env = ctx.Cfg.Child("COZY_HOME=" + ctx.Cfg.Home)
	command.Stdout = discard
	diagnostics, err := command.StderrPipe()
	if err != nil {
		discard.Close()
		return nil, exit.Internalf("cannot open the daemon startup diagnostic pipe: %s", err)
	}
	detachProcess(command)
	if err := command.Start(); err != nil {
		diagnostics.Close()
		discard.Close()
		return nil, exit.Internalf("cannot start the Cozy daemon: %s", err)
	}
	discard.Close()
	diagnosticDone := make(chan string, 1)
	go func() { diagnosticDone <- readBoundedDiagnostic(diagnostics) }()
	done := make(chan daemonExit, 1)
	go func() {
		waitErr := command.Wait()
		result := daemonExit{err: waitErr, code: exit.Internal, diagnostic: <-diagnosticDone}
		var processExit *exec.ExitError
		if errors.As(waitErr, &processExit) {
			if code := exit.Code(processExit.ExitCode()); code.Valid() && code != exit.OK {
				result.code = code
			}
		}
		done <- result
	}()
	var closeOnce sync.Once
	return &daemonChild{
		pid: command.Process.Pid, done: done,
		closeDiagnostics: func() { closeOnce.Do(func() { _ = diagnostics.Close() }) },
	}, nil
}

func readBoundedDiagnostic(reader io.Reader) string {
	kept := make([]byte, 0, maxDaemonStartupDiagnostic)
	buffer := make([]byte, 4096)
	truncated := false
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			remaining := maxDaemonStartupDiagnostic - len(kept)
			if remaining > 0 {
				if n < remaining {
					remaining = n
				}
				kept = append(kept, buffer[:remaining]...)
			}
			if len(kept) == maxDaemonStartupDiagnostic && n > remaining {
				truncated = true
			}
		}
		if err != nil {
			break
		}
	}
	diagnostic := strings.TrimSpace(string(kept))
	if truncated {
		diagnostic += "\n… (startup diagnostic truncated)"
	}
	return diagnostic
}

func daemonStartupFailure(result daemonExit) *exit.Error {
	diagnostic := strings.TrimSpace(result.diagnostic)
	if diagnostic == "" && result.err != nil {
		diagnostic = result.err.Error()
	}
	if diagnostic == "" {
		diagnostic = "the Cozy daemon exited before publishing a healthy API"
	}
	return exit.Named(result.code, "daemon_startup_failed",
		"Cozy daemon startup failed: %s", diagnostic).
		WithRemedy("correct the reported startup condition and retry `cozy up`; no persistent daemon log was created")
}

func uiReady(address string) bool {
	client := &http.Client{Timeout: 250 * time.Millisecond}
	response, err := client.Get("http://" + address + "/")
	if err != nil {
		return false
	}
	_ = response.Body.Close()
	return response.StatusCode == http.StatusOK
}
