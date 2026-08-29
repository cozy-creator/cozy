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
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

const controllerProcessName = "cozy-controller"

const maxControllerStartupDiagnostic = 16 << 10

type controllerExit struct {
	err        error
	code       exit.Code
	diagnostic string
}

type controllerChild struct {
	pid              int
	done             <-chan controllerExit
	closeDiagnostics func()
}

// ControllerProcess reports whether argv0 names the binary's private controller
// entrypoint. It is a process boundary, not a hidden public command.
func ControllerProcess(argv0 string) bool {
	return filepath.Base(argv0) == controllerProcessName
}

// RunController owns the persistent loopback service. Public `up` and stateful
// commands start this private process entrypoint.
func RunController(stdout, stderr io.Writer) int {
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
	if problem := serveController(ctx); problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		if problem.Code.Valid() && problem.Code != exit.OK {
			return int(problem.Code)
		}
		return int(exit.Internal)
	}
	return 0
}

func ensureController(ctx *Context) (service.State, bool, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return service.State{}, false, problem
	}
	staleCredential, _ := os.ReadFile(layout.Client)
	var child *controllerChild
	var childResult *controllerExit
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
		if state := service.Probe(ctx.Cfg); state.Up {
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
			return service.State{}, false, controllerStartupFailure(*childResult)
		} else if !started {
			spawned, err := startController(ctx)
			if err != nil {
				return service.State{}, false, err
			}
			started = true
			child = spawned
		}
		var done <-chan controllerExit
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
				return service.State{}, false, controllerStartupFailure(*childResult)
			}
			return service.State{}, false, exit.Named(exit.Conflict,
				"controller_startup_incomplete",
				"the Cozy controller did not become healthy within 15s").
				WithRemedy("retry `cozy up`; startup is complete only after the authenticated API and web UI answer")
		}
	}
}

func startController(ctx *Context) (*controllerChild, *exit.Error) {
	self, err := os.Executable()
	if err != nil {
		return nil, exit.Internalf("cannot locate the Cozy executable: %s", err)
	}
	// The controller is a background product process, not an attached Compose-style
	// log producer. Runtime/request diagnostics are structured state; process chatter
	// has no persistent user-facing log surface.
	discard, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return nil, exit.Internalf("cannot open the null diagnostics sink: %s", err)
	}
	command := exec.Command(self)
	command.Args[0] = controllerProcessName
	command.Env = ctx.Cfg.Child("COZY_HOME=" + ctx.Cfg.Home)
	command.Stdout = discard
	diagnostics, err := command.StderrPipe()
	if err != nil {
		discard.Close()
		return nil, exit.Internalf("cannot open the controller startup diagnostic pipe: %s", err)
	}
	detachProcess(command)
	if err := command.Start(); err != nil {
		diagnostics.Close()
		discard.Close()
		return nil, exit.Internalf("cannot start the Cozy controller: %s", err)
	}
	discard.Close()
	diagnosticDone := make(chan string, 1)
	go func() { diagnosticDone <- readBoundedDiagnostic(diagnostics) }()
	done := make(chan controllerExit, 1)
	go func() {
		waitErr := command.Wait()
		result := controllerExit{err: waitErr, code: exit.Internal, diagnostic: <-diagnosticDone}
		var processExit *exec.ExitError
		if errors.As(waitErr, &processExit) {
			if code := exit.Code(processExit.ExitCode()); code.Valid() && code != exit.OK {
				result.code = code
			}
		}
		done <- result
	}()
	var closeOnce sync.Once
	return &controllerChild{
		pid: command.Process.Pid, done: done,
		closeDiagnostics: func() { closeOnce.Do(func() { _ = diagnostics.Close() }) },
	}, nil
}

func readBoundedDiagnostic(reader io.Reader) string {
	kept := make([]byte, 0, maxControllerStartupDiagnostic)
	buffer := make([]byte, 4096)
	truncated := false
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			remaining := maxControllerStartupDiagnostic - len(kept)
			if remaining > 0 {
				if n < remaining {
					remaining = n
				}
				kept = append(kept, buffer[:remaining]...)
			}
			if len(kept) == maxControllerStartupDiagnostic && n > remaining {
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

func controllerStartupFailure(result controllerExit) *exit.Error {
	diagnostic := strings.TrimSpace(result.diagnostic)
	if diagnostic == "" && result.err != nil {
		diagnostic = result.err.Error()
	}
	if diagnostic == "" {
		diagnostic = "the background controller exited before publishing a healthy service"
	}
	return exit.Named(result.code, "controller_startup_failed",
		"Cozy controller startup failed: %s", diagnostic).
		WithRemedy("correct the reported startup condition and retry `cozy up`; no persistent controller log was created")
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
