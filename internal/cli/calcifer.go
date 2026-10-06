package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/userunit"
)

const calciferProcessName = "calcifer"

const maxCalciferStartupDiagnostic = 16 << 10

type calciferExit struct {
	err        error
	code       exit.Code
	diagnostic string
}

type calciferChild struct {
	pid              int
	done             <-chan calciferExit
	closeDiagnostics func()
}

// CalciferProcess reports whether argv0 names the binary's private daemon
// entrypoint. It is a process boundary, not a hidden public command.
func CalciferProcess(argv0 string) bool {
	return strings.TrimSuffix(filepath.Base(argv0), ".exe") == calciferProcessName
}

// RunCalcifer owns the persistent loopback daemon. Public `up` and stateful
// commands start this private process entrypoint. stderr is the parent's startup
// diagnostic pipe; everything else the daemon says goes to its own log.
func RunCalcifer(stderr io.Writer) int {
	ignoreCalciferBrokenPipe()
	cfg, problem := config.Load()
	if problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		if problem.Code.Valid() && problem.Code != exit.OK {
			return int(problem.Code)
		}
		return int(exit.Internal)
	}
	if err := raiseOpenFileLimit(); err != nil {
		fmt.Fprintf(stderr, "cannot raise calcifer's open-file limit: %v\n", err)
		return int(exit.Internal)
	}
	// THE DAEMON'S OWN LOG (cl-096). The process is detached from any terminal, so its
	// words — the orchestrator's frame-by-frame account above all — go to
	// $COZY_HOME/daemon.log, bounded by rotation on observed size (daemon.LogBytes, one
	// predecessor). `cozy daemon log` reads it. stdout is never written: the parent hands
	// this process /dev/null there, and a hand-started daemon needs no redirect.
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		fmt.Fprintln(stderr, string(startupRefusal{Error: problem}.encode()))
		return int(exit.Internal)
	}
	log, err := calcifer.OpenLog(layout.Log)
	if err != nil {
		fmt.Fprintln(stderr, string(startupRefusal{Error: exit.Internalf("%s", err)}.encode()))
		return int(exit.Internal)
	}
	defer log.Close()
	ctx := &Context{
		Inv: &Invocation{Bools: map[string]bool{}, Values: values(
			"--port", intText(cfg.Port), "--yield", cfg.Yield), Mode: output.Mode{}},
		Out: log, Err: stderr, Cfg: cfg, AccountAuth: accountauth.New(cfg),
	}
	if problem := serveCalcifer(ctx); problem != nil {
		// The parent `cozy up` reads this pipe: one typed document, so the refusal reaches
		// the human under its own name and remedy rather than as startup prose. The log
		// keeps the same words, so the refusal outlives the pipe.
		fmt.Fprintln(stderr, string(startupRefusal{Error: problem}.encode()))
		fmt.Fprintf(log, "calcifer refused to start: %s\n", problem.Error())
		if problem.Code.Valid() && problem.Code != exit.OK {
			return int(problem.Code)
		}
		return int(exit.Internal)
	}
	return 0
}

// startupRefusal is the daemon's last word on stderr when it refuses to start: the typed
// error itself, so `calciferStartupFailure` re-raises it verbatim.
type startupRefusal struct {
	Error *exit.Error `json:"error"`
}

func (r startupRefusal) encode() []byte {
	data, err := json.Marshal(r)
	if err != nil {
		return []byte(r.Error.Error())
	}
	return data
}

// startupRefusalOf reads the daemon's typed refusal back out of its startup diagnostic.
func startupRefusalOf(diagnostic string) *exit.Error {
	lines := strings.Split(strings.TrimSpace(diagnostic), "\n")
	var doc startupRefusal
	if json.Unmarshal([]byte(lines[len(lines)-1]), &doc) != nil || doc.Error == nil ||
		doc.Error.Message == "" || !doc.Error.Code.Valid() {
		return nil
	}
	return doc.Error
}

func ensureCalcifer(ctx *Context) (calcifer.State, bool, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return calcifer.State{}, false, problem
	}
	// The daemon record carries the per-launch token; a relaunch rewrites the whole
	// body, so "the bytes changed" is exactly "a fresh daemon minted a fresh token".
	staleCredential, _ := os.ReadFile(layout.Daemon)
	var child *calciferChild
	var childResult *calciferExit
	started := false
	defer func() {
		if child != nil {
			child.closeDiagnostics()
		}
	}()

	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	// Readiness follows the live owner, not elapsed time: migrations and startup
	// may legitimately take longer on a busy host. The caller can interrupt its
	// wait without stopping the detached daemon; child exits still report failure.
	for {
		if state := calcifer.Probe(ctx.Cfg); state.Up {
			if state.OperatorOwned {
				return calcifer.State{}, false, exit.Named(exit.Conflict, "daemon.operator_owned",
					"an operator process holds this Cozy root without starting a daemon").
					WithRemedy("finish or stop the operator handoff before starting Cozy")
			}
			current, _ := os.ReadFile(layout.Daemon)
			credentialReady := !started || !bytes.Equal(current, staleCredential)
			// Schema generations belong to the records reader, not this API client.
			// Missing/newer schema metadata never authorizes replacing a live owner.
			if state.Addr != "" && credentialReady {
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
			return calcifer.State{}, false, calciferStartupFailure(*childResult)
		} else if !started {
			spawned, err := startCalcifer(ctx)
			if err != nil {
				return calcifer.State{}, false, err
			}
			started = true
			child = spawned
		}
		var done <-chan calciferExit
		if child != nil {
			done = child.done
		}
		select {
		case result := <-done:
			// A clean child exit is the idempotent "another owner" path. A probe
			// can briefly hold that same flock, so re-elect if no owner remains.
			// Never replace a live owner.
			childResult, child = &result, nil
			if result.err == nil {
				started, childResult = false, nil
			}
		case <-tick.C:
		}
	}
}

func startCalcifer(ctx *Context) (*calciferChild, *exit.Error) {
	self, err := os.Executable()
	if err != nil {
		return nil, exit.Internalf("cannot locate the Cozy executable: %s", err)
	}
	entry, err := calciferEntry(self, ctx.Cfg.Home)
	if err != nil {
		return nil, exit.Internalf("cannot name Calcifer executable: %s", err)
	}
	// The daemon is a background product process, not an attached Compose-style log
	// producer: its stdout is nothing. Its words go to $COZY_HOME/daemon.log (RunCalcifer),
	// read with `cozy daemon log`; only the startup refusal crosses the stderr pipe.
	discard, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return nil, exit.Internalf("cannot open the null diagnostics sink: %s", err)
	}
	// The daemon serves every hub and each request names its own. Its default is the
	// configured hub, never a one-command --tensorhub selection.
	env := ctx.Cfg.Child("COZY_HOME="+ctx.Cfg.Home, "TENSORHUB_URL="+ctx.Cfg.ConfiguredHubURL)
	if userunit.Available() {
		discard.Close()
		return startCalciferUnit(ctx, entry, env)
	}
	command := exec.Command(entry)
	command.Args[0] = calciferProcessName
	command.Env = env
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
		return nil, exit.Internalf("cannot start Calcifer: %s", err)
	}
	discard.Close()
	diagnosticDone := make(chan string, 1)
	go func() { diagnosticDone <- readBoundedDiagnostic(diagnostics) }()
	done := make(chan calciferExit, 1)
	go func() {
		waitErr := command.Wait()
		result := calciferExit{err: waitErr, code: exit.Internal, diagnostic: <-diagnosticDone}
		var processExit *exec.ExitError
		if errors.As(waitErr, &processExit) {
			if code := exit.Code(processExit.ExitCode()); code.Valid() && code != exit.OK {
				result.code = code
			}
		}
		done <- result
	}()
	var closeOnce sync.Once
	return &calciferChild{
		pid: command.Process.Pid, done: done,
		closeDiagnostics: func() { closeOnce.Do(func() { _ = diagnostics.Close() }) },
	}, nil
}

// CalciferUnit is the systemd user unit a home's daemon runs as: calcifer for ~/.cozy.
func CalciferUnit(home string) string {
	return userunit.Name(calciferProcessName, home, home == config.DefaultHome())
}

// calciferEntry retains this executable under the controller's own name. Executing this
// path gives both detached and service launches the same OS process identity.
func calciferEntry(self, root string) (string, error) {
	entry := filepath.Join(root, calciferProcessName)
	if runtime.GOOS == "windows" {
		entry += ".exe"
	}
	source, err := os.Stat(self)
	if err != nil {
		return "", err
	}
	if held, err := os.Stat(entry); err == nil && os.SameFile(source, held) {
		return entry, nil
	}
	next := entry + ".new" + strconv.Itoa(os.Getpid())
	defer os.Remove(next)
	if err := os.Link(self, next); err != nil {
		input, err := os.Open(self)
		if err != nil {
			return "", err
		}
		defer input.Close()
		file, err := os.OpenFile(next, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o755)
		if err != nil {
			return "", err
		}
		_, err = io.Copy(file, input)
		closed := file.Close()
		if err != nil {
			return "", err
		}
		if closed != nil {
			return "", closed
		}
	}
	return entry, os.Rename(next, entry)
}

// startCalciferUnit runs the daemon as its own user unit, outside the caller's scope and
// cgroup: ending the session that started it never ends it, and `cozy down` ends the unit.
// A unit already running is the daemon, starting. Its startup refusal is its stderr, kept in
// daemon-startup.log.
func startCalciferUnit(ctx *Context, entry string, env []string) (*calciferChild, *exit.Error) {
	unit, startup := CalciferUnit(ctx.Cfg.Home), filepath.Join(ctx.Cfg.Home, "daemon-startup.log")
	if _, err := userunit.Start(userunit.Spec{Unit: unit, Argv: []string{entry}, Env: env, Stderr: startup, Fresh: true}); err != nil {
		return nil, exit.Internalf("cannot start Calcifer unit: %s", err)
	}
	done, stop := make(chan calciferExit, 1), make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(200 * time.Millisecond):
			}
			if state := userunit.Property(unit, "ActiveState"); state == "active" || state == "activating" {
				continue
			}
			raw, _ := os.ReadFile(startup)
			result := calciferExit{code: exit.Internal, diagnostic: strings.TrimSpace(string(raw[:min(len(raw), maxCalciferStartupDiagnostic)]))}
			// A unit that ended in a word refused; one that ended silently found another owner.
			if result.diagnostic != "" {
				result.err = errors.New("Calcifer unit ended")
			}
			done <- result
			return
		}
	}()
	var closeOnce sync.Once
	return &calciferChild{pid: userunit.MainPID(unit), done: done,
		closeDiagnostics: func() { closeOnce.Do(func() { close(stop) }) }}, nil
}

func readBoundedDiagnostic(reader io.Reader) string {
	kept := make([]byte, 0, maxCalciferStartupDiagnostic)
	buffer := make([]byte, 4096)
	truncated := false
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			remaining := maxCalciferStartupDiagnostic - len(kept)
			if remaining > 0 {
				if n < remaining {
					remaining = n
				}
				kept = append(kept, buffer[:remaining]...)
			}
			if len(kept) == maxCalciferStartupDiagnostic && n > remaining {
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

func calciferStartupFailure(result calciferExit) *exit.Error {
	if refusal := startupRefusalOf(result.diagnostic); refusal != nil {
		return refusal
	}
	diagnostic := strings.TrimSpace(result.diagnostic)
	if diagnostic == "" && result.err != nil {
		diagnostic = result.err.Error()
	}
	if diagnostic == "" {
		diagnostic = "Calcifer exited before publishing a healthy API"
	}
	return exit.Named(result.code, "daemon_startup_failed",
		"Calcifer startup failed: %s", diagnostic).
		WithRemedy("correct the reported startup condition and retry `cozy up`; `cozy daemon log` keeps the daemon's own words")
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
