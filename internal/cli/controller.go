package cli

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/service"
)

const controllerProcessName = "cozy-controller"

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
		return 1
	}
	ctx := &Context{
		Inv: &Invocation{Bools: map[string]bool{}, Values: values(
			"--port", intText(cfg.Port), "--yield", cfg.Yield), Mode: output.Mode{}},
		Out: stdout, Err: stderr, Cfg: cfg,
	}
	if problem := serveController(ctx); problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		return 1
	}
	return 0
}

func ensureController(ctx *Context) (service.State, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return service.State{}, problem
	}
	staleCredential, _ := os.ReadFile(layout.Client)
	var done <-chan error
	started := false
	var childErr error

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
					return state, nil
				}
			}
		} else if !started {
			child, err := startController(ctx, layout)
			if err != nil {
				return service.State{}, err
			}
			started = true
			done = child
		}
		select {
		case err := <-done:
			// Another concurrent auto-start may be the winner. Keep observing the
			// shared lock/credential readiness until it appears or the startup bound ends.
			childErr, done = err, nil
		case <-tick.C:
		case <-deadline.C:
			return service.State{}, exit.New(exit.Conflict,
				"the Cozy controller did not acquire its service lock").
				WithRemedy("the background controller child returned: %v", childErr)
		}
	}
}

func startController(ctx *Context, layout home.Layout) (<-chan error, *exit.Error) {
	self, err := os.Executable()
	if err != nil {
		return nil, exit.Internalf("cannot locate the Cozy executable: %s", err)
	}
	// The controller is a background product process, not an attached Compose-style
	// log producer. Runtime/request diagnostics are structured state; process chatter
	// has no persistent user-facing log surface.
	_ = os.Remove(filepath.Join(layout.Root, "controller.log"))
	diagnostics, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return nil, exit.Internalf("cannot open the null diagnostics sink: %s", err)
	}
	command := exec.Command(self)
	command.Args[0] = controllerProcessName
	command.Env = ctx.Cfg.Child("COZY_HOME=" + ctx.Cfg.Home)
	command.Stdout, command.Stderr = diagnostics, diagnostics
	detachProcess(command)
	if err := command.Start(); err != nil {
		diagnostics.Close()
		return nil, exit.Internalf("cannot start the Cozy controller: %s", err)
	}
	diagnostics.Close()
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	return done, nil
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
