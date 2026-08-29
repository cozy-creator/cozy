package cli

import (
	"bytes"
	"fmt"
	"io"
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

// RunController owns the persistent loopback service. The public CLI starts this
// entrypoint automatically and never exposes an up/down command.
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
	if problem := handleUp(ctx); problem != nil {
		fmt.Fprintln(stderr, problem.Error())
		return 1
	}
	return 0
}

func ensureController(ctx *Context) (service.State, *exit.Error) {
	if state := service.Probe(ctx.Cfg); state.Up {
		return state, nil
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return service.State{}, problem
	}
	staleCredential, _ := os.ReadFile(layout.Client)
	self, err := os.Executable()
	if err != nil {
		return service.State{}, exit.Internalf("cannot locate the Cozy executable: %s", err)
	}
	logFile, err := os.OpenFile(filepath.Join(layout.Root, "controller.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return service.State{}, exit.Internalf("cannot open the controller log: %s", err)
	}
	defer logFile.Close()
	command := exec.Command(self)
	command.Args[0] = controllerProcessName
	command.Env = ctx.Cfg.Child("COZY_HOME=" + ctx.Cfg.Home)
	command.Stdout, command.Stderr = logFile, logFile
	detachProcess(command)
	if err := command.Start(); err != nil {
		return service.State{}, exit.Internalf("cannot start the Cozy controller: %s", err)
	}
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	var childErr error

	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case err := <-done:
			// Another concurrent auto-start may be the winner. Keep observing the
			// shared lock/credential readiness until it appears or the startup bound ends.
			childErr, done = err, nil
		case <-tick.C:
			if state := service.Probe(ctx.Cfg); state.Up {
				current, _ := os.ReadFile(layout.Client)
				if !bytes.Equal(current, staleCredential) {
					if _, problem := api.ClientCredential(layout); problem == nil {
						return state, nil
					}
				}
			}
		case <-deadline.C:
			return service.State{}, exit.New(exit.Conflict,
				"the Cozy controller did not acquire its service lock").
				WithRemedy("inspect %s (child result: %v)", filepath.Join(layout.Root, "controller.log"), childErr)
		}
	}
}
