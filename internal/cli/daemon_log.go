package cli

import (
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

// handleDaemonLog prints the daemon's own log verbatim, and with --follow keeps printing
// what the daemon writes until interrupted. It reads a file, not the daemon: the words
// are there whether or not the process still is.
func handleDaemonLog(ctx *Context) *exit.Error {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	file, err := os.Open(layout.Log)
	if err != nil {
		if os.IsNotExist(err) {
			return exit.New(exit.NotFound, "no daemon log at %s: no Cozy daemon has run on this root",
				layout.Log).WithNext("cozy up")
		}
		return exit.Internalf("cannot open the daemon log %s: %s", layout.Log, err)
	}
	if !ctx.Inv.Bool("--follow") {
		defer file.Close()
		if _, err := io.Copy(ctx.Out, file); err != nil {
			return exit.Internalf("cannot read the daemon log %s: %s", layout.Log, err)
		}
		return nil
	}
	return followLog(ctx, layout.Log, file)
}

// followLog copies what the daemon appends as it appends it, from the start of the file
// it is handed. The poll is a sampling cadence, not a decision: it decides nothing and
// the follow ends only on the reader's interrupt. A rotation (daemon.LogBytes) replaces
// the file under the same name; the follower notices the identity change and carries on
// from the start of the new one.
func followLog(ctx *Context, path string, file *os.File) *exit.Error {
	defer func() { file.Close() }()
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)
	current, err := file.Stat()
	if err != nil {
		return exit.Internalf("cannot read the daemon log %s: %s", path, err)
	}
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		if _, err := io.Copy(ctx.Out, file); err != nil {
			return exit.Internalf("cannot read the daemon log %s: %s", path, err)
		}
		if latest, err := os.Stat(path); err == nil && !os.SameFile(current, latest) {
			if replacement, err := os.Open(path); err == nil {
				file.Close()
				file, current = replacement, latest
				continue
			}
		}
		select {
		case <-interrupt:
			return nil
		case <-tick.C:
		}
	}
}
