package cli

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
)

// An explicit --await follows the machine's durable receipt before its events.
// A signal detaches only this caller; it does not manufacture machine acceptance
// or send an implicit execution cancellation.
func waitMachineAcceptance(ctx *Context, client *localapi.Client, handle api.JobHandle) (api.JobState, bool, *exit.Error) {
	wait, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	id := handle.JobID
	state := api.JobState{Number: handle.Number, JobID: id, Package: handle.Package, Function: handle.Function, Status: "queued", MachineExecution: &api.MachineExecutionView{}}
	for {
		next, problem := client.JobContext(wait, id)
		if problem != nil {
			if wait.Err() != nil {
				return state, true, nil
			}
			return state, false, problem
		}
		state = next
		if state.MachineExecution == nil || state.MachineExecution.Accepted || settled(state.Status) || state.Status == "blocked" || state.Status == "paused" {
			return state, false, nil
		}
		select {
		case <-wait.Done():
			return state, true, nil
		case <-time.After(200 * time.Millisecond):
		}
	}
}
