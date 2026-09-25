package cli

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
)

// handleRunWatch is the one reattachment surface for every durable run kind.
// A watcher never cancels work; explicit cancellation belongs to run cancel.
func handleRunWatch(ctx *Context) *exit.Error {
	id := strings.TrimSpace(ctx.Inv.Args[0])
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	switch {
	case strings.HasPrefix(id, "job-"):
		state, problem := client.Job(id)
		if problem != nil {
			return problem
		}
		return watchJob(ctx, client, state)
	default:
		life, problem := client.Request(id)
		if problem != nil {
			return problem
		}
		if life.Kind == "job" {
			state, problem := client.Job(id)
			if problem != nil {
				return problem
			}
			return watchJob(ctx, client, state)
		}
		return watchInvocation(ctx, client, life.RequestID)
	}
}

func watchJob(ctx *Context, client *localclient.Client, state api.JobState) *exit.Error {
	return followJob(ctx, client, state.JobID, recordedRunStart(state.CreatedAt))
}

func watchInvocation(ctx *Context, client *localclient.Client, id string) *exit.Error {
	life, problem := client.Request(id)
	if problem != nil {
		return problem
	}
	began := recordedRunStart(life.CreatedAt)
	terminal, detached, problem := watchRunStream(ctx, client, id, began)
	if problem != nil {
		return problem
	}
	life, problem = client.Request(id)
	if problem != nil {
		return problem
	}
	if detached && !invocationSettled(life.Status) {
		return renderSubmittedRun(ctx, life, false)
	}
	life, problem = waitOutputExport(client, life)
	if problem != nil {
		return problem
	}
	return renderRun(ctx, life, terminal, "", exportedOutputs(life), 0)
}

func watchRunStream(ctx *Context, client *localclient.Client, id string,
	began time.Time,
) (*localclient.Event, bool, *exit.Error) {
	watchCtx, stop := context.WithCancel(context.Background())
	defer stop()
	interrupt, restoreInput, _, problem := liveSignals(ctx, nil)
	if problem != nil {
		return nil, false, problem
	}
	defer restoreInput()
	detached := make(chan struct{}, 1)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-interrupt:
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err, "\ndetached from run %s; durable work continues\n", id)
			}
			detached <- struct{}{}
			stop()
		case <-done:
		}
	}()
	lines := NewProgress(ctx, ctx.Mode().JSON, began)
	terminal, problem := client.WatchContext(watchCtx, id, 0, lines.On)
	lines.Done()
	wasDetached := false
	select {
	case <-detached:
		wasDetached = true
	default:
	}
	return terminal, wasDetached, problem
}

func recordedRunStart(value string) time.Time {
	if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return parsed
	}
	return time.Now()
}

// recordedRunWall is accepted-request time through its durable terminal event.
// Reattaching or collecting/exporting outputs later cannot extend a completed run.
// Missing timestamps are unknown, not an elapsed observer clock or a fabricated zero.
func recordedRunWall(createdAt string, terminal *localclient.Event) (int64, bool) {
	if terminal == nil || terminal.EventID <= 0 || !localclient.Terminal(terminal.Type) {
		return 0, false
	}
	began, err := time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return 0, false
	}
	ended, err := time.Parse(time.RFC3339Nano, terminal.At)
	if err != nil || ended.Before(began) {
		return 0, false
	}
	return ended.Sub(began).Milliseconds(), true
}
