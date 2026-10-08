package cli

import (
	"context"
	"fmt"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// machineRuns is a client transport and observer. Stopping it closes connections
// and upload/observation work; it never sends an execution cancellation.
type machineRuns struct {
	ctx     context.Context
	cancel  context.CancelFunc
	context *Context
	// A foreground controller closes with its CLI command. Explicit controls must
	// reach the machine before that command returns; observation remains detachable.
	foreground bool
	layout     home.Layout
	store      *records.Store
	resolver   *Resolver
	fleet      *managedRentals
	mu         sync.Mutex
	running    map[string]chan struct{}
	// hubAccess holds each machine's execution access, by hub, credential and leaf.
	hubAccess sync.Map
	placed    map[string]string // the last placement decision recorded per waiting run
	machines  *machines.Resolver
	updates   *rentalRuntimeUpdates
	// submitting stops each request's submission work in flight (upload, preparation,
	// staging) once its cancel is durable; guarded by mu.
	submitting map[string]context.CancelFunc
}

func newMachineRuns(ctx *Context, layout home.Layout, store *records.Store, resolver *Resolver, fleet *managedRentals, found *machines.Resolver) *machineRuns {
	background, cancel := context.WithCancel(context.Background())
	return &machineRuns{ctx: background, cancel: cancel, context: ctx, layout: layout, store: store, resolver: resolver, fleet: fleet, machines: found, running: map[string]chan struct{}{}, placed: map[string]string{}, submitting: map[string]context.CancelFunc{}}
}

func (m *machineRuns) Start(request records.Request) *exit.Error {
	m.mu.Lock()
	if m.ctx.Err() != nil {
		m.mu.Unlock()
		return exit.Named(exit.Unavailable, "daemon.closing", "the client observer has disconnected")
	}
	if m.running[request.ID] != nil {
		m.mu.Unlock()
		return nil
	}
	finished := make(chan struct{})
	m.running[request.ID] = finished
	m.mu.Unlock()
	go func() {
		defer m.endObservation(request.ID, finished)
		if m.machines != nil {
			m.loopV1(request)
		}
	}()
	return nil
}

func (m *machineRuns) endObservation(id string, finished chan struct{}) {
	m.mu.Lock()
	delete(m.running, id)
	delete(m.placed, id)
	close(finished)
	m.mu.Unlock()
}

func (m *machineRuns) Resume() {
	if problem := m.store.ReconcileEndedMachineExecutions(); problem != nil {
		fmt.Fprintf(m.context.Out, "machine execution loss recovery: %s\n", problem.Message)
		return
	}
	links, problem := m.store.MachineExecutions()
	if problem != nil {
		fmt.Fprintf(m.context.Out, "machine execution observation recovery: %s\n", problem.Message)
		return
	}
	for _, link := range links {
		if link.Lost {
			continue // its machine is gone: nothing of the run is left to observe
		}
		if awaiting, problem := m.store.MachinePublicationsAwaitingOwner(link.RequestID); link.Collected && !link.CancelRequested && len(link.PendingControl) == 0 && (problem != nil || len(awaiting) == 0) {
			continue
		}
		if !link.CancelRequested && m.collectionRefused(link.RequestID) {
			continue // ended with its reason; only its owner asks again
		}
		request, problem := m.store.RequestRow(link.RequestID)
		if problem == nil && request != nil {
			_ = m.Start(*request)
		}
	}
}

// PlaceReleased follows again every run back in the outbox: released from a machine proven
// gone before it confirmed the run, it is placed on another under the same identity.
func (m *machineRuns) PlaceReleased() {
	ids, problem := m.store.UnplacedMachineExecutions()
	if problem != nil {
		fmt.Fprintf(m.context.Out, "released runs: %s\n", problem.Message)
		return
	}
	for _, id := range ids {
		if request, problem := m.store.RequestRow(id); problem == nil && request != nil && !records.Settled(request.State) {
			_ = m.Start(*request)
		}
	}
}

// WakeMachine asks again about every run held on a machine that shows new evidence of life.
func (m *machineRuns) WakeMachine(machine string) {
	links, problem := m.store.MachineExecutions()
	if problem != nil {
		return
	}
	for _, link := range links {
		if link.MachineID != machine || link.Collected || link.Abandoned || link.Lost {
			continue
		}
		if request, problem := m.store.RequestRow(link.RequestID); problem == nil && request != nil {
			_ = m.Start(*request)
		}
	}
}

// Withdraw stops local submission/observation for a durable cancel or abandonment.
// The observer's next pass delivers cancellation for any accepted execution.
func (m *machineRuns) Withdraw(request string) {
	m.mu.Lock()
	stop := m.submitting[request]
	m.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// recordPlacement makes a waiting run's placement decision durable, as a queued run's is:
// the fleet line and the decision record, each once per change.
func (m *machineRuns) recordPlacement(request records.Request, decision orchestrator.PlacementDecision, line string) {
	m.mu.Lock()
	news := m.placed[request.ID] != decision.Line()+"\x00"+line
	m.placed[request.ID] = decision.Line() + "\x00" + line
	m.mu.Unlock()
	if !news {
		return
	}
	if line != "" {
		_ = m.store.AppendEvent(request.ID, "request.rentals", 0, map[string]any{"line": line})
	}
	if len(decision.Candidates) > 0 {
		m.fleet.owner.LogPlacement(request, decision)
	}
}

// runHolder names a run and what it is doing on its machine.
func (m *machineRuns) runHolder(request records.Request, doing string) string {
	return m.runName(request) + " " + doing
}

func (m *machineRuns) runName(request records.Request) string {
	if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil && numbered.Number > 0 {
		return fmt.Sprintf("run %d", numbered.Number)
	}
	return "run " + request.ID
}

// Refresh brings a run's record up to date for a reader. A run the observer is following
// already is: the observer holds its machine's next event, and the reader takes the record.
// One whose collection ended on a refusal is collected again at once.
func (m *machineRuns) Refresh(parent context.Context, request records.Request) *exit.Error {
	for {
		m.mu.Lock()
		if m.ctx.Err() != nil {
			m.mu.Unlock()
			return exit.Named(exit.Unavailable, "daemon.closing", "the client observer has disconnected")
		}
		following := m.running[request.ID]
		if following == nil {
			finished := make(chan struct{})
			m.running[request.ID] = finished
			m.mu.Unlock()
			problem := func() *exit.Error {
				defer m.endObservation(request.ID, finished)
				return m.catchUpV1(parent, request)
			}()
			// A cancel can arrive while catch-up owns the slot. Deliver that durable
			// intent even if this reader detached or its connection failed.
			started := m.Start(request)
			if problem != nil {
				return problem
			}
			return started
		}
		m.mu.Unlock()
		if !m.collectionRefused(request.ID) {
			return nil // live work is already observed; a reader never waits for it
		}
		// The previous observer may have recorded a refusal but not released its
		// slot yet. Wait for that handoff, then retry collection as its sole owner.
		select {
		case <-following:
		case <-parent.Done():
			return exit.New(exit.Canceled, "machine observation detached")
		case <-m.ctx.Done():
			return exit.Named(exit.Unavailable, "daemon.closing", "the client observer has disconnected")
		}
	}
}

// collectionRefused is whether a finished result waits on its owner. A reader asking about
// it tries the collection again at once: the cause may be fixed since the observer tried.
func (m *machineRuns) collectionRefused(id string) bool {
	link, problem := m.store.MachineExecution(id)
	if problem != nil || link == nil || link.Collected {
		return false
	}
	code, _, problem := m.store.MachineCollectionRefusal(id)
	return problem == nil && code != ""
}

// Control wakes the daemon's durable cancellation delivery. A foreground controller
// instead waits for the machine's acknowledgement before its CLI owner can close it;
// --await separately observes the terminal outcome. A stopped local machine stays
// canceling until `cozy run cancel` starts it. Pause/resume keep their synchronous path.
func (m *machineRuns) Control(parent context.Context, request records.Request, action string) *exit.Error {
	if action == "cancel" {
		if m.foreground {
			return m.controlV1(parent, request, action)
		}
		m.Withdraw(request.ID)
		return m.Start(request)
	}
	return m.controlV1(parent, request, action)
}
