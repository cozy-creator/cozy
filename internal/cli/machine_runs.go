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
	ctx      context.Context
	cancel   context.CancelFunc
	context  *Context
	layout   home.Layout
	store    *records.Store
	resolver *Resolver
	fleet    *managedRentals
	mu       sync.Mutex
	running  map[string]chan struct{}
	placed   map[string]string // the last placement decision recorded per waiting run
	machines *machines.Resolver
	updates  *rentalRuntimeUpdates
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
		defer func() {
			m.endObservation(request.ID, finished)
			// A control can be persisted after the final observation, while its
			// Start still sees this goroutine running. Read after withdrawing our
			// running marker so either that Start or this handoff owns the wakeup.
			if m.ctx.Err() != nil {
				return
			}
			link, problem := m.store.MachineExecution(request.ID)
			if problem != nil || link == nil || link.Abandoned || !link.Collected || len(link.Receipt) == 0 || !link.CancelRequested && len(link.PendingControl) == 0 {
				return
			}
			if owed, problem := m.machineWorkOwed(request.ID); problem != nil || !owed {
				return
			}
			current, problem := m.store.RequestRow(request.ID)
			if problem == nil && current != nil {
				_ = m.Start(*current)
			}
		}()
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

// machineWorkOwed is whether the execution still owes work or custody, or holds a sent
// publication only its owner can settle.
func (m *machineRuns) machineWorkOwed(request string) (bool, *exit.Error) {
	if link, problem := m.store.MachineExecution(request); problem != nil || link != nil && link.Abandoned {
		return false, problem
	}
	if owed, problem := m.store.MachineExecutionOwesWork(request); problem != nil || owed {
		return owed, problem
	}
	awaiting, problem := m.store.MachinePublicationsAwaitingOwner(request)
	return len(awaiting) > 0, problem
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
			problem := m.catchUpV1(parent, request)
			m.endObservation(request.ID, finished)
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

// Control wakes delivery of a durable cancel without waiting on its machine.
// Pause/resume retain their synchronous generation-checked control path.
func (m *machineRuns) Control(parent context.Context, request records.Request, action string) *exit.Error {
	if action == "cancel" {
		m.Withdraw(request.ID)
		if link, problem := m.store.MachineExecution(request.ID); problem == nil && link != nil {
			m.settleStoppedCancellation(link)
		}
		return m.Start(request)
	}
	return m.controlV1(parent, request, action)
}

// settleStoppedCancellation settles a requested cancel whose run is on this computer's machine
// while that machine is stopped: its unit and agent have ended, so nothing of the run executes.
// The observer still delivers the cancel when the machine next runs. `cozy machine stop` used
// to leave such runs canceling until the next start, and `--await` waited with them.
func (m *machineRuns) settleStoppedCancellation(link *records.MachineExecution) {
	if link.MachineID != machines.Local || !link.CancelRequested || link.Abandoned || m.machines == nil || m.machines.Host == nil {
		return
	}
	if status, problem := m.machines.Host.Status(); problem != nil || status.Running {
		return
	}
	if problem := m.store.SettleStoppedMachineCancellation(link.RequestID); problem != nil {
		fmt.Fprintf(m.context.Out, "machine execution %s: %s\n", link.RequestID, problem.Message)
	}
}
