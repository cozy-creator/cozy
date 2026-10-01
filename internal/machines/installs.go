package machines

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// ErrCanceled is the cause an installation's context carries when its owner canceled it,
// as opposed to the daemon stopping: only the first tells the machine to stop.
var ErrCanceled = errors.New("canceled by its owner")

// Prepare lands one download or installation on its machine and answers the bytes it
// last observed per model.
type Prepare func(context.Context, records.Operation) ([]records.ModelProgress, *exit.Error)

// Installs is the daemon's durable queue of downloads and installations, one per machine
// at a time. It observes only local records; the fleet reconciler owns provider
// observation and attachment, and no installation can allocate or replace a rental.
type Installs struct {
	store   *records.Store
	prepare Prepare
	log     io.Writer
	wake    chan struct{}
}

func NewInstalls(store *records.Store, prepare Prepare, log io.Writer) *Installs {
	if log == nil {
		log = io.Discard
	}
	return &Installs{store: store, prepare: prepare, log: log, wake: make(chan struct{}, 1)}
}

// Accept durably queues one download or installation on a machine this host knows. A
// non-empty key answers its first acceptance again.
func (q *Installs) Accept(machine, key string, selection records.InstallSelection) (*records.Operation, bool, *exit.Error) {
	hub := selection.Hub
	if machine != Local {
		row, problem := q.store.RentalRow(machine)
		if problem != nil {
			return nil, false, problem
		}
		if row == nil {
			return nil, false, exit.New(exit.NotFound, "rental %s is not recorded on this host", machine)
		}
		if problem := records.MachineEndedProblem(row.ID, row.State); problem != nil {
			return nil, false, problem
		}
		hub = row.Hub
	}
	row, fresh, problem := q.store.BeginInstall(machine, hub, key, selection)
	if problem == nil {
		q.Wake()
	}
	return row, fresh, problem
}

// InstallTarget names the machine an installation goes to: this computer's machine, or a
// rental by its machine name or id.
func InstallTarget(store *records.Store, name string) (string, *exit.Error) {
	name = strings.TrimSpace(name)
	if name == Local {
		return Local, nil
	}
	row, problem := store.RentalByMachine(name)
	if problem != nil {
		return "", problem
	}
	if row == nil {
		return "", exit.New(exit.NotFound, "no rental %q on this host", name)
	}
	return row.ID, nil
}

// InstallHub is the Tensorhub a machine installs from: a rental's own, or "" for this
// computer's machine, which installs from the hub the daemon addresses.
func InstallHub(store *records.Store, machine string) (string, *exit.Error) {
	if machine == Local {
		return "", nil
	}
	row, problem := store.RentalRow(machine)
	if problem != nil {
		return "", problem
	}
	if row == nil {
		return "", exit.New(exit.NotFound, "rental %s is not recorded on this host", machine)
	}
	return row.Hub, nil
}

// installFence is the worker boot an installation is claimed on and whether the machine
// takes it now, or why its installations end. This computer's machine is always present
// and has no boot fence.
func installFence(store *records.Store, machine string) (boot string, ready bool, ended, problem *exit.Error) {
	if machine == Local {
		return "", true, nil, nil
	}
	row, problem := store.RentalRow(machine)
	if problem != nil {
		return "", false, nil, problem
	}
	if row == nil {
		return "", false, exit.Named(exit.Unavailable, "rental.ended", "rental %s ended before installation completed", machine), nil
	}
	if ended := records.MachineEndedProblem(row.ID, row.State); ended != nil {
		return "", false, ended, nil
	}
	return row.ExpectedWorkerBootID, row.State == "ready" && row.ExpectedWorkerBootID != "", nil, nil
}

func (q *Installs) Wake() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

// Run recovers unfinished rows before observing new admissions. One preparation per
// machine runs at a time; interrupted calls replay their exact frozen inputs, and a row
// its owner canceled stops its call.
func (q *Installs) Run(ctx context.Context) {
	type active struct {
		id     string
		cancel context.CancelCauseFunc
	}
	running := map[string]active{}
	type completion struct {
		machine string
		retry   bool
	}
	done := make(chan completion)
	var workers sync.WaitGroup
	defer workers.Wait()
	q.Wake()
	for {
		select {
		case <-ctx.Done():
			return
		case finished := <-done:
			delete(running, finished.machine)
			if !finished.retry {
				q.Wake()
			}
		case <-q.wake:
			rows, problem := q.store.PendingInstalls()
			if problem != nil {
				q.report(problem)
				continue
			}
			pending := map[string]bool{}
			for _, row := range rows {
				pending[row.ID] = true
			}
			for _, held := range running {
				if !pending[held.id] {
					held.cancel(ErrCanceled)
				}
			}
			seen := map[string]bool{}
			for _, row := range rows {
				if ctx.Err() != nil {
					break
				}
				boot, ready, ended, problem := installFence(q.store, row.Machine)
				if problem != nil {
					q.report(problem)
					continue
				}
				if ended != nil {
					q.report(q.store.SettleInstall(row.ID, "failed", ended, nil))
					if held, ok := running[row.Machine]; ok && held.id == row.ID {
						held.cancel(ended)
					}
					continue
				}
				if seen[row.Machine] {
					continue
				}
				seen[row.Machine] = true
				if _, busy := running[row.Machine]; busy || !ready {
					continue
				}
				claimed, problem := q.store.StartInstall(row.ID, boot)
				if problem != nil {
					q.report(problem)
					continue
				}
				work, cancel := context.WithCancelCause(ctx)
				running[row.Machine] = active{id: row.ID, cancel: cancel}
				workers.Add(1)
				go func(row records.Operation) {
					defer workers.Done()
					defer cancel(nil)
					progress, problem := q.prepare(work, row)
					state := "succeeded"
					if problem != nil {
						state = "failed"
						if work.Err() != nil || problem.Code == exit.Unavailable || problem.Code == exit.Canceled {
							state = "queued"
						}
					}
					q.report(q.store.SettleInstall(row.ID, state, problem, progress))
					select {
					case done <- completion{machine: row.Machine, retry: state == "queued" && !errors.Is(context.Cause(work), ErrCanceled)}:
					case <-ctx.Done():
					}
				}(*claimed)
			}
		}
	}
}

func (q *Installs) report(problem *exit.Error) {
	if problem != nil {
		fmt.Fprintln(q.log, problem.Message)
	}
}
