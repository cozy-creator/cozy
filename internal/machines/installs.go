package machines

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// Installs is the daemon's durable queue of package and model installations, one per
// machine at a time. It observes only local records; the fleet reconciler owns provider
// observation and attachment, and no installation can allocate or replace a rental.
type Installs struct {
	store   *records.Store
	prepare Prepare
	log     io.Writer
	wake    chan struct{}
	// progress is each running installation's latest preparation report, by install id.
	progress sync.Map
}

// InstallProgress is the machine's latest report on one running installation.
type InstallProgress struct {
	Stage            string `json:"stage"`
	TotalBytes       uint64 `json:"total_bytes"`
	TransferredBytes uint64 `json:"transferred_bytes"`
}

// InstallStatus is one installation's durable record and, while it runs, its progress.
type InstallStatus struct {
	records.RentalInstall
	Progress *InstallProgress `json:"progress,omitempty"`
}

// Prepare makes one installation on its machine, answering what it produced, if anything.
type Prepare func(context.Context, records.RentalInstall, func(InstallProgress)) (json.RawMessage, *exit.Error)

func NewInstalls(store *records.Store, prepare Prepare, log io.Writer) *Installs {
	if log == nil {
		log = io.Discard
	}
	return &Installs{store: store, prepare: prepare, log: log, wake: make(chan struct{}, 1)}
}

// Accept durably queues one installation on a machine this host knows.
func (q *Installs) Accept(machine string, selection records.RentalInstallSelection) (*records.RentalInstall, *exit.Error) {
	if machine != Local {
		row, problem := q.store.RentalRow(machine)
		if problem != nil {
			return nil, problem
		}
		if row == nil {
			return nil, exit.New(exit.NotFound, "rental %s is not recorded on this host", machine)
		}
		if problem := records.RentalInstallStateProblem(row.ID, row.State); problem != nil {
			return nil, problem
		}
	}
	row, problem := q.store.BeginRentalInstall(machine, selection)
	if problem == nil {
		q.Wake()
	}
	return row, problem
}

// Status reads one installation on one machine, with its progress while it runs.
func (q *Installs) Status(machine, id string) (*InstallStatus, *exit.Error) {
	row, problem := q.store.RentalInstall(id)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.RentalID != machine {
		return nil, exit.New(exit.NotFound, "no installation %s on %s", id, machine)
	}
	status := &InstallStatus{RentalInstall: *row}
	if progress, ok := q.progress.Load(id); ok && row.Active() {
		reported := progress.(InstallProgress)
		status.Progress = &reported
	}
	return status, nil
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
	if ended := records.RentalInstallStateProblem(row.ID, row.State); ended != nil {
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

// Run recovers unfinished rows before observing new admissions. One preparation
// per rental runs at a time; interrupted calls replay their exact frozen inputs.
func (q *Installs) Run(ctx context.Context) {
	type active struct {
		id     string
		cancel context.CancelFunc
	}
	running := map[string]active{}
	type completion struct {
		rental string
		retry  bool
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
			delete(running, finished.rental)
			if !finished.retry {
				q.Wake()
			}
		case <-q.wake:
			rows, problem := q.store.PendingRentalInstalls()
			if problem != nil {
				q.report(problem)
				continue
			}
			seen := map[string]bool{}
			for _, row := range rows {
				if ctx.Err() != nil {
					break
				}
				boot, ready, ended, problem := installFence(q.store, row.RentalID)
				if problem != nil {
					q.report(problem)
					continue
				}
				if ended != nil {
					q.settle(row, "failed", nil, ended)
					if held, ok := running[row.RentalID]; ok && held.id == row.ID {
						held.cancel()
					}
					continue
				}
				if seen[row.RentalID] {
					continue
				}
				seen[row.RentalID] = true
				if _, busy := running[row.RentalID]; busy || !ready {
					continue
				}
				claimed, problem := q.store.StartRentalInstall(row.ID, boot)
				if problem != nil {
					q.report(problem)
					continue
				}
				work, cancel := context.WithCancel(ctx)
				running[row.RentalID] = active{id: row.ID, cancel: cancel}
				workers.Add(1)
				go func(row records.RentalInstall) {
					defer workers.Done()
					defer cancel()
					var reported sync.Mutex
					var first *InstallProgress
					moved := false
					result, problem := q.prepare(work, row, func(progress InstallProgress) {
						reported.Lock()
						if first == nil {
							first = &progress
						} else if progress.Stage != first.Stage || progress.TransferredBytes > first.TransferredBytes {
							moved = true
						}
						reported.Unlock()
						q.progress.Store(row.ID, progress)
					})
					reported.Lock()
					progressed := moved
					reported.Unlock()
					q.progress.Delete(row.ID)
					state := "succeeded"
					if problem != nil {
						// A stopping daemon leaves the installation to the next one. Otherwise it
						// runs again only after an unavailable machine let it make progress.
						state, result = "failed", nil
						if ctx.Err() != nil || progressed && problem.Code == exit.Unavailable {
							state = "queued"
						}
					}
					q.settle(row, state, result, problem)
					select {
					case done <- completion{rental: row.RentalID, retry: state == "queued"}:
					case <-ctx.Done():
					}
				}(*claimed)
			}
		}
	}
}

// settle records an installation's end, or its return to the queue, and logs each outcome.
func (q *Installs) settle(row records.RentalInstall, state string, result json.RawMessage, problem *exit.Error) {
	if failed := q.store.SettleRentalInstall(row.ID, state, result, problem); failed != nil || state == "queued" {
		q.report(failed)
		return
	}
	line := fmt.Sprintf("%s on %s %s: %s", row.ID, row.RentalID, state, row.Selection.Target())
	if problem != nil {
		line += ": " + problem.Message
	}
	fmt.Fprintln(q.log, line)
}

func (q *Installs) report(problem *exit.Error) {
	if problem != nil {
		fmt.Fprintln(q.log, problem.Message)
	}
}
