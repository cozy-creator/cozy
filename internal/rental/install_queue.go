package rental

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// InstallQueue is owned by the Creator daemon. It only observes local rental
// records; the existing fleet reconciler owns provider observation and attachment.
// No install request can allocate or replace a rental.
type InstallQueue struct {
	store   *records.Store
	prepare func(context.Context, records.RentalInstall) *exit.Error
	log     io.Writer
	wake    chan struct{}
}

func NewInstallQueue(store *records.Store, prepare func(context.Context, records.RentalInstall) *exit.Error, log io.Writer) *InstallQueue {
	if log == nil {
		log = io.Discard
	}
	return &InstallQueue{store: store, prepare: prepare, log: log, wake: make(chan struct{}, 1)}
}

func (q *InstallQueue) Accept(rental string, selection records.RentalInstallSelection) (*records.RentalInstall, *exit.Error) {
	row, problem := q.store.BeginRentalInstall(rental, selection)
	if problem == nil {
		q.Wake()
	}
	return row, problem
}

func (q *InstallQueue) Wake() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

// Run recovers unfinished rows before observing new admissions. One preparation
// per rental runs at a time; interrupted calls replay their exact frozen inputs.
func (q *InstallQueue) Run(ctx context.Context) {
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
			rows, problem := q.store.RentalInstalls("", true)
			if problem != nil {
				q.report(problem)
				continue
			}
			seen := map[string]bool{}
			for _, row := range rows {
				machine, problem := q.store.RentalRow(row.RentalID)
				if problem != nil {
					q.report(problem)
					continue
				}
				if machine == nil {
					problem = exit.Named(exit.Unavailable, "rental.ended", "rental %s ended before installation completed", row.RentalID)
				} else {
					problem = records.RentalInstallStateProblem(machine.ID, machine.State)
				}
				if problem == nil && row.WorkerBootID != "" && machine.ExpectedWorkerBootID != "" && machine.ExpectedWorkerBootID != row.WorkerBootID {
					problem = exit.Named(exit.Conflict, "rental.worker_boot_changed", "rental %s changed worker boot during installation", row.RentalID)
				}
				if problem != nil {
					q.report(q.store.SettleRentalInstall(row.ID, "failed", problem))
					if held, ok := running[row.RentalID]; ok && held.id == row.ID {
						held.cancel()
					}
					continue
				}
				if seen[row.RentalID] {
					continue
				}
				seen[row.RentalID] = true
				if _, busy := running[row.RentalID]; busy || machine.State != "ready" || machine.ExpectedWorkerBootID == "" {
					continue
				}
				claimed, problem := q.store.StartRentalInstall(row.ID, machine.ExpectedWorkerBootID)
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
					problem := q.prepare(work, row)
					state := "succeeded"
					if problem != nil {
						state = "failed"
						if work.Err() != nil || problem.Code == exit.Unavailable || problem.Code == exit.Canceled {
							state = "queued"
						}
					}
					q.report(q.store.SettleRentalInstall(row.ID, state, problem))
					select {
					case done <- completion{rental: row.RentalID, retry: state == "queued"}:
					case <-ctx.Done():
					}
				}(*claimed)
			}
		}
	}
}

func (q *InstallQueue) report(problem *exit.Error) {
	if problem != nil {
		fmt.Fprintln(q.log, problem.Message)
	}
}
