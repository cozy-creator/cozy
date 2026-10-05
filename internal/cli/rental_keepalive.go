package cli

import (
	"context"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/rental"
)

func handleRentalKeepalive(ctx *Context) *exit.Error {
	layout, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	subject, problem := rental.Resolve(store, strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if subject.Row == nil {
		return exit.New(exit.NotFound, "no current rental %q on this host", ctx.Inv.Args[0])
	}
	// The command itself resets the deadline, with the rental's recorded address, pin and key:
	// no daemon is asked, so it works whatever daemon runs and whatever the pod's machine is.
	observed, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	receipt, problem := rental.KeepAlive(observed, layout, subject.Row)
	if problem != nil {
		return problem
	}
	if problem := store.RecordRentalKeepalive(subject.Row.ID, receipt, time.Now()); problem != nil {
		return problem
	}
	result := api.RentalKeepaliveResult{Rental: subject.Row.ID, WorkerID: receipt.WorkerID, WorkerBootID: receipt.WorkerBootID,
		AcknowledgedAtUnixMS: receipt.AcknowledgedAtMS, IdleDeadlineUnixMS: receipt.IdleDeadlineMS}
	return emit(ctx, compactRecord([]output.Field{
		{K: "rental", V: result.Rental},
		{K: "acknowledged_at", V: time.UnixMilli(result.AcknowledgedAtUnixMS).UTC().Format(time.RFC3339Nano)},
		{K: "release_due", V: time.UnixMilli(result.IdleDeadlineUnixMS).UTC().Format(time.RFC3339Nano)},
	}, "rental", "acknowledged_at", "release_due"))
}

// keepalive is the daemon's route for the same reset (a cozy before the command did it
// itself still asks here), and records what the machine answered.
func (m *managedRentals) keepalive(ctx context.Context, id string) (api.RentalKeepaliveResult, *exit.Error) {
	var out api.RentalKeepaliveResult
	m.mu.Lock()
	if m.closed || m.owner == nil {
		m.mu.Unlock()
		return out, exit.Unavailablef("rental controller is unavailable")
	}
	row, problem := m.store.RentalRow(id)
	if problem == nil && (row == nil || row.State != "ready" || m.settling[id]) {
		problem = exit.New(exit.Conflict, "keepalive requires a current ready rental")
	}
	m.mu.Unlock()
	if problem != nil {
		return out, problem
	}
	ctx, cancel := context.WithTimeout(ctx, hub.Timeout)
	defer cancel()
	receipt, problem := rental.KeepAlive(ctx, m.layout, row)
	if problem == nil {
		problem = m.store.RecordRentalKeepalive(id, receipt, time.Now())
	}
	if problem != nil {
		return out, problem
	}
	return api.RentalKeepaliveResult{Rental: id, WorkerID: receipt.WorkerID, WorkerBootID: receipt.WorkerBootID, AcknowledgedAtUnixMS: receipt.AcknowledgedAtMS, IdleDeadlineUnixMS: receipt.IdleDeadlineMS}, nil
}
