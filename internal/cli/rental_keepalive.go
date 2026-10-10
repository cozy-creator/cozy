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
	// The deadline the machine answered: its clock restarted now, or none named.
	fields := append([]output.Field{{K: "rental", V: subject.Row.ID},
		{K: "acknowledged_at", V: time.UnixMilli(receipt.AcknowledgedAtMS).UTC().Format(time.RFC3339Nano)}},
		releaseDue(receipt.IdleDeadlineMS, !ctx.Mode().Human || ctx.Mode().JSON)...)
	return emit(ctx, output.Record{Fields: fields, AllFields: fields})
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
	receipt, problem := rental.KeepAlive(ctx, m.layout, row)
	if problem == nil {
		problem = m.store.RecordRentalKeepalive(id, receipt, time.Now())
	}
	if problem != nil {
		return out, problem
	}
	return api.RentalKeepaliveResult{Rental: id, WorkerID: receipt.WorkerID, WorkerBootID: receipt.WorkerBootID, AcknowledgedAtUnixMS: receipt.AcknowledgedAtMS, IdleDeadlineUnixMS: receipt.IdleDeadlineMS}, nil
}
