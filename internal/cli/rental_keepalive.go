package cli

import (
	"context"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/rental"
)

func handleRentalKeepalive(ctx *Context) *exit.Error {
	_, store, problem := rentalStores(ctx)
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
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.KeepRentalAlive(subject.Row.ID, requestKey(""))
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "rental", V: result.Rental}, {K: "request_id", V: result.RequestID},
		{K: "acknowledged_at", V: time.UnixMilli(result.AcknowledgedAtUnixMS).UTC().Format(time.RFC3339Nano)},
		{K: "release_due", V: time.UnixMilli(result.IdleDeadlineUnixMS).UTC().Format(time.RFC3339Nano)},
	}, "rental", "acknowledged_at", "release_due"))
}

// keepalive asks the rental's Host to renew its own idle deadline once, at the owner's
// explicit request, and records the Host's receipt.
func (m *managedRentals) keepalive(ctx context.Context, id, requestID string) (api.RentalKeepaliveResult, *exit.Error) {
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
	receipt, problem := m.owner.KeepRentalAlive(ctx, id, requestID)
	if problem == nil {
		problem = m.store.RecordRentalKeepalive(id, receipt, time.Now())
	}
	if problem != nil {
		return out, problem
	}
	return api.RentalKeepaliveResult{Rental: id, RequestID: receipt.RequestId, WorkerID: receipt.WorkerId, WorkerBootID: receipt.WorkerBootId, AcknowledgedAtUnixMS: receipt.AcknowledgedAtUnixMs, IdleDeadlineUnixMS: receipt.IdleDeadlineUnixMs}, nil
}
