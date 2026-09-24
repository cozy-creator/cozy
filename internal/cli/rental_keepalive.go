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

// Serialize receipt persistence with Creator's release decision. The Host also
// atomically arbitrates renewal against its independently committed expiry.
func (m *managedRentals) keepalive(ctx context.Context, id, requestID string) (api.RentalKeepaliveResult, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out api.RentalKeepaliveResult
	if m.closed || m.owner == nil {
		return out, exit.Unavailablef("rental controller is unavailable")
	}
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return out, problem
	}
	if row == nil || row.State != "ready" {
		return out, exit.New(exit.Conflict, "keepalive requires a current ready rental")
	}
	if strings.TrimRight(row.Hub, "/") != client(m.ctx).Base() {
		return out, exit.New(exit.Conflict, "rental belongs to a different Hub authority")
	}
	receipt, problem := m.owner.KeepRentalAlive(ctx, id, requestID)
	if problem != nil {
		return out, problem
	}
	if problem = m.store.RecordRentalKeepalive(id, receipt); problem != nil {
		return out, problem
	}
	m.forgetIdleLocked(id)
	return api.RentalKeepaliveResult{Rental: id, RequestID: receipt.RequestId, WorkerID: receipt.WorkerId, WorkerBootID: receipt.WorkerBootId, AcknowledgedAtUnixMS: receipt.AcknowledgedAtUnixMs, IdleDeadlineUnixMS: receipt.IdleDeadlineUnixMs}, nil
}
