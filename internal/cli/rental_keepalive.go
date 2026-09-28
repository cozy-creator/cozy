package cli

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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

// Serialize receipt persistence with Creator's release decision: idle release waits out
// a keepalive in flight, which asks the rental with the lock released. The Host also
// atomically arbitrates renewal against its independently committed expiry.
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
	if problem != nil {
		m.mu.Unlock()
		return out, problem
	}
	m.keepingLocked(id)
	m.mu.Unlock()
	receipt, problem := m.renew(ctx, id, requestID)
	if problem != nil {
		return out, problem
	}
	return api.RentalKeepaliveResult{Rental: id, RequestID: receipt.RequestId, WorkerID: receipt.WorkerId, WorkerBootID: receipt.WorkerBootId, AcknowledgedAtUnixMS: receipt.AcknowledgedAtUnixMs, IdleDeadlineUnixMS: receipt.IdleDeadlineUnixMs}, nil
}

func (m *managedRentals) keepingLocked(id string) {
	if m.keeping == nil {
		m.keeping = map[string]int{}
	}
	m.keeping[id]++
}

// renew asks the Host for one keepalive and records its receipt; the caller counted it
// in keeping, which this ends.
func (m *managedRentals) renew(ctx context.Context, id, requestID string) (*pb.KeepRentalAliveResult, *exit.Error) {
	receipt, problem := m.owner.KeepRentalAlive(ctx, id, requestID)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.keeping[id]--; m.keeping[id] == 0 {
		delete(m.keeping, id)
	}
	if problem == nil {
		problem = m.store.RecordRentalKeepalive(id, receipt, time.Now())
	}
	if problem != nil {
		return nil, problem
	}
	m.forgetIdleLocked(id)
	return receipt, nil
}

// holdLocked keeps the Host from releasing a rental while this host holds accepted work
// executing on it. The Host releases on its own clock, from what its Runtime reports;
// that report can lapse (a Runtime restarted by maintenance), and this host's accepted
// executions are then the only evidence of live work. Renewal follows the Host's own
// granted window. A worker that reported itself FAILED runs nothing, so nothing is held.
func (m *managedRentals) holdLocked(row records.Rental) {
	if m.owner == nil || m.keeping[row.ID] > 0 || m.owner.RentalWorkerFailed(row.ID) {
		return
	}
	due, problem := m.store.RentalKeepaliveDue(row, time.Now())
	if problem != nil || !due {
		return
	}
	m.keepingLocked(row.ID)
	go func() {
		if _, problem := m.renew(context.Background(), row.ID, requestKey("")); problem != nil {
			m.mu.Lock()
			m.sayLocked("hold:"+row.ID, fmt.Sprintf("rental %s (%s) runs accepted work but could not renew its Host idle deadline: %s",
				row.ID, row.MachineName, problem.Message))
			m.mu.Unlock()
		}
	}()
}
