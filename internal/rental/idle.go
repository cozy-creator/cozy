package rental

import (
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// IdleTimeout is immutable. Configuration, request options and connection traffic
// cannot change it. Only actual work and acknowledged manual keepalive reset it.
const IdleTimeout = time.Duration(pb.RentalIdleTimeoutSeconds) * time.Second

type Idleness struct {
	Queued, Running int
	// An unresolved preparation after controller loss defers to the pod guard;
	// it is uncertainty, not a claim of activity or a clock renewal.
	PendingPreparation int
	Since              time.Time
}

func (i Idleness) ReleaseAt() (time.Time, bool) {
	if i.Queued > 0 || i.Running > 0 || i.PendingPreparation > 0 || i.Since.IsZero() {
		return time.Time{}, false
	}
	return i.Since.Add(IdleTimeout), true
}

func (i Idleness) Due(now time.Time) bool {
	at, eligible := i.ReleaseAt()
	return eligible && !now.Before(at)
}

// ObserveIdle uses only this machine's real work. Retained results and unassigned
// fleet work are deliberately absent: neither is execution on this rental.
func ObserveIdle(st *records.Store, row records.Rental) (Idleness, *exit.Error) {
	var idle Idleness
	var problem *exit.Error
	idle.Queued, idle.Running, problem = st.RentalIdleRunCounts(row.ID, row.ManagedRequestID)
	if problem != nil {
		return idle, problem
	}
	if row.ReadyAt != "" {
		idle.Since, _ = time.Parse(time.RFC3339Nano, row.ReadyAt)
		if idle.Since.IsZero() {
			return idle, exit.Internalf("rental %s has an invalid ready timestamp", row.ID)
		}
	}
	last, found, problem := st.RentalLastSettlement(row.ID)
	if problem != nil {
		return idle, problem
	}
	if found && last.SettledAt.After(idle.Since) {
		idle.Since = last.SettledAt
	}
	at, pending, problem := st.RentalIdleResetAt(row)
	idle.PendingPreparation = pending
	if problem != nil {
		return idle, problem
	}
	if at.After(idle.Since) {
		idle.Since = at
	}
	return idle, nil
}
