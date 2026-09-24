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

type Idleness = records.RentalIdleState

// ObserveIdle uses the same persisted facts as the atomic release admission fence.
func ObserveIdle(st *records.Store, row records.Rental) (Idleness, *exit.Error) {
	return st.RentalIdleObservation(row)
}
