package rental

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

type Idleness = records.RentalIdleState

// ObserveIdle is this host's record of a rental's queued and running work and idle start.
func ObserveIdle(st *records.Store, row records.Rental) (Idleness, *exit.Error) {
	return st.RentalIdleObservation(row)
}
