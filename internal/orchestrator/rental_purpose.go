package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// RentalOwedBy answers whether the request this rental was bought for still owes it a
// pin: not yet settled, and not durably routed to another machine. The debt is read from
// the durable rows alone — `rentals.managed_request_id` is written with the paid create —
// so it holds across a daemon restart and across the whole boot, when the request row
// still says worker=” because the pin is routing's output (cl-092) and routing has not
// run yet. Without this, a booting or freshly ready pod looks unowed to every observer
// and an idle sweep — or any fleet hygiene reading the same records — reaps a pod whose
// buyer is still queued for it (cl-113, observed live: rental pr-b192da1a released
// mid-boot while req-ff3f79f4 waited on it).
func RentalOwedBy(st *records.Store, row records.Rental) (bool, *exit.Error) {
	if row.ManagedRequestID == "" {
		return false, nil
	}
	owing, problem := st.RequestRow(row.ManagedRequestID)
	if problem != nil {
		return false, problem
	}
	return owing != nil && !settledState(owing.State) &&
		(owing.Worker == "" || owing.Worker == row.ID), nil
}

// RentalSpent is the existing managed-rental lifetime fact, shared by idle release
// and capacity selection. A retained rental remains billable without becoming reusable.
func RentalSpent(st *records.Store, row records.Rental) (bool, *exit.Error) {
	if row.ManagedRequestID == "" {
		return false, nil
	}
	owed, problem := RentalOwedBy(st, row)
	if problem != nil || owed {
		return false, problem
	}
	last, found, problem := st.RentalLastSettlement(row.ID)
	if problem != nil {
		return false, problem
	}
	return !found || last.Kind == "job" || last.ClosedAt.IsZero(), nil
}
