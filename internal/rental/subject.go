package rental

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// Subject joins immutable rental IDs to this controller's operation history.
// It supplies custody and replay metadata, never a live name or state lookup.
type Subject struct {
	// Typed is the argument exactly as given.
	Typed string
	// RentalID is the hub's identity for it, when this host ever learned one.
	RentalID string
	// Machine is the observed host name, or the original rental alias before allocation.
	Machine string
	// Hub is the authority the ask went to; a release must not be judged by another's 404.
	Hub string
	// Row is the live local rental record, when there is one.
	Row *records.Rental
	// Operation is the paid acquisition, live or settled. It survives Row, so it is what
	// answers "this host asked for that pod" after the record has been forgotten — and it
	// exists BEFORE the POST, so it also answers for a pod whose create answer was lost.
	Operation *records.RentalOperation
}

// Recorded says this host has its own durable trace of the subject, and therefore that a
// hub 404 for it is absence rather than a failed lookup.
func (s Subject) Recorded() bool { return s.Row != nil || s.Operation != nil }

// RecordedSubject looks up exact rental IDs only. Name selection belongs to
// LiveResolver; retained aliases cannot make an ended host look live.
func RecordedSubject(st *records.Store, id string) (Subject, *exit.Error) {
	found := Subject{Typed: id}
	row, problem := st.RentalRow(id)
	if problem != nil {
		return Subject{}, problem
	}
	if row != nil {
		found.Row = row
		found.RentalID, found.Machine, found.Hub = row.ID, row.DisplayName(), row.Hub
	}
	operations, problem := st.RentalOperations()
	if problem != nil {
		return Subject{}, problem
	}
	for index := range operations {
		op := &operations[index]
		if op.RentalID != id {
			continue
		}
		found.Operation = op
		if row == nil {
			found.RentalID, found.Machine, found.Hub = op.RentalID, operationMachineName(op), op.Hub
		}
		break
	}
	return found, nil
}

// operationMachineName reads the machine word out of the persisted ask. It is deliberately
// lenient: an operation authored by an older Creator that no longer parses canonically is
// still a record that this host asked for a pod, and refusing to read it would put the
// command back where it started.
func operationMachineName(op *records.RentalOperation) string {
	var request hub.RentalRequest
	if json.Unmarshal(op.RequestBody, &request) != nil {
		return ""
	}
	return request.Name
}
