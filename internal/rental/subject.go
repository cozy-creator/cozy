package rental

import (
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// Subject is what a person typed on the command line, resolved against everything this
// host durably knows about paid pods (cl-193).
//
// It exists because a machine word — `aeirik` — is CREATOR'S alias and never a key the
// hub can look anything up by: Tensorhub identifies a rental by the opaque id it minted,
// which only the host that rented it holds. Sending the word as that id and reading the
// resulting 404 as "already released" is not a lookup, it is a guess dressed as a fact,
// and it answered `state: ended` for six H100s that were booting and billing.
//
// So the ladder below is the whole point: a 404 proves absence ONLY for an id this host
// recorded. Everything else must say it does not know.
type Subject struct {
	// Typed is the argument exactly as given.
	Typed string
	// RentalID is the hub's identity for it, when this host ever learned one.
	RentalID string
	// Machine is the machine word this host authored the ask under, when known.
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

// Resolve walks the local ladder: the rental record by machine word, then by id, then the
// paid operations by rental id and by the machine word the ask was authored under. A live
// operation outranks a settled one; among equals the newest ask wins, because that is the
// one that may still be running.
func Resolve(st *records.Store, typed string) (Subject, *exit.Error) {
	found := Subject{Typed: typed}
	row, problem := st.RentalByMachine(typed)
	if problem != nil {
		return Subject{}, problem
	}
	if row == nil {
		if row, problem = st.RentalRow(typed); problem != nil {
			return Subject{}, problem
		}
	}
	if row != nil {
		found.Row = row
		found.RentalID, found.Machine, found.Hub = row.ID, row.MachineName, row.Hub
	}
	operations, problem := st.RentalOperations()
	if problem != nil {
		return Subject{}, problem
	}
	for index := range operations {
		op := &operations[index]
		name := operationMachineName(op)
		match := op.RentalID != "" && op.RentalID == found.RentalID
		if found.Row == nil {
			match = match || op.RentalID != "" && op.RentalID == typed || strings.EqualFold(name, typed)
		}
		if !match {
			continue
		}
		if found.Operation != nil && settledOperation(*op) && !settledOperation(*found.Operation) {
			continue
		}
		found.Operation = op
	}
	// Without a record the subject is the chosen ask's pod: a word several asks used names
	// each pod in turn, and the newest one is meant, never the first that bore it.
	if op := found.Operation; found.Row == nil && op != nil {
		found.RentalID, found.Machine, found.Hub = op.RentalID, operationMachineName(op), op.Hub
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

func settledOperation(op records.RentalOperation) bool {
	return op.State == "released" || op.State == "rejected"
}
