package records

import (
	"database/sql"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RentalWorkCounts is the work queued and running on one machine, as `cozy rental list`
// counts it: work pinned there, its explicitly purchased acquisition and its
// installations, never retained terminal attempts.
func (s *Store) RentalWorkCounts(id, buyer string) (queued, running int, problem *exit.Error) {
	pinned, args := pinnedToRental("r.", id)
	// A machine's acceptance receipt is live work from that moment, before any observation
	// projects the request out of the queue.
	const accepted = `EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND length(e.receipt)>0)`
	err := s.db.QueryRow(`SELECT
 COALESCE(SUM(CASE WHEN r.state IN ('submitted','queued','requeue_pending') AND NOT `+accepted+` THEN 1 ELSE 0 END),0),
 COALESCE(SUM(CASE WHEN r.state IN ('dispatching','finalizing','pausing','canceling') OR
 r.state IN ('submitted','queued','requeue_pending') AND `+accepted+` OR
 EXISTS(SELECT 1 FROM attempts a WHERE a.request_id=r.id AND a.state IN ('preparing','offered','accepted','recovered_open')) THEN 1 ELSE 0 END),0)
 FROM requests r WHERE `+pinned+` OR
 (r.id=? AND r.worker='') OR EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND e.machine_id=?)`, append(args, buyer, id)...).Scan(&queued, &running)
	if err != nil {
		return 0, 0, exit.Internalf("cannot observe rental work: %s", err)
	}
	var queuedInstalls, runningInstalls int
	err = s.db.QueryRow(`SELECT COALESCE(SUM(state='queued'),0),COALESCE(SUM(state='installing'),0) FROM rental_installs WHERE rental_id=? AND state IN ('queued','installing')`, id).Scan(&queuedInstalls, &runningInstalls)
	if err != nil {
		return 0, 0, exit.Internalf("cannot observe queued rental installations: %s", err)
	}
	return queued + queuedInstalls, running + runningInstalls, nil
}

type rentalReader interface {
	Query(string, ...any) (*sql.Rows, error)
	QueryRow(string, ...any) *sql.Row
}

func refuseReleasedRental(reader rentalReader, id string) *exit.Error {
	var unavailable bool
	if err := reader.QueryRow(`SELECT EXISTS(SELECT 1 FROM rentals WHERE id=? AND state IN ('release_requested','released','failed'))`, id).Scan(&unavailable); err != nil {
		return exit.Internalf("cannot inspect rental work admission: %s", err)
	}
	if unavailable {
		return exit.Named(exit.Conflict, "request.rental_unavailable", "the selected rental is being released or has ended")
	}
	return nil
}
