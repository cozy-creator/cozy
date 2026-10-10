package records

import (
	"database/sql"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

const rentalIdleDDL = `CREATE TABLE IF NOT EXISTS rental_idle (
 rental_id TEXT PRIMARY KEY REFERENCES rentals(id) ON DELETE CASCADE,
 worker_id TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 acknowledged_at_ms INTEGER NOT NULL DEFAULT 0,
 idle_deadline_ms INTEGER NOT NULL DEFAULT 0,
 receipt_observed_at TEXT NOT NULL DEFAULT '',
 work_finished_at TEXT NOT NULL DEFAULT ''
)`

// RentalIdleRunCounts excludes retained terminal attempts and counts only work
// pinned to this machine, including its explicitly purchased acquisition.
func (s *Store) RentalIdleRunCounts(id, buyer string) (int, int, *exit.Error) {
	return rentalIdleRunCounts(s.db, id, buyer)
}

func rentalIdleRunCounts(reader rentalIdleReader, id, buyer string) (queued, running int, problem *exit.Error) {
	pinned, args := pinnedToRental("r.", id)
	// A machine's acceptance receipt is live work from that moment, before any observation
	// projects the request out of the queue.
	const accepted = `EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND length(e.receipt)>0)`
	err := reader.QueryRow(`SELECT
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
	err = reader.QueryRow(`SELECT COALESCE(SUM(state='queued'),0),COALESCE(SUM(state='installing'),0) FROM rental_installs WHERE rental_id=? AND state IN ('queued','installing')`, id).Scan(&queuedInstalls, &runningInstalls)
	if err != nil {
		return 0, 0, exit.Internalf("cannot observe queued rental installations: %s", err)
	}
	return queued + queuedInstalls, running + runningInstalls, nil
}

// RentalKeepalive is the machine's answer to one explicit idle reset: the boot that reset it
// and the deadline it now holds, received at AcknowledgedAtMS on this computer's clock.
type RentalKeepalive struct {
	WorkerID, WorkerBootID           string
	AcknowledgedAtMS, IdleDeadlineMS int64
}

// RecordRentalKeepalive accepts only an acknowledgment from the rental's recorded boot.
// The machine owns expiry; a zero deadline names none. An older acknowledgment never
// replaces a newer observation.
func (s *Store) RecordRentalKeepalive(id string, result RentalKeepalive, observedAt time.Time) *exit.Error {
	if observedAt.IsZero() || result.WorkerID == "" || result.WorkerBootID == "" || result.AcknowledgedAtMS <= 0 || (result.IdleDeadlineMS != 0 && result.IdleDeadlineMS <= result.AcknowledgedAtMS) {
		return exit.New(exit.Conflict, "the machine returned an invalid rental keepalive acknowledgment")
	}
	updated, err := s.db.Exec(`INSERT INTO rental_idle(rental_id,worker_id,worker_boot_id,acknowledged_at_ms,idle_deadline_ms,receipt_observed_at)
 SELECT id,expected_worker_id,expected_worker_boot_id,?,?,? FROM rentals
 WHERE id=? AND state='ready' AND expected_worker_id=? AND expected_worker_boot_id=?
 ON CONFLICT(rental_id) DO UPDATE SET worker_id=excluded.worker_id,worker_boot_id=excluded.worker_boot_id,
 acknowledged_at_ms=excluded.acknowledged_at_ms,idle_deadline_ms=excluded.idle_deadline_ms,receipt_observed_at=excluded.receipt_observed_at
 WHERE excluded.acknowledged_at_ms>rental_idle.acknowledged_at_ms`, result.AcknowledgedAtMS, result.IdleDeadlineMS, observedAt.UTC().Format(time.RFC3339Nano), id, result.WorkerID, result.WorkerBootID)
	if err != nil {
		return exit.Internalf("cannot persist rental keepalive acknowledgment: %s", err)
	}
	count, err := updated.RowsAffected()
	if err != nil {
		return exit.Internalf("cannot confirm rental keepalive record: %s", err)
	}
	if count == 1 {
		return nil
	}
	row, problem := s.RentalRow(id)
	if problem != nil {
		return problem
	}
	if row == nil || row.State != "ready" || row.ExpectedWorkerID != result.WorkerID || row.ExpectedWorkerBootID != result.WorkerBootID {
		return exit.New(exit.Conflict, "rental identity or lifetime changed before keepalive acknowledgment")
	}
	// An older acknowledgment arriving late leaves the current deadline alone.
	return nil
}

func rentalIdleResetAt(reader rentalIdleReader, row Rental) (time.Time, *exit.Error) {
	var ack int64
	var worker, boot, finished, observed string
	err := reader.QueryRow(`SELECT worker_id,worker_boot_id,acknowledged_at_ms,work_finished_at,receipt_observed_at FROM rental_idle WHERE rental_id=?`, row.ID).Scan(&worker, &boot, &ack, &finished, &observed)
	if err == sql.ErrNoRows {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, exit.Internalf("cannot read rental idle receipt: %s", err)
	}
	var at time.Time
	if ack > 0 && worker == row.ExpectedWorkerID && boot == row.ExpectedWorkerBootID {
		// Host timestamps are identity/order facts; idle starts at this computer's receipt.
		at, err = time.Parse(time.RFC3339Nano, observed)
		if err != nil {
			return time.Time{}, exit.Internalf("invalid rental keepalive observation time: %s", err)
		}
	}
	if finished != "" {
		work, err := time.Parse(time.RFC3339Nano, finished)
		if err != nil {
			return time.Time{}, exit.Internalf("invalid rental work completion time: %s", err)
		}
		if work.After(at) {
			at = work
		}
	}
	return at, nil
}

type rentalIdleReader interface {
	Query(string, ...any) (*sql.Rows, error)
	QueryRow(string, ...any) *sql.Row
}

// RentalIdleState is this host's observation of one machine's work: what it has queued and
// running, and since when it has had none. When the machine ends itself is its own to say.
type RentalIdleState struct {
	Queued, Running int
	Since           time.Time
}

func (s *Store) RentalIdleObservation(row Rental) (RentalIdleState, *exit.Error) {
	return rentalIdleObservation(s.db, row)
}
func rentalIdleObservation(reader rentalIdleReader, row Rental) (RentalIdleState, *exit.Error) {
	var idle RentalIdleState
	var problem *exit.Error
	idle.Queued, idle.Running, problem = rentalIdleRunCounts(reader, row.ID, row.ManagedRequestID)
	if problem != nil {
		return idle, problem
	}
	if row.ReadyAt != "" {
		idle.Since, _ = time.Parse(time.RFC3339Nano, row.ReadyAt)
		if idle.Since.IsZero() {
			return idle, exit.Internalf("rental %s has an invalid ready timestamp", row.ID)
		}
	}
	last, found, problem := rentalLastSettlement(reader, row.ID)
	if problem != nil {
		return idle, problem
	}
	if found && last.SettledAt.After(idle.Since) {
		idle.Since = last.SettledAt
	}
	at, problem := rentalIdleResetAt(reader, row)
	if problem != nil {
		return idle, problem
	}
	if at.After(idle.Since) {
		idle.Since = at
	}
	return idle, nil
}

func refuseReleasedRental(reader rentalIdleReader, id string) *exit.Error {
	var unavailable bool
	if err := reader.QueryRow(`SELECT EXISTS(SELECT 1 FROM rentals WHERE id=? AND state IN ('release_requested','released','failed'))`, id).Scan(&unavailable); err != nil {
		return exit.Internalf("cannot inspect rental work admission: %s", err)
	}
	if unavailable {
		return exit.Named(exit.Conflict, "request.rental_unavailable", "the selected rental is being released or has ended")
	}
	return nil
}
