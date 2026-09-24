package records

import (
	"database/sql"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const rentalIdleDDL = `CREATE TABLE IF NOT EXISTS rental_idle (
 rental_id TEXT PRIMARY KEY REFERENCES rentals(id) ON DELETE CASCADE,
 worker_id TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 request_id TEXT NOT NULL DEFAULT '',
 acknowledged_at_ms INTEGER NOT NULL DEFAULT 0,
 idle_deadline_ms INTEGER NOT NULL DEFAULT 0,
 receipt_observed_at TEXT NOT NULL DEFAULT '',
 work_finished_at TEXT NOT NULL DEFAULT '',
 preparation_pending INTEGER NOT NULL DEFAULT 0
)`

// RentalIdleRunCounts excludes retained terminal attempts and counts only work
// assigned to this machine, including its explicitly purchased acquisition.
func (s *Store) RentalIdleRunCounts(id, buyer string) (queued, running int, problem *exit.Error) {
	err := s.db.QueryRow(`SELECT
 COALESCE(SUM(CASE WHEN r.state IN ('submitted','queued','requeue_pending') THEN 1 ELSE 0 END),0),
 COALESCE(SUM(CASE WHEN r.state IN ('dispatching','finalizing','pausing','canceling') OR
 EXISTS(SELECT 1 FROM attempts a WHERE a.request_id=r.id AND a.state IN ('preparing','offered','accepted','recovered_open')) THEN 1 ELSE 0 END),0)
 FROM requests r WHERE r.worker=? OR
 (r.id=? AND r.worker='') OR EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND e.machine_id=?)`, id, buyer, id).Scan(&queued, &running)
	if err != nil {
		return 0, 0, exit.Internalf("cannot observe rental work: %s", err)
	}
	return queued, running, nil
}

// RecordRentalKeepalive accepts only a full, identity-bound Host acknowledgment.
// Creator schedules from the first local observation, not the Host clock.
// Older receipts never move that clock backward, and a replay cannot renew it.
func (s *Store) RecordRentalKeepalive(id string, result *pb.KeepRentalAliveResult, observedAt time.Time) *exit.Error {
	if observedAt.IsZero() || result == nil || result.RequestId == "" || len(result.RequestId) > pb.MaxRentalKeepaliveRequestIDBytes || result.WorkerId == "" || result.WorkerBootId == "" || result.AcknowledgedAtUnixMs <= 0 || result.IdleDeadlineUnixMs <= result.AcknowledgedAtUnixMs || result.IdleDeadlineUnixMs-result.AcknowledgedAtUnixMs != pb.RentalIdleTimeoutSeconds*1000 {
		return exit.New(exit.Conflict, "worker returned an invalid rental keepalive receipt")
	}
	updated, err := s.db.Exec(`INSERT INTO rental_idle(rental_id,worker_id,worker_boot_id,request_id,acknowledged_at_ms,idle_deadline_ms,receipt_observed_at)
 SELECT id,expected_worker_id,expected_worker_boot_id,?,?,?,? FROM rentals
 WHERE id=? AND state='ready' AND expected_worker_id=? AND expected_worker_boot_id=?
 ON CONFLICT(rental_id) DO UPDATE SET worker_id=excluded.worker_id,worker_boot_id=excluded.worker_boot_id,
 request_id=excluded.request_id,acknowledged_at_ms=excluded.acknowledged_at_ms,idle_deadline_ms=excluded.idle_deadline_ms,receipt_observed_at=excluded.receipt_observed_at
 WHERE excluded.acknowledged_at_ms>rental_idle.acknowledged_at_ms AND
 excluded.request_id!=rental_idle.request_id`, result.RequestId, result.AcknowledgedAtUnixMs, result.IdleDeadlineUnixMs, observedAt.UTC().Format(time.RFC3339Nano), id, result.WorkerId, result.WorkerBootId)
	if err != nil {
		return exit.Internalf("cannot persist rental keepalive acknowledgment: %s", err)
	}
	count, err := updated.RowsAffected()
	if err != nil {
		return exit.Internalf("cannot confirm rental keepalive record: %s", err)
	}
	if count != 1 {
		row, problem := s.RentalRow(id)
		if problem != nil {
			return problem
		}
		if row == nil || row.State != "ready" || row.ExpectedWorkerID != result.WorkerId || row.ExpectedWorkerBootID != result.WorkerBootId {
			return exit.New(exit.Conflict, "rental identity or lifetime changed before keepalive acknowledgment")
		}
		var request string
		var ack, deadline int64
		if err := s.db.QueryRow(`SELECT request_id,acknowledged_at_ms,idle_deadline_ms FROM rental_idle WHERE rental_id=?`, id).Scan(&request, &ack, &deadline); err != nil {
			return exit.Internalf("cannot confirm prior keepalive: %s", err)
		}
		if request == result.RequestId && (ack != result.AcknowledgedAtUnixMs || deadline != result.IdleDeadlineUnixMs) {
			return exit.New(exit.Conflict, "worker changed an idempotent keepalive acknowledgment")
		}
		// A delayed replay of an older receipt is acknowledged without moving
		// the current deadline backward or extending it again.
	}
	return nil
}

func (s *Store) RecordRentalPreparationStarted(id string) *exit.Error {
	result, err := s.db.Exec(`INSERT INTO rental_idle(rental_id,preparation_pending) SELECT id,1 FROM rentals WHERE id=? AND state='ready' ON CONFLICT(rental_id) DO UPDATE SET preparation_pending=preparation_pending+1`, id)
	if err != nil {
		return exit.Internalf("cannot record rental preparation: %s", err)
	}
	count, err := result.RowsAffected()
	if err != nil || count != 1 {
		return exit.New(exit.Conflict, "rental is no longer ready for preparation")
	}
	return nil
}

func (s *Store) RecordRentalWorkFinished(id string, at time.Time) *exit.Error {
	_, err := s.db.Exec(`INSERT INTO rental_idle(rental_id,work_finished_at) SELECT id,? FROM rentals WHERE id=?
 ON CONFLICT(rental_id) DO UPDATE SET work_finished_at=excluded.work_finished_at,preparation_pending=MAX(0,preparation_pending-1)`, at.UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return exit.Internalf("cannot record rental work completion: %s", err)
	}
	return nil
}

func (s *Store) RentalIdleResetAt(row Rental) (time.Time, int, *exit.Error) {
	var ack int64
	var pending int
	var worker, boot, finished, observed string
	err := s.db.QueryRow(`SELECT worker_id,worker_boot_id,acknowledged_at_ms,work_finished_at,preparation_pending,receipt_observed_at FROM rental_idle WHERE rental_id=?`, row.ID).Scan(&worker, &boot, &ack, &finished, &pending, &observed)
	if err == sql.ErrNoRows {
		return time.Time{}, 0, nil
	}
	if err != nil {
		return time.Time{}, 0, exit.Internalf("cannot read rental idle receipt: %s", err)
	}
	var at time.Time
	if ack > 0 && worker == row.ExpectedWorkerID && boot == row.ExpectedWorkerBootID {
		// Host timestamps are identity/order facts, never a laptop clock baseline.
		// Scheduling from first receipt avoids early DELETE when clocks differ.
		at, err = time.Parse(time.RFC3339Nano, observed)
		if err != nil {
			return time.Time{}, 0, exit.Internalf("invalid rental keepalive observation time: %s", err)
		}
	}
	if finished != "" {
		work, err := time.Parse(time.RFC3339Nano, finished)
		if err != nil {
			return time.Time{}, 0, exit.Internalf("invalid rental work completion time: %s", err)
		}
		if work.After(at) {
			at = work
		}
	}
	return at, pending, nil
}
