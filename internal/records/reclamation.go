package records

import "github.com/cozy-creator/cozy/internal/exit"

// Reclamation fences. TensorFS decides what its store holds: a live writer or read lease
// refuses a pass, and open ingest sessions, conversion journals and retained roots keep
// their objects. It cannot see the daemon's own work between two tfs calls (a Hub
// download's admitted objects stay unnamed until its commit), nor who will resume an
// ingest session whose writer has exited. These queries name exactly those requests, so a
// stale row fences nothing but itself.

// localLive is a request that may place work on this machine and has not stopped: retained
// stops (paused, blocked) move nothing, and a request bound to a rental writes into the
// pod's store.
const localLive = `r.worker='' AND r.rental_required=0 AND r.state IN ('submitted','queued','dispatching',
	'requeue_pending','finalizing','pausing','canceling','releasing')`

// LocalStoreWriters is every request that may be moving bytes into this machine's store
// now: an open local attempt, a local transition in flight, or the daemon's own model
// transfer between materialization and commit. A queued request has moved nothing yet.
func (s *Store) LocalStoreWriters() ([]Request, *exit.Error) {
	return s.numberedRequests(localLive + ` AND (
		r.state IN ('dispatching','finalizing','pausing','canceling','releasing')
		OR EXISTS(SELECT 1 FROM attempts a WHERE a.request_id=r.id AND a.state IN (` + openAttemptStates + `))
		OR EXISTS(SELECT 1 FROM request_model_transfers t WHERE t.request_id=r.id
		          AND t.state IN ('materializing','materialized','finalizing','canceling')))`)
}

// LocalRetainedStops is every local request stopped with its work retained. Its retry may
// resume an ingest session whose writer has exited, so reaping waits for a retry or cancel.
func (s *Store) LocalRetainedStops() ([]Request, *exit.Error) {
	return s.numberedRequests(`r.worker='' AND r.retain_work=1 AND r.state IN ('paused','blocked')`)
}

// LocalModelUsers is every live local request that reads or writes the named local model.
func (s *Store) LocalModelUsers(model string) ([]Request, *exit.Error) {
	return s.numberedRequests(localLive+` AND (
		EXISTS(SELECT 1 FROM json_each(CASE WHEN json_valid(r.models) THEN r.models ELSE '[]' END) m
		       WHERE json_extract(m.value,'$.model')=?1 OR json_extract(m.value,'$.catalog_repository')=?1)
		OR EXISTS(SELECT 1 FROM request_model_transfers t WHERE t.request_id=r.id AND (
		       json_extract(t.intent,'$.destination')=?1 OR json_extract(t.intent,'$.source')=?1
		       OR substr(json_extract(t.intent,'$.source'),1,length(?1)+1)=?1||'@')))`, model)
}

func (s *Store) numberedRequests(where string, args ...any) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`WITH numbered AS (SELECT ROW_NUMBER() OVER (ORDER BY created_at,id) AS number,
		id AS numbered_id FROM requests)
		SELECT numbered.number, `+requestCols+` FROM requests r JOIN numbered ON numbered.numbered_id=r.id
		WHERE `+where+` ORDER BY r.created_at, r.id`, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read the reclamation fence: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanNumberedRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a reclamation fence row: %s", err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the reclamation fence: %s", err)
	}
	return out, nil
}
