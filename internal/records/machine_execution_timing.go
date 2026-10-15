package records

import (
	"encoding/json"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RunV1RunningMS is how long a cozy.machine.v1 run has run on its machine, by that machine's
// clock: from its start running to its end. While live, an
// interval still open counts to now. ran says whether it ever started running.
func (s *Store) RunV1RunningMS(id string, now time.Time, live bool) (ms int64, ran bool, problem *exit.Error) {
	rows, err := s.db.Query(`SELECT type,at,payload FROM request_events WHERE request_id=?
 AND type IN ('run.in_progress','client.machine_work_finished') ORDER BY seq`, id)
	if err != nil {
		return 0, false, exit.Internalf("cannot read the run's running time: %s", err)
	}
	defer rows.Close()
	var since time.Time
	for rows.Next() {
		var kind, at string
		var raw []byte
		if err := rows.Scan(&kind, &at, &raw); err != nil {
			return 0, false, exit.Internalf("cannot read the run's running time: %s", err)
		}
		var stamps struct {
			Started  int64 `json:"started_unix_ms"`
			Finished int64 `json:"finished_unix_ms"`
		}
		_ = json.Unmarshal(raw, &stamps)
		when, _ := time.Parse(time.RFC3339Nano, at)
		if machine := max(stamps.Started, stamps.Finished); machine > 0 {
			when = time.UnixMilli(machine)
		}
		switch {
		case kind == "run.in_progress":
			ran = true
			if since.IsZero() {
				since = when
			}
		case !since.IsZero():
			ms += max(when.Sub(since).Milliseconds(), 0)
			since = time.Time{}
		}
	}
	if err := rows.Err(); err != nil {
		return 0, false, exit.Internalf("cannot read the run's running time: %s", err)
	}
	if live && !since.IsZero() {
		ms += max(now.Sub(since).Milliseconds(), 0)
	}
	return ms, ran, nil
}
