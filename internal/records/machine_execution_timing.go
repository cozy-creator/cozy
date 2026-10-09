package records

import (
	"encoding/json"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

type MachineExecutionInterval struct {
	Attempt               int64
	ResumedAt, FinishedAt string
}

// MachineExecutionIntervals reads Runtime's retained transition timestamps,
// without scanning progress payloads or inventing Creator-owned attempt rows.
// Acceptance starts the first interval; a resume admits each subsequent one.
func (s *Store) MachineExecutionIntervals(id string) ([]MachineExecutionInterval, *exit.Error) {
	rows, err := s.db.Query(`SELECT attempt,
 COALESCE(MIN(CASE WHEN type='machine.control' THEN at END),''),
 COALESCE(MAX(CASE WHEN type IN ('machine.outcome','machine.terminal') THEN at END),
 MIN(CASE WHEN type='client.machine_lost' THEN at END),'')
 FROM request_events WHERE request_id=? AND (
 type IN ('machine.outcome','machine.terminal','client.machine_lost') OR
 (type='machine.control' AND json_extract(payload,'$.action')='resume'))
 GROUP BY attempt ORDER BY attempt`, id)
	if err != nil {
		return nil, exit.Internalf("cannot read retained machine timing: %s", err)
	}
	defer rows.Close()
	var intervals []MachineExecutionInterval
	for rows.Next() {
		var interval MachineExecutionInterval
		if err := rows.Scan(&interval.Attempt, &interval.ResumedAt, &interval.FinishedAt); err != nil {
			return nil, exit.Internalf("cannot read machine attempt timing: %s", err)
		}
		intervals = append(intervals, interval)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading machine timing: %s", err)
	}
	return intervals, nil
}

// RunV1RunningMS is how long a cozy.machine.v1 run has run on its machine, by that machine's
// clock: each interval from its start running to its pause or end, summed. While live, an
// interval still open counts to now. ran says whether it ever started running.
func (s *Store) RunV1RunningMS(id string, now time.Time, live bool) (ms int64, ran bool, problem *exit.Error) {
	rows, err := s.db.Query(`SELECT type,at,payload FROM request_events WHERE request_id=?
 AND type IN ('run.in_progress','request.paused','client.machine_work_finished') ORDER BY seq`, id)
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
			Paused   int64 `json:"paused_unix_ms"`
			Finished int64 `json:"finished_unix_ms"`
		}
		_ = json.Unmarshal(raw, &stamps)
		when, _ := time.Parse(time.RFC3339Nano, at)
		if machine := max(stamps.Started, stamps.Paused, stamps.Finished); machine > 0 {
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
