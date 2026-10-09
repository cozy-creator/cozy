package records

import (
	"database/sql"
	"encoding/json"
	"math"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RunV1RunningElapsedMS is a wall-clock display for the current running interval,
// not a substitute for Runtime's measured execution counter. Only Runtime's
// explicit transition timestamp can start it; import/receipt time cannot.
func (s *Store) RunV1RunningElapsedMS(id string, attempt int64, now time.Time) (int64, bool, *exit.Error) {
	var raw []byte
	err := s.db.QueryRow(`SELECT payload FROM request_events
 WHERE request_id=? AND attempt=? AND type='run.in_progress'
 ORDER BY seq DESC LIMIT 1`, id, attempt).Scan(&raw)
	if err == sql.ErrNoRows {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, exit.Internalf("cannot read Runtime's execution start: %s", err)
	}
	var stamp struct {
		Started int64 `json:"started_unix_ms"`
	}
	if json.Unmarshal(raw, &stamp) != nil || stamp.Started <= 0 || stamp.Started > now.UnixMilli() {
		return 0, false, nil
	}
	return now.UnixMilli() - stamp.Started, true, nil
}

// MachineRunExecutionMS consumes cumulative Runtime observations. A wall interval
// cannot recover cooperative execution for an older producer or a missing attempt.
// Live values stay at the latest measurement: observing a stalled/offline machine
// must not start a client-side execution clock.
func (s *Store) MachineRunExecutionMS(id string, current int64, running bool) (int64, bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT attempt,payload FROM request_events
 WHERE request_id=? AND type='machine.run.timing' ORDER BY seq`, id)
	if err != nil {
		return 0, false, exit.Internalf("cannot read measured machine timing: %s", err)
	}
	defer rows.Close()
	type measurement struct {
		Attempt     int64    `json:"attempt"`
		ExecutionMS *float64 `json:"execution_ms"`
		Terminal    bool     `json:"terminal"`
	}
	latest := map[int64]measurement{}
	for rows.Next() {
		var attempt int64
		var raw []byte
		if err := rows.Scan(&attempt, &raw); err != nil {
			return 0, false, exit.Internalf("cannot read measured machine timing: %s", err)
		}
		var value measurement
		if json.Unmarshal(raw, &value) != nil || value.Attempt != attempt {
			value = measurement{}
		}
		latest[attempt] = value
	}
	if err := rows.Err(); err != nil {
		return 0, false, exit.Internalf("cannot finish measured machine timing: %s", err)
	}
	var total float64
	if current <= 0 || int64(len(latest)) != current {
		return 0, false, nil
	}
	for attempt := int64(1); attempt <= current; attempt++ {
		value := latest[attempt]
		if value.ExecutionMS == nil || *value.ExecutionMS < 0 || math.IsNaN(*value.ExecutionMS) || math.IsInf(*value.ExecutionMS, 0) ||
			(!value.Terminal && (attempt != current || !running)) {
			return 0, false, nil
		}
		total += *value.ExecutionMS
	}
	if total >= math.MaxInt64 {
		return 0, false, nil
	}
	return int64(math.Round(total)), true, nil
}

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

// RunV1Span is when a cozy.machine.v1 run started running, when its machine ended it, and when
// it last rested paused; each zero until it happened.
func (s *Store) RunV1Span(id string) (started, ended, paused time.Time, problem *exit.Error) {
	rows, err := s.db.Query(`SELECT type,at,payload FROM request_events WHERE request_id=?
 AND type IN ('run.in_progress','client.machine_work_finished','request.paused') ORDER BY seq`, id)
	if err != nil {
		return started, ended, paused, exit.Internalf("cannot read the run's span: %s", err)
	}
	defer rows.Close()
	for rows.Next() {
		var kind, at string
		var raw []byte
		if err := rows.Scan(&kind, &at, &raw); err != nil {
			return started, ended, paused, exit.Internalf("cannot read the run's span: %s", err)
		}
		var stamps struct {
			Started  int64 `json:"started_unix_ms"`
			Finished int64 `json:"finished_unix_ms"`
		}
		_ = json.Unmarshal(raw, &stamps)
		when, _ := time.Parse(time.RFC3339Nano, at)
		switch {
		case kind == "run.in_progress" && started.IsZero():
			started = when
			if stamps.Started > 0 {
				started = time.UnixMilli(stamps.Started)
			}
		case kind == "client.machine_work_finished":
			ended = when
			if stamps.Finished > 0 {
				ended = time.UnixMilli(stamps.Finished)
			}
		case kind == "request.paused":
			paused = when
		}
	}
	if err := rows.Err(); err != nil {
		return started, ended, paused, exit.Internalf("cannot read the run's span: %s", err)
	}
	return started, ended, paused, nil
}
