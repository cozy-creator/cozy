package records

import "github.com/cozy-creator/cozy/internal/exit"

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
