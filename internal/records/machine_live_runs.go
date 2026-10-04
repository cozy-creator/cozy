package records

import "github.com/cozy-creator/cozy/internal/exit"

// MachineLiveRuns is every run a machine holds unsettled: replacing the machine loses them.
func (s *Store) MachineLiveRuns(machine string) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`WITH numbered AS (SELECT ROW_NUMBER() OVER (ORDER BY created_at,id) AS number,
		id AS numbered_id FROM requests)
		SELECT numbered.number, `+requestCols+` FROM requests r JOIN numbered ON numbered.numbered_id=r.id
		WHERE EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND e.machine_id=?1 AND `+machineExecutionLive+`)
		ORDER BY r.created_at, r.id`, machine)
	if err != nil {
		return nil, exit.Internalf("cannot read the machine's live runs: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanNumberedRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a live run row: %s", err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the machine's live runs: %s", err)
	}
	return out, nil
}
