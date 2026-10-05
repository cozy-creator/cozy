package records

import "github.com/cozy-creator/cozy/internal/exit"

// MachineLiveRuns is every run a machine holds that has not ended: replacing the machine
// loses them. A run that ended (completed, failed, canceled, refused, abandoned) owes the
// machine only its own record of it.
func (s *Store) MachineLiveRuns(machine string) ([]Request, *exit.Error) {
	return s.machineRuns(machine, `r.state NOT IN (`+settledRequestStates+`) AND `+machineExecutionLive)
}

// MachineUncollectedRuns is every completed run on a machine that this client never finished
// collecting: an output it has not saved exists only on that machine.
func (s *Store) MachineUncollectedRuns(machine string) ([]Request, *exit.Error) {
	return s.machineRuns(machine, `r.state='succeeded' AND length(e.receipt)>0 AND e.collected=0 AND NOT `+
		machineExecutionLost+` AND NOT `+machineRetentionReleased)
}

// machineRuns is the numbered runs on machine whose execution (e) and request (r) meet where.
func (s *Store) machineRuns(machine, where string) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`WITH numbered AS (SELECT ROW_NUMBER() OVER (ORDER BY created_at,id) AS number,
		id AS numbered_id FROM requests)
		SELECT numbered.number, `+requestCols+` FROM requests r JOIN numbered ON numbered.numbered_id=r.id
		WHERE EXISTS(SELECT 1 FROM machine_executions e WHERE e.request_id=r.id AND e.machine_id=?1 AND `+where+`)
		ORDER BY r.created_at, r.id`, machine)
	if err != nil {
		return nil, exit.Internalf("cannot read the machine's runs: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanNumberedRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a machine run row: %s", err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the machine's runs: %s", err)
	}
	return out, nil
}
