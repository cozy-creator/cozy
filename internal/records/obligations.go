package records

import "github.com/cozy-creator/cozy/internal/exit"

// The idle watch asks for the census every second: each part reads only its own rows.
var obligationIndexes = []string{
	`CREATE INDEX IF NOT EXISTS requests_state ON requests(state)`,
	`CREATE INDEX IF NOT EXISTS requests_retained_work ON requests(state) WHERE retain_work=1`,
	`CREATE INDEX IF NOT EXISTS attempts_state ON attempts(state)`,
	`CREATE INDEX IF NOT EXISTS request_output_exports_state ON request_output_exports(state)`,
}

// Obligation is a durable work or retention fact: what the idle exit waits on, what
// `cozy down --all` tears down, and what a plain down reports still in flight.
type Obligation struct {
	Kind  string
	ID    string
	State string
}

// Obligations is the full automatic-management census, read in ONE statement so
// no fence can see half of the picture. A rental counts until the hub reports it failed
// or released: the hub records either only once the provider holds nothing. A rental
// operation counts until it is final, and only when no rental row already stands for it.
func (s *Store) Obligations() ([]Obligation, *exit.Error) {
	rows, err := s.db.Query(`
		SELECT CASE WHEN kind='job' THEN 'job' ELSE 'invocation' END,id,state FROM requests
		 WHERE state IN (` + activeRequestStates + `)
		UNION ALL
		SELECT 'job',r.id,r.state FROM requests r LEFT JOIN request_model_transfers t ON t.request_id=r.id
		 WHERE r.state='succeeded' AND r.retain_work=1 AND (r.child_artifacts=1 OR r.weights_outputs!='[]') AND COALESCE(json_extract(t.intent,'$.destination'),'')=''
		UNION ALL
		SELECT 'attempt',request_id||'#'||attempt,state FROM attempts
		 WHERE state IN (` + openAttemptStates + `)
		UNION ALL
		SELECT 'rental',id,state FROM rentals WHERE state NOT IN (` + absentRentalStates + `)
		UNION ALL
		SELECT 'rental_operation',operation_key,state FROM rental_operations
		 WHERE state NOT IN (` + finalRentalOperationStates + `)
		   AND rental_id NOT IN (SELECT id FROM rentals)
		UNION ALL
		SELECT 'rental_install',id,state FROM rental_installs WHERE state IN ('queued','installing')
		UNION ALL
		SELECT 'output_export',request_id,state FROM request_output_exports
		 WHERE state IN ('pending','exporting')
		ORDER BY 1,2`)
	if err != nil {
		return nil, exit.Internalf("cannot read the daemon's obligations: %s", err)
	}
	defer rows.Close()
	var out []Obligation
	for rows.Next() {
		var o Obligation
		if err := rows.Scan(&o.Kind, &o.ID, &o.State); err != nil {
			return nil, exit.Internalf("cannot read an obligation row: %s", err)
		}
		out = append(out, o)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the daemon's obligations: %s", err)
	}
	return out, nil
}

func (o Obligation) String() string { return o.Kind + " " + o.ID + " (" + o.State + ")" }

// AtRest is work that waits on its owner alone: a paused run, or a finished one keeping its
// results. Nothing happens to it until a command asks, so it holds no daemon up.
func (o Obligation) AtRest() bool { return o.State == "paused" || o.State == "succeeded" }
