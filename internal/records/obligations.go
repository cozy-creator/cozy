package records

import "github.com/cozy-creator/cozy/internal/exit"

// Obligation is a durable work or retention fact: what the idle exit waits on, what
// `cozy down --all` tears down, and what a plain down reports still in flight.
type Obligation struct {
	Kind  string
	ID    string
	State string
}

// Obligations is the full automatic-management census, read in ONE statement so
// no fence can see half of the picture. A rental counts until the hub proves it
// gone or explicitly confirms that creation never happened. Failed provisioning
// alone is not that proof. A rental operation counts until it is final, and only when no
// rental row already stands for it.
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
		SELECT 'rental',id,state FROM rentals
		 WHERE NOT (state='failed' AND failure_code='provider_create_did_not_happen')
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
