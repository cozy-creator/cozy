package records

import "github.com/cozy-creator/cozy/internal/exit"

// Obligation is one durable fact whose next transition only a running daemon can make.
// Kind is the vocabulary `cozy down` refuses in — invocation, job, rental,
// rental_operation — plus the two it drains rather than refuses: attempt (a terminal
// still owed its acknowledgement) and output_export (a --out copy in flight).
type Obligation struct {
	Kind  string
	ID    string
	State string
}

// Obligations is every durable reason this daemon may not stop, read in ONE statement so
// no fence can see half of the picture. A rental row counts in every state, including one
// whose release is requested but not yet confirmed by the hub: the row leaves only when
// the pod is proved gone. A rental operation counts until it is final, and only when no
// rental row already stands for it.
func (s *Store) Obligations() ([]Obligation, *exit.Error) {
	rows, err := s.db.Query(`
		SELECT CASE WHEN kind='job' THEN 'job' ELSE 'invocation' END,id,state FROM requests
		 WHERE state IN (` + activeRequestStates + `)
		UNION ALL
		SELECT 'job',r.id,r.state FROM requests r LEFT JOIN request_model_transfers t ON t.request_id=r.id
		 WHERE r.state='succeeded' AND r.retain_work=1 AND r.weights_outputs!='[]' AND COALESCE(json_extract(t.intent,'$.destination'),'')=''
		UNION ALL
		SELECT 'attempt',request_id||'#'||attempt,state FROM attempts
		 WHERE state IN (` + openAttemptStates + `)
		UNION ALL
		SELECT 'rental',id,state FROM rentals
		UNION ALL
		SELECT 'rental_operation',operation_key,state FROM rental_operations
		 WHERE state NOT IN (` + finalRentalOperationStates + `)
		   AND rental_id NOT IN (SELECT id FROM rentals)
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
