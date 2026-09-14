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

// ClientShutdownObligations is the explicit disconnect boundary. A machine's
// durable receipt permits its observer to stop while execution and custody stay
// with Runtime. Automatic idle release still uses the complete Obligations set.
func (s *Store) ClientShutdownObligations() ([]Obligation, *exit.Error) {
	all, problem := s.Obligations()
	if problem != nil {
		return nil, problem
	}
	links, problem := s.MachineExecutions()
	if problem != nil {
		return nil, problem
	}
	accepted, machines := map[string]bool{}, map[string]bool{}
	for _, link := range links {
		if len(link.Receipt) > 0 && len(link.PendingControl) == 0 && !link.CancelRequested {
			accepted[link.RequestID] = true
			machines[link.MachineID] = true
		}
	}
	var held []Obligation
	for _, obligation := range all {
		if (obligation.Kind == "job" || obligation.Kind == "invocation") && accepted[obligation.ID] ||
			obligation.Kind == "rental" && machines[obligation.ID] {
			continue
		}
		held = append(held, obligation)
	}
	for _, link := range links {
		if len(link.Receipt) > 0 && len(link.PendingControl) == 0 && !link.CancelRequested {
			continue
		}
		owed, problem := s.MachineExecutionOwesWork(link.RequestID)
		if problem != nil {
			return nil, problem
		}
		if !owed {
			continue
		}
		found := false
		for _, obligation := range held {
			found = found || obligation.ID == link.RequestID
		}
		if !found {
			held = append(held, Obligation{Kind: "job", ID: link.RequestID, State: "machine_control_pending"})
		}
	}
	return held, nil
}
