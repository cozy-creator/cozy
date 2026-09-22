package records

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"strings"
)

// Obligation is a durable work or retention fact. Automatic idle management and
// destructive teardown keep the complete set; explicit disconnect filters it to
// work that still requires the client online.
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
	accepted := map[string]bool{}
	for _, link := range links {
		if len(link.Receipt) > 0 && len(link.PendingControl) == 0 && !link.CancelRequested {
			accepted[link.RequestID] = true
		}
	}
	// A retained row is custody, not proof of an executing daemon child. Keep
	// unfinished handoffs guarded; remote accepted attempts can replay their
	// supervisor ledger after reconnect without running another attempt.
	attempts := map[string][]Obligation{}
	for _, obligation := range all {
		if obligation.Kind == "attempt" {
			id, _, _ := strings.Cut(obligation.ID, "#")
			attempts[id] = append(attempts[id], obligation)
		}
	}
	for _, obligation := range all {
		if obligation.Kind != "job" && obligation.Kind != "invocation" {
			continue
		}
		live, remoteAccepted := false, len(attempts[obligation.ID]) > 0
		for _, attempt := range attempts[obligation.ID] {
			live = live || attempt.State != "terminal"
			remoteAccepted = remoteAccepted && (attempt.State == "accepted" || attempt.State == "recovered_open" || attempt.State == "terminal")
		}
		if !live && (obligation.State == "paused" || obligation.State == "blocked" || obligation.State == "succeeded") {
			accepted[obligation.ID] = true
		} else if remoteAccepted {
			row, problem := s.RequestRow(obligation.ID)
			if problem != nil {
				return nil, problem
			}
			if row != nil && row.Worker != "" {
				accepted[obligation.ID] = true
			}
		}
	}
	var held []Obligation
	for _, obligation := range all {
		if obligation.Kind == "rental" || obligation.Kind == "output_export" && obligation.State == "pending" {
			// An idle rental or deferred output contract owns durable resources,
			// but no current client-side copy. Active exporters are fenced below.
			continue
		}
		id := obligation.ID
		if obligation.Kind == "attempt" {
			id, _, _ = strings.Cut(id, "#")
		}
		if (obligation.Kind == "job" || obligation.Kind == "invocation" || obligation.Kind == "attempt") && accepted[id] {
			continue
		}
		held = append(held, obligation)
	}
	for _, link := range links {
		if len(link.Receipt) > 0 && len(link.PendingControl) == 0 && !link.CancelRequested {
			continue
		}
		row, problem := s.RequestRow(link.RequestID)
		if problem != nil {
			return nil, problem
		}
		if row != nil && (Settled(row.State) || row.State == "paused" || row.State == "blocked") && len(link.PendingControl) == 0 && !link.CancelRequested {
			continue // retained custody alone does not require the client online
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
