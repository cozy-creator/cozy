package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
)

// MachineEndpoint reads the public target retained in ordinary request events.
// No global machine roster, schema change or second journal is introduced.
func (s *Store) MachineEndpoint(name string) (*machineendpoint.Endpoint, *exit.Error) {
	return decodeMachineEndpoint(s.db.QueryRow(`SELECT e.payload FROM request_events e JOIN machine_executions m ON m.request_id=e.request_id WHERE m.machine_id=? AND e.type IN ('run.created','request.submitted') ORDER BY e.seq DESC LIMIT 1`, name), name)
}

// RequestMachineEndpoint reads that request's frozen public selector. A later
// submission to the same address cannot change this request's boot authority.
func (s *Store) RequestMachineEndpoint(id, name string) (*machineendpoint.Endpoint, *exit.Error) {
	return decodeMachineEndpoint(s.db.QueryRow(`SELECT e.payload FROM request_events e WHERE e.request_id=? AND e.type IN ('run.created','request.submitted') ORDER BY e.seq DESC LIMIT 1`, id), name)
}

func decodeMachineEndpoint(row *sql.Row, name string) (*machineendpoint.Endpoint, *exit.Error) {
	var body string
	err := row.Scan(&body)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read explicit machine endpoint: %s", err)
	}
	var event struct {
		Endpoint *machineendpoint.Endpoint `json:"machine_endpoint"`
	}
	if err = json.Unmarshal([]byte(body), &event); err != nil {
		return nil, exit.Internalf("retained machine endpoint is invalid: %s", err)
	}
	if event.Endpoint == nil || event.Endpoint.Name() != name {
		return nil, exit.New(exit.Conflict, "retained explicit endpoint identity is absent")
	}
	if err = event.Endpoint.Validate(); err != nil {
		return nil, exit.New(exit.Conflict, "retained explicit endpoint is invalid: %s", err)
	}
	return event.Endpoint, nil
}
