package records

import (
	"database/sql"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/exit"
)

type MachineInput struct {
	InputID      string            `json:"input_id"`
	Manifest     ArtifactObjectRef `json:"manifest"`
	ContentBytes int64             `json:"content_bytes"`
	State        string            `json:"state"`
}

const machineInputOwed = `EXISTS(SELECT 1 FROM request_events intake
 WHERE intake.request_id=r.id AND intake.type='machine.input'
 AND json_extract(intake.payload,'$.state')!='released'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=intake.request_id
 AND newer.type=intake.type AND newer.seq>intake.seq
 AND json_extract(newer.payload,'$.input_id')=json_extract(intake.payload,'$.input_id')))`

func machineInputsIn(q interface {
	Query(string, ...any) (*sql.Rows, error)
}, request string) ([]MachineInput, *exit.Error) {
	rows, err := q.Query(`SELECT intake.payload FROM request_events intake
 WHERE intake.request_id=? AND intake.type='machine.input'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=intake.request_id
 AND newer.type=intake.type AND newer.seq>intake.seq
 AND json_extract(newer.payload,'$.input_id')=json_extract(intake.payload,'$.input_id')) ORDER BY intake.seq`, request)
	if err != nil {
		return nil, exit.Internalf("cannot read native input receipts: %s", err)
	}
	defer rows.Close()
	var result []MachineInput
	for rows.Next() {
		var raw []byte
		var input MachineInput
		if rows.Scan(&raw) != nil || json.Unmarshal(raw, &input) != nil {
			return nil, exit.Internalf("native input receipt is unreadable")
		}
		result = append(result, input)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish input receipt read: %s", err)
	}
	return result, nil
}

// MachineInputs reads retained input custody from the owner's event history.
func (s *Store) MachineInputs(request string) ([]MachineInput, *exit.Error) {
	return machineInputsIn(s.db, request)
}
