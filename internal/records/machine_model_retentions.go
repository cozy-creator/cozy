package records

import (
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/exit"
)

// This is recipient custody metadata in the existing observer journal. Native
// transaction identities come from the exact hashed Runtime outcome.
type MachineModelRetention struct {
	OutcomeID         string          `json:"outcome_id"`
	ResultPointer     string          `json:"result_pointer"`
	Artifact          json.RawMessage `json:"artifact"`
	TransactionID     string          `json:"transaction_id"`
	ReceiptDigest     []byte          `json:"receipt_digest"`
	SourceRetentionID string          `json:"source_retention_id"`
	RetentionID       string          `json:"retention_id"`
	State             string          `json:"state"`
	// Bytes is what the output's write added to the machine's disk, as its native
	// receipt reports it; inherited payload is its sources' and is not counted again.
	Bytes int64 `json:"bytes,omitempty"`
}

const machineModelRetentionOwed = `EXISTS(SELECT 1 FROM request_events hold
 WHERE hold.request_id=r.id AND hold.type='machine.model_retention'
 AND json_extract(hold.payload,'$.state')!='released'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=hold.request_id
 AND newer.type=hold.type AND newer.seq>hold.seq
 AND json_extract(newer.payload,'$.retention_id')=json_extract(hold.payload,'$.retention_id')))`

func (s *Store) MachineModelRetentions(request string) ([]MachineModelRetention, *exit.Error) {
	rows, err := s.db.Query(`SELECT hold.payload FROM request_events hold WHERE hold.request_id=? AND hold.type='machine.model_retention'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=hold.request_id AND newer.type=hold.type AND newer.seq>hold.seq
 AND json_extract(newer.payload,'$.retention_id')=json_extract(hold.payload,'$.retention_id')) ORDER BY hold.seq`, request)
	if err != nil {
		return nil, exit.Internalf("cannot read collected model custody: %s", err)
	}
	defer rows.Close()
	var holds []MachineModelRetention
	for rows.Next() {
		var raw []byte
		var hold MachineModelRetention
		if err := rows.Scan(&raw); err != nil || json.Unmarshal(raw, &hold) != nil {
			return nil, exit.Internalf("recorded model custody is unreadable")
		}
		holds = append(holds, hold)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish model custody read: %s", err)
	}
	return holds, nil
}
