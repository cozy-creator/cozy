package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
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

func (s *Store) FreezeMachineModelRetention(request string, hold MachineModelRetention) *exit.Error {
	hold.State = "pending"
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model recipient custody: %s", err)
	}
	defer tx.Rollback()
	var raw []byte
	err = tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.model_retention' AND json_extract(payload,'$.retention_id')=? ORDER BY seq DESC LIMIT 1`, request, hold.RetentionID).Scan(&raw)
	if err == nil {
		var prior MachineModelRetention
		if json.Unmarshal(raw, &prior) != nil {
			return exit.Internalf("recorded model recipient custody is unreadable")
		}
		state := prior.State
		prior.State = "pending"
		expected, _ := json.Marshal(hold)
		actual, _ := json.Marshal(prior)
		if string(expected) != string(actual) || state == "releasing" || state == "released" {
			return exit.New(exit.Conflict, "model recipient custody was changed or released")
		}
		return nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return exit.Internalf("cannot inspect prior model recipient custody: %s", err)
	}
	var outcomeBytes []byte
	var outcome pb.AttemptOutcome
	if err := tx.QueryRow(`SELECT e.outcome FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND length(e.outcome)>0 AND e.cancel_requested=0 AND r.state!='canceled'`, request).Scan(&outcomeBytes); err != nil || proto.Unmarshal(outcomeBytes, &outcome) != nil || outcome.OutcomeId != hold.OutcomeID {
		return exit.New(exit.Conflict, "model collection has no uncanceled observed outcome")
	}
	if problem := appendMachineModelRetention(tx, request, hold); problem != nil {
		return problem
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot freeze model recipient custody: %s", err)
	}
	return nil
}

func (s *Store) AdvanceMachineModelRetention(request, retention, state string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model custody update: %s", err)
	}
	defer tx.Rollback()
	var raw []byte
	var hold MachineModelRetention
	if err := tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.model_retention' AND json_extract(payload,'$.retention_id')=? ORDER BY seq DESC LIMIT 1`, request, retention).Scan(&raw); err != nil || json.Unmarshal(raw, &hold) != nil {
		return exit.New(exit.Conflict, "model recipient custody is absent or unreadable")
	}
	if hold.State == state {
		return nil
	}
	if !(hold.State == "pending" && state == "held" || (hold.State == "pending" || hold.State == "held") && state == "releasing" || hold.State == "releasing" && state == "released") {
		return exit.New(exit.Conflict, "model recipient custody transition is stale")
	}
	hold.State = state
	if problem := appendMachineModelRetention(tx, request, hold); problem != nil {
		return problem
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model custody update: %s", err)
	}
	return nil
}

func appendMachineModelRetention(tx *sql.Tx, request string, hold MachineModelRetention) *exit.Error {
	raw, err := json.Marshal(hold)
	if err != nil || len(raw) > 16<<10 {
		return exit.New(exit.Validation, "model recipient custody exceeds its metadata bound")
	}
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.model_retention',0,?,?)`, request, raw, now()); err != nil {
		return exit.Internalf("cannot record model recipient custody: %s", err)
	}
	return nil
}
