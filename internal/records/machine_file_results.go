package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// MachineFileResult records receipt-bound recipient custody. Execution attempts
// remain exclusively in Runtime; this row only describes the client's copy.
type MachineFileResult struct {
	OutcomeID   string     `json:"outcome_id"`
	RetentionID string     `json:"retention_id"`
	Source      ByteOutput `json:"source"`
	Output      Output     `json:"output"`
	State       string     `json:"state"`
	Copied      bool       `json:"copied"`
}

const machineFileResultOwed = `EXISTS(SELECT 1 FROM request_events hold
 WHERE hold.request_id=r.id AND hold.type='machine.file_result'
 AND json_extract(hold.payload,'$.state')!='released'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=hold.request_id
 AND newer.type=hold.type AND newer.seq>hold.seq
 AND json_extract(newer.payload,'$.retention_id')=json_extract(hold.payload,'$.retention_id')))`

func (s *Store) MachineFileResults(request string) ([]MachineFileResult, *exit.Error) {
	rows, err := s.db.Query(`SELECT hold.payload FROM request_events hold
 WHERE hold.request_id=? AND hold.type='machine.file_result'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=hold.request_id
 AND newer.type=hold.type AND newer.seq>hold.seq
 AND json_extract(newer.payload,'$.retention_id')=json_extract(hold.payload,'$.retention_id')) ORDER BY hold.seq`, request)
	if err != nil {
		return nil, exit.Internalf("cannot read received files: %s", err)
	}
	defer rows.Close()
	var result []MachineFileResult
	for rows.Next() {
		var raw []byte
		var file MachineFileResult
		if rows.Scan(&raw) != nil || json.Unmarshal(raw, &file) != nil {
			return nil, exit.Internalf("received file metadata is unreadable")
		}
		result = append(result, file)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish received file read: %s", err)
	}
	return result, nil
}

func (s *Store) FreezeMachineFileResult(request string, file MachineFileResult) (MachineFileResult, *exit.Error) {
	file.State = "pending"
	file.Copied = false
	tx, err := s.db.Begin()
	if err != nil {
		return file, exit.Internalf("cannot begin file custody: %s", err)
	}
	defer tx.Rollback()
	var raw []byte
	err = tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.file_result' AND json_extract(payload,'$.retention_id')=? ORDER BY seq DESC LIMIT 1`, request, file.RetentionID).Scan(&raw)
	if err == nil {
		var prior MachineFileResult
		if json.Unmarshal(raw, &prior) != nil {
			return file, exit.Internalf("received file metadata is unreadable")
		}
		comparison := prior
		comparison.State = "pending"
		comparison.Copied = false
		comparison.Output.MediaID = file.Output.MediaID
		a, _ := json.Marshal(comparison)
		b, _ := json.Marshal(file)
		if string(a) != string(b) || prior.State == "releasing" || prior.State == "released" && !prior.Copied {
			return file, exit.New(exit.Conflict, "received file custody changed or is being canceled")
		}
		return prior, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return file, exit.Internalf("cannot read file custody: %s", err)
	}
	var outcomeBytes []byte
	var outcome pb.AttemptOutcome
	if tx.QueryRow(`SELECT e.outcome FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND length(e.outcome)>0 AND e.cancel_requested=0 AND length(e.pending_control)=0 AND r.state!='canceled'`, request).Scan(&outcomeBytes) != nil || proto.Unmarshal(outcomeBytes, &outcome) != nil || outcome.OutcomeId != file.OutcomeID {
		return file, exit.New(exit.Conflict, "file collection has no uncanceled observed outcome")
	}
	if problem := appendMachineFileResult(tx, request, file); problem != nil {
		return file, problem
	}
	if err := tx.Commit(); err != nil {
		return file, exit.Internalf("cannot freeze file custody: %s", err)
	}
	return file, nil
}

func (s *Store) AdvanceMachineFileResult(request, retention, state string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin file custody update: %s", err)
	}
	defer tx.Rollback()
	var raw []byte
	var file MachineFileResult
	if tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.file_result' AND json_extract(payload,'$.retention_id')=? ORDER BY seq DESC LIMIT 1`, request, retention).Scan(&raw) != nil || json.Unmarshal(raw, &file) != nil {
		return exit.New(exit.Conflict, "received file custody is absent")
	}
	if file.State == state {
		return nil
	}
	if !(file.State == "pending" && state == "copied" || file.State == "copied" && state == "released" || (file.State == "pending" || file.State == "copied") && state == "releasing" || file.State == "releasing" && state == "released") {
		return exit.New(exit.Conflict, "file custody transition is stale")
	}
	file.State = state
	if state == "copied" {
		file.Copied = true
	}
	if problem := appendMachineFileResult(tx, request, file); problem != nil {
		return problem
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit file custody: %s", err)
	}
	return nil
}

func appendMachineFileResult(tx *sql.Tx, request string, file MachineFileResult) *exit.Error {
	raw, err := json.Marshal(file)
	if err != nil || len(raw) > 16<<10 {
		return exit.New(exit.Validation, "received file metadata exceeds its bound")
	}
	if _, err = tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.file_result',0,?,?)`, request, raw, now()); err != nil {
		return exit.Internalf("cannot record file custody: %s", err)
	}
	return nil
}
