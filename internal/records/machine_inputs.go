package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// MachineInput is the observer's receipt for one exact native input intake.
// Its execution recipient and native bytes belong to Runtime, not a client attempt.
type MachineInput struct {
	InputID      string            `json:"input_id"`
	Manifest     ArtifactObjectRef `json:"manifest"`
	ContentBytes int64             `json:"content_bytes"`
	State        string            `json:"state"`
	Receipt      []byte            `json:"receipt,omitempty"`
}

const machineInputOwed = `EXISTS(SELECT 1 FROM request_events intake
 WHERE intake.request_id=r.id AND intake.type='machine.input'
 AND json_extract(intake.payload,'$.state')!='released'
 AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=intake.request_id
 AND newer.type=intake.type AND newer.seq>intake.seq
 AND json_extract(newer.payload,'$.input_id')=json_extract(intake.payload,'$.input_id')))`

func (s *Store) MachineInputs(request string) ([]MachineInput, *exit.Error) {
	rows, err := s.db.Query(`SELECT intake.payload FROM request_events intake
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

func (s *Store) BeginMachineInput(request string, asset AssetBinding) (MachineInput, *exit.Error) {
	if asset.Snapshot == nil {
		return MachineInput{}, exit.New(exit.Conflict, "root input has no immutable snapshot")
	}
	desired := MachineInput{InputID: asset.FieldPath, Manifest: asset.Snapshot.Manifest, ContentBytes: asset.Snapshot.ContentBytes, State: "pending"}
	inputs, problem := s.MachineInputs(request)
	if problem != nil {
		return desired, problem
	}
	for _, input := range inputs {
		if input.InputID == desired.InputID {
			if input.Manifest != desired.Manifest || input.ContentBytes != desired.ContentBytes {
				return desired, exit.New(exit.Conflict, "native input intake changed its frozen subject")
			}
			return input, nil
		}
	}
	row, problem := s.RequestRow(request)
	if problem != nil {
		return desired, problem
	}
	if row == nil || row.State == "canceled" {
		return desired, exit.New(exit.Canceled, "input capture was canceled before intake")
	}
	if problem := s.AppendEvent(request, "machine.input", 0, map[string]any{"input_id": desired.InputID, "manifest": desired.Manifest, "content_bytes": desired.ContentBytes, "state": desired.State}); problem != nil {
		return desired, problem
	}
	return desired, nil
}

func (s *Store) RecordMachineInput(request string, input MachineInput, result *pb.NativeByteRetentionResult) *exit.Error {
	if result == nil {
		return exit.New(exit.Conflict, "native intake omitted its receipt")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin input receipt update: %s", err)
	}
	defer tx.Rollback()
	var data []byte
	var current MachineInput
	if tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.input' AND json_extract(payload,'$.input_id')=? ORDER BY seq DESC LIMIT 1`, request, input.InputID).Scan(&data) != nil || json.Unmarshal(data, &current) != nil {
		return exit.New(exit.Conflict, "native input has no frozen intake")
	}
	if current.Manifest != input.Manifest || current.ContentBytes != input.ContentBytes || current.State == "released" && !result.Released {
		return exit.New(exit.Conflict, "native input changed its frozen or released subject")
	}
	if len(current.Receipt) > 0 {
		var prior pb.NativeByteRetentionResult
		if proto.Unmarshal(current.Receipt, &prior) != nil || prior.RetentionId != result.RetentionId || !proto.Equal(prior.Source, result.Source) {
			return exit.New(exit.Conflict, "native input changed its receipt subject")
		}
	}
	raw, err := proto.Marshal(result)
	if err != nil {
		return exit.Internalf("cannot encode native intake receipt: %s", err)
	}
	current.Receipt, current.State = raw, "held"
	if result.Released {
		current.State = "released"
	}
	body, err := json.Marshal(current)
	if err != nil || len(body) > 16<<10 {
		return exit.New(exit.Conflict, "native input receipt exceeds its metadata bound")
	}
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.input',0,?,?)`, request, body, now()); err != nil {
		return exit.Internalf("cannot record native input receipt: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit native input receipt: %s", err)
	}
	return nil
}
