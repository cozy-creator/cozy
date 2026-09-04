package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

var modelTransferSlug = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`)

const maxModelTransferSourceFiles = 20_000
const maxExactJSONInteger = (int64(1) << 53) - 1

var modelTransferSchema = []string{`
CREATE TABLE IF NOT EXISTS request_model_transfers (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  intent           TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'pending',
  models           TEXT NOT NULL DEFAULT '[]',
  checkpoints      TEXT NOT NULL DEFAULT '{}',
  error_code       TEXT NOT NULL DEFAULT '',
  safe_error       TEXT NOT NULL DEFAULT '',
  updated_at       TEXT NOT NULL,
  CHECK (state IN ('pending','materializing','materialized','finalizing','completed','failed','canceled'))
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_files (
  request_id          TEXT NOT NULL REFERENCES requests(id),
  member              TEXT NOT NULL,
  object_id           TEXT NOT NULL,
  length              INTEGER NOT NULL CHECK(length>0),
  capability_revision INTEGER NOT NULL DEFAULT 0,
  state               TEXT NOT NULL DEFAULT 'pending',
  transferred         INTEGER NOT NULL DEFAULT 0,
  safe_code           TEXT NOT NULL DEFAULT '',
  safe_detail         TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,member)
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_outputs (
  request_id       TEXT NOT NULL REFERENCES requests(id),
  output_slot      TEXT NOT NULL,
  manifest_id      TEXT NOT NULL,
  manifest_length  INTEGER NOT NULL CHECK(manifest_length>0),
  attempt           INTEGER NOT NULL,
  invocation_digest TEXT NOT NULL,
  transaction_id    TEXT NOT NULL,
  receipt_digest    TEXT NOT NULL,
  receipt           BLOB NOT NULL,
  final_id          TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot)
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_objects (
  request_id       TEXT NOT NULL,
  attempt          INTEGER NOT NULL,
  output_slot      TEXT NOT NULL,
  object_id        TEXT NOT NULL,
  length           INTEGER NOT NULL CHECK(length>0),
  source_ref       TEXT NOT NULL,
  operation_id     TEXT NOT NULL DEFAULT '',
  grant_revision   INTEGER NOT NULL DEFAULT 0,
  update_sequence  INTEGER NOT NULL DEFAULT 0,
  state            TEXT NOT NULL DEFAULT 'pending',
  transferred      INTEGER NOT NULL DEFAULT 0,
  safe_code        TEXT NOT NULL DEFAULT '',
  safe_detail      TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot,object_id),
  FOREIGN KEY(request_id,attempt,output_slot)
    REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
)`}

type ModelTransferSourceFile struct {
	Member string `json:"member"`
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
}

type ModelTransferSourceStatus struct {
	RequestID, Member, ObjectID, State, SafeCode, SafeDetail string
	Length, CapabilityRevision, Transferred                  int64
}

type ModelTransferOutput struct {
	Name string `json:"name"`
}

// ModelTransferIntent is immutable request meaning. Credentials, URLs, workers,
// attempts, prices, progress, and retries deliberately have no field.
type ModelTransferIntent struct {
	Kind            string                    `json:"kind"`
	Destination     string                    `json:"destination"`
	Source          string                    `json:"source"`
	SourceSelection string                    `json:"source_selection"`
	SourceLicense   string                    `json:"source_license,omitempty"`
	SourceFiles     []ModelTransferSourceFile `json:"source_files,omitempty"`
	InputLane       string                    `json:"input_lane,omitempty"`
	SourceProfiles  map[string]string         `json:"source_profiles,omitempty"`
	Outputs         []ModelTransferOutput     `json:"outputs"`
	LocalOnly       bool                      `json:"local_only,omitempty"`
}

type ModelTransfer struct {
	RequestID string
	ModelTransferIntent
	State       string
	Models      []ModelRef
	Checkpoints map[string]string
	ErrorCode   string
	SafeError   string
	UpdatedAt   string
}

type ModelTransferObject struct {
	RequestID, OutputSlot, ObjectID, SourceRef string
	OperationID, State, SafeCode, SafeDetail   string
	Attempt, Length, GrantRevision             int64
	UpdateSequence, Transferred                int64
}

type ModelTransferWeights struct {
	RequestID, OutputSlot, ManifestID, FinalID     string
	InvocationDigest, TransactionID, ReceiptDigest string
	ManifestLength, Attempt                        int64
	Receipt                                        []byte
	Objects                                        []ModelTransferObject
}

func NormalizeModelTransferIntent(intent *ModelTransferIntent) *exit.Error {
	if intent == nil {
		return nil
	}
	if (intent.Kind != "model-upload" && intent.Kind != "model-download") ||
		intent.Destination == "" || intent.Source == "" || intent.SourceSelection == "" ||
		len(intent.Outputs) == 0 {
		return exit.New(exit.Validation, "model transfer intent is incomplete")
	}
	localSource := strings.HasPrefix(intent.Source, "file:") || strings.HasPrefix(intent.Source, "local/")
	if intent.LocalOnly != localSource {
		return exit.New(exit.Validation, "model transfer local_only does not match its source")
	}
	parts := strings.Split(intent.Destination, "/")
	if intent.Kind == "model-upload" {
		if len(parts) != 2 || parts[0] == "local" || !modelTransferSlug.MatchString(parts[0]) ||
			!modelTransferSlug.MatchString(parts[1]) {
			return exit.New(exit.Validation, "model upload destination is not one org/model")
		}
	} else if len(parts) != 2 || parts[0] != "local" || !modelTransferSlug.MatchString(parts[1]) {
		return exit.New(exit.Validation, "model download destination is not local/name")
	}
	intent.SourceFiles = append([]ModelTransferSourceFile(nil), intent.SourceFiles...)
	if len(intent.SourceFiles) > maxModelTransferSourceFiles {
		return exit.New(exit.Validation, "model transfer source inventory exceeds %d files",
			maxModelTransferSourceFiles)
	}
	sort.Slice(intent.SourceFiles, func(i, j int) bool {
		return intent.SourceFiles[i].Member < intent.SourceFiles[j].Member
	})
	previousFile := ""
	var sourceBytes int64
	for _, file := range intent.SourceFiles {
		_, digestErr := canonical.Raw("sha256:" + file.SHA256)
		if file.Member == "" || file.Member <= previousFile || file.Length <= 0 ||
			file.Length > maxExactJSONInteger-sourceBytes ||
			digestErr != nil {
			return exit.New(exit.Validation, "model transfer source inventory is invalid")
		}
		sourceBytes += file.Length
		previousFile = file.Member
	}
	if _, err := canonical.Raw(intent.SourceSelection); err != nil {
		return exit.New(exit.Validation, "model transfer source selection is not an exact digest")
	}
	intent.Outputs = append([]ModelTransferOutput(nil), intent.Outputs...)
	sort.Slice(intent.Outputs, func(i, j int) bool { return intent.Outputs[i].Name < intent.Outputs[j].Name })
	seen := map[string]bool{}
	for i := range intent.Outputs {
		output := &intent.Outputs[i]
		if output.Name == "" || seen[output.Name] {
			return exit.New(exit.Validation, "model transfer output names are non-empty and unique")
		}
		seen[output.Name] = true
	}
	return nil
}

func recordModelTransferTx(tx *sql.Tx, requestID string, intent *ModelTransferIntent) *exit.Error {
	if intent == nil {
		return nil
	}
	data, err := json.Marshal(intent)
	if err != nil {
		return exit.Internalf("cannot encode model transfer intent: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO request_model_transfers(request_id,intent,updated_at)
		VALUES(?,?,?)`, requestID, string(data), now()); err != nil {
		return exit.Internalf("cannot record model transfer for %s: %s", requestID, err)
	}
	for _, file := range intent.SourceFiles {
		if _, err := tx.Exec(`INSERT INTO request_model_transfer_files
			(request_id,member,object_id,length) VALUES(?,?,?,?)`, requestID, file.Member,
			"sha256:"+file.SHA256, file.Length); err != nil {
			return exit.Internalf("cannot record model transfer source %s: %s", file.Member, err)
		}
	}
	return nil
}

// PlannedSourceBytes is the summed length of every source object this request
// will pull, or 0 when the request moves no model bytes (th-152).
//
// It is read from request_model_transfer_files, which is written in the same
// transaction as the request itself and is the table the length CHECK lives
// on — never re-summed from the intent JSON, which would let two spellings of
// the same plan disagree.
//
// This is what makes a plan-sized pod possible. An ingest holds its source
// objects AND the canonical CAS output built from them in one TensorFS Store
// on the pod's container disk, so a pod bought at the serving default would
// block on a full filesystem partway through. The hub sizes the disk from this
// figure; it is stated before the rental is bought because the selection is
// resolved before any pod exists.
func (s *Store) PlannedSourceBytes(requestID string) (int64, *exit.Error) {
	var total sql.NullInt64
	if err := s.db.QueryRow(`SELECT SUM(length) FROM request_model_transfer_files
		WHERE request_id=?`, requestID).Scan(&total); err != nil {
		return 0, exit.Internalf("cannot total model transfer source bytes for %s: %s",
			requestID, err)
	}
	if !total.Valid || total.Int64 < 0 {
		return 0, nil
	}
	return total.Int64, nil
}

func (s *Store) ModelTransferOf(requestID string) (*ModelTransfer, *exit.Error) {
	var row ModelTransfer
	var intent, models, checkpoints string
	err := s.db.QueryRow(`SELECT request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at
		FROM request_model_transfers WHERE request_id=?`, requestID).Scan(&row.RequestID,
		&intent, &row.State, &models, &checkpoints, &row.ErrorCode, &row.SafeError, &row.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil || json.Unmarshal([]byte(intent), &row.ModelTransferIntent) != nil ||
		json.Unmarshal([]byte(models), &row.Models) != nil ||
		json.Unmarshal([]byte(checkpoints), &row.Checkpoints) != nil {
		return nil, exit.Internalf("cannot decode model transfer %s", requestID)
	}
	return &row, nil
}

func (s *Store) ModelTransfersOwed() ([]ModelTransfer, *exit.Error) {
	rows, err := s.db.Query(`SELECT t.request_id FROM request_model_transfers t
		JOIN requests r ON r.id=t.request_id
		WHERE t.state NOT IN ('completed','failed','canceled')
		   OR r.state NOT IN ('succeeded','failed','canceled','refused','abandoned')
		ORDER BY r.created_at,r.id`)
	if err != nil {
		return nil, exit.Internalf("cannot list owed model transfers: %s", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot decode owed model transfer: %s", err)
		}
		ids = append(ids, id)
	}
	rows.Close()
	out := make([]ModelTransfer, 0, len(ids))
	for _, id := range ids {
		row, problem := s.ModelTransferOf(id)
		if problem != nil {
			return nil, problem
		}
		if row != nil {
			out = append(out, *row)
		}
	}
	return out, nil
}

func (s *Store) BeginModelTransferMaterialization(requestID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='materializing',updated_at=?
		WHERE request_id=? AND state IN ('pending','materializing')`, now(), requestID)
	if err != nil {
		return exit.Internalf("cannot begin model transfer materialization: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	transfer, problem := s.ModelTransferOf(requestID)
	if problem != nil {
		return problem
	}
	if transfer != nil && (transfer.State == "materialized" || transfer.State == "finalizing" ||
		transfer.State == "completed") {
		return nil
	}
	return exit.New(exit.Conflict, "model transfer %s is not materializable", requestID)
}

func (s *Store) CompleteModelTransferMaterialization(requestID string, models []ModelRef) *exit.Error {
	rows := append([]ModelRef(nil), models...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Slot < rows[j].Slot })
	data, err := json.Marshal(rows)
	if err != nil {
		return exit.Internalf("cannot encode model transfer inputs: %s", err)
	}
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='materialized',models=?,updated_at=?
		WHERE request_id=? AND state IN ('pending','materializing','materialized')`, string(data), now(), requestID)
	if err != nil {
		return exit.Internalf("cannot retain model transfer inputs: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		transfer, problem := s.ModelTransferOf(requestID)
		if problem != nil || transfer == nil || string(mustJSON(transfer.Models)) != string(data) ||
			(transfer.State != "materialized" && transfer.State != "finalizing" &&
				transfer.State != "completed") {
			return exit.New(exit.Conflict, "model transfer %s is not materializable", requestID)
		}
	}
	return nil
}

func (s *Store) ModelTransferSourceStatuses(requestID string) ([]ModelTransferSourceStatus, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,member,object_id,length,capability_revision,state,
		transferred,safe_code,safe_detail FROM request_model_transfer_files
		WHERE request_id=? ORDER BY member`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot list model transfer source files: %s", err)
	}
	defer rows.Close()
	var out []ModelTransferSourceStatus
	for rows.Next() {
		var row ModelTransferSourceStatus
		if err := rows.Scan(&row.RequestID, &row.Member, &row.ObjectID, &row.Length,
			&row.CapabilityRevision, &row.State, &row.Transferred, &row.SafeCode,
			&row.SafeDetail); err != nil {
			return nil, exit.Internalf("cannot decode model transfer source status: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) RecordModelTransferSourceStatus(row ModelTransferSourceStatus) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model transfer source status: %s", err)
	}
	defer tx.Rollback()
	var held ModelTransferSourceStatus
	err = tx.QueryRow(`SELECT request_id,member,object_id,length,capability_revision,state,
		transferred,safe_code,safe_detail FROM request_model_transfer_files
		WHERE request_id=? AND member=?`, row.RequestID, row.Member).Scan(&held.RequestID,
		&held.Member, &held.ObjectID, &held.Length, &held.CapabilityRevision, &held.State,
		&held.Transferred, &held.SafeCode, &held.SafeDetail)
	if err != nil {
		return exit.Internalf("cannot read model transfer source status: %s", err)
	}
	exact := held.ObjectID == row.ObjectID && held.Length == row.Length
	replay := exact && held.CapabilityRevision == row.CapabilityRevision && held.State == row.State &&
		held.Transferred == row.Transferred && held.SafeCode == row.SafeCode && held.SafeDetail == row.SafeDetail
	if replay {
		return nil
	}
	advancedRevision := row.CapabilityRevision > held.CapabilityRevision
	advancedState := row.CapabilityRevision == held.CapabilityRevision &&
		sourceStateRank(row.State) >= sourceStateRank(held.State)
	absorbing := held.State == "verified" || (held.State == "failed" && !advancedRevision)
	if !exact || row.Transferred < held.Transferred || (!advancedRevision && !advancedState) || absorbing {
		return exit.Named(exit.Conflict, "model_transfer.source_status_changed",
			"model transfer source %s changed identity or revision", row.Member)
	}
	result, err := tx.Exec(`UPDATE request_model_transfer_files SET capability_revision=?,
		state=?,transferred=?,safe_code=?,safe_detail=? WHERE request_id=? AND member=?
		AND capability_revision=? AND state=? AND transferred=?`, row.CapabilityRevision,
		row.State, row.Transferred, row.SafeCode, row.SafeDetail, row.RequestID, row.Member,
		held.CapabilityRevision, held.State, held.Transferred)
	if err != nil {
		return exit.Internalf("cannot record model transfer source status: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_transfer.source_status_changed",
			"model transfer source %s changed concurrently", row.Member)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model transfer source status: %s", err)
	}
	return nil
}

func sourceStateRank(state string) int {
	switch state {
	case "pending":
		return 0
	case "accepted":
		return 1
	case "downloading":
		return 2
	case "verified", "failed":
		return 3
	default:
		return -1
	}
}

func (s *Store) RecordModelTransferWeights(row ModelTransferWeights) *exit.Error {
	if row.Receipt == nil {
		row.Receipt = []byte{}
	}
	objectsCopy := append([]ModelTransferObject(nil), row.Objects...)
	sort.Slice(objectsCopy, func(i, j int) bool { return objectsCopy[i].ObjectID < objectsCopy[j].ObjectID })
	row.Objects = objectsCopy
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model transfer weights: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`INSERT INTO request_model_transfer_outputs
		(request_id,output_slot,manifest_id,manifest_length,attempt,
		 invocation_digest,transaction_id,receipt_digest,receipt)
		VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(request_id,attempt,output_slot) DO NOTHING`, row.RequestID,
		row.OutputSlot, row.ManifestID, row.ManifestLength, row.Attempt,
		row.InvocationDigest, row.TransactionID, row.ReceiptDigest, row.Receipt)
	if err != nil {
		return exit.Internalf("cannot record model transfer weights: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		previous := ""
		for _, object := range row.Objects {
			if object.ObjectID == "" || object.ObjectID <= previous || object.Length <= 0 || object.SourceRef == "" {
				return exit.New(exit.Validation, "model transfer weights object inventory is invalid")
			}
			previous = object.ObjectID
			if _, err := tx.Exec(`INSERT INTO request_model_transfer_objects
				(request_id,attempt,output_slot,object_id,length,source_ref) VALUES(?,?,?,?,?,?)`,
				row.RequestID, row.Attempt, row.OutputSlot, object.ObjectID, object.Length,
				object.SourceRef); err != nil {
				return exit.Internalf("cannot record model transfer object: %s", err)
			}
		}
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot commit model transfer weights: %s", err)
		}
		return nil
	}
	_ = tx.Rollback()
	held, problem := s.ModelTransferWeights(row.RequestID, row.Attempt, row.OutputSlot)
	if problem != nil || held == nil || held.ManifestID != row.ManifestID ||
		held.ManifestLength != row.ManifestLength ||
		held.Attempt != row.Attempt || held.InvocationDigest != row.InvocationDigest ||
		held.TransactionID != row.TransactionID || held.ReceiptDigest != row.ReceiptDigest ||
		string(held.Receipt) != string(row.Receipt) ||
		string(mustJSON(held.Objects)) != string(mustJSON(row.Objects)) {
		return exit.Named(exit.Conflict, "model_transfer.weights_changed",
			"model transfer output %s replay changed identity", row.OutputSlot)
	}
	return nil
}

func mustJSON(value any) []byte { data, _ := json.Marshal(value); return data }

func (s *Store) ModelTransferWeights(requestID string, attempt int64, slot string) (*ModelTransferWeights, *exit.Error) {
	var row ModelTransferWeights
	err := s.db.QueryRow(`SELECT request_id,output_slot,manifest_id,manifest_length,
		attempt,invocation_digest,transaction_id,receipt_digest,receipt,final_id
		FROM request_model_transfer_outputs WHERE request_id=? AND attempt=? AND output_slot=?`, requestID, attempt, slot).
		Scan(&row.RequestID, &row.OutputSlot, &row.ManifestID, &row.ManifestLength,
			&row.Attempt, &row.InvocationDigest, &row.TransactionID,
			&row.ReceiptDigest, &row.Receipt, &row.FinalID)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot decode model transfer output %s/%s", requestID, slot)
	}
	objects, problem := s.ModelTransferObjects(requestID, attempt, slot)
	if problem != nil {
		return nil, problem
	}
	row.Objects = objects
	return &row, nil
}

func (s *Store) AllModelTransferWeights(requestID string, attempt int64) ([]ModelTransferWeights, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,output_slot,manifest_id,manifest_length,
		attempt,invocation_digest,transaction_id,receipt_digest,receipt,final_id
		FROM request_model_transfer_outputs WHERE request_id=? AND attempt=? ORDER BY output_slot`, requestID, attempt)
	if err != nil {
		return nil, exit.Internalf("cannot list model transfer outputs: %s", err)
	}
	var out []ModelTransferWeights
	for rows.Next() {
		var row ModelTransferWeights
		if err := rows.Scan(&row.RequestID, &row.OutputSlot, &row.ManifestID, &row.ManifestLength,
			&row.Attempt, &row.InvocationDigest, &row.TransactionID,
			&row.ReceiptDigest, &row.Receipt, &row.FinalID); err != nil {
			return nil, exit.Internalf("cannot decode model transfer output row")
		}
		out = append(out, row)
	}
	rows.Close()
	for i := range out {
		var problem *exit.Error
		out[i].Objects, problem = s.ModelTransferObjects(requestID, attempt, out[i].OutputSlot)
		if problem != nil {
			return nil, problem
		}
	}
	return out, nil
}

func (s *Store) ModelTransferObjects(requestID string, attempt int64,
	slot string,
) ([]ModelTransferObject, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,attempt,output_slot,object_id,length,source_ref,
		operation_id,grant_revision,update_sequence,state,transferred,safe_code,safe_detail
		FROM request_model_transfer_objects WHERE request_id=? AND attempt=? AND output_slot=?
		ORDER BY object_id`, requestID, attempt, slot)
	if err != nil {
		return nil, exit.Internalf("cannot list model transfer objects: %s", err)
	}
	defer rows.Close()
	var out []ModelTransferObject
	for rows.Next() {
		var row ModelTransferObject
		if err := rows.Scan(&row.RequestID, &row.Attempt, &row.OutputSlot, &row.ObjectID,
			&row.Length, &row.SourceRef, &row.OperationID, &row.GrantRevision,
			&row.UpdateSequence, &row.State, &row.Transferred, &row.SafeCode,
			&row.SafeDetail); err != nil {
			return nil, exit.Internalf("cannot decode model transfer object: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) RecordModelTransferObjectStatus(row ModelTransferObject) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model transfer object status: %s", err)
	}
	defer tx.Rollback()
	var held ModelTransferObject
	err = tx.QueryRow(`SELECT request_id,attempt,output_slot,object_id,length,source_ref,
		operation_id,grant_revision,update_sequence,state,transferred,safe_code,safe_detail
		FROM request_model_transfer_objects WHERE request_id=? AND attempt=? AND output_slot=?
		AND object_id=?`, row.RequestID, row.Attempt, row.OutputSlot, row.ObjectID).Scan(
		&held.RequestID, &held.Attempt, &held.OutputSlot, &held.ObjectID, &held.Length,
		&held.SourceRef, &held.OperationID, &held.GrantRevision, &held.UpdateSequence,
		&held.State, &held.Transferred, &held.SafeCode, &held.SafeDetail)
	if err != nil {
		return exit.Internalf("cannot read model transfer object status: %s", err)
	}
	replay := held.Length == row.Length && held.OperationID == row.OperationID &&
		held.GrantRevision == row.GrantRevision && held.UpdateSequence == row.UpdateSequence &&
		held.State == row.State && held.Transferred == row.Transferred && held.SafeCode == row.SafeCode &&
		held.SafeDetail == row.SafeDetail
	if replay {
		return nil
	}
	advanced := row.GrantRevision > held.GrantRevision ||
		(row.GrantRevision == held.GrantRevision && row.UpdateSequence > held.UpdateSequence)
	absorbing := held.State == "uploaded" || held.State == "already_present" || held.State == "held"
	stateRegressed := row.GrantRevision == held.GrantRevision &&
		objectStateRank(row.State) < objectStateRank(held.State)
	absorbing = absorbing || held.State == "failed" && row.GrantRevision == held.GrantRevision
	if row.Length != held.Length || row.Transferred < held.Transferred || !advanced || absorbing ||
		stateRegressed {
		return exit.Named(exit.Conflict, "model_transfer.object_status_changed",
			"model transfer object %s status moved backwards or changed identity", row.ObjectID)
	}
	result, err := tx.Exec(`UPDATE request_model_transfer_objects SET operation_id=?,
		grant_revision=?,update_sequence=?,state=?,transferred=?,safe_code=?,safe_detail=?
		WHERE request_id=? AND attempt=? AND output_slot=? AND object_id=?
		AND grant_revision=? AND update_sequence=?`, row.OperationID, row.GrantRevision,
		row.UpdateSequence, row.State, row.Transferred, row.SafeCode, row.SafeDetail,
		row.RequestID, row.Attempt, row.OutputSlot, row.ObjectID, held.GrantRevision,
		held.UpdateSequence)
	if err != nil {
		return exit.Internalf("cannot record model transfer object status: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_transfer.object_status_changed",
			"model transfer object %s changed concurrently", row.ObjectID)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model transfer object status: %s", err)
	}
	return nil
}

func objectStateRank(state string) int {
	switch state {
	case "pending":
		return 0
	case "accepted":
		return 1
	case "reading", "uploading":
		return 2
	case "uploaded", "already_present", "held", "failed":
		return 3
	default:
		return -1
	}
}

func (s *Store) BeginModelTransferFinalization(requestID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='finalizing',updated_at=?
		WHERE request_id=? AND state IN ('materialized','finalizing')`, now(), requestID)
	if err != nil {
		return exit.Internalf("cannot begin model transfer finalization: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	transfer, problem := s.ModelTransferOf(requestID)
	if problem != nil {
		return problem
	}
	if transfer != nil && transfer.State == "completed" {
		return nil
	}
	return exit.New(exit.Conflict, "model transfer %s is not finalizable", requestID)
}

func (s *Store) CompleteModelTransferOutput(requestID string, attempt int64, slot, finalID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfer_outputs SET final_id=?
		WHERE request_id=? AND attempt=? AND output_slot=? AND (final_id='' OR final_id=?)`, finalID,
		requestID, attempt, slot, finalID)
	if err != nil {
		return exit.Internalf("cannot retain model transfer output: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	held, problem := s.ModelTransferWeights(requestID, attempt, slot)
	if problem != nil || held == nil || held.FinalID != finalID {
		return exit.Named(exit.Conflict, "model_transfer.final_id_changed",
			"model transfer output %s already names another finalization", slot)
	}
	return nil
}

func (s *Store) CompleteModelTransfer(requestID string, checkpoints map[string]string) *exit.Error {
	data, err := json.Marshal(checkpoints)
	if err != nil {
		return exit.Internalf("cannot encode model transfer checkpoints: %s", err)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model transfer completion: %s", err)
	}
	defer tx.Rollback()
	var state, intentJSON string
	var attempt, retained int
	if err := tx.QueryRow(`SELECT t.state,t.intent,r.ordinal,
		(SELECT COUNT(*) FROM request_model_transfer_outputs o
		 WHERE o.request_id=t.request_id AND o.attempt=r.ordinal AND o.final_id<>'')
		FROM request_model_transfers t JOIN requests r ON r.id=t.request_id
		WHERE t.request_id=?`, requestID).Scan(&state, &intentJSON, &attempt, &retained); err != nil {
		return exit.Internalf("cannot read model transfer completion: %s", err)
	}
	_ = attempt
	var intent ModelTransferIntent
	if json.Unmarshal([]byte(intentJSON), &intent) != nil || retained != len(intent.Outputs) ||
		len(checkpoints) != len(intent.Outputs) {
		return exit.Named(exit.Conflict, "model_transfer.outputs_incomplete",
			"model transfer %s has not retained every accepted output", requestID)
	}
	result, err := tx.Exec(`UPDATE request_model_transfers SET state='completed',checkpoints=?,
		error_code='',safe_error='',updated_at=? WHERE request_id=? AND state='finalizing'`,
		string(data), now(), requestID)
	if err != nil {
		return exit.Internalf("cannot complete model transfer: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot commit model transfer completion: %s", err)
		}
		return nil
	}
	if state == "completed" {
		return nil
	}
	return exit.New(exit.Conflict, "model transfer %s cannot complete", requestID)
}

func (s *Store) FailModelTransfer(requestID, code, detail string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='failed',error_code=?,safe_error=?,updated_at=?
		WHERE request_id=? AND state NOT IN ('completed','canceled')`, code, detail, now(), requestID)
	if err != nil {
		return exit.Internalf("cannot fail model transfer: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	transfer, problem := s.ModelTransferOf(requestID)
	if problem == nil && transfer != nil && transfer.State == "failed" &&
		transfer.ErrorCode == code && transfer.SafeError == detail {
		return nil
	}
	return exit.New(exit.Conflict, "model transfer %s cannot fail from its current state", requestID)
}

func (s *Store) RequestModelTransferCancellation(requestID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='canceled',
		error_code='CLIENT_CANCELED',safe_error='model transfer finalization canceled by client',
		updated_at=? WHERE request_id=? AND state='finalizing'`, now(), requestID)
	if err != nil {
		return exit.Internalf("cannot request model transfer cancellation: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	transfer, problem := s.ModelTransferOf(requestID)
	if problem == nil && transfer != nil && transfer.State == "canceled" {
		return nil
	}
	return exit.New(exit.Conflict, "model transfer %s is not finalizing", requestID)
}

// SettleModelTransferRequest atomically projects destination retention into the
// ordinary request terminal and its absorbing event. Provider teardown happens first.
func (s *Store) SettleModelTransferRequest(requestID string, attempt int64) (string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin model transfer settlement: %s", err)
	}
	defer tx.Rollback()
	var requestState, transferState, code, detail string
	if err := tx.QueryRow(`SELECT r.state,t.state,t.error_code,t.safe_error FROM requests r
		JOIN request_model_transfers t ON t.request_id=r.id WHERE r.id=?`, requestID).
		Scan(&requestState, &transferState, &code, &detail); err != nil {
		return "", exit.Internalf("cannot read model transfer settlement: %s", err)
	}
	if requestState == "canceled" {
		return "canceled", nil
	}
	if requestState == "succeeded" || requestState == "failed" {
		return requestState, nil
	}
	state, eventType := "failed", "request.failed"
	payload := map[string]any{"status": "FAILED", "cause": code,
		"error_type": code, "error": detail, "outputs": []any{}, "requeuing": false}
	if transferState == "completed" {
		state, eventType = "succeeded", "request.completed"
		payload = map[string]any{"status": "SUCCEEDED", "cause": "COMPLETED",
			"outputs": []any{}, "requeuing": false}
	} else if transferState == "canceled" {
		state, eventType = "canceled", "request.canceled"
		payload = map[string]any{"status": "CANCELED", "cause": "CLIENT_CANCELED",
			"error_type": "CLIENT_CANCELED", "error": detail,
			"outputs": []any{}, "requeuing": false}
	} else if transferState != "failed" {
		return "", exit.New(exit.Conflict, "model transfer %s is %s before settlement",
			requestID, transferState)
	}
	result, err := tx.Exec(`UPDATE requests SET state=? WHERE id=? AND state IN ('submitted','finalizing')`,
		state, requestID)
	if err != nil {
		return "", exit.Internalf("cannot settle model transfer request: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return "", exit.New(exit.Conflict, "model transfer request changed before settlement")
	}
	if err := appendEventTx(tx, requestID, eventType, attempt, payload); err != nil {
		return "", exit.Internalf("cannot append model transfer terminal: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return "", exit.Internalf("cannot commit model transfer settlement: %s", err)
	}
	return state, nil
}

// FailModelTransferRequest atomically settles a pre-attempt hook failure and its
// ordinary absorbing event. It never overwrites cancellation or a completed request.
func (s *Store) FailModelTransferRequest(requestID, code, detail string,
	payload map[string]any,
) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin model transfer failure: %s", err)
	}
	defer tx.Rollback()
	var state string
	if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, requestID).Scan(&state); err != nil {
		return false, exit.Internalf("cannot read model transfer request %s: %s", requestID, err)
	}
	if settledRequestState(state) {
		return false, nil
	}
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='failed',error_code=?,
		safe_error=?,updated_at=? WHERE request_id=? AND state NOT IN ('completed','canceled')`,
		code, detail, now(), requestID); err != nil {
		return false, exit.Internalf("cannot fail model transfer sidecar: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state='failed' WHERE id=?`, requestID); err != nil {
		return false, exit.Internalf("cannot fail model transfer request: %s", err)
	}
	if err := appendEventTx(tx, requestID, "request.failed", 0, payload); err != nil {
		return false, exit.Internalf("cannot append model transfer failure: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit model transfer failure: %s", err)
	}
	return true, nil
}
