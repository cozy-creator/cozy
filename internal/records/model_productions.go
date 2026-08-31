package records

import (
	"bytes"
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

var modelProductionSchema = []string{`
CREATE TABLE IF NOT EXISTS model_productions (
  id          TEXT PRIMARY KEY,
  plan_digest TEXT NOT NULL,
  plan        BLOB NOT NULL,
  state       TEXT NOT NULL,
  node_index  INTEGER NOT NULL DEFAULT 0,
  rental_id   TEXT NOT NULL DEFAULT '',
  selected_sku TEXT NOT NULL DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
  safe_code   TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
)`, `
CREATE TABLE IF NOT EXISTS model_production_source_files (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  member TEXT NOT NULL,
  object_id TEXT NOT NULL,
  length INTEGER NOT NULL CHECK(length>0),
  capability_revision INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending',
  transferred_bytes INTEGER NOT NULL DEFAULT 0,
  safe_code TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(operation_id,member)
)`, `
CREATE TABLE IF NOT EXISTS model_production_sources (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  slot TEXT NOT NULL,
  profile TEXT NOT NULL,
  manifest_id TEXT NOT NULL,
  manifest_length INTEGER NOT NULL CHECK(manifest_length>0),
  release_evidence BLOB NOT NULL,
  PRIMARY KEY(operation_id,slot)
)`, `
CREATE TABLE IF NOT EXISTS model_production_nodes (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  node_index INTEGER NOT NULL,
  node_name TEXT NOT NULL,
  request_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'pending',
  PRIMARY KEY(operation_id,node_index),
  UNIQUE(operation_id,node_name)
)`, `
CREATE TABLE IF NOT EXISTS model_production_artifacts (
  operation_id TEXT NOT NULL REFERENCES model_productions(id),
  node_name TEXT NOT NULL,
  output_slot TEXT NOT NULL,
  request_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  invocation_digest TEXT NOT NULL,
  transaction_id TEXT NOT NULL,
  writer_generation INTEGER NOT NULL CHECK(writer_generation>0),
  receipt_digest TEXT NOT NULL,
  receipt BLOB NOT NULL,
  manifest_id TEXT NOT NULL,
  manifest_length INTEGER NOT NULL CHECK(manifest_length>0),
  release_evidence BLOB NOT NULL,
  publication_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'received',
  PRIMARY KEY(operation_id,node_name,output_slot),
  UNIQUE(request_id,attempt,output_slot)
)`, `
CREATE TABLE IF NOT EXISTS model_production_objects (
  operation_id TEXT NOT NULL,
  node_name TEXT NOT NULL,
  output_slot TEXT NOT NULL,
  object_id TEXT NOT NULL,
  length INTEGER NOT NULL CHECK(length>0),
  source_ref TEXT NOT NULL,
  transfer_operation_id TEXT NOT NULL DEFAULT '',
  grant_revision INTEGER NOT NULL DEFAULT 0,
  update_sequence INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending',
  transferred_bytes INTEGER NOT NULL DEFAULT 0,
  safe_code TEXT NOT NULL DEFAULT '',
  safe_detail TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(operation_id,node_name,output_slot,object_id),
  FOREIGN KEY(operation_id,node_name,output_slot)
    REFERENCES model_production_artifacts(operation_id,node_name,output_slot)
)`}

// ModelProductionOperation is one durable source-to-release instruction. Plan
// contains only immutable identities; transient access and execution facts stay
// in their owning ledgers and joins.
type ModelProductionOperation struct {
	ID                   string
	PlanDigest           string
	Plan                 []byte
	State                string
	NodeIndex            int64
	RentalID             string
	SelectedSKU          string
	CancelRequested      bool
	SafeCode, SafeDetail string
	CreatedAt            string
	UpdatedAt            string
}

const modelProductionCols = `id,plan_digest,plan,state,node_index,rental_id,selected_sku,
	cancel_requested,safe_code,safe_detail,created_at,updated_at`

func scanModelProduction(row interface{ Scan(...any) error }) (ModelProductionOperation, error) {
	var operation ModelProductionOperation
	err := row.Scan(&operation.ID, &operation.PlanDigest, &operation.Plan, &operation.State,
		&operation.NodeIndex, &operation.RentalID, &operation.SelectedSKU,
		&operation.CancelRequested, &operation.SafeCode, &operation.SafeDetail,
		&operation.CreatedAt, &operation.UpdatedAt)
	return operation, err
}

// BeginModelProduction creates the operation before any worker/rental act. An
// exact replay returns the same row; an identity collision always conflicts.
func (s *Store) BeginModelProduction(operation ModelProductionOperation) (ModelProductionOperation, bool, *exit.Error) {
	if operation.ID == "" || operation.PlanDigest == "" || len(operation.Plan) == 0 {
		return ModelProductionOperation{}, false, exit.New(exit.Validation,
			"model production needs one id, plan digest, and restart plan")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return ModelProductionOperation{}, false, exit.Internalf("cannot begin model production: %s", err)
	}
	defer tx.Rollback()
	stored, err := scanModelProduction(tx.QueryRow(
		`SELECT `+modelProductionCols+` FROM model_productions WHERE id=?`, operation.ID))
	if err == nil {
		if stored.PlanDigest != operation.PlanDigest || !bytes.Equal(stored.Plan, operation.Plan) {
			return ModelProductionOperation{}, false, exit.Named(exit.Conflict,
				"model_production.identity_conflict",
				"model production %s already binds different immutable plan bytes", operation.ID)
		}
		if err := tx.Commit(); err != nil {
			return ModelProductionOperation{}, false, exit.Internalf("cannot read replayed model production: %s", err)
		}
		return stored, true, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return ModelProductionOperation{}, false, exit.Internalf("cannot read model production: %s", err)
	}
	stamp := now()
	if _, err := tx.Exec(`INSERT INTO model_productions(`+modelProductionCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`, operation.ID, operation.PlanDigest, operation.Plan,
		"accepted", 0, "", "", false, "", "", stamp, stamp); err != nil {
		return ModelProductionOperation{}, false, exit.Internalf("cannot record model production: %s", err)
	}
	stored, err = scanModelProduction(tx.QueryRow(
		`SELECT `+modelProductionCols+` FROM model_productions WHERE id=?`, operation.ID))
	if err != nil {
		return ModelProductionOperation{}, false, exit.Internalf("cannot read recorded model production: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return ModelProductionOperation{}, false, exit.Internalf("cannot commit model production: %s", err)
	}
	return stored, false, nil
}

func (s *Store) ModelProduction(id string) (*ModelProductionOperation, *exit.Error) {
	operation, err := scanModelProduction(s.db.QueryRow(
		`SELECT `+modelProductionCols+` FROM model_productions WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read model production %s: %s", id, err)
	}
	return &operation, nil
}

func (s *Store) ActiveModelProductions() ([]ModelProductionOperation, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + modelProductionCols + ` FROM model_productions
		WHERE state NOT IN ('completed','failed','canceled') ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot list active model productions: %s", err)
	}
	defer rows.Close()
	var out []ModelProductionOperation
	for rows.Next() {
		row, err := scanModelProduction(rows)
		if err != nil {
			return nil, exit.Internalf("cannot scan active model production: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) SelectModelProductionSKU(id, sku string) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_productions SET selected_sku=?,updated_at=?
		WHERE id=? AND (selected_sku='' OR selected_sku=?)`, sku, now(), id, sku)
	if err != nil {
		return exit.Internalf("cannot select model production SKU: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_production.sku_conflict",
			"model production %s already selected another rental SKU", id)
	}
	return nil
}

func (s *Store) RequestModelProductionCancel(id string) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_productions SET cancel_requested=1,updated_at=?
		WHERE id=? AND state NOT IN ('completed','failed','canceled')`, now(), id)
	if err != nil {
		return exit.Internalf("cannot request model production cancellation: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		if current, problem := s.ModelProduction(id); problem != nil {
			return problem
		} else if current == nil {
			return exit.New(exit.NotFound, "model production %s is absent", id)
		}
	}
	return nil
}

func (s *Store) FailModelProduction(id, from, code, detail string) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_productions SET state='failed',safe_code=?,
		safe_detail=?,updated_at=? WHERE id=? AND state=?`, code, detail, now(), id, from)
	if err != nil {
		return exit.Internalf("cannot fail model production %s: %s", id, err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	current, problem := s.ModelProduction(id)
	if problem != nil {
		return problem
	}
	if current != nil && current.State == "failed" && current.SafeCode == code &&
		current.SafeDetail == detail {
		return nil
	}
	return exit.Named(exit.Conflict, "model_production.state_conflict",
		"model production %s could not fail from %s", id, from)
}

// AdvanceModelProduction is a compare-and-swap over the one coarse lifecycle.
// Node detail remains in ordinary job attempts; node_index is only the restart cursor.
func (s *Store) AdvanceModelProduction(id, from, to string, nodeIndex int64, rentalID string) *exit.Error {
	if !modelProductionTransition(from, to) || nodeIndex < 0 {
		return exit.Named(exit.Validation, "model_production.transition_invalid",
			"model production transition %s -> %s at node %d is invalid", from, to, nodeIndex)
	}
	result, err := s.db.Exec(`UPDATE model_productions SET state=?,node_index=?,
		rental_id=CASE WHEN rental_id='' THEN ? ELSE rental_id END,updated_at=?
		WHERE id=? AND state=? AND node_index<=? AND (rental_id='' OR ?='' OR rental_id=?)`,
		to, nodeIndex, rentalID, now(), id, from, nodeIndex, rentalID, rentalID)
	if err != nil {
		return exit.Internalf("cannot advance model production %s: %s", id, err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	current, problem := s.ModelProduction(id)
	if problem != nil {
		return problem
	}
	if current == nil {
		return exit.New(exit.NotFound, "model production %s is absent", id)
	}
	if current.State == to && current.NodeIndex == nodeIndex &&
		(rentalID == "" || current.RentalID == rentalID) {
		return nil
	}
	return exit.Named(exit.Conflict, "model_production.state_conflict",
		"model production %s is %s at node %d, not %s", id,
		current.State, current.NodeIndex, from)
}

func modelProductionTransition(from, to string) bool {
	if to == "failed" || to == "canceled" {
		return from != "completed" && from != "failed" && from != "canceled"
	}
	switch from + "\x00" + to {
	case "accepted\x00source_preparing",
		"source_preparing\x00source_prepared",
		"source_prepared\x00node_running",
		"node_running\x00node_running",
		"node_running\x00outputs_preparing",
		"outputs_preparing\x00release_cut",
		"release_cut\x00cleanup_pending",
		"cleanup_pending\x00completed":
		return true
	}
	return false
}
