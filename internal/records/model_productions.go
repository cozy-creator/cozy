package records

import (
	"bytes"
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

const modelProductionsDDL = `
CREATE TABLE IF NOT EXISTS model_productions (
  id          TEXT PRIMARY KEY,
  plan_digest TEXT NOT NULL,
  plan        BLOB NOT NULL,
  state       TEXT NOT NULL,
  node_index  INTEGER NOT NULL DEFAULT 0,
  rental_id   TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
)`

// ModelProductionOperation is one durable source-to-release instruction. Plan
// contains only immutable identities; transient access and execution facts stay
// in their owning ledgers and joins.
type ModelProductionOperation struct {
	ID         string
	PlanDigest string
	Plan       []byte
	State      string
	NodeIndex  int64
	RentalID   string
	CreatedAt  string
	UpdatedAt  string
}

const modelProductionCols = `id,plan_digest,plan,state,node_index,rental_id,created_at,updated_at`

func scanModelProduction(row interface{ Scan(...any) error }) (ModelProductionOperation, error) {
	var operation ModelProductionOperation
	err := row.Scan(&operation.ID, &operation.PlanDigest, &operation.Plan, &operation.State,
		&operation.NodeIndex, &operation.RentalID, &operation.CreatedAt, &operation.UpdatedAt)
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
		VALUES(?,?,?,?,?,?,?,?)`, operation.ID, operation.PlanDigest, operation.Plan,
		"accepted", 0, "", stamp, stamp); err != nil {
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
