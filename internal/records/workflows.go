package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

var workflowSchema = []string{`
CREATE TABLE IF NOT EXISTS workflow_executions (
  id                   TEXT PRIMARY KEY,
  idem_key             TEXT NOT NULL UNIQUE,
  body_digest          TEXT NOT NULL,
  execution_digest     TEXT NOT NULL,
  creative_plan_digest TEXT NOT NULL,
  plan                 BLOB NOT NULL,
  state                TEXT NOT NULL,
  cancel_requested_at  TEXT NOT NULL DEFAULT '',
  terminal_code        TEXT NOT NULL DEFAULT '',
  terminal_message     TEXT NOT NULL DEFAULT '',
  created_at           TEXT NOT NULL,
  settled_at           TEXT NOT NULL DEFAULT ''
)`, `
CREATE TABLE IF NOT EXISTS workflow_steps (
  workflow_id             TEXT NOT NULL REFERENCES workflow_executions(id),
  ordinal                 INTEGER NOT NULL CHECK (ordinal BETWEEN 1 AND 16),
  install_id              TEXT REFERENCES install_generations(id),
  worker                  TEXT NOT NULL DEFAULT '',
  authored_assets         TEXT NOT NULL DEFAULT '[]',
  materialized_submission BLOB,
  materialized_digest     TEXT NOT NULL DEFAULT '',
  resolved_bindings       BLOB,
  child_key               TEXT UNIQUE,
  child_request_id        TEXT UNIQUE REFERENCES requests(id),
  PRIMARY KEY (workflow_id, ordinal)
)`}

type WorkflowExecution struct {
	ID                 string
	IdemKey            string
	BodyDigest         string
	ExecutionDigest    string
	CreativePlanDigest string
	Plan               []byte
	State              string
	CancelRequestedAt  string
	TerminalCode       string
	TerminalMessage    string
	CreatedAt          string
	SettledAt          string
	StepCount          int
}

type WorkflowStep struct {
	WorkflowID             string
	Ordinal                int
	InstallID              string
	Worker                 string
	AuthoredAssets         []AssetBinding
	MaterializedSubmission []byte
	MaterializedDigest     string
	ResolvedBindings       []ResolvedBinding
	ChildKey               string
	ChildRequestID         string
}

type ResolvedBinding struct {
	FieldPath         string `json:"field_path"`
	PriorStep         int    `json:"prior_step"`
	PriorRequestID    string `json:"prior_request_id"`
	PriorAttempt      int64  `json:"prior_attempt"`
	OutputName        string `json:"output_name"`
	Digest            string `json:"digest"`
	Length            int64  `json:"length"`
	MediaType         string `json:"media_type"`
	ExpectedMediaKind string `json:"expected_media_kind"`
}

const workflowCols = `w.id,w.idem_key,w.body_digest,w.execution_digest,w.creative_plan_digest,
	w.plan,w.state,w.cancel_requested_at,w.terminal_code,w.terminal_message,w.created_at,w.settled_at,
	(SELECT COUNT(*) FROM workflow_steps s WHERE s.workflow_id=w.id)`

func scanWorkflow(row interface{ Scan(...any) error }) (WorkflowExecution, error) {
	var w WorkflowExecution
	err := row.Scan(&w.ID, &w.IdemKey, &w.BodyDigest, &w.ExecutionDigest,
		&w.CreativePlanDigest, &w.Plan, &w.State, &w.CancelRequestedAt, &w.TerminalCode,
		&w.TerminalMessage, &w.CreatedAt, &w.SettledAt, &w.StepCount)
	return w, err
}

func (s *Store) CreateWorkflow(w WorkflowExecution, assets map[int][]AssetBinding,
	installs, workers map[int]string, stepCount int) (WorkflowExecution, bool, *exit.Error) {
	existing, err := scanWorkflow(s.db.QueryRow(`SELECT `+workflowCols+`
		FROM workflow_executions w WHERE w.idem_key=?`, w.IdemKey))
	if err == nil {
		if existing.BodyDigest != w.BodyDigest {
			return WorkflowExecution{}, false, exit.New(exit.Conflict,
				"idempotency key %s already names a workflow with a different body", w.IdemKey)
		}
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return WorkflowExecution{}, false, exit.Internalf("cannot read workflow %s: %s", w.IdemKey, err)
	}
	if stepCount < 1 || stepCount > 16 {
		return WorkflowExecution{}, false, exit.New(exit.Validation,
			"a workflow has 1-16 steps; got %d", stepCount)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return WorkflowExecution{}, false, exit.Internalf("cannot begin workflow creation: %s", err)
	}
	defer tx.Rollback()
	w.CreatedAt, w.State = now(), "running"
	if _, err := tx.Exec(`INSERT INTO workflow_executions(id,idem_key,body_digest,
		execution_digest,creative_plan_digest,plan,state,created_at) VALUES(?,?,?,?,?,?,?,?)`,
		w.ID, w.IdemKey, w.BodyDigest, w.ExecutionDigest, w.CreativePlanDigest,
		w.Plan, w.State, w.CreatedAt); err != nil {
		return WorkflowExecution{}, false, exit.Internalf("cannot record workflow %s: %s", w.ID, err)
	}
	for ordinal := 1; ordinal <= stepCount; ordinal++ {
		raw, err := json.Marshal(assets[ordinal])
		if err != nil {
			return WorkflowExecution{}, false, exit.Internalf(
				"cannot encode workflow %s step %d assets: %s", w.ID, ordinal, err)
		}
		if _, err := tx.Exec(`INSERT INTO workflow_steps(workflow_id,ordinal,install_id,worker,authored_assets)
			VALUES(?,?,?,?,?)`, w.ID, ordinal, nullable(installs[ordinal]), workers[ordinal], string(raw)); err != nil {
			return WorkflowExecution{}, false, exit.Internalf(
				"cannot record workflow %s step %d: %s", w.ID, ordinal, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return WorkflowExecution{}, false, exit.Internalf("cannot commit workflow %s: %s", w.ID, err)
	}
	w.StepCount = stepCount
	return w, true, nil
}

func (s *Store) Workflow(id string) (*WorkflowExecution, *exit.Error) {
	w, err := scanWorkflow(s.db.QueryRow(`SELECT `+workflowCols+`
		FROM workflow_executions w WHERE w.id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read workflow %s: %s", id, err)
	}
	return &w, nil
}

func (s *Store) WorkflowByIdempotencyKey(key string) (*WorkflowExecution, *exit.Error) {
	w, err := scanWorkflow(s.db.QueryRow(`SELECT `+workflowCols+`
		FROM workflow_executions w WHERE w.idem_key=?`, key))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read workflow for idempotency key %s: %s", key, err)
	}
	return &w, nil
}

func scanWorkflowStep(row interface{ Scan(...any) error }) (WorkflowStep, error) {
	var step WorkflowStep
	var assets string
	var resolved []byte
	err := row.Scan(&step.WorkflowID, &step.Ordinal, &step.InstallID, &step.Worker, &assets,
		&step.MaterializedSubmission, &step.MaterializedDigest, &resolved,
		&step.ChildKey, &step.ChildRequestID)
	if err == nil && assets != "" {
		err = json.Unmarshal([]byte(assets), &step.AuthoredAssets)
	}
	if err == nil && len(resolved) > 0 {
		err = json.Unmarshal(resolved, &step.ResolvedBindings)
	}
	return step, err
}

const workflowStepCols = `workflow_id,ordinal,COALESCE(install_id,''),worker,authored_assets,
	COALESCE(materialized_submission,x''),materialized_digest,COALESCE(resolved_bindings,x''),
	COALESCE(child_key,''),COALESCE(child_request_id,'')`

func (s *Store) WorkflowSteps(id string) ([]WorkflowStep, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+workflowStepCols+` FROM workflow_steps
		WHERE workflow_id=? ORDER BY ordinal`, id)
	if err != nil {
		return nil, exit.Internalf("cannot read workflow %s steps: %s", id, err)
	}
	defer rows.Close()
	var out []WorkflowStep
	for rows.Next() {
		step, err := scanWorkflowStep(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode a workflow %s step: %s", id, err)
		}
		out = append(out, step)
	}
	return out, nil
}

func (s *Store) ActiveWorkflows() ([]WorkflowExecution, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + workflowCols + ` FROM workflow_executions w
		WHERE w.state IN ('running','canceling') ORDER BY w.created_at,w.id`)
	if err != nil {
		return nil, exit.Internalf("cannot read active workflows: %s", err)
	}
	defer rows.Close()
	var out []WorkflowExecution
	for rows.Next() {
		w, err := scanWorkflow(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode an active workflow: %s", err)
		}
		out = append(out, w)
	}
	return out, nil
}

func (s *Store) PrepareWorkflowStep(workflowID string, ordinal int, installID string,
	materialized []byte, digest string, resolved []ResolvedBinding, childKey string) *exit.Error {
	rawResolved, err := json.Marshal(resolved)
	if err != nil {
		return exit.Internalf("cannot encode workflow %s step %d bindings: %s", workflowID, ordinal, err)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin workflow step preparation: %s", err)
	}
	defer tx.Rollback()
	var existingInstall, existingDigest, existingKey string
	if err := tx.QueryRow(`SELECT COALESCE(install_id,''),materialized_digest,COALESCE(child_key,'') FROM workflow_steps
		WHERE workflow_id=? AND ordinal=?`, workflowID, ordinal).
		Scan(&existingInstall, &existingDigest, &existingKey); err != nil {
		return exit.Internalf("cannot read workflow %s step %d: %s", workflowID, ordinal, err)
	}
	if existingInstall != "" && existingInstall != installID {
		return exit.New(exit.Conflict,
			"workflow %s step %d is pinned to install %s, not %s",
			workflowID, ordinal, existingInstall, installID)
	}
	if existingDigest != "" {
		if existingDigest != digest || existingKey != childKey {
			return exit.New(exit.Conflict,
				"workflow %s step %d was already materialized with different bytes", workflowID, ordinal)
		}
		return nil
	}
	if _, err := tx.Exec(`UPDATE workflow_steps SET install_id=?,materialized_submission=?,
		materialized_digest=?,resolved_bindings=?,child_key=?
		WHERE workflow_id=? AND ordinal=? AND materialized_digest=''`,
		nullable(installID), materialized, digest, rawResolved, childKey, workflowID, ordinal); err != nil {
		return exit.Internalf("cannot prepare workflow %s step %d: %s", workflowID, ordinal, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit workflow %s step %d preparation: %s", workflowID, ordinal, err)
	}
	return nil
}

func (s *Store) LinkWorkflowChild(workflowID string, ordinal int, childKey, requestID string) *exit.Error {
	res, err := s.db.Exec(`UPDATE workflow_steps SET child_request_id=?
		WHERE workflow_id=? AND ordinal=? AND child_key=?
		AND (child_request_id IS NULL OR child_request_id=?)`,
		requestID, workflowID, ordinal, childKey, requestID)
	if err != nil {
		return exit.Internalf("cannot link workflow %s step %d child: %s", workflowID, ordinal, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "workflow %s step %d child link disagrees", workflowID, ordinal)
	}
	return nil
}

func (s *Store) RequestWorkflowCancel(id string) (*WorkflowExecution, bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, false, exit.Internalf("cannot begin workflow cancellation: %s", err)
	}
	defer tx.Rollback()
	w, err := scanWorkflow(tx.QueryRow(`SELECT `+workflowCols+`
		FROM workflow_executions w WHERE w.id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, exit.Internalf("cannot read workflow %s: %s", id, err)
	}
	if workflowTerminal(w.State) || w.CancelRequestedAt != "" {
		return &w, false, nil
	}
	w.CancelRequestedAt, w.State = now(), "canceling"
	if _, err := tx.Exec(`UPDATE workflow_executions SET state='canceling',cancel_requested_at=?
		WHERE id=? AND state='running'`, w.CancelRequestedAt, id); err != nil {
		return nil, false, exit.Internalf("cannot request workflow %s cancellation: %s", id, err)
	}
	if err := tx.Commit(); err != nil {
		return nil, false, exit.Internalf("cannot commit workflow %s cancellation: %s", id, err)
	}
	return &w, true, nil
}

func (s *Store) SettleWorkflow(id, state, code, message string) *exit.Error {
	if !workflowTerminal(state) {
		return exit.Internalf("cannot settle workflow %s to nonterminal state %q", id, state)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin workflow settlement: %s", err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE workflow_executions SET state=?,terminal_code=?,
		terminal_message=?,settled_at=? WHERE id=? AND state IN ('running','canceling')`,
		state, code, message, now(), id); err != nil {
		return exit.Internalf("cannot settle workflow %s: %s", id, err)
	}
	if _, err := tx.Exec(`UPDATE workflow_steps SET install_id=NULL WHERE workflow_id=?`, id); err != nil {
		return exit.Internalf("cannot release workflow %s installs: %s", id, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit workflow %s settlement: %s", id, err)
	}
	return nil
}

func workflowTerminal(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled":
		return true
	}
	return false
}

func (s *Store) workflowAssetInUse(digest string) (bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT s.authored_assets FROM workflow_steps s
		JOIN workflow_executions w ON w.id=s.workflow_id
		WHERE w.state IN ('running','canceling') AND s.authored_assets <> '[]'`)
	if err != nil {
		return false, exit.Internalf("cannot read live workflow asset ownership: %s", err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return false, exit.Internalf("cannot read a workflow asset binding: %s", err)
		}
		var assets []AssetBinding
		if err := json.Unmarshal([]byte(raw), &assets); err != nil {
			return false, exit.Internalf("cannot decode a workflow asset binding: %s", err)
		}
		for _, asset := range assets {
			if asset.Digest == digest {
				return true, nil
			}
		}
	}
	return false, nil
}
