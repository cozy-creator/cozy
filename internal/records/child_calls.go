package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// SubmitChild records an ordinary request under the parent/index identity. The
// parent attempt and its captured executable authority are checked in the same
// transaction that acquires the child. Runtime supplies no placement or scope.
func (s *Store) SubmitChild(r Request, parentAttempt int64, parentSpec, parentSession string) (Request, bool, *exit.Error) {
	if r.ParentRequestID == "" || r.ParentCallIndex < 0 || r.ParentCallIndex >= 32 || parentAttempt <= 0 {
		return Request{}, false, exit.New(exit.Validation, "child call must name a current parent attempt and bounded call index")
	}
	if _, err := canonical.Raw(r.ChildIntentDigest); err != nil {
		return Request{}, false, exit.New(exit.Validation, "child intent digest is malformed")
	}
	if _, err := canonical.Raw(r.ChildTargetDigest); err != nil {
		return Request{}, false, exit.New(exit.Validation, "child target digest is malformed")
	}
	r, assets, models, exports, problem := prepareRequest(r)
	if problem != nil {
		return Request{}, false, problem
	}
	tx, err := s.db.Begin()
	if err != nil {
		return Request{}, false, exit.Internalf("cannot begin child admission: %s", err)
	}
	defer tx.Rollback()
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, r.ParentRequestID))
	if err != nil {
		return Request{}, false, exit.Named(exit.Conflict, "child.parent_unavailable", "the child call has no retained parent request")
	}
	var owned bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND attempt=? AND session_id=? AND invocation_digest=? AND state IN ('offered','accepted','recovered_open'))`, parent.ID, parentAttempt, parentSession, parentSpec).Scan(&owned); err != nil {
		return Request{}, false, exit.Internalf("cannot inspect child caller authority: %s", err)
	}
	if !owned || !parent.IsJob() || !parent.RetainWork || parent.State != "dispatching" || parent.Ordinal != parentAttempt {
		return Request{}, false, exit.Named(exit.Conflict, "child.parent_stopped", "the parent attempt is not authorized to create child work")
	}
	existing, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE parent_request_id=? AND parent_call_index=?`, parent.ID, r.ParentCallIndex))
	if err == nil {
		if existing.ChildIntentDigest != r.ChildIntentDigest || existing.ChildTargetDigest != r.ChildTargetDigest {
			return Request{}, false, exit.Named(exit.Conflict, "child.intent_changed", "a parent call index already names a different target or input")
		}
		existing.Number, err = requestNumber(tx, existing)
		if err != nil {
			return Request{}, false, exit.Internalf("cannot number child replay: %s", err)
		}
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return Request{}, false, exit.Internalf("cannot inspect child replay: %s", err)
	}
	r.RetainWork = true
	r.ReuseScope = parent.ReuseScope
	if r.ReuseScope == "" {
		r.ReuseScope = parent.ID
	}
	// Scalar result reuse has no native byte ownership to transfer. Artifact and
	// weights children continue through ordinary execution until their native
	// adoption contract supplies independent new-request ownership.
	if r.Outputs == "" && (r.WeightsOutputs == "" || r.WeightsOutputs == "[]") && parent.RetryOf != "" {
		var previous string
		err = tx.QueryRow(`SELECT COALESCE(NULLIF(reused_from,''),id) FROM requests WHERE parent_request_id=? AND parent_call_index=?
			AND state='succeeded' AND child_intent_digest=? AND child_target_digest=? AND reuse_scope=?
			AND EXISTS(SELECT 1 FROM attempts WHERE request_id=COALESCE(NULLIF(requests.reused_from,''),requests.id) AND state='closed' AND terminal_status='SUCCEEDED')
			ORDER BY created_at DESC,id DESC LIMIT 1`, parent.RetryOf, r.ParentCallIndex, r.ChildIntentDigest, r.ChildTargetDigest, r.ReuseScope).Scan(&previous)
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return Request{}, false, exit.Internalf("cannot inspect reusable child result: %s", err)
		}
		if err == nil {
			r.ReusedFrom = previous
		}
	}
	recorded, fresh, problem := submitRequestTx(tx, r, assets, models, exports)
	if problem != nil {
		return Request{}, false, problem
	}
	if !fresh {
		return recorded, false, nil
	}
	event := "request.submitted"
	payload := map[string]any{"parent_request_id": parent.ID, "call_index": r.ParentCallIndex, "child_intent_digest": r.ChildIntentDigest, "child_target_digest": r.ChildTargetDigest}
	if r.ReusedFrom != "" {
		if _, err := tx.Exec(`UPDATE requests SET state='succeeded' WHERE id=?`, r.ID); err != nil {
			return Request{}, false, exit.Internalf("cannot acquire reused child result: %s", err)
		}
		recorded.State = "succeeded"
		event = "request.completed"
		payload["status"], payload["reused_from"] = "SUCCEEDED", r.ReusedFrom
	}
	if err := appendEventTx(tx, r.ID, event, 0, payload); err != nil {
		return Request{}, false, exit.Internalf("cannot journal child admission: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return Request{}, false, exit.Internalf("cannot commit child admission: %s", err)
	}
	return recorded, true, nil
}

func (s *Store) Children(parent string) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+requestCols+` FROM requests WHERE parent_request_id=? ORDER BY parent_call_index`, parent)
	if err != nil {
		return nil, exit.Internalf("cannot read child calls: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		row, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read child call: %s", err)
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish child call census: %s", err)
	}
	return out, nil
}
