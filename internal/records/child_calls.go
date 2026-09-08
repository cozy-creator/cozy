package records

import (
	"database/sql"
	"encoding/json"
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
	for _, asset := range r.Assets {
		held := false
		for _, original := range parent.Assets {
			if asset.Digest == original.Digest && asset.Length == original.Length &&
				asset.MediaType == original.MediaType && asset.LocalPath == original.LocalPath {
				held = true
				break
			}
		}
		if !held {
			return Request{}, false, exit.Named(exit.Conflict, "child.asset_ungranted", "child input bytes are not owned by the current parent")
		}
	}
	existing, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE parent_request_id=? AND parent_call_index=?`, parent.ID, r.ParentCallIndex))
	if err == nil {
		if existing.ChildIntentDigest != r.ChildIntentDigest || existing.ChildTargetDigest != r.ChildTargetDigest || existing.ChildReusable != r.ChildReusable || existing.ChildArtifacts != r.ChildArtifacts {
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
	r.ReusedFrom = "" // Only an authenticated workspace lookup may adopt old results.
	r.ReuseScope = parent.ReuseScope
	if r.ReuseScope == "" {
		r.ReuseScope = parent.ID
	}
	recorded, fresh, problem := submitRequestTx(tx, r, assets, models, exports)
	if problem != nil {
		return Request{}, false, problem
	}
	if !fresh {
		return recorded, false, nil
	}
	var arguments map[string]json.RawMessage
	if len(r.Models) > 0 && json.Unmarshal(r.Payload, &arguments) != nil {
		return Request{}, false, exit.New(exit.Validation, "child model payload is not JSON")
	}
	for _, model := range r.Models {
		artifact, problem := DecodeModelArtifact(arguments[model.Slot])
		if problem != nil {
			return Request{}, false, problem
		}
		if artifact == nil || artifact.Manifest.Digest != model.Manifest || artifact.Manifest.Length != model.ManifestLength {
			return Request{}, false, exit.Named(exit.Conflict, "child.model_changed", "child model grant differs from its canonical artifact handle")
		}
		var ordinal int64
		if err := tx.QueryRow(`SELECT attempt FROM request_model_transfer_outputs WHERE request_id=? AND output_slot=? AND manifest_id=? AND manifest_length=?
			AND json_extract(CAST(receipt AS TEXT),'$.tensorfs_receipt_digest')=? ORDER BY attempt DESC LIMIT 1`, artifact.ProducerRequestID, artifact.OutputSlot, artifact.Manifest.Digest, artifact.Manifest.Length, artifact.TensorFSReceiptDigest).Scan(&ordinal); err != nil {
			return Request{}, false, exit.Named(exit.Conflict, "child.artifact_unowned", "child input no longer has its original native provenance")
		}
		retention := WeightsRetention{RequestID: r.ID, Kind: "input", Slot: "result/" + model.Slot, ProducerRequestID: artifact.ProducerRequestID, ProducerAttempt: ordinal, ProducerOutputSlot: artifact.OutputSlot,
			RetentionID: ArtifactRetentionID(r.ID, "input", "result/"+model.Slot, *artifact)}
		if _, problem := recordWeightsRetentionTx(tx, retention); problem != nil {
			return Request{}, false, problem
		}
	}
	event := "request.submitted"
	payload := map[string]any{"parent_request_id": parent.ID, "call_index": r.ParentCallIndex, "child_intent_digest": r.ChildIntentDigest, "child_target_digest": r.ChildTargetDigest}
	if err := appendEventTx(tx, r.ID, event, 0, payload); err != nil {
		return Request{}, false, exit.Internalf("cannot journal child admission: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return Request{}, false, exit.Internalf("cannot commit child admission: %s", err)
	}
	return recorded, true, nil
}

func (s *Store) CompleteReusedChild(id string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin reused result completion: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='succeeded' WHERE id=? AND reused_from<>'' AND ordinal=0
		AND (state='finalizing' OR (state IN ('submitted','queued') AND EXISTS(SELECT 1 FROM request_operation_lookups l WHERE l.request_id=requests.id AND l.state='hit')))
		AND NOT EXISTS(SELECT 1 FROM request_weights_retentions h WHERE h.request_id=requests.id AND h.kind='result' AND h.state!='held')
		AND (child_artifacts=0 OR EXISTS(SELECT 1 FROM request_weights_retentions h WHERE h.request_id=requests.id AND h.kind='result' AND h.state='held'))
		AND (weights_outputs IN ('','[]') OR (SELECT COUNT(*) FROM request_weights_retentions h WHERE h.request_id=requests.id AND h.kind='result' AND h.slot LIKE 'weights/%' AND h.state='held')=json_array_length(weights_outputs))`, id)
	if err != nil {
		return exit.Internalf("cannot complete reused child result: %s", err)
	}
	if n, _ := result.RowsAffected(); n == 0 {
		var state string
		if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&state); err != nil {
			return exit.Internalf("cannot read reused result completion: %s", err)
		}
		if state == "succeeded" {
			return nil
		}
		if state == "finalizing" {
			return exit.Named(exit.Unavailable, "child.result_custody_pending", "the reused result still awaits independent artifact custody")
		}
		return exit.Named(exit.Canceled, "child.result_stopped", "the reused result was stopped before publication")
	}
	if err := appendEventTx(tx, id, "request.completed", 0, map[string]any{"status": "SUCCEEDED", "reused": true}); err != nil {
		return exit.Internalf("cannot journal reused child result: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit reused child result: %s", err)
	}
	return nil
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

const retainedDescendantsSQL = `WITH RECURSIVE descendants(id) AS (
		SELECT id FROM requests WHERE parent_request_id=?
		UNION ALL SELECT r.id FROM requests r JOIN descendants d ON r.parent_request_id=d.id
	) SELECT EXISTS(SELECT 1 FROM descendants d JOIN requests r ON r.id=d.id
		LEFT JOIN request_model_transfers t ON t.request_id=r.id WHERE r.retain_work=1 AND
		(r.state IN (` + activeRequestStates + `) OR (r.state='succeeded' AND
		(r.child_artifacts=1 OR r.weights_outputs NOT IN ('','[]')) AND COALESCE(json_extract(t.intent,'$.destination'),'')='')))`

// RequestRetaining includes work delegated by a script that returns no result.
// Descendants remain ordinary requests; this is only the parent's retention view.
func (s *Store) RequestRetaining(request Request) (bool, *exit.Error) {
	if !request.RetainWork || (Settled(request.State) && request.State != "succeeded") {
		return false, nil
	}
	if request.State != "succeeded" || request.RetainsLocalOutputs() {
		return true, nil
	}
	var retained bool
	err := s.db.QueryRow(retainedDescendantsSQL, request.ID).Scan(&retained)
	if err != nil {
		return false, exit.Internalf("cannot inspect retained child work: %s", err)
	}
	return retained, nil
}
