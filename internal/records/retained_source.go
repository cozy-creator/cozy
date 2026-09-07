package records

import (
	"database/sql"
	"encoding/json"
	"reflect"

	"github.com/cozy-creator/cozy/internal/exit"
)

// SameSourceAcquisition compares source computation only. Producer code, output
// declarations and destination publication are independently allowed to change.
func SameSourceAcquisition(a, b *ModelTransferIntent) bool {
	if a == nil || b == nil || !a.HasAcquisition() || !b.HasAcquisition() {
		return false
	}
	left, right := *a, *b
	left.Kind, right.Kind = "", ""
	left.Destination, right.Destination = "", ""
	left.Outputs, right.Outputs = nil, nil
	return reflect.DeepEqual(left, right)
}

// RetriedSourceCheckpoints returns only acknowledged checkpoints from the exact
// predecessor acquisition. Merely observed bytes are never promoted to custody.
func (s *Store) RetriedSourceCheckpoints(id string) (string, []ModelCheckpointProgress, *exit.Error) {
	request, problem := s.RequestRow(id)
	if problem != nil || request == nil || request.RetryOf == "" {
		return "", nil, problem
	}
	prior, problem := s.RequestRow(request.RetryOf)
	if problem != nil || prior == nil || !SameSourceAcquisition(request.ModelTransfer, prior.ModelTransfer) {
		return "", nil, problem
	}
	progress, problem := s.ModelSourceProgress(prior.ID)
	if problem != nil {
		return "", nil, problem
	}
	retained := progress[:0]
	for _, item := range progress {
		if item.Acknowledged != nil {
			retained = append(retained, item)
		}
	}
	return prior.ID, retained, nil
}

// AdoptRetriedSourceCheckpoint follows independently verified custody under the
// new request's publication holds. Native restore still checks the full source
// plan and content before dispatch; this copies no worker status or model handle.
func (s *Store) AdoptRetriedSourceCheckpoint(id, priorID string, checkpoint ModelCheckpoint) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin source checkpoint adoption: %s", err)
	}
	defer tx.Rollback()
	var parent string
	if err := tx.QueryRow(`SELECT retry_of FROM requests WHERE id=? AND retain_work=1 AND state IN (`+activeRequestStates+`)`, id).Scan(&parent); err != nil || parent != priorID {
		return exit.Named(exit.Conflict, "request.source_adoption_refused", "source adoption must name the admitted retained predecessor")
	}
	var priorRaw, currentRaw string
	if err := tx.QueryRow(`SELECT p.intent,c.intent FROM request_model_transfers p,request_model_transfers c
		WHERE p.request_id=? AND c.request_id=?`, priorID, id).Scan(&priorRaw, &currentRaw); err != nil {
		return exit.Internalf("cannot read source adoption identities: %s", err)
	}
	var prior, current ModelTransferIntent
	if json.Unmarshal([]byte(priorRaw), &prior) != nil || json.Unmarshal([]byte(currentRaw), &current) != nil || !SameSourceAcquisition(&prior, &current) {
		return exit.Named(exit.Conflict, "request.source_adoption_changed", "source adoption changed the selected source or conversion profile")
	}
	if problem := validateSourceCheckpoint(current, checkpoint); problem != nil {
		return problem
	}
	data, _ := json.Marshal(checkpoint)
	var held string
	if err := tx.QueryRow(`SELECT acknowledged FROM request_model_checkpoints WHERE request_id=? AND kind='source' AND slot=?`, priorID, checkpoint.Slot).Scan(&held); err != nil || held != string(data) {
		return exit.Named(exit.Conflict, "request.source_adoption_changed", "source adoption must preserve the predecessor's exact acknowledged checkpoint")
	}
	var observed, acknowledged string
	err = tx.QueryRow(`SELECT observed,acknowledged FROM request_model_checkpoints WHERE request_id=? AND kind='source' AND slot=?`, id, checkpoint.Slot).Scan(&observed, &acknowledged)
	if err == nil {
		// A restart after this commit, or later progress, already owns its result.
		var existing ModelCheckpoint
		if json.Unmarshal([]byte(acknowledged), &existing) == nil && existing.PlanDigest == checkpoint.PlanDigest && existing.Index >= checkpoint.Index {
			return nil
		}
		return exit.Named(exit.Conflict, "request.source_adoption_changed", "source adoption cannot replace independently started progress")
	}
	if err != sql.ErrNoRows {
		return exit.Internalf("cannot read source adoption progress: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO request_model_checkpoints(request_id,slot,worker_boot_id,observed,acknowledged)
		VALUES(?,?,'',?,?)`, id, checkpoint.Slot, string(data), string(data)); err != nil {
		return exit.Internalf("cannot adopt source checkpoint: %s", err)
	}
	if err := appendEventTx(tx, id, "request.source_reused", 0, map[string]any{"retry_of": priorID, "slot": checkpoint.Slot, "head": checkpoint.HeadID, "bytes": checkpoint.Bytes}); err != nil {
		return exit.Internalf("cannot journal source reuse: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source checkpoint adoption: %s", err)
	}
	return nil
}

// RetriedSourceCustodyPending keeps predecessor holds alive until each dependent
// revision acquires its own verified checkpoint custody or is abandoned.
func (s *Store) RetriedSourceCustodyPending(priorID string) (bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT id FROM requests WHERE retry_of=? AND retain_work=1 AND state IN (`+activeRequestStates+`) AND state NOT IN ('canceling','releasing')`, priorID)
	if err != nil {
		return false, exit.Internalf("cannot inspect dependent source custody: %s", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return false, exit.Internalf("cannot read dependent source custody: %s", err)
		}
		ids = append(ids, id)
	}
	rows.Close()
	for _, id := range ids {
		_, prior, problem := s.RetriedSourceCheckpoints(id)
		if problem != nil {
			return false, problem
		}
		current, problem := s.ModelSourceProgress(id)
		if problem != nil {
			return false, problem
		}
		bySlot := map[string]*ModelCheckpoint{}
		for _, progress := range current {
			bySlot[progress.Observed.Slot] = progress.Acknowledged
		}
		for _, progress := range prior {
			owned := bySlot[progress.Observed.Slot]
			if owned == nil || owned.PlanDigest != progress.Acknowledged.PlanDigest || owned.Index < progress.Acknowledged.Index {
				return true, nil
			}
		}
	}
	return false, nil
}
