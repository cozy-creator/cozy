package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"reflect"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

type sourceAdoption struct {
	PriorID     string `json:"retry_of"`
	PriorHead   string `json:"prior_head"`
	AdoptedHead string `json:"adopted_head"`
}

// RecordRetriedSourceAdoption records the native owner's independent new roots.
// Checkpoint byte counters include journal history and are not a progress measure
// across different chains; explicit provenance is the release barrier instead.
func (s *Store) RecordRetriedSourceAdoption(id, priorID, bootID string, previous, adopted []ModelCheckpoint) *exit.Error {
	if len(previous) == 0 || len(previous) != len(adopted) {
		return exit.Named(exit.Conflict, "request.source_adoption_incomplete", "source adoption must retain each selected old head")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin source adoption: %s", err)
	}
	defer tx.Rollback()
	var valid bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests child JOIN requests parent ON child.retry_of=parent.id
		WHERE child.id=? AND parent.id=? AND child.retain_work=1 AND child.worker=parent.worker)`, id, priorID).Scan(&valid); err != nil || !valid {
		return exit.Named(exit.Conflict, "request.source_adoption_refused", "source adoption has no admitted same-machine lineage")
	}
	for i, old := range previous {
		fresh := adopted[i]
		if old.Slot != fresh.Slot || old.PlanDigest != fresh.PlanDigest || old.HeadID == fresh.HeadID {
			return exit.Named(exit.Conflict, "request.source_adoption_changed", "source adoption must produce an independent head of the same slot and plan")
		}
		var oldRaw, newRaw, oldBoot, newBoot string
		if err := tx.QueryRow(`SELECT p.observed,c.observed,p.worker_boot_id,c.worker_boot_id
			FROM request_model_checkpoints p JOIN request_model_checkpoints c ON c.slot=p.slot AND c.kind=p.kind
			WHERE p.request_id=? AND c.request_id=? AND p.kind='source' AND p.slot=?`, priorID, id, old.Slot).
			Scan(&oldRaw, &newRaw, &oldBoot, &newBoot); err != nil {
			return exit.Internalf("cannot verify observed source adoption: %s", err)
		}
		var heldOld, heldNew ModelCheckpoint
		if json.Unmarshal([]byte(oldRaw), &heldOld) != nil || json.Unmarshal([]byte(newRaw), &heldNew) != nil || heldOld != old || heldNew != fresh || oldBoot != bootID || newBoot != bootID {
			return exit.Named(exit.Conflict, "request.source_adoption_changed", "source adoption observations changed before ownership transfer")
		}
		provenance, _ := json.Marshal(sourceAdoption{priorID, old.HeadID, fresh.HeadID})
		if _, err := tx.Exec(`UPDATE request_model_checkpoints SET subject=? WHERE request_id=? AND kind='source' AND slot=?`, provenance, id, old.Slot); err != nil {
			return exit.Internalf("cannot record source adoption ownership: %s", err)
		}
	}
	if err := appendEventTx(tx, id, "request.source_reused", 0, map[string]any{"retry_of": priorID, "slots": len(adopted)}); err != nil {
		return exit.Internalf("cannot journal source adoption: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source adoption: %s", err)
	}
	return nil
}

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

// RetriedSourceCheckpoints selects the exact predecessor acquisition. Private
// work uses observed local heads; the caller must also prove the original boot
// still holds them. These observations never become remote custody acknowledgments.
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
	if request.RetainWork {
		complete := true
		for _, item := range progress {
			adopted, problem := s.sourceAlreadyAdopted(id, prior.ID, item)
			if problem != nil {
				return "", nil, problem
			}
			if !adopted {
				complete = false
			}
		}
		if complete {
			return prior.ID, nil, nil
		}
		return prior.ID, progress, nil
	}
	retained := progress[:0]
	for _, item := range progress {
		if item.Acknowledged != nil {
			retained = append(retained, item)
		}
	}
	return prior.ID, retained, nil
}

// Once a retry owns an independent native head it resumes that head, even after
// the predecessor is canceled. Later checkpoints keep the recorded provenance.
func (s *Store) sourceAlreadyAdopted(id, priorID string, prior ModelCheckpointProgress) (bool, *exit.Error) {
	var raw, observed []byte
	var boot string
	err := s.db.QueryRow(`SELECT subject,observed,worker_boot_id FROM request_model_checkpoints
		WHERE request_id=? AND kind='source' AND slot=?`, id, prior.Observed.Slot).Scan(&raw, &observed, &boot)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, exit.Internalf("cannot read adopted source custody: %s", err)
	}
	var provenance sourceAdoption
	var current ModelCheckpoint
	if json.Unmarshal(raw, &provenance) != nil || json.Unmarshal(observed, &current) != nil {
		return false, nil
	}
	_, err = canonical.Raw(provenance.AdoptedHead)
	return err == nil && provenance.PriorID == priorID && provenance.PriorHead == prior.Observed.HeadID &&
		provenance.AdoptedHead != prior.Observed.HeadID && boot == prior.WorkerBootID &&
		current.Slot == prior.Observed.Slot && current.PlanDigest == prior.Observed.PlanDigest && current.HeadID != prior.Observed.HeadID, nil
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
	err = rows.Err()
	rows.Close()
	if err != nil {
		return false, exit.Internalf("cannot finish dependent source custody census: %s", err)
	}
	for _, id := range ids {
		_, prior, problem := s.RetriedSourceCheckpoints(id)
		if problem != nil {
			return false, problem
		}
		if len(prior) > 0 {
			return true, nil
		}
	}
	return false, nil
}
