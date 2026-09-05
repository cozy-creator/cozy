package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const modelSourceCheckpointSchema = `
CREATE TABLE IF NOT EXISTS request_model_source_checkpoints (
  request_id TEXT NOT NULL REFERENCES requests(id),
  slot TEXT NOT NULL,
  worker_boot_id TEXT NOT NULL,
  observed TEXT NOT NULL,
  acknowledged TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,slot)
)`

// ModelSourceCheckpoint is a TensorFS recovery head. Observing it says nothing
// about remote custody; only the separately acknowledged head may restore a pod.
type ModelSourceCheckpoint struct {
	Slot       string `json:"slot"`
	HeadID     string `json:"head_id"`
	HeadLength int64  `json:"head_length"`
	PlanDigest string `json:"plan_digest"`
	Index      int64  `json:"index"`
	Bytes      int64  `json:"bytes"`
}

type ModelSourceProgress struct {
	WorkerBootID string
	Observed     ModelSourceCheckpoint
	Acknowledged *ModelSourceCheckpoint
}

func scanModelSourceProgress(row interface{ Scan(...any) error }) (ModelSourceProgress, error) {
	var progress ModelSourceProgress
	var observed, acknowledged string
	if err := row.Scan(&progress.WorkerBootID, &observed, &acknowledged); err != nil {
		return progress, err
	}
	if err := json.Unmarshal([]byte(observed), &progress.Observed); err != nil {
		return progress, err
	}
	if acknowledged != "" {
		progress.Acknowledged = &ModelSourceCheckpoint{}
		if err := json.Unmarshal([]byte(acknowledged), progress.Acknowledged); err != nil {
			return progress, err
		}
	}
	return progress, nil
}

func (s *Store) ModelSourceProgress(requestID string) ([]ModelSourceProgress, *exit.Error) {
	rows, err := s.db.Query(`SELECT worker_boot_id,observed,acknowledged
		FROM request_model_source_checkpoints WHERE request_id=? ORDER BY slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read source checkpoint progress: %s", err)
	}
	defer rows.Close()
	var out []ModelSourceProgress
	for rows.Next() {
		progress, err := scanModelSourceProgress(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode source checkpoint progress: %s", err)
		}
		out = append(out, progress)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish source checkpoint progress: %s", err)
	}
	return out, nil
}

func sourceCheckpointOwner(tx *sql.Tx, requestID, selection string) (ModelTransferIntent, *exit.Error) {
	var intent ModelTransferIntent
	var raw, state string
	err := tx.QueryRow(`SELECT intent,state FROM request_model_transfers WHERE request_id=?`, requestID).
		Scan(&raw, &state)
	if err != nil || json.Unmarshal([]byte(raw), &intent) != nil {
		return intent, exit.Internalf("cannot read source checkpoint owner")
	}
	if intent.SourceSelection != selection || state == "failed" || state == "canceled" || state == "completed" {
		return intent, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_stale",
			"source checkpoint does not belong to the active selected operation")
	}
	return intent, nil
}

func validateSourceCheckpoint(intent ModelTransferIntent, checkpoint ModelSourceCheckpoint) *exit.Error {
	_, headErr := canonical.Raw(checkpoint.HeadID)
	_, planErr := canonical.Raw(checkpoint.PlanDigest)
	if intent.SourceProfiles[checkpoint.Slot] == "" || headErr != nil || planErr != nil ||
		checkpoint.HeadLength <= 0 || checkpoint.Index < 0 || checkpoint.Bytes < 0 {
		return exit.Named(exit.Validation, "model_transfer.source_checkpoint_invalid",
			"source checkpoint must name a declared slot and exact bounded progress")
	}
	return nil
}

// ObserveModelSourceCheckpoints freezes each slot's plan and advances only its
// current worker's local watermark. A replacement boot may start from the last
// acknowledged head, never from unacknowledged progress left by the old pod.
func (s *Store) ObserveModelSourceCheckpoints(requestID, selection, bootID string,
	checkpoints []ModelSourceCheckpoint,
) *exit.Error {
	if bootID == "" {
		return exit.New(exit.Validation, "source checkpoint observer has no worker boot")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin source checkpoint observation: %s", err)
	}
	defer tx.Rollback()
	intent, problem := sourceCheckpointOwner(tx, requestID, selection)
	if problem != nil {
		return problem
	}
	seen := map[string]bool{}
	for _, checkpoint := range checkpoints {
		if problem := validateSourceCheckpoint(intent, checkpoint); problem != nil {
			return problem
		}
		if seen[checkpoint.Slot] {
			return exit.New(exit.Validation, "source checkpoint repeats a slot")
		}
		seen[checkpoint.Slot] = true
		prior, err := scanModelSourceProgress(tx.QueryRow(`SELECT worker_boot_id,observed,acknowledged
			FROM request_model_source_checkpoints WHERE request_id=? AND slot=?`, requestID, checkpoint.Slot))
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return exit.Internalf("cannot read prior source checkpoint: %s", err)
		}
		if err == nil {
			if prior.Observed.PlanDigest != checkpoint.PlanDigest {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_plan_changed",
					"source checkpoint changed the slot's frozen TensorFS plan")
			}
			if prior.Acknowledged != nil && (checkpoint.Index < prior.Acknowledged.Index ||
				(checkpoint.Index == prior.Acknowledged.Index && checkpoint != *prior.Acknowledged)) {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_required",
					"source checkpoint precedes or changes acknowledged recovery progress")
			}
			if prior.WorkerBootID == bootID {
				if checkpoint.Index < prior.Observed.Index {
					continue // replayed older observation; it cannot move the watermark back
				}
				if (checkpoint.Index == prior.Observed.Index && checkpoint != prior.Observed) ||
					checkpoint.Bytes < prior.Observed.Bytes {
					return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_changed",
						"source checkpoint changed an existing index or reduced its cumulative bytes")
				}
			}
		}
		data, _ := json.Marshal(checkpoint)
		if _, err := tx.Exec(`INSERT INTO request_model_source_checkpoints
			(request_id,slot,worker_boot_id,observed) VALUES(?,?,?,?)
			ON CONFLICT(request_id,slot) DO UPDATE SET worker_boot_id=excluded.worker_boot_id,observed=excluded.observed`,
			requestID, checkpoint.Slot, bootID, string(data)); err != nil {
			return exit.Internalf("cannot record source checkpoint observation: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source checkpoint observation: %s", err)
	}
	return nil
}

// AcknowledgeModelSourceCheckpoint follows verified custody of every link object.
// The caller supplies its previous acknowledged head so a stale uploader cannot
// overwrite a newer acknowledgment; a replaced worker also loses this authority.
func (s *Store) AcknowledgeModelSourceCheckpoint(requestID, selection, bootID, previousHead string,
	checkpoint ModelSourceCheckpoint,
) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin source checkpoint acknowledgment: %s", err)
	}
	defer tx.Rollback()
	intent, problem := sourceCheckpointOwner(tx, requestID, selection)
	if problem != nil {
		return problem
	}
	if problem := validateSourceCheckpoint(intent, checkpoint); problem != nil {
		return problem
	}
	progress, err := scanModelSourceProgress(tx.QueryRow(`SELECT worker_boot_id,observed,acknowledged
		FROM request_model_source_checkpoints WHERE request_id=? AND slot=?`, requestID, checkpoint.Slot))
	if err != nil {
		return exit.Internalf("cannot read checkpoint acknowledgment target: %s", err)
	}
	if progress.Acknowledged != nil && *progress.Acknowledged == checkpoint {
		return nil
	}
	priorHead := ""
	if progress.Acknowledged != nil {
		priorHead = progress.Acknowledged.HeadID
		if checkpoint.Index <= progress.Acknowledged.Index || checkpoint.Bytes < progress.Acknowledged.Bytes {
			return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded",
				"source checkpoint acknowledgment cannot move backwards")
		}
	}
	if priorHead != previousHead || progress.WorkerBootID != bootID ||
		progress.Observed.PlanDigest != checkpoint.PlanDigest || checkpoint.Index > progress.Observed.Index ||
		(checkpoint.Index == progress.Observed.Index && checkpoint != progress.Observed) {
		return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded",
			"source checkpoint acknowledgment no longer matches the observed operation")
	}
	data, _ := json.Marshal(checkpoint)
	if _, err := tx.Exec(`UPDATE request_model_source_checkpoints SET acknowledged=? WHERE request_id=? AND slot=?`,
		string(data), requestID, checkpoint.Slot); err != nil {
		return exit.Internalf("cannot record source checkpoint acknowledgment: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source checkpoint acknowledgment: %s", err)
	}
	return nil
}
