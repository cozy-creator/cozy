package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const modelCheckpointSchema = `
CREATE TABLE IF NOT EXISTS request_model_checkpoints (
  request_id TEXT NOT NULL REFERENCES requests(id),
  kind TEXT NOT NULL DEFAULT 'source' CHECK(kind IN ('source','weights')),
  slot TEXT NOT NULL,
  subject BLOB NOT NULL DEFAULT x'',
  attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt>=0),
  worker_boot_id TEXT NOT NULL,
  observed TEXT NOT NULL DEFAULT '',
  acknowledged TEXT NOT NULL DEFAULT '',
  grant_revision INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(request_id,kind,slot)
)`

const modelCheckpointPublicationSchema = `
CREATE TABLE IF NOT EXISTS request_model_checkpoint_publications (
  request_id TEXT NOT NULL REFERENCES requests(id),
  operation TEXT NOT NULL,
  objects BLOB NOT NULL,
  opened INTEGER NOT NULL DEFAULT 0,
  released INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(request_id,operation)
)`

// ModelCheckpoint is a TensorFS recovery head. Observing it says nothing
// about remote custody; only the separately acknowledged head may restore a pod.
type ModelCheckpoint struct {
	Slot       string `json:"slot"`
	HeadID     string `json:"head_id"`
	HeadLength int64  `json:"head_length"`
	PlanDigest string `json:"plan_digest"`
	Index      int64  `json:"index"`
	Bytes      int64  `json:"bytes"`
}

type ModelCheckpointProgress struct {
	WorkerBootID string
	Observed     ModelCheckpoint
	Acknowledged *ModelCheckpoint
}

func scanModelCheckpointProgress(row interface{ Scan(...any) error }) (ModelCheckpointProgress, error) {
	var progress ModelCheckpointProgress
	var observed, acknowledged string
	if err := row.Scan(&progress.WorkerBootID, &observed, &acknowledged); err != nil {
		return progress, err
	}
	if err := json.Unmarshal([]byte(observed), &progress.Observed); err != nil {
		return progress, err
	}
	if acknowledged != "" {
		progress.Acknowledged = &ModelCheckpoint{}
		if err := json.Unmarshal([]byte(acknowledged), progress.Acknowledged); err != nil {
			return progress, err
		}
	}
	return progress, nil
}

func (s *Store) ModelSourceProgress(requestID string) ([]ModelCheckpointProgress, *exit.Error) {
	rows, err := s.db.Query(`SELECT worker_boot_id,observed,acknowledged
		FROM request_model_checkpoints WHERE request_id=? AND kind='source' ORDER BY slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read source checkpoint progress: %s", err)
	}
	defer rows.Close()
	var out []ModelCheckpointProgress
	for rows.Next() {
		progress, err := scanModelCheckpointProgress(rows)
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
	if intent.SourceSelection != selection || state == "failed" || state == "canceling" || state == "canceled" || state == "completed" {
		return intent, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_stale",
			"source checkpoint does not belong to the active selected operation")
	}
	return intent, nil
}

func validateSourceCheckpoint(intent ModelTransferIntent, checkpoint ModelCheckpoint) *exit.Error {
	_, headErr := canonical.Raw(checkpoint.HeadID)
	_, planErr := canonical.Raw(checkpoint.PlanDigest)
	if intent.SourceProfiles[checkpoint.Slot] == "" || headErr != nil || planErr != nil ||
		checkpoint.HeadLength <= 0 || checkpoint.Index < 0 || checkpoint.Bytes < 0 {
		return exit.Named(exit.Validation, "model_transfer.source_checkpoint_invalid",
			"source checkpoint must name a declared slot and exact bounded progress")
	}
	return nil
}

// ObserveModelCheckpoints freezes each slot's plan and advances only its
// current worker's local watermark. A replacement boot may start from the last
// acknowledged head, never from unacknowledged progress left by the old pod.
func (s *Store) ObserveModelSourceCheckpoints(requestID, selection, bootID string,
	checkpoints []ModelCheckpoint,
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
		prior, err := scanModelCheckpointProgress(tx.QueryRow(`SELECT worker_boot_id,observed,acknowledged
			FROM request_model_checkpoints WHERE request_id=? AND kind='source' AND slot=?`, requestID, checkpoint.Slot))
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return exit.Internalf("cannot read prior source checkpoint: %s", err)
		}
		if err == nil {
			checkpoint, problem = advanceCheckpoint(prior, bootID, checkpoint)
			if problem != nil {
				return problem
			}
		}

		data, _ := json.Marshal(checkpoint)
		if _, err := tx.Exec(`INSERT INTO request_model_checkpoints
			(request_id,slot,worker_boot_id,observed) VALUES(?,?,?,?)
			ON CONFLICT(request_id,kind,slot) DO UPDATE SET worker_boot_id=excluded.worker_boot_id,observed=excluded.observed`,
			requestID, checkpoint.Slot, bootID, string(data)); err != nil {
			return exit.Internalf("cannot record source checkpoint observation: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source checkpoint observation: %s", err)
	}
	return nil
}

// AcknowledgeModelCheckpoint follows verified custody of every link object.
// The caller supplies its previous acknowledged head so a stale uploader cannot
// overwrite a newer acknowledgment; a replaced worker also loses this authority.
func (s *Store) AcknowledgeModelSourceCheckpoint(requestID, selection, bootID, previousHead string,
	checkpoint ModelCheckpoint,
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
	progress, err := scanModelCheckpointProgress(tx.QueryRow(`SELECT worker_boot_id,observed,acknowledged
		FROM request_model_checkpoints WHERE request_id=? AND kind='source' AND slot=?`, requestID, checkpoint.Slot))
	if err != nil {
		return exit.Internalf("cannot read checkpoint acknowledgment target: %s", err)
	}
	var foreignBoot bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_model_transfer_files WHERE request_id=? AND worker_boot_id<>?)`, requestID, bootID).Scan(&foreignBoot); err != nil {
		return exit.Internalf("cannot fence source checkpoint worker: %s", err)
	}
	if foreignBoot {
		return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded", "source checkpoint uploader no longer owns the declared worker")
	}
	if progress.WorkerBootID != bootID {
		return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded", "checkpoint uploader no longer owns the observed worker")
	}
	if problem := checkCheckpointAcknowledgment(progress, previousHead, checkpoint); problem != nil {
		return problem
	}

	data, _ := json.Marshal(checkpoint)
	if _, err := tx.Exec(`UPDATE request_model_checkpoints SET acknowledged=? WHERE request_id=? AND kind='source' AND slot=?`,
		string(data), requestID, checkpoint.Slot); err != nil {
		return exit.Internalf("cannot record source checkpoint acknowledgment: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit source checkpoint acknowledgment: %s", err)
	}
	return nil
}

// NextCheckpointGrantRevision fences refreshed capabilities across daemon restarts.
// The counter covers one typed checkpoint slot; individual transfer identities remain exact.
func (s *Store) NextCheckpointGrantRevision(requestID, kind, slot string) (uint64, *exit.Error) {
	var revision int64
	if err := s.db.QueryRow(`UPDATE request_model_checkpoints
		SET grant_revision=grant_revision+1 WHERE request_id=? AND kind=? AND slot=? AND grant_revision<9223372036854775807
		RETURNING grant_revision`, requestID, kind, slot).Scan(&revision); err != nil {
		return 0, exit.Internalf("cannot allocate a source checkpoint grant revision: %s", err)
	}
	return uint64(revision), nil
}

// RecordCheckpointPublication precedes the external open so cancellation can release
// every hold even if the daemon dies before the checkpoint is acknowledged.
func (s *Store) RecordCheckpointPublication(requestID, operation string, objects []byte) *exit.Error {
	if _, err := s.db.Exec(`INSERT OR IGNORE INTO request_model_checkpoint_publications(request_id,operation,objects) VALUES(?,?,?)`,
		requestID, operation, objects); err != nil {
		return exit.Internalf("cannot record source publication intent: %s", err)
	}
	var stored []byte
	if err := s.db.QueryRow(`SELECT objects FROM request_model_checkpoint_publications WHERE request_id=? AND operation=?`, requestID, operation).Scan(&stored); err != nil {
		return exit.Internalf("cannot read source publication intent: %s", err)
	}
	if !bytes.Equal(stored, objects) {
		return exit.Named(exit.Conflict, "model_transfer.source_publication_changed", "source publication changed its fixed object set")
	}
	return nil
}

type CheckpointPublication struct {
	Operation string
	Objects   json.RawMessage
	Opened    bool
}

func (s *Store) OpenedCheckpointPublication(requestID, operation string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE request_model_checkpoint_publications SET opened=1 WHERE request_id=? AND operation=?`, requestID, operation); err != nil {
		return exit.Internalf("cannot confirm source publication open: %s", err)
	}
	return nil
}

func (s *Store) CheckpointPublications(requestID string) ([]CheckpointPublication, *exit.Error) {
	rows, err := s.db.Query(`SELECT operation,objects,opened FROM request_model_checkpoint_publications
		WHERE request_id=? AND released=0 ORDER BY operation LIMIT 128`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read source publication holds: %s", err)
	}
	defer rows.Close()
	var out []CheckpointPublication
	for rows.Next() {
		var operation CheckpointPublication
		if err := rows.Scan(&operation.Operation, &operation.Objects, &operation.Opened); err != nil {
			return nil, exit.Internalf("cannot read source publication operation: %s", err)
		}
		out = append(out, operation)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish source publication holds: %s", err)
	}
	return out, nil
}

func (s *Store) ReleaseCheckpointPublication(requestID, operation string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE request_model_checkpoint_publications SET released=1 WHERE request_id=? AND operation=?`,
		requestID, operation); err != nil {
		return exit.Internalf("cannot record source publication release: %s", err)
	}
	return nil
}

func (s *Store) CheckpointRequests() ([]string, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id FROM request_model_checkpoint_publications WHERE released=0
		UNION SELECT c.request_id FROM request_model_checkpoints c JOIN request_model_transfers t ON t.request_id=c.request_id
		WHERE c.acknowledged<>c.observed AND t.state IN ('materializing','materialized','finalizing')`)
	if err != nil {
		return nil, exit.Internalf("cannot read owed source checkpoint work: %s", err)
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var requestID string
		if err := rows.Scan(&requestID); err != nil {
			return nil, exit.Internalf("cannot read source checkpoint request: %s", err)
		}
		out = append(out, requestID)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish source checkpoint requests: %s", err)
	}
	return out, nil
}

func advanceCheckpoint(prior ModelCheckpointProgress, boot string, next ModelCheckpoint) (ModelCheckpoint, *exit.Error) {
	if prior.Observed.HeadID != "" {
		if prior.Observed.HeadID == next.HeadID && prior.Observed != next {
			return next, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_changed", "immutable checkpoint changed its metadata")
		}
		if prior.Observed.PlanDigest != next.PlanDigest {
			return next, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_plan_changed", "checkpoint changed the frozen native plan")
		}
	}
	if prior.Acknowledged != nil && (next.Index < prior.Acknowledged.Index || next.Index == prior.Acknowledged.Index && next != *prior.Acknowledged) {
		return next, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_required", "checkpoint precedes acknowledged custody")
	}
	if prior.WorkerBootID == boot && prior.Observed.HeadID != "" {
		if next.Index < prior.Observed.Index {
			return prior.Observed, nil
		}
		if next.Index == prior.Observed.Index && next != prior.Observed || next.Bytes < prior.Observed.Bytes {
			return next, exit.Named(exit.Conflict, "model_transfer.source_checkpoint_changed", "checkpoint changed its index or reduced cumulative bytes")
		}
	}
	return next, nil
}

func checkCheckpointAcknowledgment(progress ModelCheckpointProgress, previous string, checkpoint ModelCheckpoint) *exit.Error {
	if progress.Acknowledged != nil && *progress.Acknowledged == checkpoint {
		return nil
	}
	priorHead := ""
	if progress.Acknowledged != nil {
		priorHead = progress.Acknowledged.HeadID
		if checkpoint.Index <= progress.Acknowledged.Index || checkpoint.Bytes < progress.Acknowledged.Bytes {
			return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded", "checkpoint acknowledgment cannot move backwards")
		}
	}
	if checkpoint.HeadID == "" || previous != priorHead || checkpoint.Slot != progress.Observed.Slot || progress.Observed.PlanDigest != checkpoint.PlanDigest || checkpoint.Index > progress.Observed.Index || checkpoint.Index == progress.Observed.Index && checkpoint != progress.Observed {
		return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_superseded", "checkpoint acknowledgment no longer matches observed progress")
	}
	return nil
}
