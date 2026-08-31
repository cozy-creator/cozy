package records

import (
	"bytes"
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ModelProductionSourceFile is the credential-free source transfer journal. Provider URLs
// never enter this row; capability_revision only fences refreshed memory-only access.
type ModelProductionSourceFile struct {
	OperationID, SelectionDigest, Member, ObjectID, State, SafeCode, SafeDetail string
	Length, TransferredBytes                                                    int64
	CapabilityRevision                                                          uint64
}

type PreparedModelSource struct {
	OperationID     string `json:"operation_id"`
	Slot            string `json:"slot"`
	Profile         string `json:"profile"`
	ManifestID      string `json:"manifest_id"`
	ManifestLength  int64  `json:"manifest_length"`
	ReleaseEvidence []byte `json:"release_evidence"`
}

type ModelProductionStep struct {
	OperationID, StepName, RequestID, State string
	StepIndex                               int64
}

type ModelProductionArtifact struct {
	OperationID, StepName, OutputSlot, RequestID, InvocationDigest string
	TransactionID, ReceiptDigest, ManifestID, PublicationID, State string
	Attempt, WriterGeneration, ManifestLength                      int64
	Receipt, ReleaseEvidence                                       []byte
}

type ModelProductionObject struct {
	OperationID, StepName, OutputSlot, ObjectID, SourceRef  string
	TransferOperationID, State, SafeCode, SafeDetail        string
	Length, GrantRevision, UpdateSequence, TransferredBytes int64
}

func (s *Store) RecordModelProductionSourceFiles(operationID string,
	files []ModelProductionSourceFile,
) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model source file journal: %s", err)
	}
	defer tx.Rollback()
	for _, file := range files {
		var selection, objectID string
		var length int64
		err := tx.QueryRow(`SELECT selection_digest,object_id,length FROM model_production_source_files
			WHERE operation_id=? AND member=?`, operationID, file.Member).Scan(&selection, &objectID, &length)
		switch {
		case errors.Is(err, sql.ErrNoRows):
			if _, err := tx.Exec(`INSERT INTO model_production_source_files
				(operation_id,selection_digest,member,object_id,length) VALUES(?,?,?,?,?)`,
				operationID, file.SelectionDigest, file.Member, file.ObjectID, file.Length); err != nil {
				return exit.Internalf("cannot journal model source %s: %s", file.Member, err)
			}
		case err != nil:
			return exit.Internalf("cannot read model source %s: %s", file.Member, err)
		case selection != file.SelectionDigest || objectID != file.ObjectID || length != file.Length:
			return exit.Named(exit.Conflict, "model_production.source_file_conflict",
				"model source member %s already binds another ObjectRef", file.Member)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model source file journal: %s", err)
	}
	return nil
}

func (s *Store) RecordModelProductionSourceStatus(file ModelProductionSourceFile) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_production_source_files SET
		capability_revision=?,state=?,transferred_bytes=?,safe_code=?,safe_detail=?
		WHERE operation_id=? AND member=? AND object_id=? AND length=?
		AND capability_revision<=? AND transferred_bytes<=?`, file.CapabilityRevision,
		file.State, file.TransferredBytes, file.SafeCode, file.SafeDetail, file.OperationID,
		file.Member, file.ObjectID, file.Length, file.CapabilityRevision, file.TransferredBytes)
	if err != nil {
		return exit.Internalf("cannot record model source status: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_production.source_status_conflict",
			"model source status changed identity or moved backwards")
	}
	return nil
}

func (s *Store) ModelProductionSourceFiles(operationID string) ([]ModelProductionSourceFile, *exit.Error) {
	rows, err := s.db.Query(`SELECT operation_id,selection_digest,member,object_id,length,capability_revision,
		state,transferred_bytes,safe_code,safe_detail FROM model_production_source_files
		WHERE operation_id=? ORDER BY member`, operationID)
	if err != nil {
		return nil, exit.Internalf("cannot read model source files: %s", err)
	}
	defer rows.Close()
	var out []ModelProductionSourceFile
	for rows.Next() {
		var row ModelProductionSourceFile
		if err := rows.Scan(&row.OperationID, &row.SelectionDigest, &row.Member, &row.ObjectID, &row.Length,
			&row.CapabilityRevision, &row.State, &row.TransferredBytes, &row.SafeCode,
			&row.SafeDetail); err != nil {
			return nil, exit.Internalf("cannot scan model source file: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) RecordModelProductionProfiles(operationID string,
	profiles []PreparedModelSource,
) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model source profile journal: %s", err)
	}
	defer tx.Rollback()
	for _, profile := range profiles {
		if _, err := tx.Exec(`INSERT INTO model_production_sources(operation_id,slot,profile)
			VALUES(?,?,?) ON CONFLICT(operation_id,slot) DO NOTHING`, operationID,
			profile.Slot, profile.Profile); err != nil {
			return exit.Internalf("cannot journal model source profile %s: %s", profile.Slot, err)
		}
		var held string
		if err := tx.QueryRow(`SELECT profile FROM model_production_sources
			WHERE operation_id=? AND slot=?`, operationID, profile.Slot).Scan(&held); err != nil ||
			held != profile.Profile {
			return exit.Named(exit.Conflict, "model_production.source_profile_conflict",
				"model source slot %s already binds another profile", profile.Slot)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model source profiles: %s", err)
	}
	return nil
}

func (s *Store) RecordPreparedModelSources(operationID string,
	sources []PreparedModelSource,
) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin prepared model source journal: %s", err)
	}
	defer tx.Rollback()
	for _, source := range sources {
		var profile, manifest string
		var length int64
		var evidence []byte
		err := tx.QueryRow(`SELECT profile,manifest_id,manifest_length,release_evidence
			FROM model_production_sources WHERE operation_id=? AND slot=?`, operationID,
			source.Slot).Scan(&profile, &manifest, &length, &evidence)
		switch {
		case errors.Is(err, sql.ErrNoRows):
			if _, err := tx.Exec(`INSERT INTO model_production_sources
				(operation_id,slot,profile,manifest_id,manifest_length,release_evidence)
				VALUES(?,?,?,?,?,?)`, operationID, source.Slot, source.Profile,
				source.ManifestID, source.ManifestLength, source.ReleaseEvidence); err != nil {
				return exit.Internalf("cannot journal prepared model source %s: %s", source.Slot, err)
			}
		case err != nil:
			return exit.Internalf("cannot read prepared model source %s: %s", source.Slot, err)
		case profile != source.Profile:
			return exit.Named(exit.Conflict, "model_production.prepared_source_conflict",
				"prepared model source %s changed its requested profile", source.Slot)
		case manifest == "":
			if _, err := tx.Exec(`UPDATE model_production_sources SET manifest_id=?,
				manifest_length=?,release_evidence=? WHERE operation_id=? AND slot=? AND manifest_id=''`,
				source.ManifestID, source.ManifestLength, source.ReleaseEvidence, operationID,
				source.Slot); err != nil {
				return exit.Internalf("cannot complete prepared model source %s: %s", source.Slot, err)
			}
		case manifest != source.ManifestID || length != source.ManifestLength ||
			!bytes.Equal(evidence, source.ReleaseEvidence):
			return exit.Named(exit.Conflict, "model_production.prepared_source_conflict",
				"prepared model source %s replay changed its profile, Manifest, or evidence", source.Slot)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit prepared model sources: %s", err)
	}
	return nil
}

func (s *Store) PreparedModelSources(operationID string) ([]PreparedModelSource, *exit.Error) {
	rows, err := s.db.Query(`SELECT operation_id,slot,profile,manifest_id,manifest_length,
		release_evidence FROM model_production_sources WHERE operation_id=? ORDER BY slot`, operationID)
	if err != nil {
		return nil, exit.Internalf("cannot read prepared model sources: %s", err)
	}
	defer rows.Close()
	var out []PreparedModelSource
	for rows.Next() {
		var row PreparedModelSource
		if err := rows.Scan(&row.OperationID, &row.Slot, &row.Profile, &row.ManifestID,
			&row.ManifestLength, &row.ReleaseEvidence); err != nil {
			return nil, exit.Internalf("cannot scan prepared model source: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) BeginModelProductionStep(row ModelProductionStep) (ModelProductionStep, *exit.Error) {
	_, err := s.db.Exec(`INSERT INTO model_production_steps
		(operation_id,step_index,step_name,request_id,state) VALUES(?,?,?,?,?)
		ON CONFLICT(operation_id,step_index) DO NOTHING`, row.OperationID, row.StepIndex,
		row.StepName, row.RequestID, row.State)
	if err != nil {
		return row, exit.Internalf("cannot begin model production step %s: %s", row.StepName, err)
	}
	var held ModelProductionStep
	err = s.db.QueryRow(`SELECT operation_id,step_index,step_name,request_id,state
		FROM model_production_steps WHERE operation_id=? AND step_index=?`, row.OperationID,
		row.StepIndex).Scan(&held.OperationID, &held.StepIndex, &held.StepName,
		&held.RequestID, &held.State)
	if err != nil {
		return row, exit.Internalf("cannot read model production step %s: %s", row.StepName, err)
	}
	if held.StepName != row.StepName || row.RequestID != "" && held.RequestID != row.RequestID {
		return held, exit.Named(exit.Conflict, "model_production.step_conflict",
			"model production step %d already binds different work", row.StepIndex)
	}
	return held, nil
}

func (s *Store) ModelProductionStepByRequest(requestID string) (*ModelProductionStep, *exit.Error) {
	var row ModelProductionStep
	err := s.db.QueryRow(`SELECT operation_id,step_index,step_name,request_id,state
		FROM model_production_steps WHERE request_id=?`, requestID).Scan(&row.OperationID,
		&row.StepIndex, &row.StepName, &row.RequestID, &row.State)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read model production step for request %s: %s", requestID, err)
	}
	return &row, nil
}

func (s *Store) SetModelProductionStepRequest(operationID string, stepIndex int64,
	stepName, requestID, state string,
) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_production_steps SET request_id=?,state=?
		WHERE operation_id=? AND step_index=? AND step_name=?
		AND (request_id='' OR request_id=?)`, requestID, state, operationID, stepIndex,
		stepName, requestID)
	if err != nil {
		return exit.Internalf("cannot bind model production step request: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_production.step_request_conflict",
			"model production step %s already binds different work", stepName)
	}
	return nil
}

func (s *Store) RecordModelProductionArtifact(artifact ModelProductionArtifact,
	objects []ModelProductionObject,
) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin model artifact journal: %s", err)
	}
	defer tx.Rollback()
	var held ModelProductionArtifact
	err = tx.QueryRow(`SELECT operation_id,step_name,output_slot,request_id,attempt,
		invocation_digest,transaction_id,writer_generation,receipt_digest,receipt,
		manifest_id,manifest_length,release_evidence,publication_id,state
		FROM model_production_artifacts WHERE operation_id=? AND step_name=? AND output_slot=?`,
		artifact.OperationID, artifact.StepName, artifact.OutputSlot).Scan(&held.OperationID,
		&held.StepName, &held.OutputSlot, &held.RequestID, &held.Attempt,
		&held.InvocationDigest, &held.TransactionID, &held.WriterGeneration,
		&held.ReceiptDigest, &held.Receipt, &held.ManifestID, &held.ManifestLength,
		&held.ReleaseEvidence, &held.PublicationID, &held.State)
	if err == nil {
		if held.RequestID != artifact.RequestID || held.Attempt != artifact.Attempt ||
			held.InvocationDigest != artifact.InvocationDigest || held.TransactionID != artifact.TransactionID ||
			held.WriterGeneration != artifact.WriterGeneration || held.ReceiptDigest != artifact.ReceiptDigest ||
			!bytes.Equal(held.Receipt, artifact.Receipt) || held.ManifestID != artifact.ManifestID ||
			held.ManifestLength != artifact.ManifestLength ||
			!bytes.Equal(held.ReleaseEvidence, artifact.ReleaseEvidence) {
			return exit.Named(exit.Conflict, "model_production.artifact_conflict",
				"artifact receipt replay changed exact output %s.%s", artifact.StepName, artifact.OutputSlot)
		}
		rows, queryErr := tx.Query(`SELECT object_id,length,source_ref FROM model_production_objects
			WHERE operation_id=? AND step_name=? AND output_slot=? ORDER BY object_id`,
			artifact.OperationID, artifact.StepName, artifact.OutputSlot)
		if queryErr != nil {
			return exit.Internalf("cannot read replayed model production objects: %s", queryErr)
		}
		var replayed []ModelProductionObject
		for rows.Next() {
			var row ModelProductionObject
			if scanErr := rows.Scan(&row.ObjectID, &row.Length, &row.SourceRef); scanErr != nil {
				rows.Close()
				return exit.Internalf("cannot scan replayed model production object: %s", scanErr)
			}
			replayed = append(replayed, row)
		}
		rows.Close()
		if len(replayed) != len(objects) {
			return exit.Named(exit.Conflict, "model_production.artifact_conflict",
				"artifact receipt replay changed the object inventory of %s.%s",
				artifact.StepName, artifact.OutputSlot)
		}
		for i := range replayed {
			if replayed[i].ObjectID != objects[i].ObjectID || replayed[i].Length != objects[i].Length ||
				replayed[i].SourceRef != objects[i].SourceRef {
				return exit.Named(exit.Conflict, "model_production.artifact_conflict",
					"artifact receipt replay changed object %d of %s.%s", i,
					artifact.StepName, artifact.OutputSlot)
			}
		}
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot commit exact model production replay: %s", err)
		}
		return nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return exit.Internalf("cannot read model production artifact: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO model_production_artifacts
		(operation_id,step_name,output_slot,request_id,attempt,invocation_digest,
		transaction_id,writer_generation,receipt_digest,receipt,manifest_id,manifest_length,
		release_evidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)`, artifact.OperationID,
		artifact.StepName, artifact.OutputSlot, artifact.RequestID, artifact.Attempt,
		artifact.InvocationDigest, artifact.TransactionID, artifact.WriterGeneration,
		artifact.ReceiptDigest, artifact.Receipt, artifact.ManifestID, artifact.ManifestLength,
		artifact.ReleaseEvidence); err != nil {
		return exit.Internalf("cannot insert model production artifact: %s", err)
	}
	for _, object := range objects {
		if _, err := tx.Exec(`INSERT INTO model_production_objects
			(operation_id,step_name,output_slot,object_id,length,source_ref)
			VALUES(?,?,?,?,?,?)`, artifact.OperationID, artifact.StepName,
			artifact.OutputSlot, object.ObjectID, object.Length, object.SourceRef); err != nil {
			return exit.Internalf("cannot journal model production object %s: %s", object.ObjectID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit model production artifact: %s", err)
	}
	return nil
}

func (s *Store) ModelProductionArtifacts(operationID string) ([]ModelProductionArtifact, *exit.Error) {
	rows, err := s.db.Query(`SELECT operation_id,step_name,output_slot,request_id,attempt,
		invocation_digest,transaction_id,writer_generation,receipt_digest,receipt,
		manifest_id,manifest_length,release_evidence,publication_id,state
		FROM model_production_artifacts WHERE operation_id=? ORDER BY step_name,output_slot`, operationID)
	if err != nil {
		return nil, exit.Internalf("cannot read model production artifacts: %s", err)
	}
	defer rows.Close()
	var out []ModelProductionArtifact
	for rows.Next() {
		var row ModelProductionArtifact
		if err := rows.Scan(&row.OperationID, &row.StepName, &row.OutputSlot, &row.RequestID,
			&row.Attempt, &row.InvocationDigest, &row.TransactionID, &row.WriterGeneration,
			&row.ReceiptDigest, &row.Receipt, &row.ManifestID, &row.ManifestLength,
			&row.ReleaseEvidence, &row.PublicationID, &row.State); err != nil {
			return nil, exit.Internalf("cannot scan model production artifact: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) ModelProductionObjects(operationID, stepName,
	outputSlot string,
) ([]ModelProductionObject, *exit.Error) {
	rows, err := s.db.Query(`SELECT operation_id,step_name,output_slot,object_id,length,
		source_ref,transfer_operation_id,grant_revision,update_sequence,state,
		transferred_bytes,safe_code,safe_detail FROM model_production_objects
		WHERE operation_id=? AND step_name=? AND output_slot=? ORDER BY object_id`,
		operationID, stepName, outputSlot)
	if err != nil {
		return nil, exit.Internalf("cannot read model production objects: %s", err)
	}
	defer rows.Close()
	var out []ModelProductionObject
	for rows.Next() {
		var row ModelProductionObject
		if err := rows.Scan(&row.OperationID, &row.StepName, &row.OutputSlot, &row.ObjectID,
			&row.Length, &row.SourceRef, &row.TransferOperationID, &row.GrantRevision,
			&row.UpdateSequence, &row.State, &row.TransferredBytes, &row.SafeCode,
			&row.SafeDetail); err != nil {
			return nil, exit.Internalf("cannot scan model production object: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func (s *Store) RecordModelProductionObjectStatus(row ModelProductionObject) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_production_objects SET transfer_operation_id=?,
		grant_revision=?,update_sequence=?,state=?,transferred_bytes=?,safe_code=?,safe_detail=?
		WHERE operation_id=? AND step_name=? AND output_slot=? AND object_id=? AND length=?
		AND (transfer_operation_id='' OR transfer_operation_id=?)
		AND grant_revision<=? AND (grant_revision<? OR update_sequence<=?)`,
		row.TransferOperationID, row.GrantRevision, row.UpdateSequence, row.State,
		row.TransferredBytes, row.SafeCode, row.SafeDetail, row.OperationID, row.StepName,
		row.OutputSlot, row.ObjectID, row.Length, row.TransferOperationID,
		row.GrantRevision, row.GrantRevision, row.UpdateSequence)
	if err != nil {
		return exit.Internalf("cannot record model artifact transfer status: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_production.artifact_status_conflict",
			"model artifact transfer status changed identity or moved backwards")
	}
	return nil
}

func (s *Store) MarkModelProductionArtifactPublished(operationID, stepName, outputSlot,
	publicationID string,
) *exit.Error {
	result, err := s.db.Exec(`UPDATE model_production_artifacts SET publication_id=?,state='prepared'
		WHERE operation_id=? AND step_name=? AND output_slot=?
		AND (publication_id='' OR publication_id=?)`, publicationID, operationID, stepName,
		outputSlot, publicationID)
	if err != nil {
		return exit.Internalf("cannot record prepared model publication: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "model_production.publication_conflict",
			"model production output %s.%s already names another publication", stepName, outputSlot)
	}
	return nil
}
