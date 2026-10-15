package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/custody"
	"github.com/cozy-creator/cozy/internal/exit"
)

var modelTransferSlug = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`)

const maxModelTransferSourceFiles = 20_000
const maxExactJSONInteger = (int64(1) << 53) - 1

var modelTransferSchema = []string{`
CREATE TABLE IF NOT EXISTS request_model_transfers (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  intent           TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'pending',
  models           TEXT NOT NULL DEFAULT '[]',
  checkpoints      TEXT NOT NULL DEFAULT '{}',
  error_code       TEXT NOT NULL DEFAULT '',
  safe_error       TEXT NOT NULL DEFAULT '',
  updated_at       TEXT NOT NULL,
  models_worker_boot_id TEXT NOT NULL DEFAULT '',
  CHECK (state IN ('pending','materializing','materialized','finalizing','completed','failed','canceling','canceled'))
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_files (
  request_id          TEXT NOT NULL REFERENCES requests(id),
  member              TEXT NOT NULL,
  object_id           TEXT NOT NULL,
  length              INTEGER NOT NULL CHECK(length>0),
  capability_revision INTEGER NOT NULL DEFAULT 0,
  state               TEXT NOT NULL DEFAULT 'pending',
  transferred         INTEGER NOT NULL DEFAULT 0,
  safe_code           TEXT NOT NULL DEFAULT '',
  safe_detail         TEXT NOT NULL DEFAULT '',
  worker_boot_id      TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,member)
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_outputs (
  request_id       TEXT NOT NULL REFERENCES requests(id),
  output_slot      TEXT NOT NULL,
  manifest_id      TEXT NOT NULL,
  manifest_length  INTEGER NOT NULL CHECK(manifest_length>0),
  attempt           INTEGER NOT NULL,
  invocation_digest TEXT NOT NULL,
  transaction_id    TEXT NOT NULL,
  receipt_digest    TEXT NOT NULL,
  receipt           BLOB NOT NULL,
  final_id          TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot)
)`, `
CREATE TABLE IF NOT EXISTS request_model_transfer_objects (
  request_id       TEXT NOT NULL,
  attempt          INTEGER NOT NULL,
  output_slot      TEXT NOT NULL,
  object_id        TEXT NOT NULL,
  length           INTEGER NOT NULL CHECK(length>0),
  source_ref       TEXT NOT NULL,
  operation_id     TEXT NOT NULL DEFAULT '',
  grant_revision   INTEGER NOT NULL DEFAULT 0,
  update_sequence  INTEGER NOT NULL DEFAULT 0,
  state            TEXT NOT NULL DEFAULT 'pending',
  transferred      INTEGER NOT NULL DEFAULT 0,
  safe_code        TEXT NOT NULL DEFAULT '',
  safe_detail      TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(request_id,attempt,output_slot,object_id),
  FOREIGN KEY(request_id,attempt,output_slot)
    REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
)`, `
CREATE TABLE IF NOT EXISTS request_planned_sources (
  request_id   TEXT PRIMARY KEY REFERENCES requests(id),
  source_bytes INTEGER NOT NULL CHECK(source_bytes>0)
)`}

type ModelTransferSourceFile struct {
	Member string `json:"member"`
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
	Header []byte `json:"header,omitempty"`
}

type ModelTransferOutput struct {
	Name string `json:"name"`
}

// ModelTransferIntent is immutable request meaning. Credentials, URLs, workers,
// attempts, prices, progress, and retries deliberately have no field.
type ModelTransferIntent struct {
	Kind            string                    `json:"kind"`
	Destination     string                    `json:"destination"`
	Source          string                    `json:"source,omitempty"`
	SourceSelection string                    `json:"source_selection,omitempty"`
	SourceLicense   string                    `json:"source_license,omitempty"`
	SourceFiles     []ModelTransferSourceFile `json:"source_files,omitempty"`
	InputLane       string                    `json:"input_lane,omitempty"`
	SourceProfiles  map[string]string         `json:"source_profiles,omitempty"`
	Outputs         []ModelTransferOutput     `json:"outputs"`
	LocalOnly       bool                      `json:"local_only,omitempty"`
}

// HasAcquisition distinguishes source ingestion from output publication for a
// job whose inputs are already selected. Partial acquisition fields still count,
// so validation cannot mistake an incomplete source declaration for their absence.
func (intent *ModelTransferIntent) HasAcquisition() bool {
	return intent != nil && (intent.Source != "" || intent.SourceSelection != "" ||
		intent.SourceLicense != "" || len(intent.SourceFiles) > 0 || intent.InputLane != "" ||
		len(intent.SourceProfiles) > 0 || intent.LocalOnly)
}

type ModelTransfer struct {
	RequestID string
	ModelTransferIntent
	State              string
	Models             []ModelRef
	Checkpoints        map[string]string
	ErrorCode          string
	SafeError          string
	UpdatedAt          string
	ModelsWorkerBootID string
}

func NormalizeModelTransferIntent(intent *ModelTransferIntent) *exit.Error {
	if intent == nil {
		return nil
	}
	if (intent.Kind != "model-upload" && intent.Kind != "model-download") || len(intent.Outputs) == 0 {
		return exit.New(exit.Validation, "model transfer intent is incomplete")
	}
	if intent.HasAcquisition() {
		if intent.Source == "" || intent.SourceSelection == "" {
			return exit.New(exit.Validation, "model source acquisition requires both source and source_selection")
		}
	} else if intent.Kind != "model-upload" {
		return exit.New(exit.Validation, "output-only publication requires a model-upload destination")
	}
	localSource := strings.HasPrefix(intent.Source, "file:") || strings.HasPrefix(intent.Source, "local/")
	if intent.LocalOnly != localSource {
		return exit.New(exit.Validation, "model transfer local_only does not match its source")
	}
	parts := strings.Split(intent.Destination, "/")
	if intent.Destination == "" {
		if intent.Kind != "model-upload" || !intent.HasAcquisition() {
			return exit.New(exit.Validation, "an unpublished model result requires a source acquisition")
		}
	} else if intent.Kind == "model-upload" {
		if len(parts) != 2 || parts[0] == "local" || !modelTransferSlug.MatchString(parts[0]) ||
			!modelTransferSlug.MatchString(parts[1]) {
			return exit.New(exit.Validation, "model upload destination is not one org/model")
		}
	} else if len(parts) != 2 || parts[0] != "local" || !modelTransferSlug.MatchString(parts[1]) {
		return exit.New(exit.Validation, "model download destination is not local/name")
	}
	intent.SourceFiles = append([]ModelTransferSourceFile(nil), intent.SourceFiles...)
	if len(intent.SourceFiles) > maxModelTransferSourceFiles {
		return exit.New(exit.Validation, "model transfer source inventory exceeds %d files",
			maxModelTransferSourceFiles)
	}
	sort.Slice(intent.SourceFiles, func(i, j int) bool {
		return intent.SourceFiles[i].Member < intent.SourceFiles[j].Member
	})
	previousFile := ""
	var sourceBytes int64
	for _, file := range intent.SourceFiles {
		_, digestErr := canonical.Raw("sha256:" + file.SHA256)
		if file.Member == "" || file.Member <= previousFile || file.Length <= 0 ||
			file.Length > maxExactJSONInteger-sourceBytes ||
			len(file.Header) > custody.MaxSourceHeaderBytes || int64(len(file.Header)) > file.Length ||
			digestErr != nil {
			return exit.New(exit.Validation, "model transfer source inventory is invalid")
		}
		sourceBytes += file.Length
		previousFile = file.Member
	}
	if intent.HasAcquisition() {
		if _, err := canonical.Raw(intent.SourceSelection); err != nil {
			return exit.New(exit.Validation, "model transfer source selection is not an exact digest")
		}
	}
	intent.Outputs = append([]ModelTransferOutput(nil), intent.Outputs...)
	sort.Slice(intent.Outputs, func(i, j int) bool { return intent.Outputs[i].Name < intent.Outputs[j].Name })
	seen := map[string]bool{}
	for i := range intent.Outputs {
		output := &intent.Outputs[i]
		if output.Name == "" || seen[output.Name] {
			return exit.New(exit.Validation, "model transfer output names are non-empty and unique")
		}
		seen[output.Name] = true
	}
	return nil
}

func recordModelTransferTx(tx *sql.Tx, requestID string, intent *ModelTransferIntent) *exit.Error {
	if intent == nil {
		return nil
	}
	data, err := json.Marshal(intent)
	if err != nil {
		return exit.Internalf("cannot encode model transfer intent: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO request_model_transfers(request_id,intent,updated_at)
		VALUES(?,?,?)`, requestID, string(data), now()); err != nil {
		return exit.Internalf("cannot record model transfer for %s: %s", requestID, err)
	}
	for _, file := range intent.SourceFiles {
		if _, err := tx.Exec(`INSERT INTO request_model_transfer_files
			(request_id,member,object_id,length) VALUES(?,?,?,?)`, requestID, file.Member,
			"sha256:"+file.SHA256, file.Length); err != nil {
			return exit.Internalf("cannot record model transfer source %s: %s", file.Member, err)
		}
	}
	return nil
}

// PlannedSourceBytes is the summed length of every source object this request
// will pull, or 0 when the request moves no model bytes (th-152). A request that
// ingests through its own script declares the figure at submission instead.
//
// It is read from request_model_transfer_files, which is written in the same
// transaction as the request itself and is the table the length CHECK lives
// on — never re-summed from the intent JSON, which would let two spellings of
// the same plan disagree.
//
// This is what makes a plan-sized pod possible. An ingest holds its source
// objects AND the canonical CAS output built from them in one TensorFS Store
// on the pod's container disk, so a pod bought at the serving default would
// block on a full filesystem partway through. The hub sizes the disk from this
// figure; it is stated before the rental is bought because the selection is
// resolved before any pod exists.
func (s *Store) PlannedSourceBytes(requestID string) (int64, *exit.Error) {
	var total sql.NullInt64
	if err := s.db.QueryRow(`SELECT COALESCE((SELECT SUM(length) FROM request_model_transfer_files WHERE request_id=?),0)
		+ COALESCE((SELECT source_bytes FROM request_planned_sources WHERE request_id=?),0)`, requestID, requestID).Scan(&total); err != nil {
		return 0, exit.Internalf("cannot total model transfer source bytes for %s: %s",
			requestID, err)
	}
	if !total.Valid || total.Int64 < 0 {
		return 0, nil
	}
	return total.Int64, nil
}

func recordPlannedSourcesTx(tx *sql.Tx, requestID string, sourceBytes int64) *exit.Error {
	if sourceBytes <= 0 {
		return nil
	}
	if _, err := tx.Exec(`INSERT INTO request_planned_sources(request_id,source_bytes) VALUES(?,?)`, requestID, sourceBytes); err != nil {
		return exit.Internalf("cannot record the planned source bytes of %s: %s", requestID, err)
	}
	return nil
}

func (s *Store) ModelTransferOf(requestID string) (*ModelTransfer, *exit.Error) {
	var row ModelTransfer
	var intent, models, checkpoints string
	err := s.db.QueryRow(`SELECT request_id,intent,state,models,checkpoints,error_code,safe_error,updated_at,models_worker_boot_id
		FROM request_model_transfers WHERE request_id=?`, requestID).Scan(&row.RequestID,
		&intent, &row.State, &models, &checkpoints, &row.ErrorCode, &row.SafeError, &row.UpdatedAt, &row.ModelsWorkerBootID)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil || json.Unmarshal([]byte(intent), &row.ModelTransferIntent) != nil ||
		json.Unmarshal([]byte(models), &row.Models) != nil ||
		json.Unmarshal([]byte(checkpoints), &row.Checkpoints) != nil {
		return nil, exit.Internalf("cannot decode model transfer %s", requestID)
	}
	return &row, nil
}

// DeclaredServingModels is the set of Hub-resolved models this request will make
// its pod hold, or empty when it names none (th-155/cl-130).
//
// It is the SERVING half of what PlannedSourceBytes does for an ingest, and it
// exists for the same reason: the container disk is bought once and cannot grow
// (any PATCH of a running RunPod pod destroys it), so the pod must be sized for
// its whole workload up front. These rows are committed with the request, before
// any pod is asked for, and the SAME rows become the pod's desired download set
// — so declaring them costs no extra round trip and cannot disagree with what
// the pod later fetches.
//
// Operation-local refs are dropped through the same Downloadable predicate as
// the worker download set. Retained Hub checkpoints need no release to be sized.
func (s *Store) DeclaredServingModels(requestID string) ([]ModelRef, *exit.Error) {
	request, problem := s.RequestRow(requestID)
	if problem != nil || request == nil {
		return nil, problem
	}
	out := make([]ModelRef, 0, len(request.Models))
	seen := map[string]bool{}
	add := func(model ModelRef) {
		key := model.Model + "\x00" + model.Release + "\x00" + model.Lane + "\x00" + model.Manifest
		if model.Downloadable() && !seen[key] {
			seen[key] = true
			out = append(out, model)
		}
	}
	for _, model := range request.Models {
		add(model)
		// Zero-strength adapters are still validated/leased by Runtime and their
		// immutable bytes are part of the declared disk closure.
		for _, a := range model.Adapters {
			add(ModelRef{Model: a.Model, Release: a.Release, Lane: a.Lane, Manifest: a.Manifest, HubCheckpoint: true})
		}
	}
	return out, nil
}

// Downloadable identifies a Hub-resolved checkpoint that can be measured and
// fetched. Operation-local manifests remain private to their prepared worker.
func (model ModelRef) Downloadable() bool {
	return model.Model != "" && model.Manifest != "" && (model.Release != "" || model.HubCheckpoint)
}
