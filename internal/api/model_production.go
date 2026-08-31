package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type ModelProductionController interface {
	SubmitModelProduction(context.Context, modelproduction.Instruction) (
		records.ModelProductionOperation, bool, *exit.Error)
	CancelModelProduction(string) *exit.Error
}

type ModelProductionState struct {
	ID              string   `json:"id"`
	Kind            string   `json:"kind"`
	Model           string   `json:"model"`
	Release         string   `json:"release"`
	Source          string   `json:"source"`
	Producer        string   `json:"producer,omitempty"`
	Status          string   `json:"status"`
	NodeIndex       int64    `json:"node_index"`
	Nodes           int      `json:"nodes"`
	Lanes           []string `json:"lanes,omitempty"`
	ManifestIDs     []string `json:"manifest_ids,omitempty"`
	Rental          string   `json:"rental,omitempty"`
	CancelRequested bool     `json:"cancel_requested"`
	Committed       bool     `json:"committed"`
	Cleanup         string   `json:"cleanup"`
	ErrorCode       string   `json:"error_code,omitempty"`
	Error           string   `json:"error,omitempty"`
	CreatedAt       string   `json:"created_at"`
	UpdatedAt       string   `json:"updated_at"`
	Changed         bool     `json:"changed"`
}

func (s *Server) submitModelProduction(w http.ResponseWriter, r *http.Request) {
	if s.modelProductions == nil {
		s.refuse(w, r, http.StatusServiceUnavailable, "model_production.owner_absent",
			"this daemon has no model-production owner", "restart the Cozy daemon")
		return
	}
	var body struct {
		Instruction modelproduction.Instruction `json:"instruction"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, MaxBody+1))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"model production submission is not one closed instruction: "+err.Error(), "")
		return
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"model production submission carries trailing JSON", "")
		return
	}
	operation, changed, problem := s.modelProductions.SubmitModelProduction(r.Context(), body.Instruction)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	state, problem := s.modelProductionState(operation, changed)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, state)
}

func (s *Server) getModelProduction(w http.ResponseWriter, r *http.Request) {
	operation, problem := s.store.ModelProduction(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if operation == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no model production "+r.PathValue("id")+" on this host", "")
		return
	}
	state, problem := s.modelProductionState(*operation, false)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, state)
}

func (s *Server) listModelProductions(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if raw := r.URL.Query().Get("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 500 {
			s.refuse(w, r, http.StatusBadRequest, "invalid_limit",
				"model production limit must be between 1 and 500", "")
			return
		}
		limit = parsed
	}
	rows, problem := s.store.ModelProductions(strings.TrimSpace(r.URL.Query().Get("status")), limit)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	out := make([]ModelProductionState, 0, len(rows))
	for _, operation := range rows {
		state, stateProblem := s.modelProductionState(operation, false)
		if stateProblem != nil {
			s.refuseTyped(w, r, stateProblem)
			return
		}
		out = append(out, state)
	}
	s.ok(w, r, http.StatusOK, map[string]any{"model_productions": out, "count": len(out)})
}

func (s *Server) cancelModelProduction(w http.ResponseWriter, r *http.Request) {
	if s.modelProductions == nil {
		s.refuse(w, r, http.StatusServiceUnavailable, "model_production.owner_absent",
			"this daemon has no model-production owner", "restart the Cozy daemon")
		return
	}
	if problem := s.modelProductions.CancelModelProduction(r.PathValue("id")); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	operation, problem := s.store.ModelProduction(r.PathValue("id"))
	if problem != nil || operation == nil {
		if problem != nil {
			s.refuseTyped(w, r, problem)
		} else {
			s.refuse(w, r, http.StatusNotFound, "not_found",
				"no model production "+r.PathValue("id")+" on this host", "")
		}
		return
	}
	state, problem := s.modelProductionState(*operation, true)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, state)
}

func (s *Server) modelProductionState(operation records.ModelProductionOperation,
	changed bool,
) (ModelProductionState, *exit.Error) {
	state := ModelProductionState{ID: operation.ID, Kind: "model-publication",
		Status: operation.State, NodeIndex: operation.NodeIndex, Rental: operation.RentalID,
		CancelRequested: operation.CancelRequested, ErrorCode: operation.SafeCode,
		Error: operation.SafeDetail, CreatedAt: operation.CreatedAt, UpdatedAt: operation.UpdatedAt,
		Changed: changed}
	if operation.State == "resolving" {
		instruction, err := modelproduction.ParseInstruction(operation.Plan)
		if err != nil {
			return state, exit.Internalf("cannot read model production %s instruction: %s", operation.ID, err)
		}
		state.Model, state.Release, state.Source = instruction.Destination, instruction.Release, instruction.Source
		state.Producer = instruction.Producer
	} else {
		plan, err := modelproduction.Parse(operation.Plan)
		if err != nil {
			return state, exit.Internalf("cannot read model production %s plan: %s", operation.ID, err)
		}
		state.Model, state.Release, state.Source = plan.Destination, plan.Release, plan.Source
		state.Producer, state.Lanes, state.Nodes = plan.Producer, plan.Lanes(), len(plan.Jobs)
	}
	switch operation.State {
	case "release_cut", "cleanup_pending", "completed":
		state.Committed = true
	}
	switch {
	case operation.State == "cleanup_pending" || operation.State == "release_cut":
		state.Cleanup = "pending"
	case operation.State == "completed" && operation.RentalID != "":
		state.Cleanup = "provider_absent"
	case operation.State == "failed" || operation.State == "canceled":
		state.Cleanup = "settled"
	default:
		state.Cleanup = "not_started"
	}
	artifacts, problem := s.store.ModelProductionArtifacts(operation.ID)
	if problem != nil {
		return state, problem
	}
	seen := map[string]bool{}
	for _, artifact := range artifacts {
		if artifact.ManifestID != "" && !seen[artifact.ManifestID] {
			seen[artifact.ManifestID] = true
			state.ManifestIDs = append(state.ManifestIDs, artifact.ManifestID)
		}
	}
	return state, nil
}

// ModelProductionAction is the one private daemon seam used by `cozy model
// publish --rental`. The descriptor-derived graph never crosses this route: the
// CLI asks the connected record owner to perform one typed host exchange at a time.
type ModelProductionAction struct {
	Action   string                         `json:"action"`
	RentalID string                         `json:"rental_id"`
	Source   *ModelProductionSourceAction   `json:"source,omitempty"`
	Artifact *ModelProductionArtifactAction `json:"artifact,omitempty"`
}

type ModelProductionSourceAction struct {
	SelectionDigest string                              `json:"selection_digest"`
	SourceURI       string                              `json:"source_uri"`
	DeclaredLicense string                              `json:"declared_license"`
	Files           []orchestrator.ProductionSourceFile `json:"files"`
	Profiles        map[string]string                   `json:"profiles"`
	Capabilities    []ModelProductionSourceCapability   `json:"capabilities"`
}

type ModelProductionSourceCapability struct {
	Member        string `json:"member"`
	ObjectID      string `json:"object_id"`
	Length        int64  `json:"length"`
	Provider      string `json:"provider"`
	URL           string `json:"url"`
	ExpiresAtUnix uint64 `json:"expires_at_unix"`
}

type ModelProductionArtifactAction struct {
	NodeName            string                                  `json:"node_name"`
	OutputSlot          string                                  `json:"output_slot"`
	TransferOperationID string                                  `json:"transfer_operation_id"`
	Decisions           []orchestrator.ArtifactTransferDecision `json:"decisions"`
}

type ModelProductionActionResult struct {
	Prepared []records.PreparedModelSource `json:"prepared,omitempty"`
	Changed  bool                          `json:"changed"`
}

func (s *Server) modelProductionAction(w http.ResponseWriter, r *http.Request) {
	if !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"model production host exchanges require the OS-protected CLI credential", "")
		return
	}
	var action ModelProductionAction
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil || len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"model production action exceeds "+strconv.Itoa(MaxBody)+" bytes", "")
		return
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&action); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"model production action is not one closed JSON object: "+err.Error(), "")
		return
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"model production action carries trailing JSON", "")
		return
	}
	operationID := r.PathValue("id")
	switch action.Action {
	case "prepare_source":
		if action.Source == nil || action.Artifact != nil {
			s.refuse(w, r, http.StatusBadRequest, "model_production.action_invalid",
				"prepare_source requires exactly one source action", "")
			return
		}
		capabilities := make([]orchestrator.ModelSourceCapability, 0,
			len(action.Source.Capabilities))
		for _, capability := range action.Source.Capabilities {
			provider := map[string]pb.ModelSourceProvider{
				"huggingface": pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE,
				"civitai":     pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_CIVITAI,
			}[capability.Provider]
			if provider == pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED {
				s.refuse(w, r, http.StatusBadRequest, "model_production.provider_invalid",
					"source capability names an unsupported provider", "")
				return
			}
			capabilities = append(capabilities, orchestrator.ModelSourceCapability{
				Member: capability.Member, ObjectID: capability.ObjectID, Length: capability.Length,
				Provider: provider, URL: capability.URL, ExpiresAtUnix: capability.ExpiresAtUnix,
			})
		}
		prepared, problem := s.orchestrator.PrepareProductionSources(r.Context(), operationID,
			action.RentalID, orchestrator.ProductionSourcePlan{
				SelectionDigest: action.Source.SelectionDigest, SourceURI: action.Source.SourceURI,
				DeclaredLicense: action.Source.DeclaredLicense, Files: action.Source.Files,
				Profiles: action.Source.Profiles,
			}, capabilities)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		s.ok(w, r, http.StatusOK, ModelProductionActionResult{Prepared: prepared, Changed: true})
	case "transfer_artifact":
		if action.Artifact == nil || action.Source != nil {
			s.refuse(w, r, http.StatusBadRequest, "model_production.action_invalid",
				"transfer_artifact requires exactly one artifact action", "")
			return
		}
		problem := s.orchestrator.TransferProductionArtifact(r.Context(), operationID,
			action.Artifact.NodeName, action.Artifact.OutputSlot, action.RentalID,
			action.Artifact.TransferOperationID, action.Artifact.Decisions)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		s.ok(w, r, http.StatusOK, ModelProductionActionResult{Changed: true})
	default:
		s.refuse(w, r, http.StatusBadRequest, "model_production.action_invalid",
			"model production action must be prepare_source or transfer_artifact", "")
	}
}
