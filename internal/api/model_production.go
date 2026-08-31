package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
