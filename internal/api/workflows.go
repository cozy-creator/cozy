package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/rental"
	"github.com/cozy-creator/cozy-creator-v2/internal/workflow"
)

type WorkflowController interface {
	Submit(workflow.Submission) (records.WorkflowExecution, bool, *exit.Error)
	State(id string) (*workflow.Snapshot, *exit.Error)
	Receipt(id string) (*workflow.Receipt, *exit.Error)
	Cancel(id string) (*workflow.Snapshot, bool, *exit.Error)
}

type WorkflowSubmission struct {
	Plan    json.RawMessage            `json:"plan"`
	Assets  []WorkflowAssetResolution  `json:"assets,omitempty"`
	Targets []WorkflowTargetResolution `json:"targets,omitempty"`
}

type WorkflowTargetResolution struct {
	Step   int    `json:"step"`
	Worker string `json:"worker"`
}

// WorkflowAssetResolution names only a staged content object. The private path is derived
// by the LocalService and cannot enter an API payload or workflow identity.
type WorkflowAssetResolution struct {
	Step      int    `json:"step"`
	FieldPath string `json:"field_path"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	Order     uint32 `json:"order"`
}

type WorkflowHandle struct {
	WorkflowID string `json:"workflow_id"`
	Status     string `json:"status"`
	StatusURL  string `json:"status_url"`
	CancelURL  string `json:"cancel_url"`
	Replay     bool   `json:"idempotent_replay"`
}

type WorkflowState struct {
	WorkflowID         string              `json:"workflow_id"`
	Status             string              `json:"status"`
	ExecutionDigest    string              `json:"execution_digest"`
	CreativePlanDigest string              `json:"creative_plan_digest"`
	CancelRequestedAt  string              `json:"cancel_requested_at,omitempty"`
	TerminalCode       string              `json:"terminal_code,omitempty"`
	TerminalMessage    string              `json:"terminal_message,omitempty"`
	CreatedAt          string              `json:"created_at"`
	SettledAt          string              `json:"settled_at,omitempty"`
	Steps              []WorkflowStepState `json:"steps"`
}

type WorkflowStepState struct {
	Ordinal            int                          `json:"ordinal"`
	Status             string                       `json:"status"`
	Materialized       bool                         `json:"materialized"`
	MaterializedDigest string                       `json:"materialized_submission_digest,omitempty"`
	MaterializedAssets []workflow.MaterializedAsset `json:"materialized_assets"`
	ResolvedBindings   []records.ResolvedBinding    `json:"resolved_bindings"`
	ChildKey           string                       `json:"child_key,omitempty"`
	RentalID           string                       `json:"rental_id,omitempty"`
	ChildRequest       string                       `json:"child_request_id,omitempty"`
	Outputs            []MediaRef                   `json:"outputs"`
}

type WorkflowReceipt struct {
	Workflow          WorkflowState           `json:"workflow"`
	CanonicalPlan     []byte                  `json:"canonical_plan_bytes"`
	CanonicalCreative []byte                  `json:"canonical_creative_plan_bytes,omitempty"`
	Steps             []WorkflowReceiptStep   `json:"steps"`
	Rentals           []WorkflowReceiptRental `json:"rentals"`
}

type WorkflowReceiptStep struct {
	Ordinal                int    `json:"ordinal"`
	MaterializedSubmission []byte `json:"materialized_submission_bytes,omitempty"`
}

type WorkflowReceiptRental struct {
	RentalID                  string   `json:"rental_id"`
	Endpoint                  string   `json:"endpoint"`
	Accelerator               string   `json:"accelerator"`
	ObservedAccelerator       string   `json:"observed_accelerator"`
	ObservedAcceleratorCount  int      `json:"observed_accelerator_count"`
	ObservedBackend           string   `json:"observed_backend"`
	ObservedWorkerInstance    string   `json:"observed_worker_instance"`
	ObservedWorkerBootID      string   `json:"observed_worker_boot_id"`
	ObservedAt                string   `json:"observed_at"`
	ControlSnapshotDigest     string   `json:"control_snapshot_digest"`
	ControlSnapshotLength     int64    `json:"control_snapshot_length"`
	ExactControlSnapshotBytes []byte   `json:"exact_control_snapshot_bytes"`
	EndpointExecutionDigest   string   `json:"endpoint_execution_digest"`
	ArtifactObjectSetDigest   string   `json:"artifact_object_set_digest"`
	ModelRootDigests          []string `json:"model_root_digests"`
	EndpointReleaseID         string   `json:"endpoint_release_id"`
	DescriptorDigest          string   `json:"descriptor_digest"`
	EnvironmentSpecDigest     string   `json:"environment_spec_digest"`
	InstalledReceiptDigest    string   `json:"installed_environment_receipt_digest"`
	PlacementSetDigest        string   `json:"placement_set_digest"`
	BindingPlanDigests        []string `json:"binding_plan_digests"`
}

func (s *Server) submitWorkflow(w http.ResponseWriter, r *http.Request) {
	if s.workflows == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this LocalService has no workflow controller"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil {
		s.refuse(w, r, http.StatusBadRequest, "unreadable_body",
			"the workflow body could not be read: "+err.Error(), "")
		return
	}
	if len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"a workflow submission is bounded at "+strconv.Itoa(MaxBody)+" bytes",
			"assets are staged content identities and never ride the JSON body")
		return
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var sub WorkflowSubmission
	if err := decoder.Decode(&sub); err != nil || len(sub.Plan) == 0 {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the workflow submission is not a strict object with one plan", "")
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the workflow submission carries trailing JSON", "")
		return
	}
	if len(sub.Assets) > 0 && !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"workflow asset resolutions require the OS-protected CLI credential",
			"the browser has no staged-asset upload surface")
		return
	}
	assets := map[int][]records.AssetBinding{}
	for _, asset := range sub.Assets {
		if _, err := canonical.Raw(asset.Digest); err != nil {
			s.refuse(w, r, http.StatusBadRequest, "invalid_asset_digest",
				"a workflow asset digest is not a lowercase sha256 identity", "")
			return
		}
		assets[asset.Step] = append(assets[asset.Step], records.AssetBinding{
			FieldPath: asset.FieldPath, LocalPath: s.layout.InputAsset(asset.Digest),
			Digest: asset.Digest, Length: asset.Length, MediaType: asset.MediaType, Order: asset.Order,
		})
	}
	workers := map[int]string{}
	for _, target := range sub.Targets {
		if target.Step < 1 || strings.TrimSpace(target.Worker) == "" || workers[target.Step] != "" {
			s.refuse(w, r, http.StatusBadRequest, "invalid_workflow_target",
				"each workflow target needs one unique positive step and worker id", "")
			return
		}
		workers[target.Step] = strings.TrimSpace(target.Worker)
	}
	row, fresh, problem := s.workflows.Submit(workflow.Submission{
		IdempotencyKey: strings.TrimSpace(r.Header.Get("Idempotency-Key")), Plan: sub.Plan,
		Assets: assets, Workers: workers,
	})
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	handle := WorkflowHandle{WorkflowID: row.ID, Status: row.State,
		StatusURL: "/v1/local/workflows/" + row.ID,
		CancelURL: "/v1/local/workflows/" + row.ID + "/cancel", Replay: !fresh}
	status := http.StatusAccepted
	if !fresh {
		status = http.StatusOK
	}
	s.ok(w, r, status, handle)
}

func (s *Server) getWorkflow(w http.ResponseWriter, r *http.Request) {
	if s.workflows == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this LocalService has no workflow controller"))
		return
	}
	snapshot, problem := s.workflows.State(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if snapshot == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no workflow "+r.PathValue("id")+" on this host", "")
		return
	}
	s.ok(w, r, http.StatusOK, workflowStateOf(snapshot))
}

func (s *Server) getWorkflowReceipt(w http.ResponseWriter, r *http.Request) {
	if s.workflows == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this LocalService has no workflow controller"))
		return
	}
	receipt, problem := s.workflows.Receipt(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if receipt == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no workflow "+r.PathValue("id")+" on this host", "")
		return
	}
	out := WorkflowReceipt{Workflow: workflowStateOf(&receipt.Snapshot),
		CanonicalPlan:     append([]byte(nil), receipt.Plan...),
		CanonicalCreative: append([]byte(nil), receipt.CreativePlan...),
		Steps:             make([]WorkflowReceiptStep, 0, len(receipt.Steps)),
		Rentals:           make([]WorkflowReceiptRental, 0, len(receipt.Rentals))}
	for _, step := range receipt.Steps {
		out.Steps = append(out.Steps, WorkflowReceiptStep{Ordinal: step.Ordinal,
			MaterializedSubmission: append([]byte(nil), step.MaterializedSubmission...)})
	}
	for _, control := range receipt.Rentals {
		summary, summaryProblem := rental.Summarize(records.Rental{
			ID: control.RentalID, EndpointRef: control.EndpointRef,
			ControlSnapshotDigest: control.ControlSnapshotDigest,
			ControlSnapshotLength: control.ControlSnapshotLength,
			ControlSnapshotBytes:  control.ControlSnapshotBytes,
		})
		if summaryProblem != nil {
			s.refuseTyped(w, r, summaryProblem)
			return
		}
		out.Rentals = append(out.Rentals, WorkflowReceiptRental{
			RentalID: control.RentalID, Endpoint: control.EndpointRef,
			Accelerator:               control.AcceleratorModel,
			ObservedAccelerator:       control.ObservedAccelerator,
			ObservedAcceleratorCount:  control.ObservedAcceleratorCount,
			ObservedBackend:           control.ObservedBackend,
			ObservedWorkerInstance:    control.ObservedWorkerInstance,
			ObservedWorkerBootID:      control.ObservedWorkerBootID,
			ObservedAt:                control.ObservedAt,
			ControlSnapshotDigest:     control.ControlSnapshotDigest,
			ControlSnapshotLength:     control.ControlSnapshotLength,
			ExactControlSnapshotBytes: append([]byte(nil), control.ControlSnapshotBytes...),
			EndpointExecutionDigest:   summary.EndpointExecutionDigest,
			ArtifactObjectSetDigest:   summary.ArtifactObjectSetDigest,
			ModelRootDigests:          append([]string(nil), summary.ModelRootDigests...),
			EndpointReleaseID:         summary.EndpointReleaseID,
			DescriptorDigest:          summary.DescriptorDigest,
			EnvironmentSpecDigest:     summary.EnvironmentSpecDigest,
			InstalledReceiptDigest:    summary.InstalledEnvironmentReceiptDigest,
			PlacementSetDigest:        summary.PlacementSetDigest,
			BindingPlanDigests:        append([]string(nil), summary.BindingPlanDigests...),
		})
	}
	s.ok(w, r, http.StatusOK, out)
}

func (s *Server) cancelWorkflow(w http.ResponseWriter, r *http.Request) {
	if s.workflows == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this LocalService has no workflow controller"))
		return
	}
	snapshot, changed, problem := s.workflows.Cancel(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if snapshot == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no workflow "+r.PathValue("id")+" on this host", "")
		return
	}
	status := http.StatusOK
	if changed && snapshot.Execution.State == "canceling" {
		status = http.StatusAccepted
	}
	s.ok(w, r, status, workflowStateOf(snapshot))
}

func workflowStateOf(snapshot *workflow.Snapshot) WorkflowState {
	row := snapshot.Execution
	out := WorkflowState{WorkflowID: row.ID, Status: row.State,
		ExecutionDigest: row.ExecutionDigest, CreativePlanDigest: row.CreativePlanDigest,
		CancelRequestedAt: row.CancelRequestedAt, TerminalCode: row.TerminalCode,
		TerminalMessage: row.TerminalMessage, CreatedAt: row.CreatedAt, SettledAt: row.SettledAt,
		Steps: make([]WorkflowStepState, 0, len(snapshot.Steps))}
	for _, step := range snapshot.Steps {
		status := step.State
		if status != "pending" && status != "prepared" && status != "missing" {
			status = contractStatus(status)
		}
		one := WorkflowStepState{Ordinal: step.Ordinal, Status: status,
			Materialized: step.Materialized, MaterializedDigest: step.MaterializedDigest,
			MaterializedAssets: append([]workflow.MaterializedAsset{}, step.MaterializedAssets...),
			ResolvedBindings:   append([]records.ResolvedBinding{}, step.ResolvedBindings...),
			ChildKey:           step.ChildKey, RentalID: step.RentalID, Outputs: []MediaRef{}}
		if step.Child != nil {
			one.ChildRequest = step.Child.ID
		}
		for _, output := range step.Outputs {
			one.Outputs = append(one.Outputs, MediaRef{OutputID: output.OutputID,
				MediaID: output.MediaID, URL: "/v1/media/" + output.MediaID,
				MimeType: output.MimeType, Length: output.Length, Digest: output.Digest})
		}
		out.Steps = append(out.Steps, one)
	}
	return out
}
