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
	"github.com/cozy-creator/cozy-creator-v2/internal/workflow"
)

type WorkflowController interface {
	Submit(workflow.Submission) (records.WorkflowExecution, bool, *exit.Error)
	State(id string) (*workflow.Snapshot, *exit.Error)
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
	Ordinal      int        `json:"ordinal"`
	Status       string     `json:"status"`
	Materialized bool       `json:"materialized"`
	ChildRequest string     `json:"child_request_id,omitempty"`
	Outputs      []MediaRef `json:"outputs"`
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
			Materialized: step.Materialized, Outputs: []MediaRef{}}
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
