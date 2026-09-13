package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ExecutionCapture is supplied only by the fixed Host-owned coordinator. It
// imports exact package custody and resolves the same ordinary root submission.
type ExecutionCapture func(context.Context, string, []byte) (orchestrator.Submission, *exit.Error)

func (s *Server) submitExecutionCapture(w http.ResponseWriter, r *http.Request) {
	if s.executionCapture == nil || s.executionGrantDigest == "" {
		s.refuseTyped(w, r, exit.New(exit.Unavailable, "this daemon is not a private execution owner"))
		return
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, pb.MaxInlineControlBytes+1))
	if err != nil || len(raw) == 0 || len(raw) > pb.MaxInlineControlBytes {
		s.refuseTyped(w, r, exit.New(exit.Validation, "execution capture is unreadable or exceeds its bound"))
		return
	}
	key := r.Header.Get("Idempotency-Key")
	row, fresh, problem := s.AdmitExecutionCapture(r.Context(), key, raw)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if fresh {
		defer s.activateRecorded(row)
	}
	s.replyJob(w, r, row, fresh)
}

// AdmitExecutionCapture is the shared initial/repeated admission boundary. The
// fixed bootstrap and authenticated Host bridge both use this same importer and
// ordinary record/activation split.
func (s *Server) AdmitExecutionCapture(ctx context.Context, key string, raw []byte) (records.Request, bool, *exit.Error) {
	if s.executionCapture == nil || s.executionGrantDigest == "" || key == "" || len(raw) == 0 || len(raw) > pb.MaxInlineControlBytes {
		return records.Request{}, false, exit.New(exit.Validation, "private execution admission is unavailable or incomplete")
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	existing, problem := s.store.RequestByIdempotencyKey(key)
	if problem != nil {
		return records.Request{}, false, problem
	}
	if existing != nil {
		if existing.BodyDigest != digest || existing.ParentRequestID != "" || !existing.IsJob() || existing.ExecutionGrantDigest != s.executionGrantDigest {
			return records.Request{}, false, exit.New(exit.Conflict, "execution idempotency key names a different root capture")
		}
		return *existing, false, nil
	}
	s.executionImports.Add(1)
	defer s.executionImports.Add(-1)
	spec, problem := s.executionCapture(ctx, key, raw)
	if problem != nil {
		return records.Request{}, false, problem
	}
	if spec.IdemKey != key || spec.BodyDigest != digest || spec.Kind != "job" || spec.ExecutionGrantDigest != s.executionGrantDigest {
		return records.Request{}, false, exit.New(exit.Conflict, "capture importer changed the root admission identity")
	}
	return s.recordSubmission(spec, true)
}

func (s *Server) executionActivity(w http.ResponseWriter, r *http.Request) {
	if s.executionGrantDigest == "" {
		s.refuseTyped(w, r, exit.New(exit.NotFound, "this daemon has no private execution generation"))
		return
	}
	// Admission cannot commit between the durable and live halves of this read.
	// Imports still preparing their bytes are represented by executionImports.
	s.shutdownAdmission.Lock()
	defer s.shutdownAdmission.Unlock()
	durable, problem := s.store.Obligations()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	held, problem := s.orchestrator.Managing()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	for _, obligation := range durable {
		held = append(held, obligation.String())
	}
	if s.executionImports.Load() != 0 {
		held = append(held, "capture import")
	}
	sort.Strings(held)
	s.ok(w, r, http.StatusOK, map[string]any{"execution_grant_digest": s.executionGrantDigest, "busy": len(held) != 0, "obligations": held})
}

// executionRoot owns the ancestry policy used by the Host bridge. Parent edges
// and root scope are immutable; existing status/event/cancel routes consume the
// returned exact subject without duplicating this walk.
func (s *Server) executionRoot(w http.ResponseWriter, r *http.Request) {
	if s.executionGrantDigest == "" {
		s.refuseTyped(w, r, exit.New(exit.NotFound, "this daemon has no private execution generation"))
		return
	}
	row, problem := s.store.RequestByReference(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if row == nil {
		s.refuseTyped(w, r, exit.New(exit.NotFound, "execution request is absent"))
		return
	}
	subject := row.ID
	seen := map[string]bool{}
	for row.ParentRequestID != "" {
		if seen[row.ID] || len(seen) >= 1024 {
			s.refuseTyped(w, r, exit.New(exit.Conflict, "execution ancestry is invalid"))
			return
		}
		seen[row.ID] = true
		row, problem = s.store.RequestRow(row.ParentRequestID)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		if row == nil {
			s.refuseTyped(w, r, exit.New(exit.NotFound, "execution ancestry is unavailable"))
			return
		}
	}
	if row.ExecutionGrantDigest != s.executionGrantDigest || !row.IsJob() {
		s.refuseTyped(w, r, exit.New(exit.NotFound, "request is outside this private execution generation"))
		return
	}
	value := map[string]any{"subject_request_id": subject, "request_id": row.ID,
		"capsule_digest": row.BodyDigest, "execution_grant_digest": row.ExecutionGrantDigest}
	collection, problem := s.store.ExecutionCollection(row.ID, row.ExecutionGrantDigest)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if collection != nil {
		value["terminal_event_id"], value["attempt"], value["collected"] = collection.TerminalEventID, collection.Attempt, collection.Collected
	}
	s.ok(w, r, http.StatusOK, value)
}

func (s *Server) collectExecution(w http.ResponseWriter, r *http.Request) {
	if s.executionGrantDigest == "" {
		s.refuseTyped(w, r, exit.New(exit.NotFound, "this daemon has no private execution generation"))
		return
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, 4097))
	if err != nil || len(raw) > 4096 {
		s.refuseTyped(w, r, exit.New(exit.Validation, "collection acknowledgement exceeds its bound"))
		return
	}
	var body struct {
		TerminalEventID int64 `json:"terminal_event_id"`
		Attempt         int64 `json:"attempt"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&body) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		s.refuseTyped(w, r, exit.New(exit.Validation, "collection acknowledgement differs from its closed schema"))
		return
	}
	receipt, problem := s.store.CollectExecution(r.PathValue("id"), s.executionGrantDigest, body.TerminalEventID, body.Attempt)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, receipt)
}
