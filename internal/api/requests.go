package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The request-level contract: a 202 handle out of submit, a lifecycle document out of
// status, and explicit cancellation. Cozy implements this surface today; other hosts
// require their own implementation and conformance proof.

// Submission is the request body. `input` is the package's own typed payload and is
// carried VERBATIM: the orchestrator digests exactly the bytes the client sent, so a
// re-submit under one key compares the same request identity.
type Submission struct {
	Package  string          `json:"package"`
	Function string          `json:"function"`
	Input    json.RawMessage `json:"input"`
	Outputs  []string        `json:"outputs,omitempty"`
	PlanID   string          `json:"plan_id,omitempty"`
	// LocalAssets is the local API's out-of-band input set. Each source path is ingested into
	// the daemon-owned immutable input store before the request row exists; it never
	// crosses the worker protocol. The typed payload carries only its opaque reference.
	LocalAssets []records.AssetBinding `json:"local_assets,omitempty"`
	// Worker is Cozy's local addition: it pins this request to an attached rental id.
	Worker     string `json:"worker,omitempty"`
	AttemptKey string `json:"-"`
}

// Handle is the 202 answer: the request's id and where to go next. Verbatim from the
// AsyncRequestHandle cozy.art consumes, plus `attempt` (which a local client can act on
// because there is no queue-position story to tell yet).
type Handle struct {
	RequestID     string `json:"request_id"`
	Status        string `json:"status"`
	Attempt       uint64 `json:"attempt"`
	StatusURL     string `json:"status_url"`
	ResponseURL   string `json:"response_url"`
	CancelURL     string `json:"cancel_url"`
	EventsURL     string `json:"events_url"`
	QueuePosition *int   `json:"queue_position,omitempty"`
	// Replay is true when this key was already recorded: the SAME request answers, and
	// nothing new was started. A client that retried a timed-out POST needs to know it
	// did not create a second execution, and inferring it from equal ids is a guess.
	Replay bool `json:"idempotent_replay"`
}

func (s *Server) submit(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil {
		s.refuse(w, r, http.StatusBadRequest, "unreadable_body",
			"the request body could not be read: "+err.Error(), "")
		return
	}
	if len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"a submission body is bounded at "+strconv.Itoa(MaxBody)+" bytes",
			"typed input only; assets arrive as references")
		return
	}
	var sub Submission
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	err = decoder.Decode(&sub)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
	}
	if err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the submission is not a JSON object: "+err.Error(), "")
		return
	}
	if sub.Package == "" || sub.Function == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"a submission names a package and a function",
			`{"package":"org/name","function":"denoise","input":{…}}`)
		return
	}
	if len(sub.LocalAssets) > 0 && !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"local_assets may name host filesystem paths and require the OS-protected CLI credential",
			"use `cozy invoke run --asset <field-path>=<file>`; this build exposes no browser asset-upload route")
		return
	}
	// Staging bytes and recording their request are one ownership handoff even though the
	// filesystem and SQLite cannot share a transaction. Terminal GC takes the same short
	// guard, so it cannot remove a digest in the gap between those two operations.
	if len(sub.LocalAssets) > 0 {
		unlock := inputasset.Guard()
		defer unlock()
	}
	key := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	existing, e := s.store.RequestByIdempotencyKey(key)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	var spec orchestrator.Submission
	if existing != nil {
		// Resolve an existing key from the durable request identity, not from resources
		// retained only while it can execute. The caller's asset claims are still hashed
		// below, so a different body conflicts, but neither its source nor the reclaimed
		// staging object is opened merely to answer an already-recorded request.
		spec = replaySubmission(sub, *existing)
	} else {
		spec, e = s.resolvePlan(sub)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		spec.Assets, e = s.stageAssets(spec.Assets)
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		if len(spec.Assets) > 0 {
			defer func() {
				if e := inputasset.DropUnowned(s.layout, s.store, spec.Assets); e != nil {
					fmt.Fprintf(s.log, "input asset cleanup deferred: %s\n", e.Message)
				}
			}()
		}
	}
	spec.IdemKey = key
	// THE BODY DIGEST is over the whole submission the key names, not over the payload
	// alone: one key that named a different FUNCTION must conflict as loudly as one
	// that named different input. The canonical preimage is the digest's subject, so
	// two clients that spell the same submission differently still agree.
	digest, e := submissionDigest(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.BodyDigest = digest

	requestID, attempt, fresh, e := s.orchestrator.SubmitDetail(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	row, e := s.store.RequestRow(requestID)
	if e != nil || row == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal",
			"the request was recorded and cannot be read back", "")
		return
	}
	handle := s.handleOf(*row, attempt)
	handle.Replay = !fresh
	// 202 means "this host started work", and it may not be said twice for one key.
	// A recorded answer is a 200 with the same handle: the client already has it.
	status := http.StatusAccepted
	if handle.Replay {
		status = http.StatusOK
	}
	s.ok(w, r, status, handle)
}

func replaySubmission(sub Submission, recorded records.Request) orchestrator.Submission {
	planID := sub.PlanID
	if planID == "" {
		planID = recorded.PlanID
	}
	outputs := append([]string(nil), sub.Outputs...)
	if len(outputs) == 0 && recorded.Outputs != "" {
		outputs = strings.Split(recorded.Outputs, ",")
	}
	assets := append([]records.AssetBinding(nil), sub.LocalAssets...)
	sort.Slice(assets, func(i, j int) bool { return assets[i].FieldPath < assets[j].FieldPath })
	payload := []byte(sub.Input)
	if len(payload) == 0 {
		payload = []byte("{}")
	}
	return orchestrator.Submission{
		Package: sub.Package, Entrypoint: sub.Function, Payload: payload,
		Outputs: outputs, PlanID: planID, Worker: sub.Worker, Assets: assets,
	}
}

// submissionDigest is the canonical identity of one submission. It uses the SAME writer
// the protocol documents use, so the digest a client can reproduce is the digest the
// authority recorded.
func submissionDigest(spec orchestrator.Submission) (string, *exit.Error) {
	doc := map[string]canonical.Value{
		"kind":     "serve",
		"package":  spec.Package,
		"function": spec.Entrypoint,
		"plan_id":  spec.PlanID,
		"input":    base64.StdEncoding.EncodeToString(spec.Payload),
		"outputs":  strings.Join(spec.Outputs, ","),
	}
	assets := make([]canonical.Value, 0, len(spec.Assets))
	for _, asset := range spec.Assets {
		assets = append(assets, map[string]canonical.Value{
			"field_path": asset.FieldPath,
			"digest":     asset.Digest,
			"length":     asset.Length,
			"media_type": asset.MediaType,
			"order":      int64(asset.Order),
		})
	}
	if len(assets) > 0 {
		doc["assets"] = assets
	}
	// The pinned rental is NOT in it: a worker id says WHERE the same work runs, and two
	// submissions of one key that differ only in placement are the same request. What the
	// row records is what a requeue re-derives; the digest is about meaning.
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the submission: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot digest the submission: %s", err)
	}
	return spelled, nil
}

func (s *Server) handleOf(row records.Request, attempt uint64) Handle {
	base := "/v1/requests/" + row.ID
	h := Handle{
		RequestID: row.ID, Status: contractStatus(row.State), Attempt: attempt,
		StatusURL: base, ResponseURL: base, CancelURL: base + "/cancel",
		EventsURL: base + "/events",
	}
	if h.Status == "queued" {
		if position := s.orchestrator.QueuePosition(row.ID); position > 0 {
			h.QueuePosition = &position
		}
	}
	return h
}

// contractStatus projects the authority's own state names onto the contract's vocabulary.
// The two are deliberately not the same word list: the store records what happened to a
// row, the contract says what a client should do next.
func contractStatus(state string) string {
	switch state {
	case "submitted", "queued", "requeue_pending":
		// `queued` is BOTH "never dispatched" and "an attempt ended and the orchestrator
		// is minting the next ordinal". From a client's seat those are the same fact:
		// work is owed and nothing has settled.
		return "queued"
	case "dispatching":
		return "in_progress"
	case "succeeded":
		return "completed"
	case "failed", "abandoned", "refused":
		return "failed"
	case "canceled":
		return "canceled"
	}
	return state
}

// resolvePlan turns the client's package/function into the orchestrator's Submission.
// The plan id is resolved through the LOCAL resolver — the same object `start` uses — so
// a client never names a plan digest and a submission can never bind a binding this host
// did not install.
func (s *Server) resolvePlan(sub Submission) (orchestrator.Submission, *exit.Error) {
	out := orchestrator.Submission{
		Package: sub.Package, Entrypoint: sub.Function, Payload: []byte(sub.Input),
		Outputs: sub.Outputs, PlanID: sub.PlanID, Worker: sub.Worker, Assets: sub.LocalAssets,
	}
	if len(out.Payload) == 0 {
		out.Payload = []byte("{}")
	}
	// THE PIN IS RESOLVED BEFORE A ROW EXISTS. A rental this host does not hold cannot be
	// placed on any later attempt either, so recording the request would hand the client
	// an id for work that is already known to be unplaceable.
	var remotePlacement *orchestrator.DesiredPlacement
	if out.Worker != "" {
		if s.rentals == nil {
			return out, exit.Unavailablef("this Cozy daemon attaches no remote workers")
		}
		var e *exit.Error
		remotePlacement, e = s.rentals(out.Worker)
		if e != nil {
			return out, e
		}
		if remotePlacement.Package != sub.Package {
			return out, exit.Named(exit.Conflict, "rental.package_mismatch",
				"rental %s carries exact control for %s, not %s", out.Worker,
				remotePlacement.Package, sub.Package)
		}
	}
	if remotePlacement != nil {
		expected, outputs, e := placementPlan(*remotePlacement, sub.Function)
		if e != nil {
			return out, e
		}
		if out.PlanID != "" && out.PlanID != expected {
			return out, exit.Named(exit.Conflict, "rental.plan_mismatch",
				"rental %s binds function %s to %s, not caller-supplied %s",
				out.Worker, sub.Function, expected, out.PlanID)
		}
		out.PlanID = expected
		if len(out.Outputs) == 0 {
			out.Outputs = outputs
		}
		// The rental's FROZEN descriptor is the schema; no local install is consulted.
		descriptor, e := rental.PackageDescriptor(s.store, out.Worker)
		if e != nil {
			return out, e
		}
		entrypoint, e := descriptor.Function(sub.Function)
		if e != nil {
			return out, e
		}
		if e := validateInputs(entrypoint, &out); e != nil {
			return out, e
		}
	} else {
		var placement orchestrator.DesiredPlacement
		if s.packages == nil {
			return out, exit.Unavailablef("this Cozy daemon resolves no packages")
		}
		var e *exit.Error
		placement, e = s.packages.ResolvePlacement(sub.Package)
		if e != nil {
			return out, e
		}
		planID, outputs, e := placementPlan(placement, sub.Function)
		if e != nil {
			return out, e
		}
		if out.PlanID != "" && out.PlanID != planID {
			return out, exit.Named(exit.Conflict, "plan_mismatch",
				"%s/%s resolves plan %s, not caller-supplied %s",
				sub.Package, sub.Function, planID, out.PlanID)
		}
		out.PlanID = planID
		out.InstallID = placement.InstallID
		if len(out.Outputs) == 0 {
			out.Outputs = outputs
		}
		entrypoint, e := s.packages.Entrypoint(placement.InstallID, sub.Function)
		if e != nil {
			return out, e
		}
		if e := validateInputs(entrypoint, &out); e != nil {
			return out, e
		}
	}
	return out, nil
}

// validateInputs checks the payload and every local asset against the entrypoint that
// will run it — the same law for a local install and a rental's frozen descriptor.
func validateInputs(entrypoint *launch.Entrypoint, out *orchestrator.Submission) *exit.Error {
	if e := launch.ValidatePayload(entrypoint, out.Payload); e != nil {
		return e
	}
	for index := range out.Assets {
		asset := &out.Assets[index]
		assetSpec, ok := launch.AssetSpec(entrypoint, asset.FieldPath)
		if !ok || assetSpec.MaxBytes <= 0 || asset.Length > assetSpec.MaxBytes {
			return exit.Named(exit.Validation, "input_asset_bound",
				"input asset %s is %d B and its pinned field admits %d B",
				asset.FieldPath, asset.Length, assetSpec.MaxBytes)
		}
		if !assetSpec.AcceptsMediaType(asset.MediaType) {
			return exit.Named(exit.Validation, "input_asset_media_type",
				"input asset %s is %s and its pinned field does not admit that media type",
				asset.FieldPath, asset.MediaType)
		}
		asset.MaxBytes = assetSpec.MaxBytes
	}
	return nil
}

func placementPlan(placement orchestrator.DesiredPlacement, function string) (string, []string, *exit.Error) {
	for _, binding := range placement.Bindings {
		if binding.Entrypoint != function {
			continue
		}
		id, e := binding.PlanID()
		if e != nil {
			return "", nil, e
		}
		return id, append([]string(nil), binding.Outputs...), nil
	}
	return "", nil, exit.New(exit.NotFound, "%s has no function %q", placement.Package, function).
		WithRemedy("GET /v1/local/packages lists the functions this target serves")
}

func (s *Server) stageAssets(assets []records.AssetBinding) ([]records.AssetBinding, *exit.Error) {
	if len(assets) == 0 {
		return nil, nil
	}
	out := make([]records.AssetBinding, 0, len(assets))
	rollback := func() {
		if e := inputasset.DropUnowned(s.layout, s.store, out); e != nil {
			fmt.Fprintf(s.log, "partial input asset cleanup deferred: %s\n", e.Message)
		}
	}
	seen := map[string]bool{}
	for _, asset := range assets {
		if seen[asset.FieldPath] {
			rollback()
			return nil, exit.New(exit.Validation,
				"input asset field %q was supplied more than once", asset.FieldPath)
		}
		seen[asset.FieldPath] = true
		maxBytes := asset.MaxBytes
		if maxBytes <= 0 {
			maxBytes = inputasset.MaxBytes
		}
		staged, e := inputasset.Stage(s.layout, asset, maxBytes)
		if e != nil {
			rollback()
			return nil, e
		}
		out = append(out, staged)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].FieldPath < out[j].FieldPath })
	return out, nil
}

// ------------------------------------------------------------------------- status

// Lifecycle is the status document. Verbatim from RequestLifecycleResponse plus the
// fields a local client has and a cloud one does not need to presign: the typed result,
// the visible media by OPAQUE id, and the triage handle.
type Lifecycle struct {
	Kind        string         `json:"kind"`
	RequestID   string         `json:"request_id"`
	Status      string         `json:"status"`
	Package     string         `json:"package"`
	Function    string         `json:"function"`
	Attempt     uint64         `json:"attempt"`
	Attempts    int            `json:"attempts"`
	ResponseURL string         `json:"response_url"`
	Metrics     map[string]any `json:"metrics,omitempty"`
	ErrorType   string         `json:"error_type,omitempty"`
	Error       string         `json:"error,omitempty"`
	Result      any            `json:"result,omitempty"`
	Outputs     []MediaRef     `json:"outputs"`
	Triage      *TriageRef     `json:"triage,omitempty"`
	CreatedAt   string         `json:"created_at"`
}

// MediaRef is how bytes are named in EVERY document this API emits: an opaque id and its
// facts. There is no `path` field, and there is no route that would accept one.
type MediaRef struct {
	OutputID string `json:"output_id"`
	MediaID  string `json:"media_id"`
	URL      string `json:"url"`
	MimeType string `json:"mime_type"`
	Length   int64  `json:"length"`
	Digest   string `json:"digest"`
}

// TriageRef is the handle onto a retained bundle. `attempt_key` is the orchestrator's own
// opaque key; `subject_id` is the runtime's. Neither is a path.
type TriageRef struct {
	AttemptKey string `json:"attempt_key"`
	SubjectID  string `json:"subject_id"`
	URL        string `json:"url"`
	Length     int64  `json:"length"`
	Kept       bool   `json:"kept"`
	Fault      string `json:"fault,omitempty"`
}

func (s *Server) getRequest(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	row, e := s.store.RequestRow(id)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no request "+id+" on this host", "")
		return
	}
	s.ok(w, r, http.StatusOK, s.lifecycleOf(*row))
}

func (s *Server) lifecycleOf(row records.Request) Lifecycle {
	kind := "invocation"
	if row.IsJob() {
		kind = "job"
	}
	life := Lifecycle{
		Kind: kind, RequestID: row.ID, Status: contractStatus(row.State), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal),
		ResponseURL: "/v1/requests/" + row.ID, CreatedAt: row.CreatedAt,
		Outputs: []MediaRef{},
	}
	attempts, _ := s.store.Attempts(row.ID)
	life.Attempts = len(attempts)
	outs, _ := s.store.VisibleOutputs(row.ID)
	for _, o := range outs {
		life.Outputs = append(life.Outputs, MediaRef{
			OutputID: o.OutputID, MediaID: o.MediaID, URL: "/v1/media/" + o.MediaID,
			MimeType: o.MimeType, Length: o.Length, Digest: o.Digest,
		})
	}
	if len(attempts) == 0 {
		return life
	}
	last := attempts[len(attempts)-1]
	if last.TriageSubject != "" {
		life.Triage = &TriageRef{
			AttemptKey: last.AttemptKey, SubjectID: last.TriageSubject,
			URL:    "/v1/local/attempts/" + last.AttemptKey + "/triage",
			Length: last.TriageLength, Kept: last.TriagePath != "",
		}
	}
	if len(last.TerminalBody) == 0 {
		return life
	}
	doc, err := canonical.Read(last.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return life
	}
	metrics := doc.Sub("metrics")
	life.Metrics = map[string]any{
		"runtime_ms": metrics.Int("runtime_ms"), "queue_ms": metrics.Int("queue_ms"),
		"handler_ms": metrics.Int("handler_ms"), "device_lease_ms": metrics.Int("device_lease_ms"),
		"finalization_ms": metrics.Int("finalization_ms"),
		"peak_vram_bytes": metrics.Int("peak_vram_bytes"),
	}
	if last.TerminalStatus != "SUCCEEDED" {
		life.ErrorType, life.Error = last.TerminalCause, last.SafeMessage
	}
	// The typed result rides the terminal INLINE (cl-001's proof). It is base64 in the
	// canonical document; the contract hands the client the decoded JSON so a browser
	// never decodes a transport encoding to read a result.
	if inline := doc.Sub("result").Str("inline_result"); inline != "" {
		if decoded, err := base64.StdEncoding.DecodeString(inline); err == nil {
			var typed any
			if json.Unmarshal(decoded, &typed) == nil {
				life.Result = typed
			}
		}
	}
	return life
}

func (s *Server) listRequests(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}
	state := ""
	switch strings.TrimSpace(r.URL.Query().Get("status")) {
	case "", "any":
	case "queued":
		state = "submitted"
	case "in_progress":
		state = "dispatching"
	case "completed":
		state = "succeeded"
	case "failed":
		state = "failed"
	case "canceled":
		state = "canceled"
	default:
		s.refuse(w, r, http.StatusBadRequest, "invalid_status",
			"unknown status filter", "any | queued | in_progress | completed | failed | canceled")
		return
	}
	rows, e := s.store.Requests(state, limit)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	out := make([]Lifecycle, 0, len(rows))
	for _, row := range rows {
		out = append(out, s.lifecycleOf(row))
	}
	s.ok(w, r, http.StatusOK, map[string]any{"requests": out, "count": len(out)})
}

// ------------------------------------------------------------------------- cancel

func (s *Server) cancelRequest(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	row, e := s.store.RequestRow(id)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no request "+id+" on this host", "")
		return
	}
	if status := contractStatus(row.State); status == "completed" || status == "failed" || status == "canceled" {
		s.ok(w, r, http.StatusOK, s.lifecycleOf(*row))
		return
	}
	attempts, e := s.store.Attempts(id)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if len(attempts) == 0 {
		if e := s.orchestrator.CancelQueued(id); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(id)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued request was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.lifecycleOf(*updated))
		return
	}
	last := attempts[len(attempts)-1]
	if last.State == "closed" || last.State == "dispatch_aborted" {
		if e := s.orchestrator.CancelQueued(id); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(id)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued request was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.lifecycleOf(*updated))
		return
	}
	if last.State == "terminal" {
		s.refuse(w, r, http.StatusConflict, "terminal_ack_pending",
			"the current attempt has a terminal whose retry/settlement projection is not acknowledged yet",
			"retry cancellation after the terminal ack; no new attempt can dispatch before that projection")
		return
	}
	grace := orchestrator.ClientCancelGraceMS
	if v := r.URL.Query().Get("grace_ms"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 120000 {
			grace = n
		}
	}
	if e := s.orchestrator.Cancel(id, uint64(last.Attempt), pb.CancelReason_CANCEL_REASON_CLIENT, grace); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	// A cancel is a REQUEST for cancellation, never a verdict. The attempt's own
	// journaled terminal is what settles it, and a non-cooperative handler may still
	// succeed. Saying "canceled" here would be exactly the lie the protocol refuses.
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"request_id": id, "attempt": last.Attempt, "status": "cancel_requested",
		"note": "the attempt's own journaled terminal settles it; watch the event stream",
	})
}
