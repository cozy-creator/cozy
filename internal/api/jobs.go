package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The job family is mounted under /v1/local/ because its typed input trees are local
// directories the caller already owns. Keeping filesystem paths out of the proposed
// common core preserves that boundary.
//
// The event plane is reused, and that is not an exception: a job is a request row in the
// one lifecycle authority, so `GET /v1/requests/{id}/events` streams a job's lifecycle
// with no second event authority anywhere. `cozy run list` is that route's client.

// JobSubmission is the job submit body.
type JobSubmission struct {
	Package   string          `json:"package"`
	Function  string          `json:"function"`
	Input     json.RawMessage `json:"input"`
	InstallID string          `json:"install_id,omitempty"`
	// Org is the publishing org whose SCRATCH repo this job lands in
	// (`<org>/_job-<request-id>`). It defaults to `local` — a local host has no identity
	// plane yet (decisions #229) and inventing one would be a fake account.
	Org string `json:"org,omitempty"`
	// Trees are the typed input trees as `ref=<directory>`. The grant is the read
	// capability: a `Tree` field naming a ref that is not here never hydrates.
	Trees []string `json:"trees,omitempty"`
}

// JobHandle is the 202 answer.
type JobHandle struct {
	JobID     string `json:"job_id"`
	Status    string `json:"status"`
	Attempt   uint64 `json:"attempt"`
	Package   string `json:"package"`
	Function  string `json:"function"`
	Repo      string `json:"publication_repo"`
	StatusURL string `json:"status_url"`
	CancelURL string `json:"cancel_url"`
	EventsURL string `json:"events_url"`
	Replay    bool   `json:"idempotent_replay"`
}

func (s *Server) submitJob(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil {
		s.refuse(w, r, http.StatusBadRequest, "unreadable_body",
			"the request body could not be read: "+err.Error(), "")
		return
	}
	if len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"a submission body is bounded at "+strconv.Itoa(MaxBody)+" bytes",
			"typed input only; a job's bulk input arrives as a materialized tree")
		return
	}
	var sub JobSubmission
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	err = decoder.Decode(&sub)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
	}
	if err != io.EOF {
		detail := "multiple JSON values"
		if err != nil {
			detail = err.Error()
		}
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the submission is not one closed JSON object: "+detail, "")
		return
	}
	if sub.Package == "" || sub.Function == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"a job submission names a package and a job function",
			`{"package":"org/name","function":"census","input":{…}}`)
		return
	}
	// A job's trees name HOST DIRECTORIES that become read/write worker grants — the same
	// authority local_assets carry on /v1/requests (requests.go) — so they take the same
	// gate: a browser bearer must never name host paths (credentials.go).
	if len(sub.Trees) > 0 && !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"trees name host filesystem directories and require the OS-protected CLI credential",
			"use `cozy run --input-tree <ref>=<dir>`; this build exposes no browser tree-upload route")
		return
	}
	spec, e := s.resolveJob(sub)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.IdemKey = strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	digest, e := jobSubmissionDigest(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	spec.BodyDigest = digest

	// NO CAPACITY IS A STATE, never a refusal — this route cannot answer "busy". The
	// orchestrator records the row, queues it and makes the worker resident; several jobs
	// submitted at once queue against ONE worker and drain in submission order
	// (owner directive, decisions #394 / cr-019).
	jobID, attempt, fresh, e := s.orchestrator.SubmitDetail(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	row, e := s.store.RequestRow(jobID)
	if e != nil || row == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal",
			"the job was recorded and cannot be read back", "")
		return
	}
	handle := JobHandle{
		JobID: jobID, Status: contractStatus(row.State), Attempt: attempt,
		Package: row.Package, Function: row.Entrypoint,
		Repo:      home.ScratchRepo(row.Org, row.ID),
		StatusURL: "/v1/local/jobs/" + jobID,
		CancelURL: "/v1/local/jobs/" + jobID + "/cancel",
		EventsURL: "/v1/requests/" + jobID + "/events",
		Replay:    !fresh,
	}
	status := http.StatusAccepted
	if handle.Replay {
		status = http.StatusOK
	}
	s.ok(w, r, status, handle)
}

// resolveJob turns package+function into the orchestrator's Submission. The
// `job_descriptor_id` is resolved HERE, from the installed generation's own descriptor —
// a client never names a digest, exactly as it never names a binding plan id.
func (s *Server) resolveJob(sub JobSubmission) (orchestrator.Submission, *exit.Error) {
	out := orchestrator.Submission{
		Kind: "job", Package: sub.Package, Entrypoint: sub.Function,
		Payload: []byte(sub.Input), Org: strings.TrimSpace(sub.Org),
	}
	if len(out.Payload) == 0 {
		out.Payload = []byte("{}")
	}
	if out.Org == "" {
		out.Org = "local"
	}
	if e := validOrg(out.Org); e != nil {
		return out, e
	}
	if s.packages == nil {
		return out, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	var jobs []launch.JobFacts
	var e *exit.Error
	if sub.InstallID != "" {
		var spec orchestrator.WorkerLaunchSpec
		spec, e = s.packages.ResolveInstall(sub.InstallID)
		if e == nil && spec.Placement.Package != sub.Package {
			e = exit.Named(exit.Conflict, "install_package_mismatch",
				"install %s serves %s, not %s", sub.InstallID, spec.Placement.Package, sub.Package)
		}
		if e == nil {
			jobs, e = s.packages.JobsInstall(sub.InstallID)
			out.InstallID = sub.InstallID
		}
	} else {
		jobs, e = s.packages.Jobs(sub.Package)
	}
	if e != nil {
		return out, e
	}
	names := make([]string, 0, len(jobs))
	for _, job := range jobs {
		names = append(names, job.Name)
		if job.Name != sub.Function {
			continue
		}
		out.PlanID = job.DescriptorID
		out.Outputs = job.Outputs
		out.ArtifactOutputs = job.ArtifactOutputs
	}
	if out.PlanID == "" {
		return out, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", sub.Package, sub.Function).
			WithRemedy("it registers: %s", strings.Join(names, ", "))
	}
	for _, pair := range sub.Trees {
		ref, dir, ok := strings.Cut(pair, "=")
		if !ok || ref == "" || dir == "" {
			return out, exit.Usagef("%q is not ref=<directory>", pair)
		}
		abs, err := filepath.Abs(dir)
		if err != nil {
			return out, exit.Usagef("%q is not a resolvable directory: %s", dir, err)
		}
		out.Trees = append(out.Trees, ref+"="+abs)
	}
	return out, nil
}

// validOrg keeps the SCRATCH REPO's name spellable. The org is one path segment of
// `<org>/_job-<request-id>`, so a separator or a dot component in it would move the
// publication root — which is the escape the destination fence exists to refuse. Refusing
// the org outright is the stronger form: the bad name never reaches a path at all.
func validOrg(org string) *exit.Error {
	for _, c := range org {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-':
		default:
			return exit.Named(exit.Validation, "invalid_org",
				"%q is not an org name: %q is not [A-Za-z0-9-]", org, string(c)).
				WithRemedy("the org is one segment of the scratch repo <org>/_job-<id>")
		}
	}
	if org == "" {
		return exit.Named(exit.Validation, "invalid_org", "an org name is not empty")
	}
	return nil
}

func jobSubmissionDigest(spec orchestrator.Submission) (string, *exit.Error) {
	artifactOutputs := make([]canonical.Value, 0, len(spec.ArtifactOutputs))
	for _, output := range spec.ArtifactOutputs {
		artifactOutputs = append(artifactOutputs, map[string]canonical.Value{
			"max_bytes": int64(output.MaxBytes), "mime_type": output.MimeType,
			"output_id": output.OutputID,
		})
	}
	doc := map[string]canonical.Value{
		"kind":             "job",
		"package":          spec.Package,
		"function":         spec.Entrypoint,
		"plan_id":          spec.PlanID,
		"install_id":       spec.InstallID,
		"org":              spec.Org,
		"input":            base64.StdEncoding.EncodeToString(spec.Payload),
		"outputs":          strings.Join(spec.Outputs, ","),
		"artifact_outputs": artifactOutputs,
		"trees":            strings.Join(spec.Trees, ","),
	}
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize the job submission: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot digest the job submission: %s", err)
	}
	return spelled, nil
}

// ------------------------------------------------------------------------- status

// JobState is one job's document: the lifecycle a request has, plus the two facts only a
// job has — its publication and its running bill.
type JobState struct {
	JobID    string `json:"job_id"`
	Status   string `json:"status"`
	Package  string `json:"package"`
	Function string `json:"function"`
	Attempt  uint64 `json:"attempt"`
	Attempts int    `json:"attempts"`
	// Queued is the job's position in the dispatch queue while it waits for a worker,
	// counted from 1. Absent once it has an attempt — a running job is not queued.
	QueuePosition *int `json:"queue_position,omitempty"`
	// Requeues and RetryBudget are the orchestrator's RETRY PROJECTION made visible: how
	// much of the durable budget the neutral outcomes have already spent, and what the
	// bound is. A settlement that exhausted it names the budget in `error`.
	Requeues    int64           `json:"requeues"`
	RetryBudget int64           `json:"retry_budget"`
	Progress    map[string]any  `json:"progress,omitempty"`
	Stage       string          `json:"stage,omitempty"`
	ElapsedMS   int64           `json:"elapsed_ms"`
	Metrics     map[string]any  `json:"metrics,omitempty"`
	ErrorType   string          `json:"error_type,omitempty"`
	Error       string          `json:"error,omitempty"`
	Result      any             `json:"result,omitempty"`
	Outputs     []MediaRef      `json:"outputs"`
	Artifacts   []ArtifactRef   `json:"artifacts,omitempty"`
	Checkpoints []JobCheckpoint `json:"checkpoints,omitempty"`
	Publication *PublicationRef `json:"publication,omitempty"`
	// Bill is ABSENT unless this host was configured with an explicit local rate. There
	// is no `$0.00`: a fabricated zero is a claim about money nobody made (cl-004).
	Bill      *JobBill   `json:"bill,omitempty"`
	Triage    *TriageRef `json:"triage,omitempty"`
	CreatedAt string     `json:"created_at"`
	EventsURL string     `json:"events_url"`
}

// ArtifactRef is Cozy's durable scratch adoption projection. It exposes no path or
// TensorFS internals: the exact Runtime receipt digest and Cozy-derived private root id
// are the handles a later explicit promotion will consume.
type ArtifactRef struct {
	Attempt       int64  `json:"attempt"`
	OutputSlot    string `json:"output_slot"`
	Disposition   string `json:"disposition"`
	Outcome       string `json:"outcome,omitempty"`
	ReceiptDigest string `json:"artifact_receipt_digest,omitempty"`
	ScratchRootID string `json:"scratch_root_id,omitempty"`
}

// PublicationRef is the durable publication, as a client sees it. `root` is this host's
// own path and is a LOCAL-module fact: the durable location is the point of the thing,
// and a local caller is the one who owns the directory.
type PublicationRef struct {
	Repo    string `json:"repo"`
	Root    string `json:"root"`
	Status  string `json:"status"`
	Cause   string `json:"cause,omitempty"`
	Entries int64  `json:"entries"`
	Bytes   int64  `json:"bytes"`
	// CHECKPOINTS ARE ABSENT ON PURPOSE. A job that produces canonical bytes publishes
	// them through the runtime's own publication transaction, and this host will root and
	// project ONE typed receipt from that border when it exists (cr-009 seam). Guessing at
	// one from an author's result field would be a second publication protocol.
	CommittedAt string `json:"committed_at"`
}

// JobCheckpoint is one journaled checkpoint DECLARATION. Not a durable-save receipt: no
// component stores the bytes yet (#553a), and the attempt is part of the identity because
// two attempts of one request are two runs with two checkpoint sets.
type JobCheckpoint struct {
	Attempt       int64  `json:"attempt"`
	OperationKey  string `json:"operation_key"`
	LogicalKey    string `json:"logical_key"`
	ContentDigest string `json:"content_digest"`
	ReceiptID     string `json:"receipt_id"`
	Outcome       string `json:"outcome"`
}

// JobBill is the running cost, and it exists only where a rate does.
type JobBill struct {
	RateMicroUSDPerHour int64  `json:"rate_micro_usd_per_hour"`
	MicroUSD            int64  `json:"micro_usd"`
	Source              string `json:"source"`
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	s.ok(w, r, http.StatusOK, s.jobStateOf(row))
}

func (s *Server) jobRow(w http.ResponseWriter, r *http.Request) (records.Request, bool) {
	id := r.PathValue("id")
	row, e := s.store.RequestRow(id)
	if e != nil {
		s.refuseTyped(w, r, e)
		return records.Request{}, false
	}
	if row == nil || !row.IsJob() {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no job "+id+" on this host",
			"`cozy run list` lists the jobs this host recorded")
		return records.Request{}, false
	}
	return *row, true
}

func (s *Server) jobStateOf(row records.Request) JobState {
	state := JobState{
		JobID: row.ID, Status: contractStatus(row.State), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal),
		Requeues: row.Requeues, RetryBudget: orchestrator.MaxRequeues,
		Outputs: []MediaRef{}, CreatedAt: row.CreatedAt,
		EventsURL: "/v1/requests/" + row.ID + "/events",
	}
	if row.Ordinal == 0 && (row.State == "submitted" || row.State == "queued") {
		if n := s.orchestrator.QueuePosition(row.ID); n > 0 {
			state.QueuePosition = &n
		}
	}
	attempts, _ := s.store.Attempts(row.ID)
	state.Attempts = len(attempts)
	outs, _ := s.store.VisibleOutputs(row.ID)
	for _, o := range outs {
		state.Outputs = append(state.Outputs, MediaRef{
			OutputID: o.OutputID, MediaID: o.MediaID, URL: "/v1/media/" + o.MediaID,
			MimeType: o.MimeType, Length: o.Length, Digest: o.Digest,
		})
	}
	if rows, e := s.store.Checkpoints(row.ID); e == nil {
		for _, c := range rows {
			state.Checkpoints = append(state.Checkpoints, JobCheckpoint{
				Attempt: c.Attempt, OperationKey: c.OperationKey, LogicalKey: c.LogicalKey,
				ContentDigest: c.ContentDigest, ReceiptID: c.ReceiptID, Outcome: c.Outcome,
			})
		}
	}
	if p, e := s.store.PublicationOf(row.ID); e == nil && p != nil {
		state.Publication = &PublicationRef{
			Repo: p.Repo, Root: p.Root, Status: p.Status, Cause: p.Cause,
			Entries: p.Entries, Bytes: p.Bytes, CommittedAt: p.CommittedAt,
		}
	}
	if artifacts, e := s.store.ArtifactFinalizationsOf(row.ID); e == nil {
		for _, artifact := range artifacts {
			receiptDigest := artifact.ReceiptDigest
			if receiptDigest == "" {
				receiptDigest = artifact.ResultReceiptDigest
			}
			state.Artifacts = append(state.Artifacts, ArtifactRef{
				Attempt: artifact.Attempt, OutputSlot: artifact.OutputSlot,
				Disposition: artifact.Disposition, Outcome: artifact.ResultOutcome,
				ReceiptDigest: receiptDigest, ScratchRootID: artifact.ScratchRootID,
			})
		}
	}
	// THE ELAPSED CLOCK is the authority's own timestamps, and the LIVE progress is the
	// lossy lane's latest tick — replayed on connect, never durable, never load-bearing.
	state.ElapsedMS = elapsedMS(row, attempts)
	if frame, ok := s.orchestrator.LatestFrame(row.ID); ok {
		if value, ok := frame.Value.(map[string]any); ok {
			state.Progress = value
			if stage, ok := value["stage"].(string); ok {
				state.Stage = stage
			}
		}
	}
	// THE BILL. Absent unless a rate was configured — see JobBill. `$0.00` for a job
	// that cost real electricity is a fabricated fact, and this host does not make one.
	if rate := s.cfg.LocalRateMicroUSDPerHour; rate > 0 {
		state.Bill = &JobBill{
			RateMicroUSDPerHour: rate,
			MicroUSD:            rate * state.ElapsedMS / 3_600_000,
			Source:              "COZY_LOCAL_RATE_MICRO_USD_PER_HOUR",
		}
	}
	if len(attempts) == 0 {
		return state
	}
	last := attempts[len(attempts)-1]
	if last.TriageSubject != "" {
		state.Triage = &TriageRef{
			AttemptKey: last.AttemptKey, SubjectID: last.TriageSubject,
			URL:    "/v1/local/attempts/" + last.AttemptKey + "/triage",
			Length: last.TriageLength, Kept: last.TriagePath != "",
		}
	}
	if len(last.TerminalBody) == 0 {
		return state
	}
	doc, err := canonical.Read(last.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return state
	}
	metrics := doc.Sub("metrics")
	state.Metrics = map[string]any{
		"runtime_ms": metrics.Int("runtime_ms"), "queue_ms": metrics.Int("queue_ms"),
		"handler_ms": metrics.Int("handler_ms"),
	}
	if last.TerminalStatus != "SUCCEEDED" {
		state.ErrorType, state.Error = last.TerminalCause, last.SafeMessage
	}
	if inline := doc.Sub("result").Str("inline_result"); inline != "" {
		if decoded, err := base64.StdEncoding.DecodeString(inline); err == nil {
			var typed any
			if json.Unmarshal(decoded, &typed) == nil {
				state.Result = typed
			}
		}
	}
	return state
}

// parseStamp reads the authority's own RFC3339Nano timestamps. An unreadable one answers
// the zero time, which the caller renders as "no elapsed clock" rather than as zero
// milliseconds — the difference between "not known" and "instant".
func parseStamp(s string) time.Time {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Time{}
	}
	return t
}

func elapsedMS(row records.Request, attempts []records.Attempt) int64 {
	began := parseStamp(row.CreatedAt)
	if began.IsZero() {
		return 0
	}
	end := time.Now().UTC()
	if len(attempts) > 0 {
		last := attempts[len(attempts)-1]
		if last.ClosedAt != "" && (row.State == "succeeded" || row.State == "failed" ||
			row.State == "canceled" || row.State == "refused" || row.State == "abandoned") {
			if closed := parseStamp(last.ClosedAt); !closed.IsZero() {
				end = closed
			}
		}
	}
	return end.Sub(began).Milliseconds()
}

// ------------------------------------------------------------------------- cancel

func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if status := contractStatus(row.State); status == "completed" || status == "failed" || status == "canceled" {
		s.ok(w, r, http.StatusOK, s.jobStateOf(row))
		return
	}
	attempts, e := s.store.Attempts(row.ID)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if len(attempts) == 0 {
		// A QUEUED job has nothing running, and cancelling it is still a real act: it
		// leaves the queue and settles, so a client that asked never has to wonder.
		if e := s.orchestrator.CancelQueued(row.ID); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(row.ID)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued job was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.jobStateOf(*updated))
		return
	}
	last := attempts[len(attempts)-1]
	if last.State == "closed" || last.State == "dispatch_aborted" {
		if e := s.orchestrator.CancelQueued(row.ID); e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		updated, e := s.store.RequestRow(row.ID)
		if e != nil || updated == nil {
			s.refuse(w, r, http.StatusInternalServerError, "internal",
				"the queued job was canceled and cannot be read back", "")
			return
		}
		s.ok(w, r, http.StatusOK, s.jobStateOf(*updated))
		return
	}
	if last.State == "terminal" {
		s.refuse(w, r, http.StatusConflict, "terminal_ack_pending",
			"the current job attempt has a terminal whose retry/settlement projection is not acknowledged yet",
			"retry cancellation after the terminal ack")
		return
	}
	grace := orchestrator.ClientCancelGraceMS
	if v := r.URL.Query().Get("grace_ms"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 120000 {
			grace = n
		}
	}
	if e := s.orchestrator.CancelClient(row.ID, uint64(last.Attempt), grace); e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"job_id": row.ID, "attempt": last.Attempt, "status": "cancel_requested",
		"note": "the attempt's own journaled terminal settles it; watch the event stream",
	})
}
