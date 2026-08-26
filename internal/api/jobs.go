package api

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// THE JOB FAMILY (cl-004), mounted under /v1/local/ and deliberately NOT in the shared
// contract core. Two reasons, and both are about honesty rather than taste:
//
//   - cl-004 is LOCAL ONLY. The hub's job plane arrives with th-008, and a core route
//     that only one of the three hosts served would make `docs/client-contract.md` a
//     document about this host rather than about the contract.
//   - A job's typed input TREES are local directories the caller already owns. That is
//     the design (cr-009: a tree's path rides the field VALUE), and it is exactly the
//     kind of parameter the CORE must never take — a cloud host materializes trees from
//     digests instead. Keeping it local keeps the core's "no client-supplied path"
//     property intact.
//
// The EVENT plane is shared, and that is not an exception: a job IS a request row in the
// one lifecycle authority, so `GET /v1/requests/{id}/events` streams a job's lifecycle
// with no second event authority anywhere. `cozy job follow` is that route's client.

// JobSubmission is the job submit body.
type JobSubmission struct {
	Endpoint string          `json:"endpoint"`
	Function string          `json:"function"`
	Input    json.RawMessage `json:"input"`
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
	Endpoint  string `json:"endpoint"`
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
	if err := json.Unmarshal(body, &sub); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the submission is not a JSON object: "+err.Error(), "")
		return
	}
	if sub.Endpoint == "" || sub.Function == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"a job submission names an endpoint and a job function",
			`{"endpoint":"org/name","function":"census","input":{…}}`)
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
		Endpoint: row.Endpoint, Function: row.Entrypoint,
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

// resolveJob turns endpoint+function into the orchestrator's Submission. The
// `job_descriptor_id` is resolved HERE, from the installed generation's own descriptor —
// a client never names a digest, exactly as it never names a binding plan id.
func (s *Server) resolveJob(sub JobSubmission) (orchestrator.Submission, *exit.Error) {
	out := orchestrator.Submission{
		Kind: "job", Endpoint: sub.Endpoint, Entrypoint: sub.Function,
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
	if s.endpoints == nil {
		return out, exit.Unavailablef("this LocalService resolves no endpoints")
	}
	jobs, e := s.endpoints.Jobs(sub.Endpoint)
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
	}
	if out.PlanID == "" {
		return out, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", sub.Endpoint, sub.Function).
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
	doc := map[string]canonical.Value{
		"format":   "cozy.client.JobSubmission/1",
		"endpoint": spec.Endpoint,
		"function": spec.Entrypoint,
		"plan_id":  spec.PlanID,
		"org":      spec.Org,
		"input":    base64.StdEncoding.EncodeToString(spec.Payload),
		"outputs":  strings.Join(spec.Outputs, ","),
		"trees":    strings.Join(spec.Trees, ","),
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
	Endpoint string `json:"endpoint"`
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
	Checkpoints []JobCheckpoint `json:"checkpoints,omitempty"`
	Publication *PublicationRef `json:"publication,omitempty"`
	// Bill is ABSENT unless this host was configured with an explicit local rate. There
	// is no `$0.00`: a fabricated zero is a claim about money nobody made (cl-004).
	Bill      *JobBill   `json:"bill,omitempty"`
	Triage    *TriageRef `json:"triage,omitempty"`
	CreatedAt string     `json:"created_at"`
	EventsURL string     `json:"events_url"`
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

// JobCheckpoint is one journaled durable-save identity.
type JobCheckpoint struct {
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
			"`cozy job ls` lists the jobs this host recorded")
		return records.Request{}, false
	}
	return *row, true
}

func (s *Server) listJobs(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}
	state, e := storeState(strings.TrimSpace(r.URL.Query().Get("status")))
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	rows, err := s.store.RequestsOfKind("job", state, limit)
	if err != nil {
		s.refuseTyped(w, r, err)
		return
	}
	endpoint := strings.TrimSpace(r.URL.Query().Get("endpoint"))
	out := make([]JobState, 0, len(rows))
	counts := map[string]int{}
	for _, row := range rows {
		if endpoint != "" && row.Endpoint != endpoint {
			continue
		}
		state := s.jobStateOf(row)
		counts[state.Status]++
		out = append(out, state)
	}
	s.ok(w, r, http.StatusOK, map[string]any{
		"jobs": out, "count": len(out), "states": counts})
}

// storeState maps the contract's status vocabulary onto the authority's own names.
func storeState(status string) (string, *exit.Error) {
	switch status {
	case "", "any":
		return "", nil
	case "queued":
		return "submitted", nil
	case "in_progress":
		return "dispatching", nil
	case "completed":
		return "succeeded", nil
	case "failed", "canceled":
		return status, nil
	}
	return "", exit.Named(exit.Validation, "invalid_status", "unknown status filter %q", status).
		WithRemedy("any | queued | in_progress | completed | failed | canceled")
}

func (s *Server) jobStateOf(row records.Request) JobState {
	state := JobState{
		JobID: row.ID, Status: contractStatus(row.State), Endpoint: row.Endpoint,
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
				OperationKey: c.OperationKey, LogicalKey: c.LogicalKey,
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
	attempts, e := s.store.Attempts(row.ID)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if len(attempts) == 0 {
		// A QUEUED job has nothing running, and cancelling it is still a real act: it
		// leaves the queue and settles, so a client that asked never has to wonder.
		s.orchestrator.CancelQueued(row.ID)
		s.ok(w, r, http.StatusOK, s.jobStateOf(row))
		return
	}
	last := attempts[len(attempts)-1]
	if last.State == "terminal" || last.State == "closed" {
		s.ok(w, r, http.StatusOK, s.jobStateOf(row))
		return
	}
	grace := uint64(5000)
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
