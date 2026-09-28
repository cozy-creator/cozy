package orchestrator

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Submission is one local request. The orchestrator owns everything in it that decides
// WHAT runs; the runtime owns everything about HOW.
type Submission struct {
	// Hub is the Tensorhub origin the request belongs to; empty is the daemon's default.
	Hub                      string
	RequestID                string // server-reserved identity for request-owned input capture
	AllowPublish             []string
	MachineExecutionObserver bool
	TimeoutMS                int64
	DeadlineUnixMS           uint64
	IdemKey                  string // the caller's idempotency key
	Package                  string // org/name
	Entrypoint               string // the function
	PlanID                   string // the entrypoint_binding_plan_id this attempt binds
	Release                  string // immutable remote package release; empty for local execution
	// LocalInstallationID is the exact staged wheel-set identity for one editable rental.
	LocalInstallationID string
	Models              []ModelRef

	// Payload is the request body, verbatim. It rides the DeliveryGrant as the input
	// `payload` — a grant input, never a wire field, so refreshing the grant can never
	// substitute it.
	Payload []byte
	// Assets are immutable, daemon-staged files bound to exact payload field paths.
	// The request row keeps them so every requeue derives the same spec and grant.
	Assets []records.AssetBinding

	// Outputs is one destination per RESULT FIELD PATH (`image`, `detail.thumb`). Binding
	// by field path with exact set equality is what makes a two-output result
	// unswappable; a positional grant would silently cross them (decisions #248).
	Outputs []string
	// WeightsOutputs is the explicit Runtime-authored WeightsSink subset. Rev5's generic
	// OutputBinding carries no kind, so this is persisted beside the InvocationSpec and is
	// never inferred from an arriving receipt. The M0 lane is weights-only: when non-empty,
	// this set is the complete output set for the job.
	WeightsOutputs []WeightsOutput
	ProducerParams []string

	// BodyDigest is the caller's own digest of the WHOLE submission it is making
	// idempotent, not merely of the payload. cl-006 supplies the digest of
	// (package, function, input, outputs) so that one key naming a different PACKAGE
	// conflicts as loudly as one naming different input — a digest over the payload
	// alone would let a key be reused across functions and mean two different things.
	// Empty falls back to the payload's digest.
	BodyDigest string

	// Kind is the ATTEMPT CLASS: "" or `serving`, or `job`. A job carries two more facts
	// a serving request has no version of.
	Kind                string
	RetainWork          bool
	ReleaseImplicitWork bool
	RetryOf             string
	ChildReusable       bool
	ChildArtifacts      bool
	// Org is the publishing org whose scratch repo this job publishes into.
	Org string
	// Trees are the job's typed input TREES as `ref=dir`, one grant input each.
	Trees []string
	// NeedsAccelerator is derived from the selected package's immutable dependency facts.
	// It is the only machine-class decision retained on a request.
	NeedsAccelerator bool

	// Worker pins this request to an ATTACHED remote worker (a rental id resolved
	// through Options.Rentals). Empty = any local worker.
	Worker          string
	RequestedRental string
	// InstallID pins a durable request to one immutable local install resolution.
	// Remote requests instead carry their immutable Release.
	InstallID string
	// Rental authorizes placement on Creator-managed rented capacity.
	Rental bool
	// RentalRequired is the explicit development/E2E override that forbids local capacity.
	// It implies Rental and survives queue/restart scheduling in the request row.
	RentalRequired bool
	// RentNew requires an acquisition owned by this request, never an existing fleet rental.
	RentNew bool
	// PlannedSourceBytes is what a script ingest will pull; a rental bought for it is
	// sized to it.
	PlannedSourceBytes int64
	// OutputDirectory is the caller's explicit --out; empty means the package's store.
	// It is part of the submission's identity, where the derived intent below is not.
	OutputDirectory string
	// AttentionKernel is an optional developer execution-path pin. It affects only the
	// InvocationSpec and is intentionally excluded from placement and model resolution.
	AttentionKernel string
	// OutputExport is the derived publication obligation: the directory (explicit or
	// default) and the result-file contract. It changes no execution fact and is settled
	// independently after the terminal mirror.
	OutputExport *records.OutputExportIntent
	// ModelTransfer is a privileged materializer/finalizer attached to this
	// ordinary request. It changes no lifecycle, placement, attempt, or event fact.
	ModelTransfer *records.ModelTransferIntent
}

const WeightsManifestMime = "application/vnd.cozy.model-manifest"

// WeightsOutput is one bounded WeightsSink slot projected from the installed job
// descriptor. MaxBytes bounds only newly written table/config bytes, not inherited closure.
type WeightsOutput struct {
	OutputID string `json:"output_id"`
	MimeType string `json:"mime_type"`
	MaxBytes uint64 `json:"max_bytes"`
}

// Result is what one closed attempt produced.
type Result struct {
	RequestID  string
	Attempt    uint64
	AttemptKey string
	Status     string
	Cause      string
	Outputs    []records.Output
	Body       []byte // the TerminalBody document, exactly as it was digested
}

// Submit records the request and dispatches its FIRST attempt. It is idempotent in the
// strong sense: the same key with the same body answers with the recorded request and
// its current attempt, and never starts a second execution. Re-dispatch is the
// orchestrator's requeue PROJECTION over a terminal, never a client repeating itself.
func (c *Orchestrator) Submit(s Submission) (string, uint64, *exit.Error) {
	id, attempt, _, e := c.SubmitDetail(s)
	return id, attempt, e
}

// SubmitDetail is Submit plus the one fact an HTTP host must not guess: whether THIS
// call started the work. A client that retried a timed-out POST needs "202, I started it"
// and "200, this key was already yours" to be different answers, and inferring it from
// equal request ids is a race.
func (c *Orchestrator) SubmitDetail(s Submission) (string, uint64, bool, *exit.Error) {
	req, fresh, e := c.RecordSubmission(s)
	if e != nil {
		return "", 0, false, e
	}
	if !fresh {
		return req.ID, uint64(req.Ordinal), false, nil
	}
	attempt, e := c.activateRecorded(req)
	return req.ID, attempt, true, e
}

// ActivateRecordedRequest is the second half for callers that transactionally
// attach request sidecars before ordinary queue activation.
func (c *Orchestrator) ActivateRecordedRequest(req records.Request) (uint64, *exit.Error) {
	return c.activateRecorded(req)
}

// RecordSubmission crosses the durable ordinary-request boundary.
func (c *Orchestrator) RecordSubmission(s Submission) (records.Request, bool, *exit.Error) {
	req, event, e := requestRecord(s)
	if e != nil {
		return records.Request{}, false, e
	}
	if req.Hub == "" {
		req.Hub = c.opt.Cfg.HubURL
	}
	req, fresh, e := c.opt.Store.SubmitWithEvent(req, event)
	if e != nil {
		return records.Request{}, false, e
	}
	if !fresh {
		c.logRecordedReplay(req, s.IdemKey)
		return req, false, nil
	}
	return req, true, nil
}

func requestRecord(s Submission) (records.Request, map[string]any, *exit.Error) {
	if s.RentNew {
		if s.RequestedRental != "" || s.RetryOf != "" {
			return records.Request{}, nil, exit.Usagef("a fresh rental cannot reuse a selected machine or retained run")
		}
		s.RentalRequired = true
	}
	if s.RequestedRental != "" {
		if s.Worker != "" && s.Worker != s.RequestedRental {
			return records.Request{}, nil, exit.New(exit.Conflict, "assigned rental differs from requested rental")
		}
		s.RentalRequired = true
	}
	if s.RentalRequired {
		s.Rental = true
	}
	if s.RetainWork && s.Kind != "job" {
		return records.Request{}, nil, exit.Named(exit.Validation, "retain_work_not_job",
			"retained work is an ordinary unpublished package job capability")
	}
	weightsOutputs, weightsBytes, e := normalizeWeightsOutputs(s)
	if e != nil {
		return records.Request{}, nil, e
	}
	s.WeightsOutputs = weightsOutputs
	bodyDigest := s.BodyDigest
	if bodyDigest == "" {
		identity := s.Payload
		if s.Rental {
			document := map[string]canonical.Value{
				"payload": base64.StdEncoding.EncodeToString(s.Payload),
				"rental":  true, "rental_required": s.RentalRequired,
			}
			if s.RentNew {
				document["rent_new"] = true
			}
			if s.RequestedRental != "" {
				document["requested_rental"] = s.RequestedRental
			}
			encoded, err := canonical.Write(document)
			if err != nil {
				return records.Request{}, nil, exit.Internalf("cannot encode request budget identity: %s", err)
			}
			identity = encoded
		}
		spelled, err := canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot digest the request body: %s", err)
		}
		bodyDigest = spelled
	}
	if s.AttentionKernel != "" {
		identity, err := canonical.Write(map[string]canonical.Value{
			"body_digest":      bodyDigest,
			"attention_kernel": s.AttentionKernel,
		})
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot encode pinned request identity: %s", err)
		}
		bodyDigest, err = canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot digest pinned request identity: %s", err)
		}
	}
	if s.LocalInstallationID != "" {
		if s.InstallID == "" || s.LocalInstallationID == "" {
			return records.Request{}, nil, exit.Named(exit.Structural,
				"local_package_request_invalid",
				"a local package request requires its editable installation")
		}
		identity, err := canonical.Write(map[string]canonical.Value{
			"body_digest":           bodyDigest,
			"local_installation_id": s.LocalInstallationID,
		})
		if err != nil {
			return records.Request{}, nil, exit.Internalf(
				"cannot encode the local package request identity: %s", err)
		}
		bodyDigest, err = canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf(
				"cannot digest the local package request identity: %s", err)
		}
	}
	if s.Kind == "job" && hasModelAdapters(s.Models) {
		return records.Request{}, nil, exit.Usagef("model adapters apply only to serving requests")
	}
	id := s.RequestID
	if id == "" {
		id = records.NewID("req")
		if s.Kind == "job" {
			id = records.NewID("job")
		}
	}
	req := records.Request{
		MachineExecutionObserver: s.MachineExecutionObserver,
		DeadlineUnixMS:           s.DeadlineUnixMS,
		ID:                       id, IdemKey: s.IdemKey, BodyDigest: bodyDigest, Hub: s.Hub,
		Package: s.Package, Entrypoint: s.Entrypoint, PlanID: s.PlanID, Payload: s.Payload,
		Release:             s.Release,
		LocalInstallationID: s.LocalInstallationID,
		Outputs:             strings.Join(s.Outputs, ","),
		Assets:              s.Assets, WeightsOutputs: string(weightsBytes),
		Kind: s.Kind, RetainWork: s.RetainWork, ReleaseImplicitWork: s.ReleaseImplicitWork, RetryOf: s.RetryOf, ChildArtifacts: s.ChildArtifacts, NeedsAccelerator: s.NeedsAccelerator, Org: s.Org, Trees: strings.Join(s.Trees, ","),
		RequestedRental: s.RequestedRental,
		AttentionKernel: s.AttentionKernel,
		Worker:          s.Worker, InstallID: s.InstallID, Rental: s.Rental,
		RentalRequired: s.RentalRequired, RentNew: s.RentNew, Models: s.Models,
		OutputExport: s.OutputExport, ModelTransfer: s.ModelTransfer, PlannedSourceBytes: s.PlannedSourceBytes,
	}
	event := map[string]any{
		"retain_work": s.RetainWork,
		"package":     s.Package, "function": s.Entrypoint,
		"body_digest": bodyDigest, "plan_id": s.PlanID, "outputs": s.Outputs,
		"weights_outputs": weightsOutputs,
	}
	if s.TimeoutMS > 0 {
		event["timeout_ms"] = s.TimeoutMS
		event["deadline_unix_ms"] = s.DeadlineUnixMS
	}
	if len(s.AllowPublish) > 0 {
		event["allow_publish"] = s.AllowPublish
	}
	if s.AttentionKernel != "" {
		event["attention_kernel"] = s.AttentionKernel
	}
	if s.Rental {
		event["rental"] = true
		event["rental_required"] = s.RentalRequired
		event["rent_new"] = s.RentNew
		event["release"] = s.Release
		if s.LocalInstallationID != "" {
			event["local_installation_id"] = s.LocalInstallationID
		}
	}
	return req, event, nil
}

func normalizeWeightsOutputs(s Submission) ([]WeightsOutput, []byte, *exit.Error) {
	if len(s.WeightsOutputs) == 0 {
		return nil, []byte("[]"), nil
	}
	if s.Kind != "job" {
		return nil, nil, exit.Named(exit.Validation, "weights_output_not_job",
			"weights outputs are valid only on a job submission")
	}
	if len(s.WeightsOutputs) > pb.MaxWeightsReceipts {
		return nil, nil, exit.Named(exit.Validation, "weights_output_count_cap",
			"%d weights outputs exceeds the protocol cap of %d",
			len(s.WeightsOutputs), pb.MaxWeightsReceipts)
	}
	rows := append([]WeightsOutput(nil), s.WeightsOutputs...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].OutputID < rows[j].OutputID })
	ids := make(map[string]bool, len(rows))
	for _, row := range rows {
		if row.OutputID == "" || ids[row.OutputID] {
			return nil, nil, exit.Named(exit.Validation, "weights_output_identity",
				"weights output slots are non-empty and unique; %q is repeated or empty", row.OutputID)
		}
		if row.MimeType != WeightsManifestMime || row.MaxBytes > (uint64(1)<<53)-1 {
			return nil, nil, exit.Named(exit.Validation, "weights_output_contract",
				"weights output %s must declare MIME %s and a new-byte cap in 0..2^53-1",
				row.OutputID, WeightsManifestMime)
		}
		ids[row.OutputID] = true
	}
	// OutputBinding/1 has no kind. Until that schema gap is closed, an WeightsSink job is
	// weights-only so a missing receipt can be classified without guessing about asset slots.
	if len(ids) != len(s.Outputs) {
		return nil, nil, exit.Named(exit.Structural, "mixed_job_output_kinds",
			"this rev5 lane requires weights-only jobs; %d weights slots do not close %d outputs",
			len(ids), len(s.Outputs))
	}
	for _, id := range s.Outputs {
		if !ids[id] {
			return nil, nil, exit.Named(exit.Structural, "mixed_job_output_kinds",
				"job output %q is not in the explicit weights-output set", id)
		}
	}
	data, err := json.Marshal(rows)
	if err != nil {
		return nil, nil, exit.Internalf("cannot persist weights output declarations: %s", err)
	}
	return rows, data, nil
}

func decodeWeightsOutputs(data string) ([]WeightsOutput, *exit.Error) {
	if data == "" {
		return nil, nil
	}
	var rows []WeightsOutput
	if err := json.Unmarshal([]byte(data), &rows); err != nil {
		return nil, exit.Internalf("cannot decode persisted weights output declarations: %s", err)
	}
	return rows, nil
}

func (c *Orchestrator) logRecordedReplay(req records.Request, idempotencyKey string) {
	// The recorded answer. A settled request is settled; a live one is already
	// running the attempt this call would otherwise duplicate.
	c.logf("request %s is the recorded answer for idempotency key %s (state %s, attempt %d)",
		req.ID, idempotencyKey, req.State, req.Ordinal)
}

func (c *Orchestrator) activateRecorded(req records.Request) (uint64, *exit.Error) {
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil {
		return 0, problem
	}
	if link, problem := c.opt.Store.MachineExecution(req.ID); problem != nil {
		return 0, problem
	} else if link != nil {
		if c.opt.StartMachineExecution == nil {
			return 0, exit.Unavailablef("this client cannot reconnect its Runtime-owned execution")
		}
		return 0, c.startMachineExecution(*current)
	}
	if current.State != "submitted" && current.State != "queued" {
		return uint64(current.Ordinal), nil
	}
	req = *current
	if req.ModelTransfer != nil && req.Package == "cozy/platform" &&
		req.Entrypoint == "model-pass-through" {
		// Narrow attempt-zero exception: an unchanged verified Manifest has no code
		// to execute and no bytes to reproduce. The boundary hooks settle the same
		// ordinary request/events/watch/cancel surface; producer transfers never enter here.
		go c.runModelPassThrough(req)
		return 0, nil
	}
	// A SUBMISSION NEVER OVERTAKES WORK ALREADY WAITING. Dispatching straight from submit
	// is what keeps a warm request fast, and it is exactly what breaks FIFO when a queue
	// exists: a request arriving while six are parked would take the free slot the head of
	// the queue is waiting for. Found live by cl-004's depth pass — the LAST of six
	// submissions settled FIRST. So: with a queue, join it; the drain below still runs
	// immediately, so the head goes out now rather than at the next Report (cr-019).
	if c.queueDepth() > 0 {
		if !c.enqueue(req.ID) {
			return 0, nil
		}
		position, depth := c.QueueState(req.ID)
		c.emit(req.ID, "request.queued", 0, waitFacts{cause: WaitQueueAhead}.decorate(map[string]any{
			"reason":   "the dispatch queue is not empty; this request joins it in submission order",
			"position": position, "depth": depth,
		}, req))
		c.logf("%s QUEUED behind %d waiting request(s)", req.ID, c.QueuePosition(req.ID)-1)
		c.selectOrStart(req)
		go c.drain()
		return 0, nil
	}
	attempt, e := c.dispatch(req)
	if e != nil {
		// NO CAPACITY is a STATE, not a failure (cl-006): the request row is already
		// durable, so refusing it here would mean the client holds an id for something
		// that never runs. It waits for capacity exactly as a requeue does — one queue,
		// one projection. Every OTHER refusal (an open recovered obligation, a live
		// attempt) is a real conflict and still refuses.
		if e.Code != exit.Unavailable {
			c.failQueued(req.ID, e, "")
			return 0, e
		}
		if !c.enqueue(req.ID) {
			return 0, nil
		}
		c.emit(req.ID, "request.queued", 0,
			c.waitOf(req).decorate(map[string]any{"reason": e.Message}, req))
		c.logf("%s QUEUED for capacity: %s", req.ID, e.Message)
		c.selectOrStart(req)
		go c.drain()
		return 0, nil
	}
	return attempt, nil
}

// RequeueForCapacity returns a request a worker could not admit right now to the queue.
// It PARKS there: only a worker reporting capacity re-dispatches it, so the wait is driven
// by an observation rather than a cadence, and it spends nothing — the worker has not
// attempted anything. The local path draws the same line: `device_envelope_held` keeps the
// request queued until a later idle-capacity report re-enters select-or-start.
func (c *Orchestrator) RequeueForCapacity(requestID, why string) {
	started, e := c.opt.Store.BeginRequeue(requestID)
	if e != nil {
		c.logf("%s NOT requeued (%s): %s", requestID, why, e.Message)
		return
	}
	if !started {
		return
	}
	req, e := c.opt.Store.RequestRow(requestID)
	if e != nil || req == nil {
		return
	}
	// Announced before dispatch, which may or may not find capacity, so a client can tell
	// a first dispatch from a re-offer.
	c.emit(requestID, "request.requeued", 0, map[string]any{"cause": why})
	attempt, e := c.dispatch(*req)
	if e != nil {
		// No capacity yet: the request WAITS, and dispatch resumes the moment a worker
		// reports the binding ready.
		c.enqueue(requestID)
		c.emit(requestID, "request.queued", 0,
			c.waitOf(*req).decorate(map[string]any{"reason": e.Message}, *req))
		c.logf("%s requeued and QUEUED for capacity: %s", requestID, e.Message)
		c.selectOrStart(*req)
		return
	}
	c.logf("%s requeued as attempt %d (cause %s)", requestID, attempt, why)
}

// selectOrStart makes a queued request's package resident. It is the half of `cozy run`
// that "cold and warm traverse the same states" rests on: SELECT the worker that already
// advertises the binding, or START one — never a second invocation mechanism, and never a
// client's job. `cozy run` is the same act made explicit for prewarming.
//
// It runs off the caller's goroutine because a cold start is a 4.782 GiB fill, and the
// submitting client is already watching the event stream that will say when it lands. The
// dispatch itself is still `drain`'s, triggered by the worker reporting READY: this
// function never dispatches, so there is exactly one placement path.
//
// A worker that cannot become dispatchable is a TERMINAL condition for the request, not a
// longer wait. A request that queues forever behind a worker that died on boot is the
// worst of both: no output and no answer.
// startMachineExecution admits the async observer before releasing the closing
// fence. A delayed activation after HTTP admission may not cross a later down.
func (c *Orchestrator) startMachineExecution(req records.Request) *exit.Error {
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "daemon.closing", "the daemon is closing; retained work will reconnect on startup")
	}
	token := "machine/" + req.ID
	if c.starting[token] {
		c.mu.Unlock()
		return nil
	}
	c.starting[token] = true
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.starting, token); c.mu.Unlock() }()
	return c.opt.StartMachineExecution(req)
}

func (c *Orchestrator) selectOrStart(req records.Request) {
	c.prepare(req)
}

// prepare is select-or-start's body: a machine execution starts on its machine.
func (c *Orchestrator) prepare(req records.Request) bool {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return false
	}
	if link, problem := c.opt.Store.MachineExecution(req.ID); problem != nil || link != nil {
		if problem == nil && c.opt.StartMachineExecution != nil {
			_ = c.startMachineExecution(req)
		}
		return false
	}
	// Every request runs as a machine execution; one without its link was accepted for the
	// retired classic worker and ends here, named.
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil || (current.State != "submitted" && current.State != "queued") {
		return false
	}
	c.failPreparation(*current, ClassicRetired(), "")
	return false
}

// AwaitRental says why a queued request waits while the fleet buys its machine.
func (c *Orchestrator) AwaitRental(requestID, machine string) {
	req, problem := c.opt.Store.RequestRow(requestID)
	position := c.QueuePosition(requestID)
	if problem != nil || req == nil || position == 0 {
		return
	}
	c.park(*req, position-1, waitFacts{cause: WaitRental, on: machine},
		"waiting for rental "+machine+" to become ready")
}

// requestSlot names what a request needs resident before assignment. Explicit
// rental affinity already identifies an independent machine at this point.
func requestSlot(req records.Request) string {
	slot := pinnedPackage(req.Package, req.RequestedRental)
	if req.RentNew {
		slot += "/fresh/" + req.ID
	}
	if req.InstallID != "" {
		slot += "/install/" + req.InstallID
	}
	if req.IsJob() {
		slot += "/job/" + req.Entrypoint
	}
	return slot
}

// LogPlacement is the decision log for the capacity half: no worker held the placement,
// so the fleet placed the run on an attached rental or a bought pod (cl-165), and this
// request is pinned there until its placement reports.
func (c *Orchestrator) LogPlacement(req records.Request, decision PlacementDecision) {
	c.logf("%s: %s; candidates: %s", req.ID, decision.Line(), decision.verdicts())
	c.emit(req.ID, "request.placement", 0, decision.payload())
}

// staged answers whether this worker was launched with the given plan id staged for it.
// It reads what the LAUNCHER wrote, not what the worker has got around to advertising: a
// worker still filling is capacity, a worker holding a different digest is not.
func staged(w *worker, planID string) bool {
	for _, id := range w.planIDs {
		if id == planID {
			return true
		}
	}
	return false
}

func stagedFor(w *worker, req records.Request) bool {
	if req.IsJob() && w.spec.Connection != nil {
		// Exact job inputs include their count; an unknown older selection
		// cannot satisfy a requested model through the serving wildcard.
		return staged(w, req.PlanID) && exactJobSelection(w.spec.Placement, req)
	}
	return staged(w, req.PlanID) && selectionServes(req.Models, w.spec.Placement.Models)
}

func exactJobSelection(placement DesiredPlacement, req records.Request) bool {
	if req.InstallID != "" && placement.InstallID != req.InstallID ||
		req.LocalInstallationID != "" && placement.InstallationID != req.LocalInstallationID ||
		req.Release != "" && placement.Release != req.Release || len(placement.Models) != len(req.Models) {
		return false
	}
	for _, requested := range req.Models {
		found := false
		for _, held := range placement.Models {
			if held.Slot == requested.Slot && held.Manifest == requested.Manifest && held.ManifestLength == requested.ManifestLength {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// selectionServes is the model half of the match (cl-114): a placement that holds a slot
// the request binds must hold it under the request's exact manifest. The plan id hashes
// the entrypoint's interface, not its weights, so two selections of one package share a
// plan — matching on the plan alone dispatched an fp8 request onto the warm bf16
// placement. A request that binds no models (an editable install's frozen selection, an
// unmodeled package) accepts whatever the placement holds, and a placement whose set
// binds no row for a slot has no selection to disagree with — the request's model rows
// are then facts for the row, not residency evidence.
func selectionServes(requested, held []ModelRef) bool {
	if len(requested) == 0 || len(held) == 0 {
		return !hasModelAdapters(requested)
	}
	holds := make(map[string]ModelRef, len(held))
	for _, m := range held {
		holds[m.BindingSlot()] = m
	}
	for _, m := range requested {
		current, ok := holds[m.BindingSlot()]
		if !ok {
			if len(m.Adapters) > 0 {
				return false
			}
			continue
		}
		if !records.SameAdapters(m.Adapters, current.Adapters) {
			return false
		}
		if _, fits := rungHolding(m, current.Manifest); !fits {
			return false
		}
	}
	return true
}

// Readiness and the final offer both require facts from the worker's actual
// PlacementSet. A protocol minor or a copied request is not feature support.
func requireAdapterEcho(requested, observed []ModelRef) *exit.Error {
	if (hasModelAdapters(requested) || hasModelAdapters(observed)) && !selectionServes(requested, observed) {
		return exit.Named(exit.Structural, "model_adapters_preparation_mismatch",
			"worker preparation omitted or changed the requested LoRA stack")
	}
	return nil
}

func requireAdapterPlacementEcho(requested []ModelRef, placement DesiredPlacement) *exit.Error {
	doc, err := canonical.Read(placement.PlacementSetBytes, &pb.PlacementSet{})
	if err != nil {
		return exit.Named(exit.Structural, "model_adapters_preparation_mismatch", "prepared model bindings are not readable")
	}
	for _, row := range doc.List("placements") {
		if row.Str("bindings_digest") == placement.BindingsDigest &&
			(placement.PlacementIDValue == "" || row.Str("placement_id") == placement.PlacementIDValue) {
			return requireAdapterEcho(requested, placementModels(placement.Package, row))
		}
	}
	return exit.Named(exit.Structural, "model_adapters_preparation_mismatch", "prepared model binding identity is absent")
}

// rungHolding answers whether a held manifest is one the request's ref accepts: its own
// pin, or — unpinned — any rung of its ladder (cl-166). The rung is what a dispatch onto
// that placement pins the request to.
func rungHolding(m ModelRef, manifest string) (records.ModelRung, bool) {
	if m.Pinned() {
		return records.ModelRung{GPUs: m.GPUs, Lane: m.Lane, Manifest: m.Manifest, Bytes: m.Bytes}, m.Manifest == manifest
	}
	for _, rung := range m.Ladder {
		if rung.Manifest == manifest {
			return rung, true
		}
	}
	return records.ModelRung{}, false
}

// settledState answers whether the authority has already recorded this request's outcome.
func settledState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// failQueued settles a request that can never be placed. It is a request-level terminal:
// no offer crossed to a worker, so there is no worker terminal to replay and the request
// row is what settles. A closed dispatch_aborted row may remain as preparation history.
func (c *Orchestrator) failQueued(requestID string, cause *exit.Error, workerToStop string) {
	c.failQueuedSelection(requestID, nil, cause, workerToStop)
}

func (c *Orchestrator) failPreparation(expected records.Request, cause *exit.Error, workerToStop string) {
	c.failQueuedSelection(expected.ID, &expected, cause, workerToStop)
}

func (c *Orchestrator) failQueuedSelection(requestID string, expected *records.Request, cause *exit.Error, workerToStop string) {
	payload := records.QueuedFailure(cause)
	// The preparation waiter can outlive an accepted attempt or its successful
	// finalization. The store is the sole authority to fail queued work; neither
	// cleanup nor worker/provider teardown may run before that transaction wins.
	var applied bool
	var problem *exit.Error
	if expected == nil {
		applied, problem = c.opt.Store.FailQueuedRequest(requestID, payload)
	} else {
		applied, problem = c.opt.Store.FailQueuedPreparation(*expected, payload)
	}
	if problem != nil {
		if problem.Code == exit.Conflict {
			c.logf("%s preparation failure no longer applies: %s", requestID, problem.Message)
			return
		}
		c.logf("%s could not be settled: %s", requestID, problem.Message)
		time.AfterFunc(2*time.Second, func() { c.failQueuedSelection(requestID, expected, cause, workerToStop) })
		return
	}
	if !applied {
		c.logf("%s preparation failure no longer applies: %s", requestID, cause.Message)
		return
	}
	c.forget(requestID)
	if row, problem := c.opt.Store.RequestRow(requestID); problem == nil && row != nil && row.RetainWork {
		c.logf("%s BLOCKED (%s): %s", requestID, cause.ErrName(), cause.Message)
		c.signalClosed(requestWaitKey(requestID), cause)
		return
	}
	// A local worker that cannot make this binding resident would hold its device grant
	// forever. A rental's worker is the shared control lane for every package on that
	// machine; one request's preparation failure never detaches it.
	if workerToStop != "" {
		c.mu.Lock()
		w := c.workers[workerToStop]
		shared := w != nil && w.spec.Connection != nil
		c.mu.Unlock()
		if !shared {
			c.ShutdownWorker(workerToStop, StopGrace)
		}
	}
	// An output obligation exists before attempt one. Settle it as skipped so
	// the caller does not wait for bytes a failed preparation cannot produce.
	c.RetryOutputExport(requestID)
	if row, problem := c.opt.Store.RequestRow(requestID); problem == nil && row != nil {
		go c.cleanupRequestAssets(*row)
		if row.ModelTransfer != nil {
			c.forgetTransferProgress(requestID)
		}
		c.releaseManaged(*row)
	}
	c.logf("%s FAILED before any offer: %s", requestID, cause.Message)
	c.signalClosed(requestWaitKey(requestID), cause)
}

func (c *Orchestrator) releaseManaged(req records.Request) {
	if !req.Rental || req.Worker == "" || c.opt.ReleaseManagedRental == nil {
		return
	}
	go func() {
		if problem := c.releaseManagedNow(req); problem != nil {
			c.logf("rental %s release deferred: %s", req.Worker, problem.Message)
		}
	}()
}

func (c *Orchestrator) releaseManagedNow(req records.Request) *exit.Error {
	if !req.Rental || req.Worker == "" || c.opt.ReleaseManagedRental == nil {
		return nil
	}
	line, problem := c.opt.ReleaseManagedRental(req.Worker)
	if problem != nil {
		return problem
	}
	if line != "" {
		c.emit(req.ID, "request.rentals", 0, map[string]any{"line": line})
		c.logf("%s", line)
	}
	return nil
}

func (c *Orchestrator) dispatch(req records.Request) (uint64, *exit.Error) {
	if link, problem := c.opt.Store.MachineExecution(req.ID); problem != nil {
		return 0, problem
	} else if link != nil {
		return 0, exit.New(exit.Conflict, "Runtime-owned execution cannot create a local attempt")
	}
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return 0, problem
	}
	if current == nil || (current.State != "submitted" && current.State != "queued") {
		return 0, exit.Named(exit.Conflict, "request.execution_stopped", "request %s is not queued for execution", req.ID)
	}
	req = *current

	// A warm worker must not bypass the same immutable-capture check used by
	// preparation. Replays of already terminal/live attempts never enter here.
	if (req.Rental || req.Worker != "") && req.InstallID != "" && req.LocalInstallationID != "" {
		if c.opt.Packages == nil {
			return 0, exit.Unavailablef("captured package owner is unavailable")
		}
		if problem := c.opt.Packages.ValidateExecutionCapture(req); problem != nil {
			return 0, problem
		}
	}

	// PLACEMENT is the orchestrator's: the caller names the binding, and dispatch picks a
	// worker whose placement advertises it as DISPATCHABLE now and whose admission fence
	// is open. `pick` also returns the admission epoch it OBSERVED, which is what
	// makes a stale offer refuse deterministically rather than race.
	target, e := c.pick(req)
	if e != nil {
		return 0, e
	}
	w, sess, admissionEpoch, reservation := target.worker, target.sess, target.admissionEpoch, target.reservation
	reserved := true
	defer func() {
		if reserved {
			c.releaseDispatch(reservation)
		}
	}()
	if hit, problem := c.lookupOperationOn(req, sess); hit || problem != nil {
		return 0, problem
	}
	current, problem = c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return 0, problem
	}
	if current == nil || (current.State != "submitted" && current.State != "queued") {
		return 0, exit.Named(exit.Conflict, "request.execution_stopped", "request stopped while its operation lookup was in progress")
	}
	if req.ParentRequestID != "" {
		if problem := c.retainChildInputs(req); problem != nil {
			return 0, problem
		}
	}
	if w.spec.Connection != nil && req.Worker == "" {
		// THE PIN IS ROUTING'S OUTPUT (cl-092 step 4): the argmin was a rental, so the
		// request is pinned to it now — durably, before its identity is bound to that
		// rental's placement — and everything below reads the pinned request. A request
		// that settled first has no worker to pin; it is not dispatched.
		rentalID := w.spec.Connection.RentalID
		pinned, e := c.opt.Store.PinRental(req.ID, rentalID, nil)
		if e != nil {
			return 0, e
		}
		if !pinned {
			return 0, exit.New(exit.Conflict, "request %s settled before it could be pinned to rental %s",
				req.ID, rentalID)
		}
		req.Worker = rentalID
		target.routed.pinned = rentalID
	}
	if req.ModelTransfer != nil && req.Rental && w.spec.Connection == nil && req.Release != "" &&
		w.spec.Placement.Release != req.Release {
		return 0, exit.Unavailablef("ready local producer does not exactly match frozen %s@%s",
			req.Package, req.Release)
	}
	// A model transfer materializes its verified model inputs only after ordinary
	// placement selected the exact worker, but before InvocationSpec identity is minted.
	req, e = c.materializeModelTransfer(req, w)
	if e != nil {
		return 0, e
	}

	// The InvocationSpec DOCUMENT (#439): everything that gives the invocation meaning —
	// the payload digest, the ORDERED input identities, the output contracts, the
	// deadline — lives INSIDE the digest. Its key set is closed: no human model ref, no
	// service class, no local extension has a slot.
	installationID, e := c.invocationIdentity(w, req)
	if e != nil {
		return 0, e
	}
	weightsOutputs, e := decodeWeightsOutputs(req.WeightsOutputs)
	if e != nil {
		return 0, e
	}
	payloadDigest := spellOf(canonical.Digest(req.Payload))
	outputLimit := c.maxOutputBytes()
	var servingPlacement DesiredPlacement
	if !req.IsJob() {
		c.mu.Lock()
		servingPlacement = w.spec.Placement
		c.mu.Unlock()
		if servingPlacement.BindingsDigest == "" || len(servingPlacement.PlacementSetBytes) == 0 {
			return 0, exit.Named(exit.Conflict, "serving.placement_evidence_absent", "serving dispatch needs the exact prepared model bindings")
		}
	}
	if !req.IsJob() {
		if problem := requireAdapterPlacementEcho(req.Models, servingPlacement); problem != nil {
			return 0, problem
		}
	}
	spec := &pb.InvocationSpec{
		// `image_digest` is GONE, renamed to what it always meant (#483): "image" is wrong
		// for a native install with no OCI image at all. The value is the same one this
		// daemon was frozen with — a request cannot choose the environment it runs under.
		InstallationId:  installationID,
		PayloadDigest:   payloadDigest,
		Inputs:          inputBindings(req, payloadDigest),
		Outputs:         invocationOutputBindings(splitList(req.Outputs), weightsOutputs, outputLimit),
		AttentionKernel: req.AttentionKernel,
		Spec: &pb.InvocationSpec_Serving{Serving: &pb.ServingInvocationSpec{
			EntrypointBindingDigest: req.PlanID,
			BindingsDigest:          servingPlacement.BindingsDigest,
			// With no adapters the binding IS the plan, so the two ids are equal by
			// construction rather than by copying a value around.
			AttemptBindingId: req.PlanID,
		}},
	}
	if req.IsJob() {
		if !w.spec.IsJob() {
			return 0, exit.Internalf(
				"worker %s holds no job plan to dispatch job request %s against", w.instanceID, req.ID)
		}
		// ONE mode names ONE spec. The per-attempt publication contract names THIS
		// request's scratch repo, which is why a queue-serving worker can hold one
		// directive and still publish each attempt into its own place.
		spec.Spec = &pb.InvocationSpec_Job{Job: &pb.JobInvocationSpec{
			InstallationId:  w.spec.Placement.Jobs[0].InstallationID,
			JobDescriptorId: req.PlanID,
			PublicationContract: &pb.PublicationContract{
				GrantId: home.ScratchRepo(req.Org, req.ID),
				Outputs: invocationOutputBindings(splitList(req.Outputs), weightsOutputs, outputLimit),
			},
		}}
	}
	if req.Capture != "" {
		var capture pb.ActivationCapture
		if err := json.Unmarshal([]byte(req.Capture), &capture); err != nil {
			return 0, exit.New(exit.Validation, "recorded capture options are invalid")
		}
		spec.Capture = &capture
	}
	canonicalBytes, digest, err := canonical.Identity(spec)
	if err != nil {
		return 0, exit.Internalf("cannot mint the InvocationSpec document: %s", err)
	}
	spelled, _ := canonical.Spell(digest)

	// THE LAW AND THE ORDINAL, in ONE transaction that also journals the assignment: a
	// terminal crossing a restart is authorized by the persisted assignment, and an ordinal
	// that is minted outside the write that records it is a race (cl-003's finding — two
	// `drain()` goroutines minted 1 and 2 for one request across the old split call).
	ordinal, e := c.opt.Store.Dispatch(records.Attempt{
		RequestID: req.ID, InstanceID: w.instanceID,
		SessionID: w.bootID, InvocationDigest: spelled, InvocationCanonical: canonicalBytes,
		WeightsOutputs:      req.WeightsOutputs,
		ServingPlacementSet: servingPlacement.PlacementSetBytes,
	})
	if e != nil {
		return 0, e
	}
	attempt := uint64(ordinal)
	c.logRouting(req.ID, attempt, target.routed)

	// The grant names this attempt's own directory, so it is built after the ordinal is
	// real. ACCESS ONLY (#439): urls under the ids the spec declares — a refresh can
	// re-derive access and structurally cannot substitute meaning.
	//
	// A JOB's grant names the DURABLE PUBLICATION ROOT instead, and the destination fence
	// runs HERE — before the offer, so an escaping destination is never a capability
	// anybody held.
	grant, e := c.grantFor(req, attempt, w)
	if e != nil {
		return 0, c.abortDispatch(req.ID, attempt, w.bootID, e)
	}
	grant.InvocationSpecDigest = digest

	c.mu.Lock()
	placementID := w.placementID
	laneID := reservation.laneID
	var laneDevices []string
	if l := w.lanes.get(laneID); l != nil {
		laneDevices = devicesOf(l, w.spec.Devices)
	}
	c.mu.Unlock()
	if req.IsJob() {
		placementID = "" // job mode routes by the directive, not a placement (#446/#481)
	}
	// AN OFFER, NOT A START (#481): receiving one begins VALIDATION, not execution, which
	// is why refusing it is an ordinary journaled outcome rather than an exception. EVERY
	// offer gets AttemptAccepted or a journaled AttemptOutcome(REFUSED) — never silence.
	offer := &pb.AttemptOffer{
		RequestId: req.ID, AttemptOrdinal: attempt, InvocationSpecDigest: digest,
		Grant: grant, InvocationSpecCanonicalBytes: canonicalBytes, PlacementId: placementID,
		// THE EPOCH THIS OWNER OBSERVED WHEN IT DISPATCHED. The worker admits only if
		// this is still current; a stale echo refuses deterministically — same input, same
		// verdict, no race window — instead of running under capacity meaning that moved.
		AdmissionEpoch: admissionEpoch,
	}
	offer.RecordOwnerEpoch, offer.ControlStreamEpoch, offer.WorkerBootId =
		recordOwnerEpoch, sess.epoch, sess.bootID
	if !c.commitDispatch(reservation, req.ID, attempt) {
		cause := exit.Unavailablef("the worker selected for %s#%d left before its offer", req.ID, attempt)
		c.rollbackGrant(req, attempt, w)
		return 0, c.abortDispatch(req.ID, attempt, w.bootID, cause)
	}
	reserved = false
	if e := c.opt.Store.OfferDispatch(req.ID, int64(attempt), w.bootID); e != nil {
		c.settleDispatch(req.ID, attempt, false)
		c.rollbackGrant(req, attempt, w)
		return 0, c.abortDispatch(req.ID, attempt, w.bootID, e)
	}
	if !sess.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_AttemptOffer{AttemptOffer: offer}}) {
		c.settleDispatch(req.ID, attempt, false)
		c.rollbackGrant(req, attempt, w)
		cause := exit.Unavailablef("the control stream for %s#%d closed before its offer", req.ID, attempt)
		return 0, c.abortDispatch(req.ID, attempt, w.bootID, cause)
	}
	c.logf("AttemptOffer %s#%d spec=%s (%d canonical bytes) placement=%s lane=%s devices=%s "+
		"admission=%d outputs=%s on %s", req.ID, attempt, shortDigest(spelled),
		len(canonicalBytes), placementID, orNone(laneID), strings.Join(laneDevices, ","),
		admissionEpoch, req.Outputs, w.instanceID)
	event := map[string]any{"instance_id": w.instanceID, "invocation_digest": spelled}
	if placementID != "" {
		event["placement_id"] = placementID
	}
	// THE LANE THE OFFER DRAWS FROM, and the granted devices it covers, ride the durable
	// event (proto-024): a run's device is a fact a user can read back, not a log line.
	if laneID != "" {
		event["device_lane"] = laneID
		event["devices"] = laneDevices
	}
	c.emit(req.ID, "request.dispatched", attempt, event)
	return attempt, nil
}

func (c *Orchestrator) abortDispatch(requestID string, attempt uint64, sessionID string, cause *exit.Error) *exit.Error {
	if abort := c.opt.Store.AbortDispatch(requestID, int64(attempt), sessionID, cause.Message); abort != nil {
		c.logf("%s#%d preparation failed (%s), and its dispatch could not be aborted: %s",
			requestID, attempt, cause.Message, abort.Message)
		return abort
	}
	c.emit(requestID, "request.dispatch_aborted", attempt, map[string]any{
		"cause": cause.ErrName(), "error": cause.Message,
	})
	c.logf("%s#%d dispatch ABORTED before offer: %s", requestID, attempt, cause.Message)
	return cause
}

// rollbackGrant undoes what minting the grant created. A local serving grant creates
// nothing — it names the store directory and carries the payload inline — so only a
// remote reservation or a job's publication stage has anything to roll back.
func (c *Orchestrator) rollbackGrant(req records.Request, attempt uint64, w *worker) {
	if w.media != nil {
		c.cleanupRemote(req.ID, attempt, w)
		return
	}
	if !req.IsJob() {
		return
	}
	if err := os.RemoveAll(c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)); err != nil {
		c.logf("%s#%d local grant rollback failed: %s", req.ID, attempt, err)
		return
	}
	_ = os.Remove(filepath.Join(c.opt.Layout.PublicationRoot(req.Org, req.ID), ".staging"))
	prunePublicationParents(c.opt.Layout, c.opt.Layout.PublicationRoot(req.Org, req.ID))
}

// DefaultMaxOutputMiB is the per-output bound when Options.MaxOutputMiB is unset. It
// matches the Runtime/package media object envelope; attempt and pod quotas still bound
// the aggregate.
const DefaultMaxOutputMiB int64 = 512

func (c *Orchestrator) maxOutputBytes() uint64 {
	maxBytes := c.opt.MaxOutputMiB
	if maxBytes <= 0 {
		maxBytes = DefaultMaxOutputMiB
	}
	return uint64(maxBytes) << 20
}

// invocationIdentity selects the installed environment retained for this attempt.
// The handle orders updates and routes execution; it is not a content or memo key.
func (c *Orchestrator) invocationIdentity(w *worker,
	req records.Request) (string, *exit.Error) {
	c.mu.Lock()
	placement, remote, instanceID := w.spec.Placement, w.spec.Connection != nil, w.instanceID
	c.mu.Unlock()
	if remote && (placement.Release != req.Release ||
		(req.LocalInstallationID != "" && placement.InstallationID != req.LocalInstallationID)) {
		return "", exit.Named(exit.Conflict, "request_installation_changed",
			"worker %s no longer holds the installation selected for request %s", instanceID, req.ID)
	}
	identifier := placement.InstallationID
	if req.IsJob() && len(placement.Jobs) == 1 {
		identifier = placement.Jobs[0].InstallationID
	}
	if identifier == "" {
		return "", exit.Named(exit.Structural, "placement_installation_missing",
			"worker %s carries no installed package handle", instanceID)
	}
	if remote {
		if req.InstallationID == "" {
			if problem := c.opt.Store.BindRemoteInvocation(req.ID, req.PlanID, identifier); problem != nil {
				return "", problem
			}
		} else if req.InstallationID != identifier {
			return "", exit.Named(exit.Conflict, "request_installation_changed",
				"worker %s no longer holds the installation retained for request %s", instanceID, req.ID)
		}
	}
	return identifier, nil
}

func spellOf(raw []byte) string {
	s, err := canonical.Spell(raw)
	if err != nil {
		return ""
	}
	return s
}

// inputBindings is the spec's ORDERED input identity list (#439): the payload always,
// a job's exact Model manifests, plus a job's materialized trees (their verification is
// the store's own, so no digest). A SERVING request never declares a Model input: its
// models arrive through the placement's package-set lane, and the worker refuses a
// serving spec that carries one (grant_model_serving_refused).
func inputBindings(req records.Request, payloadDigest string) []*pb.InputBinding {
	rows := []*pb.InputBinding{{
		InputId:  "payload",
		Digest:   payloadDigest,
		Length:   uint64(len(req.Payload)),
		KindMime: "application/json",
		Order:    0,
	}}
	for _, asset := range req.Assets {
		rows = append(rows, &pb.InputBinding{
			InputId: asset.FieldPath, Digest: asset.Digest, Length: uint64(asset.Length),
			KindMime: asset.MediaType, Order: asset.Order,
		})
	}
	models := jobModels(req)
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	for _, model := range models {
		rows = append(rows, &pb.InputBinding{
			InputId: "model:" + model.Slot, Digest: model.Manifest,
			Length:   uint64(model.ManifestLength),
			KindMime: "application/vnd.cozy.model-manifest",
			Order:    0, // each Model is a scalar parameter, not an element of one shared list
		})
	}
	order := uint32(len(rows))
	for _, pair := range splitList(req.Trees) {
		ref, _, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		rows = append(rows, &pb.InputBinding{
			InputId: "tree:" + ref, KindMime: "inode/directory", Order: order,
		})
		order++
	}
	return rows
}

func downloadModelRefs(models []ModelRef) []*pb.DownloadModelRef {
	out := make([]*pb.DownloadModelRef, 0, len(models))
	for _, model := range models {
		// Operation-local manifests are already held by this worker. A retained
		// Hub checkpoint can be downloaded even when no release names it.
		if !model.Downloadable() {
			continue
		}
		// ONE PLACEMENT PER CONSTRUCTION (h3a-018): the selection rides under every slot
		// that shares its bytes, so the pod binds each of those entrypoints in the one
		// placement it prepares.
		for _, slot := range append([]string{model.BindingSlot()}, model.SharedSlots...) {
			out = append(out, &pb.DownloadModelRef{Package: model.Package, Slot: slot,
				Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, Adapters: downloadAdapters(model.Adapters)})
		}
	}
	return out
}

// DownloadModelRefs projects the exact model bindings captured for a published
// request onto the worker's package preparation contract.  Keeping this
// projection in the orchestrator ensures every preparation path (ordinary run,
// explicit prefetch, and recovery) carries the same slot, manifest, and adapter
// identity to Runtime.
func DownloadModelRefs(models []ModelRef) []*pb.DownloadModelRef {
	return downloadModelRefs(models)
}

func invocationOutputBindings(ids []string, weights []WeightsOutput, defaultMax uint64) []*pb.OutputBinding {
	byID := map[string]WeightsOutput{}
	for _, output := range weights {
		byID[output.OutputID] = output
	}
	out := make([]*pb.OutputBinding, 0, len(ids))
	for _, id := range ids {
		binding := &pb.OutputBinding{OutputId: id, MaxBytes: defaultMax}
		if weights, ok := byID[id]; ok {
			binding.MimeType = weights.MimeType
			binding.MaxBytes = weights.MaxBytes
		}
		out = append(out, binding)
	}
	return out
}

// pick resolves the worker and lane an offer for this request draws from: `route`'s
// argmin over every claimed worker whose PLACEMENT is DISPATCHABLE for the binding and
// whose lane has room (route.go), with the seat reserved. Compatibility and capacity
// matching stay here, in the orchestrator, exactly as they do in the cloud.
//
// TWO GATES, TWO OWNERS (#472e/#482). Dispatchability is a PLACEMENT property — the
// serving axis and `dispatchable_plan_ids`. Capacity is a WORKER property — the admission
// state, its epoch, and the one shared seat window. Per-placement `attempt_credits`
// are DELETED because N counters over ONE serialized device advertise N times the real
// capacity, and that defect is arithmetic rather than a race.
type dispatchReservation struct {
	worker *worker
	job    bool
	planID string
	// laneID is the lane the reserved seat draws from, "" for a worker that reports none.
	laneID string
}

// offerTarget is pick's answer: the worker, its live session, the admission epoch this
// owner observed, the reservation the offer holds, and the routing that chose it.
type offerTarget struct {
	worker         *worker
	sess           *session
	admissionEpoch uint64
	reservation    *dispatchReservation
	routed         routing
}

func (c *Orchestrator) pick(req records.Request) (offerTarget, *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing {
		return offerTarget{}, exit.Named(exit.Unavailable, "daemon.closing", "the daemon is closing; no attempt can be dispatched")
	}
	routed := c.route(req)
	pick := routed.pick()
	if pick == nil {
		return offerTarget{}, routed.noCapacity(req)
	}
	w := pick.worker
	target := offerTarget{worker: w, sess: c.sessions[w.bootID], admissionEpoch: w.admissionEpoch,
		routed: routed}
	if w.spec.IsJob() {
		// RESERVE the seat this dispatch is about to consume. The worker's own next
		// observed state is still the authority — this only stops ONE drain pass from
		// handing two queued jobs to a one-attempt worker on one reading.
		w.reservedJobs++
		w.jobsAvail = max(0, w.reportedJobs-w.reservedJobs)
		if w.jobsAvail <= 0 {
			w.dispatchable[req.PlanID] = false
		}
		target.reservation = &dispatchReservation{worker: w, job: true, planID: req.PlanID}
		return target, nil
	}
	// THE SEAT IS THE PLACEMENT'S LANE'S (proto-024). CLOSED, OPEN-with-no-seat and a
	// saturated lane are ALL "not now", distinguished on the worker facts rather than
	// here: this side's answer is the same either way — the request parks in the deep
	// queue, which is this owner's and never the worker's.
	w.reserveSeat(pick.laneID)
	target.reservation = &dispatchReservation{worker: w, laneID: pick.laneID}
	return target, nil
}

// A reservation starts while an offer is prepared and remains subtracted after emission
// until the worker causally answers Accepted or Refused. A Report can race an upload or an
// outbound frame, so it is never evidence that this particular offer has been observed.
func (c *Orchestrator) commitDispatch(r *dispatchReservation, requestID string, attempt uint64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if r == nil || c.workers[r.worker.instanceID] != r.worker {
		return false
	}
	c.offers[key(requestID, attempt)] = r
	return true
}

func (c *Orchestrator) releaseDispatch(r *dispatchReservation) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if r == nil || c.workers[r.worker.instanceID] != r.worker {
		return
	}
	if r.job {
		r.worker.reservedJobs = max(0, r.worker.reservedJobs-1)
		r.worker.observeJobs(r.worker.reportedJobs)
		if r.worker.jobsAvail > 0 {
			r.worker.dispatchable[r.planID] = true
		}
		return
	}
	r.worker.releaseSeat(r.laneID)
}

// settleDispatch consumes a seat on Accepted/executed outcome and returns it on a
// pre-execution Refused outcome. It is idempotent because only the first causal answer
// finds the offer reservation.
func (c *Orchestrator) settleDispatch(requestID string, attempt uint64, consumed bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	k := key(requestID, attempt)
	r := c.offers[k]
	delete(c.offers, k)
	if r == nil || c.workers[r.worker.instanceID] != r.worker {
		return
	}
	if r.job {
		r.worker.reservedJobs = max(0, r.worker.reservedJobs-1)
		if consumed {
			r.worker.reportedJobs = max(0, r.worker.reportedJobs-1)
			r.worker.held++
		}
		r.worker.observeJobs(r.worker.reportedJobs)
		if !consumed && r.worker.jobsAvail > 0 {
			r.worker.dispatchable[r.planID] = true
		}
		return
	}
	r.worker.settleSeat(r.laneID, consumed)
}

// grantFor picks the lane's grant. The lanes differ in exactly one thing that matters —
// WHERE the destinations are — and every one of those differences is a fact about a
// machine: a bounded job's writes must outlive the reclaim that ends it, and a REMOTE
// attempt's destinations must be on the pod, because that is where the process writing
// them is.
func (c *Orchestrator) grantFor(req records.Request, attempt uint64, w *worker) (*pb.DeliveryGrant, *exit.Error) {
	if w.media == nil {
		if e := localGrantSupport(runtime.GOOS); e != nil {
			return nil, e
		}
	}
	if req.IsJob() && w.media == nil {
		g, _, e := c.jobGrant(req, attempt)
		return g, e
	}
	if w.media != nil {
		return c.remoteGrant(req, attempt, w)
	}
	return c.grant(req.ID, attempt, req)
}

func localGrantSupport(goos string) *exit.Error {
	if goos != "windows" {
		return nil
	}
	return exit.Named(exit.Structural, "local_file_grant_unsupported",
		"local worker grants are not yet supported on Windows").
		WithRemedy("use --rental; local file URL authorization is currently POSIX-only")
}

// MediaReservationBytes prices only file outputs. Native Weights are written to
// TensorFS under Runtime admission, so their slot cannot reserve the media byte
// ceiling as well. File-count/OutputAccess identities remain unchanged.
func MediaReservationBytes(req records.Request, perOutput uint64) (int64, *exit.Error) {
	weights, problem := decodeWeightsOutputs(req.WeightsOutputs)
	if problem != nil {
		return 0, problem
	}
	native := make(map[string]bool, len(weights))
	for _, row := range weights {
		native[row.OutputID] = true
	}
	count := uint64(0)
	for _, id := range splitList(req.Outputs) {
		if !native[id] {
			count++
		}
	}
	if count > 0 && perOutput > uint64(math.MaxInt64)/count {
		return 0, exit.New(exit.Validation, "the output grant for %s exceeds the media plane's byte range", req.ID)
	}
	return int64(perOutput * count), nil
}

// remoteGrant builds the grant for an attempt that will run on a POD (cl-014/#506b).
//
// It is the same grant the local lane mints and every address in it is on the other
// machine: the payload is UPLOADED to the pod's media server first and the grant names
// where the pod says it landed; the output destinations are a directory the pod reserved.
// Nothing here composes a pod path — every one of them is an answer from the pod — and
// nothing falls back to this host's disk, because a `file://` rooted here is a destination
// the pod cannot reach and the whole class of defect #493.3 found.
//
// The BYTES MOVE BEFORE THE ATTEMPT EXISTS. `dispatch` calls this after the ordinal is
// journaled and before `StartAttempt` is sent, so a pod that cannot be fed refuses the
// dispatch rather than accepting an attempt whose inputs are unreachable.
func (c *Orchestrator) remoteGrant(req records.Request, attempt uint64, w *worker) (*pb.DeliveryGrant, *exit.Error) {
	slot := media.Slot(req.ID, attempt)
	outputIDs := splitList(req.Outputs)
	reservedOutputBytes, problem := MediaReservationBytes(req, c.maxOutputBytes())
	if problem != nil {
		return nil, problem
	}
	complete := false
	defer func() {
		if !complete {
			c.cleanupRemote(req.ID, attempt, w)
		}
	}()
	dir, e := w.media.ReserveOutputs(slot, reservedOutputBytes, len(outputIDs))
	if e != nil {
		return nil, e
	}
	path, e := w.media.PutInput(slot, "payload", req.Payload)
	if e != nil {
		return nil, e
	}
	c.logf("%s#%d: %d payload bytes crossed to %s at %s; outputs reserved at %s",
		req.ID, attempt, len(req.Payload), w.media.Addr(), path, dir)
	g := &pb.DeliveryGrant{
		FileBaseUrl:   "file://" + dir,
		ExpiresAtUnix: 0,
		Inputs:        []*pb.InputAccess{{InputId: "payload", Url: "file://" + path}},
	}
	g.Inputs = append(g.Inputs, modelAccess(req)...)
	for index, asset := range req.Assets {
		if asset.Native != nil {
			g.Inputs = append(g.Inputs, &pb.InputAccess{InputId: asset.FieldPath, NativeTree: &pb.NativeByteRetentionRequest{Source: asset.Native.Output.NativeRef(), RetentionId: asset.Native.RetentionID}})
			continue
		}
		path, e := w.media.PutInputFile(slot, "input-"+strconv.Itoa(index),
			asset.LocalPath, asset.Digest, asset.Length)
		if e != nil {
			return nil, e
		}
		g.Inputs = append(g.Inputs, &pb.InputAccess{InputId: asset.FieldPath, Url: "file://" + path})
		c.logf("%s#%d: input asset %s (%d B, %s) crossed to %s at %s",
			req.ID, attempt, asset.FieldPath, asset.Length, asset.Digest, w.media.Addr(), path)
	}
	for _, id := range outputIDs {
		if e := FenceOutputID(id); e != nil {
			return nil, e
		}
		g.Outputs = append(g.Outputs, &pb.OutputAccess{
			OutputId: id, Url: "file://" + dir + "/" + id,
		})
	}
	complete = true
	return g, nil
}

// jobModels is the set of Model inputs a request declares: a job's bound models with an
// exact manifest; none for a serving request (see inputBindings).
func jobModels(req records.Request) []ModelRef {
	if !req.IsJob() {
		return nil
	}
	var models []ModelRef
	for _, model := range req.OwnModels() { // callee defaults are never the root's inputs
		if model.ManifestLength > 0 {
			models = append(models, model)
		}
	}
	return models
}

// modelAccess is the access half of every Model input the spec declares (jobModels):
// one row per model, the same on every grant, or the worker refuses
// grant_binding_mismatch.
func modelAccess(req records.Request) []*pb.InputAccess {
	var rows []*pb.InputAccess
	for _, model := range jobModels(req) {
		rows = append(rows, &pb.InputAccess{
			InputId: "model:" + model.Slot, Url: "model://" + model.Manifest,
		})
	}
	return rows
}

// grant builds the LOCAL delivery grant: the payload INLINE, every bound model, every
// input asset at its borrowed original path, and one destination per result field path —
// the store directory itself (`outputs/<org>-<package>/` or the caller's --out), which
// the worker names the file in by its content digest, `<sha256>.<ext>`. Nothing is
// staged on this side and nothing is copied afterwards: the granted directory is where
// the file lives for good, and the terminal is verified against exactly that name. There
// is no credential — a local grant is a CAS root plus a directory, and a fabricated
// token would be a lie about authority nobody issued.
func (c *Orchestrator) grant(requestID string, attempt uint64, req records.Request) (*pb.DeliveryGrant, *exit.Error) {
	dir, e := c.outputDirectory(req)
	if e != nil {
		return nil, e
	}
	g := &pb.DeliveryGrant{
		FileBaseUrl: "file://" + dir + "/",
		// NO EXPIRY, because this host mints no deadline to derive one from. A grant lasts
		// as long as the attempt it was minted for (cr-009), and the attempt's bound is the
		// caller's deadline — which `dispatch` never sets, because there is no wire field
		// for it here and `--timeout` is enforced client-side by CANCELLING. 10 minutes was
		// therefore a ceiling on how long a local attempt could be, invented at the one
		// place nobody was asked. The worker reads 0 as "does not expire" (`grants.expired`).
		ExpiresAtUnix: 0,
		// ACCESS ONLY (#439): the identities (digest, length, media kind) live in the
		// spec's bindings, inside the invocation digest.
		Inputs: []*pb.InputAccess{{InputId: "payload", Url: payloadURL(req.Payload)}},
	}
	g.Inputs = append(g.Inputs, modelAccess(req)...)
	assets, problem := localAssetAccess(req)
	if problem != nil {
		return nil, problem
	}
	g.Inputs = append(g.Inputs, assets...)

	for _, id := range splitList(req.Outputs) {
		if e := FenceOutputID(id); e != nil {
			return nil, e
		}
		g.Outputs = append(g.Outputs, &pb.OutputAccess{OutputId: id, Url: "file://" + dir + "/"})
	}
	return g, nil
}

func localAssetAccess(req records.Request) ([]*pb.InputAccess, *exit.Error) {
	var inputs []*pb.InputAccess
	for _, asset := range req.Assets {
		if asset.Native != nil {
			if req.ParentRequestID == "" {
				return nil, exit.New(exit.Conflict, "native byte inputs require a private child grant")
			}
			if asset.LocalPath != "" || asset.Digest != asset.Native.Output.Digest || asset.Length != asset.Native.Output.Length || asset.MediaType != asset.Native.Output.MimeType {
				return nil, exit.New(exit.Validation, "native input differs from recorded byte output")
			}
			inputs = append(inputs, &pb.InputAccess{InputId: asset.FieldPath, NativeTree: &pb.NativeByteRetentionRequest{Source: asset.Native.Output.NativeRef(), RetentionId: asset.Native.RetentionID}})
			continue
		}
		limit := asset.MaxBytes
		if limit <= 0 {
			limit = asset.Length
		}
		if e := inputasset.Verify(asset, limit); e != nil {
			return nil, e
		}
		inputs = append(inputs, &pb.InputAccess{
			InputId: asset.FieldPath, Url: (&url.URL{Scheme: "file", Path: asset.LocalPath}).String(),
		})
	}
	return inputs, nil
}

// payloadURL is the request document as a `data:` URL: the grant IS the bytes, so a
// worker on this host reads the payload from the grant and nothing is written to disk
// to hand it over. The worker still verifies them against the spec's digest and length.
func payloadURL(payload []byte) string {
	return "data:application/json;base64," + base64.StdEncoding.EncodeToString(payload)
}

// outputDirectory is where one serving request's result files live: the directory its
// export row names (the caller's --out, else the package's store) — the same directory
// the grant hands the worker and the terminal is verified against.
func (c *Orchestrator) outputDirectory(req records.Request) (string, *exit.Error) {
	export, e := c.opt.Store.OutputExportOf(req.ID)
	if e != nil {
		return "", e
	}
	if export != nil {
		return export.Directory, nil
	}
	return c.opt.Layout.PackageOutputs(req.Package), nil
}

// Await blocks until the attempt is closed — the terminal accepted, its outputs visible,
// and the ack sent. The error it returns is the orchestrator's PROJECTION of the neutral
// terminal, never a worker-authored retryability claim.
func (c *Orchestrator) Await(requestID string, attempt uint64, timeout time.Duration) (*Result, *exit.Error) {
	w, release := c.acquireWait(key(requestID, attempt))
	defer release()
	if row, e := c.opt.Store.AttemptRow(requestID, int64(attempt)); e == nil && row != nil && row.State == "closed" {
		w.markClosed(outcomeError(row.TerminalStatus, row.TerminalCause, row.SafeMessage))
	}
	select {
	case <-w.closed:
	case <-time.After(timeout):
		return nil, exit.New(exit.Deadline, "%s#%d did not reach a terminal in %s",
			requestID, attempt, timeout)
	}
	row, e := c.opt.Store.AttemptRow(requestID, int64(attempt))
	if e != nil {
		return nil, e
	}
	if row == nil {
		return nil, exit.Internalf("%s#%d closed with no row", requestID, attempt)
	}
	outs, e := c.opt.Store.VisibleOutputs(requestID)
	if e != nil {
		return nil, e
	}
	return &Result{
		RequestID: requestID, Attempt: attempt, AttemptKey: row.AttemptKey,
		Status: row.TerminalStatus, Cause: row.TerminalCause,
		Outputs: outs, Body: row.TerminalBody,
	}, w.err
}

// AwaitSettled blocks until the REQUEST settles — through however many requeued
// ordinals the orchestrator's projection minted. This is what a caller waits on; an
// individual attempt is the orchestrator's business.
func (c *Orchestrator) AwaitSettled(requestID string, timeout time.Duration) (*Result, *exit.Error) {
	w, release := c.acquireWait(requestWaitKey(requestID))
	defer release()
	if row, e := c.opt.Store.RequestRow(requestID); e == nil && row != nil && settledState(row.State) {
		attempts, read := c.opt.Store.Attempts(requestID)
		if read == nil && len(attempts) > 0 {
			last := attempts[len(attempts)-1]
			w.markClosed(outcomeError(last.TerminalStatus, last.TerminalCause, last.SafeMessage))
		} else {
			w.markClosed(exit.Named(exit.Failed, row.State, "request %s settled %s before an attempt", requestID, row.State))
		}
	}
	select {
	case <-w.closed:
	case <-time.After(timeout):
		return nil, exit.New(exit.Deadline, "%s did not settle in %s", requestID, timeout)
	}
	row, e := c.opt.Store.RequestRow(requestID)
	if e != nil || row == nil {
		return nil, exit.Internalf("%s settled with no row", requestID)
	}
	attempts, e := c.opt.Store.Attempts(requestID)
	if e != nil || len(attempts) == 0 {
		return nil, exit.Internalf("%s settled with no attempt", requestID)
	}
	last := attempts[len(attempts)-1]
	outs, e := c.opt.Store.VisibleOutputs(requestID)
	if e != nil {
		return nil, e
	}
	return &Result{
		RequestID: requestID, Attempt: uint64(last.Attempt), AttemptKey: last.AttemptKey,
		Status: last.TerminalStatus, Cause: last.TerminalCause, Outputs: outs,
		Body: last.TerminalBody,
	}, w.err
}

// AwaitAccepted blocks until AttemptAccepted is journaled — the acceptance boundary a
// submit→accepted benchmark measures.
func (c *Orchestrator) AwaitAccepted(requestID string, attempt uint64, timeout time.Duration) *exit.Error {
	w, release := c.acquireWait(key(requestID, attempt))
	defer release()
	if row, e := c.opt.Store.AttemptRow(requestID, int64(attempt)); e == nil && row != nil {
		if row.AcceptedAt != "" {
			w.markAccepted()
		}
		if row.State == "closed" {
			w.markClosed(outcomeError(row.TerminalStatus, row.TerminalCause, row.SafeMessage))
		}
	}
	select {
	case <-w.accepted:
		return nil
	case <-w.closed:
		return w.err
	case <-time.After(timeout):
		return exit.New(exit.Deadline, "%s#%d was not accepted in %s", requestID, attempt, timeout)
	}
}

// ClientCancelGraceMS is the one cooperative attempt-cancellation policy. It is a
// cancellation budget carried to Runtime, never a stall or caller deadline.
const ClientCancelGraceMS uint64 = 5000

// CancelClient is the client-reason cancel, for the callers that have no business
// naming a protocol enum. The actor is journaled BEFORE the cancel frame goes out
// (cl-108): the attempt's canceled terminal arrives later from the worker, and joining
// it back to WHO asked must survive a daemon restart in between.
func (c *Orchestrator) CancelClient(requestID string, attempt, graceMS uint64, actor string) *exit.Error {
	if actor == "" {
		actor = "an unnamed client"
	}
	if e := c.opt.Store.AppendEvent(requestID, "request.cancel_requested", int64(attempt),
		map[string]any{"actor": actor, "grace_ms": graceMS}); e != nil {
		return e
	}
	return c.Cancel(requestID, attempt, pb.CancelReason_CANCEL_REASON_CLIENT, graceMS)
}

// Cancel is the only way to supersede a live attempt: explicit, digest-fenced, and
// followed by a journaled terminal. Silent supersession does not exist in this protocol,
// and an attempt is never killed to improve queue latency.
func (c *Orchestrator) Cancel(requestID string, attempt uint64, reason pb.CancelReason, graceMS uint64) *exit.Error {
	row, e := c.opt.Store.AttemptRow(requestID, int64(attempt))
	if e != nil {
		return e
	}
	if row == nil {
		return exit.New(exit.NotFound, "no attempt %s#%d to cancel", requestID, attempt)
	}
	raw, err := canonical.Raw(row.InvocationDigest)
	if err != nil {
		return exit.Internalf("the journaled spec digest is unreadable: %s", err)
	}
	c.mu.Lock()
	sess := c.sessions[row.SessionID]
	w := c.workers[row.InstanceID]
	c.mu.Unlock()
	if sess == nil || w == nil {
		return exit.Unavailablef("the stream that holds %s#%d is gone", requestID, attempt)
	}
	cancel := &pb.CancelAttempt{
		RequestId: requestID, AttemptOrdinal: attempt, Reason: reason,
		GraceMs: graceMS, InvocationSpecDigest: raw,
	}
	cancel.RecordOwnerEpoch, cancel.ControlStreamEpoch, cancel.WorkerBootId =
		recordOwnerEpoch, sess.epoch, sess.bootID
	if !sess.trySend(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_CancelAttempt{CancelAttempt: cancel}}) {
		return exit.Unavailablef("the control stream for %s#%d cannot accept cancellation now",
			requestID, attempt)
	}
	c.logf("CancelAttempt %s#%d reason=%s", requestID, attempt,
		pb.CancelReason_name[int32(reason)])
	return nil
}
