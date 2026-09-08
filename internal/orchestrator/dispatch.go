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
	IdemKey    string // the caller's idempotency key
	Package    string // org/name
	Entrypoint string // the function
	PlanID     string // the entrypoint_binding_plan_id this attempt binds
	Release    string // immutable remote package release; empty for local execution
	// LocalPackageDigest is the exact staged wheel-set identity for one editable rental.
	LocalPackageDigest string
	Models             []ModelRef

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
	Kind           string
	RetainWork     bool
	RetryOf        string
	ChildReusable  bool
	ChildArtifacts bool
	// Org is the publishing org whose scratch repo this job publishes into.
	Org string
	// Trees are the job's typed input TREES as `ref=dir`, one grant input each.
	Trees []string
	// NeedsAccelerator is derived from the selected package's immutable dependency facts.
	// It is the only machine-class decision retained on a request.
	NeedsAccelerator bool

	// Worker pins this request to an ATTACHED remote worker (a rental id resolved
	// through Options.Rentals). Empty = any local worker.
	Worker string
	// InstallID pins a durable request to one immutable local install resolution.
	// Remote requests instead carry their immutable Release.
	InstallID string
	// Rental authorizes placement on Creator-managed rented capacity.
	Rental bool
	// RentalRequired is the explicit development/E2E override that forbids local capacity.
	// It implies Rental and survives queue/restart scheduling in the request row.
	RentalRequired bool
	// OutputDirectory is the caller's explicit --out; empty means the package's store.
	// It is part of the submission's identity, where the derived intent below is not.
	OutputDirectory string
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

// MaxRequeues is the durable per-request requeue bound. Retry is bounded and the bound
// is a ROW, not a counter in memory: a worker that dies on every attempt exhausts it
// instead of dispatching forever.
const MaxRequeues = 3

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
	req, fresh, e := c.opt.Store.Submit(req)
	if e != nil {
		return records.Request{}, false, e
	}
	if !fresh {
		c.logRecordedReplay(req, s.IdemKey)
		return req, false, nil
	}
	c.emit(req.ID, "request.submitted", 0, event)
	return req, true, nil
}

func requestRecord(s Submission) (records.Request, map[string]any, *exit.Error) {
	if s.RentalRequired {
		s.Rental = true
	}
	if s.RetainWork && s.Kind != "job" {
		return records.Request{}, nil, exit.Named(exit.Validation, "retain_work_not_job",
			"retained work is an ordinary private job capability")
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
			encoded, err := canonical.Write(map[string]canonical.Value{
				"payload":         base64.StdEncoding.EncodeToString(s.Payload),
				"rental":          true,
				"rental_required": s.RentalRequired,
			})
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
	if s.LocalPackageDigest != "" {
		if s.InstallID == "" || !validDigest(s.LocalPackageDigest) {
			return records.Request{}, nil, exit.Named(exit.Structural,
				"local_package_request_invalid",
				"a local package revision requires one exact editable install")
		}
		identity, err := canonical.Write(map[string]canonical.Value{
			"body_digest":          bodyDigest,
			"local_package_digest": s.LocalPackageDigest,
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
	id := records.NewID("req")
	if s.Kind == "job" {
		id = records.NewID("job")
	}
	req := records.Request{
		ID: id, IdemKey: s.IdemKey, BodyDigest: bodyDigest,
		Package: s.Package, Entrypoint: s.Entrypoint, PlanID: s.PlanID, Payload: s.Payload,
		Release:            s.Release,
		LocalPackageDigest: s.LocalPackageDigest,
		Outputs:            strings.Join(s.Outputs, ","),
		Assets:             s.Assets, WeightsOutputs: string(weightsBytes),
		Kind: s.Kind, RetainWork: s.RetainWork, RetryOf: s.RetryOf, ChildArtifacts: s.ChildArtifacts, NeedsAccelerator: s.NeedsAccelerator, Org: s.Org, Trees: strings.Join(s.Trees, ","),
		Worker: s.Worker, InstallID: s.InstallID, Rental: s.Rental,
		RentalRequired: s.RentalRequired, Models: s.Models,
		OutputExport: s.OutputExport, ModelTransfer: s.ModelTransfer,
	}
	event := map[string]any{
		"package": s.Package, "function": s.Entrypoint,
		"body_digest": bodyDigest, "plan_id": s.PlanID, "outputs": s.Outputs,
		"weights_outputs": weightsOutputs,
	}
	if s.Rental {
		event["rental"] = true
		event["rental_required"] = s.RentalRequired
		event["release"] = s.Release
		if s.LocalPackageDigest != "" {
			event["local_package_digest"] = s.LocalPackageDigest
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
		c.emit(req.ID, "request.queued", 0, waitFacts{cause: WaitQueueAhead}.decorate(map[string]any{
			"reason":   "the dispatch queue is not empty; this request joins it in submission order",
			"position": c.QueuePosition(req.ID),
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

// Requeue is the orchestrator's PROJECTION over a neutral terminal: an ABANDONED attempt
// or an infra-class failure earns a NEW ordinal, a fresh grant and a fresh execution —
// never a patch to the one that died. It is charged against the request's durable budget.
func (c *Orchestrator) Requeue(requestID, why string) { c.requeue(requestID, why, true) }

// RequeueForCapacity returns a request to the queue WITHOUT spending a life.
//
// The two are not the same event and treating them as one killed healthy runs. The budget
// exists so a worker that faults on every attempt terminates the request rather than
// dispatching forever. A worker that answered NO_CAPACITY has not attempted anything: it
// said "not now", which is a fact about the machine's moment and not about the request.
// Charging it means three "not now"s in a row settle a run that executed nothing —
// observed as four attempts on one request, every one of them refused before execution.
//
// The local path has always drawn this line: `device_envelope_held` in selectOrStart keeps
// the durable request on the queue and charges nothing, because "a later idle-capacity
// report re-enters select-or-start". This is the same rule reaching the remote path.
//
// It is not an unbounded retry. The request returns to the queue and PARKS; only a worker
// reporting capacity re-dispatches it, so the loop is driven by an observation rather than
// by a cadence — and while it waits, the phase lane says so instead of showing silence.
func (c *Orchestrator) RequeueForCapacity(requestID, why string) {
	c.requeue(requestID, why, false)
}

func (c *Orchestrator) requeue(requestID, why string, charge bool) {
	n, started, canceled, e := c.opt.Store.BeginRequeue(requestID, MaxRequeues, charge)
	if e != nil {
		// THE REQUEST ENDS HERE, and it has to SAY so. Settling the row without emitting a
		// terminal event left a client watching the durable stream with `attempt_failed
		// (requeuing: true)` as its last frame and nothing after it — the contract's
		// terminal-stop rule never fired, and `cozy run` waited on a request that had been
		// settled for ten minutes. Observed live, in cl-003's ARM 3.
		c.logf("%s NOT requeued (%s): %s", requestID, why, e.Message)
		c.forget(requestID)
		payload := map[string]any{
			"status": "FAILED", "cause": "REQUEUE_BUDGET_EXHAUSTED",
			"error_type": e.ErrName(), "error": e.Message,
			"outputs": []any{}, "requeuing": false,
		}
		row, read := c.opt.Store.RequestRow(requestID)
		if read == nil && row != nil && row.RetainWork {
			_, _ = c.opt.Store.BlockRetainedWork(requestID, "REQUEUE_BUDGET_EXHAUSTED", e.Message)
			return
		}
		if read == nil && row != nil && row.ModelTransfer != nil {
			if problem := c.releaseManagedNow(*row); problem != nil {
				c.logf("%s model transfer provider cleanup remains pending: %s", requestID, problem.Message)
				time.AfterFunc(2*time.Second, func() { c.Requeue(requestID, why) })
				return
			}
			_, _ = c.opt.Store.FailModelTransferRequest(requestID, e.ErrName(), e.Message, payload)
		} else {
			_ = c.opt.Store.SettleRequest(requestID, "failed")
			c.emit(requestID, "request.failed", 0, payload)
		}
		c.RetryOutputExport(requestID)
		if row != nil && row.ModelTransfer != nil {
			c.forgetTransferProgress(requestID)
		} else {
			c.frames.forget(requestID)
		}
		c.signalClosed(requestWaitKey(requestID), e)
		if row, read := c.opt.Store.RequestRow(requestID); read == nil && row != nil {
			go c.cleanupRequestAssets(*row)
		}
		return
	}
	if canceled {
		c.forget(requestID)
		if row, read := c.opt.Store.RequestRow(requestID); read == nil && row != nil &&
			row.ModelTransfer != nil {
			c.forgetTransferProgress(requestID)
		} else {
			c.frames.forget(requestID)
		}
		c.signalClosed(requestWaitKey(requestID),
			exit.New(exit.Canceled, "%s was canceled before its requeue", requestID))
		if row, read := c.opt.Store.RequestRow(requestID); read == nil && row != nil {
			go c.cleanupRequestAssets(*row)
		}
		return
	}
	if !started {
		return
	}
	req, e := c.opt.Store.RequestRow(requestID)
	if e != nil || req == nil {
		return
	}
	// THE REQUEUE IS A FACT THE MOMENT THE BUDGET IS CHARGED, and it is announced here —
	// before dispatch, which may or may not find capacity. Announcing it only on the
	// successful branch lost the fact exactly when it mattered most: the orchestrator-kill
	// arm requeued into a worker that was still loading, so the stream said `queued` and
	// never said WHY, and a client could not tell a first dispatch from a retry.
	c.emit(requestID, "request.requeued", 0, map[string]any{
		"cause": why, "requeues": n, "budget": MaxRequeues,
	})
	attempt, e := c.dispatch(*req)
	if e != nil {
		// No capacity yet: the request WAITS. A requeue that cannot be placed is queued,
		// never dropped — dispatch resumes the moment a worker reports the binding ready.
		c.enqueue(requestID)
		c.emit(requestID, "request.queued", 0,
			c.waitOf(*req).decorate(map[string]any{"reason": e.Message, "requeues": n}, *req))
		c.logf("%s requeued %d/%d and QUEUED for capacity: %s", requestID, n, MaxRequeues, e.Message)
		c.selectOrStart(*req)
		return
	}
	c.logf("%s requeued as attempt %d (%d/%d of the budget, cause %s)",
		requestID, attempt, n, MaxRequeues, why)
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
func (c *Orchestrator) selectOrStart(req records.Request) {
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil || (current.State != "submitted" && current.State != "queued") {
		return
	}
	req = *current
	// A queued pin can predate a client upgrade or the peer's first ClaimAck.
	// Replan only work which has never been offered; keep old attempts/data intact.
	if req.Rental && req.Worker != "" {
		c.mu.Lock()
		w := c.workers[rentalInstanceID(req.Worker)]
		older := w != nil && !w.supportsCurrentProtocol()
		c.mu.Unlock()
		reason := ""
		if older {
			reason = ExcludedProtocol
		} else {
			retained, problem := c.opt.Store.RentalHasRetainedJob(req.Worker)
			if problem != nil {
				c.logf("%s cannot assess retained rental job: %s", req.ID, problem.Message)
				return
			}
			if retained {
				reason = ExcludedModeConflict
			}
			row, problem := c.opt.Store.RentalRow(req.Worker)
			if problem != nil {
				c.logf("%s cannot assess pinned rental: %s", req.ID, problem.Message)
				return
			}
			if row != nil {
				spent, problem := RentalSpent(c.opt.Store, *row)
				if problem != nil {
					c.logf("%s cannot assess pinned rental purpose: %s", req.ID, problem.Message)
					return
				}
				if spent {
					reason = ExcludedSpent
				}
			}
		}
		if reason != "" {
			if req.RetainWork {
				c.failQueued(req.ID, exit.Named(exit.Conflict, "request.retained_rental_unavailable", "the retained rental cannot execute this transaction (%s)", reason), "")
				return
			}
			attempts, problem := c.opt.Store.Attempts(req.ID)
			if problem != nil || len(attempts) != 0 {
				return
			}
			unpinned, problem := c.opt.Store.UnpinRentalWork(req.ID, req.Worker)
			if problem != nil || !unpinned {
				return
			}
			c.logf("%s replans queued work from rental %s: %s", req.ID, req.Worker, reason)
			req.Worker = ""
		}
	}
	var guards []string
	if req.Rental && req.Worker == "" {
		// This request is QUEUED — no lane, local or rental, could take it at routing
		// time. An attached rental that staged the plan is capacity `drain` routes onto
		// the moment its lane has room, so the request WAITS unpinned and takes its pin
		// from dispatch. Otherwise the fleet chooses the rental to stage the placement
		// on, by what each store already holds, or BUYS a pod (owner ruling 2026-09-03:
		// --rental is permission AND intent to spend — a busy local lane is what the
		// flag is for, and local disk holdings never veto the buy). That rental keeps
		// serving its other tenants while the manifests land, and this request routes to
		// it once its placement reports DISPATCHABLE.
		if c.opt.RentalFleet == nil || c.opt.AcquireManagedRental == nil {
			c.failQueued(req.ID, exit.Named(exit.Unavailable, "rental.acquisition_unavailable",
				"this Cozy daemon cannot acquire managed rentals"), "")
			return
		}
		guard := "rental/" + requestSlot(req)
		c.mu.Lock()
		rentalHolds := c.rentalHeld(req)
		staging := c.starting[guard]
		if !rentalHolds && !staging {
			c.starting[guard] = true
		}
		c.mu.Unlock()
		if rentalHolds || staging {
			return
		}
		guards = append(guards, guard)
		unguard := func() {
			c.mu.Lock()
			delete(c.starting, guard)
			c.mu.Unlock()
		}
		line, problem := c.opt.RentalFleet()
		if problem != nil {
			unguard()
			if deferred, _ := c.deferUnavailable(req, problem); deferred {
				return
			}
			c.failQueued(req.ID, problem, "")
			return
		}
		c.emit(req.ID, "request.rentals", 0, map[string]any{"line": line})
		// --rental is permission AND intent to spend (owner ruling 2026-09-03): local
		// could not take the request now and no ready rental holds its placement, so the
		// fleet places it by expected time and cost — on an attached rental or a bought
		// pod (placement-economics.md).
		decision, after, problem := c.opt.AcquireManagedRental(req)
		if problem != nil {
			unguard()
			// A refusal is a decision too (cl-174): its record is durable before the
			// request parks or fails, once per distinct park.
			deferred, news := c.deferUnavailable(req, problem)
			if len(decision.Candidates) > 0 && (news || !deferred) {
				c.logPlacement(req, decision)
			}
			if deferred {
				return
			}
			c.failQueued(req.ID, problem, "")
			return
		}
		if after != "" {
			c.emit(req.ID, "request.rentals", 0, map[string]any{"line": after})
		}
		if decision.RentalID == "" {
			// The fleet waits on a fitting rental whose worker has not attached (cl-170):
			// the record is written once per distinct wait, and the fleet's next
			// observation re-asks.
			unguard()
			if c.parkFor(req, decision.Line()) {
				c.logPlacement(req, decision)
			}
			return
		}
		req.Worker = decision.RentalID
		if decision.Models != nil {
			req.Models = decision.Models
		}
		c.logPlacement(req, decision)
	}
	// A JOB names its own slot — one worker per (package, job function) — so the
	// "already starting" and "already resident" questions are asked about that slot and
	// not about the package. Without this, submitting a job while a serving worker of
	// the same package is up would decide a job worker already existed.
	// A request PINNED to a rental asks its questions about the rental's own slot: the
	// attached worker's spec carries the pinned package name, so comparing against the
	// bare one would decide no worker was resident and attach a second control stream to
	// the pod on every request.
	slot := pinnedPackage(req.Package, req.Worker)
	if req.InstallID != "" {
		slot += "/install/" + req.InstallID
	}
	if req.IsJob() {
		slot += "/job/" + req.Entrypoint
	}
	c.mu.Lock()
	if c.starting[slot] {
		for _, guard := range guards {
			delete(c.starting, guard)
		}
		c.mu.Unlock()
		return
	}
	stale := ""
	for _, w := range c.workers {
		if w.exited || w.stopping || w.spec.IsJob() != req.IsJob() {
			continue
		}
		if req.Worker != "" && w.spec.Connection != nil {
			if w.instanceID != rentalInstanceID(req.Worker) {
				continue
			}
		} else if w.spec.Placement.Package != pinnedPackage(req.Package, req.Worker) {
			continue
		}
		// An install pins a LOCAL relaunch. On a rental the exact identity is the sealed
		// local revision the request carries, and stagedFor holds the worker to it.
		if req.InstallID != "" && w.spec.Connection == nil && w.spec.Placement.InstallID != req.InstallID {
			continue
		}
		if !req.IsJob() || w.spec.Placement.Jobs[0].Function == req.Entrypoint {
			// A local package_set request learns its binding digest from the worker.
			// Until that happens an empty plan is neither staged nor stale; resolveFor
			// sends the logical set and binds the observed answer.
			if req.PlanID == "" {
				continue
			}
			// RESIDENCY IS ABOUT THE PLAN, not about the slot. A worker that STAGED this
			// request's plan is already resident or loading, and its READY drains the
			// queue — which is exactly how several submitted jobs queue against ONE
			// worker (cr-019). A worker in the same slot that staged a DIFFERENT plan is
			// STALE: the install's surface moved under it, and treating it as capacity
			// queues the request behind a worker that will never advertise what it needs.
			// Found live by cl-004's escape arm, which changes the descriptor and
			// therefore the job descriptor id: six queued jobs waited on a worker holding
			// the previous digest, forever.
			// A worker that has already answered that this placement cannot run is not
			// warm capacity. This occurs naturally when the host Runtime is upgraded while
			// the long-lived daemon still owns a worker loaded from the previous install.
			// Replace that process once; the new launch below either serves the request with
			// the current Runtime or gives the request its own prompt terminal answer.
			if stagedFor(w, req) && retirementGround(w) == "" {
				for _, guard := range guards {
					delete(c.starting, guard)
				}
				c.mu.Unlock()
				return
			}
			if req.Worker != "" && w.spec.Connection != nil {
				// The rental is the slot. A different package is an addition to this
				// machine, not evidence that its existing worker is stale.
				continue
			}
			if staged(w, req.PlanID) && retirementGround(w) == "" {
				// SAME PLAN, DIFFERENT MODEL SELECTION (cl-114). The process is fine;
				// only its placement's selection differs from the request's. The launch
				// below resolves the request's own selection and EnsureWorker re-stages
				// this worker's placement under it — the desired set is authoritative and
				// the worker's own residency arbitration vacates the old selection.
				continue
			}
			stale = w.instanceID
		}
	}
	c.starting[slot] = true
	guards = append(guards, slot)
	c.mu.Unlock()
	if stale != "" {
		c.logf("worker %s staged no plan for %s and is STALE; replacing it", stale, req.PlanID)
		c.ShutdownWorker(stale, StopGrace)
	}

	go func() {
		// THE LAUNCH FLAG IS CLEARED BEFORE THE QUEUE IS RE-ASKED, and the order is the
		// whole point: `selectOrStart` returns early while `starting[slot]` is set, so a
		// revive that ran under a deferred clear would ALWAYS no-op on its own guard.
		// Found live — six queued jobs arrived during an in-flight launch for a different
		// plan, every one of them bounced off the flag, and the revive at the end of that
		// launch bounced off it too.
		done := func() {
			c.mu.Lock()
			for _, guard := range guards {
				delete(c.starting, guard)
			}
			c.mu.Unlock()
		}
		spec, planID, e := c.resolveFor(req)
		if e != nil {
			done()
			if e.ErrName() == "device_envelope_held" {
				c.logf("%s remains QUEUED while serving preparation awaits the local device envelope", req.ID)
				return
			}
			if deferred, _ := c.deferUnavailable(req, e); deferred {
				return
			}
			c.failQueued(req.ID, autoRentalGate(req, e), "")
			return
		}
		if planID != "" && planID != req.PlanID {
			// The worker resolved the binding this request routes on. The row is what the
			// drain routes from, so the plan is durable before the queue is re-asked.
			if e := c.opt.Store.BindRequestPlan(req.ID, planID); e != nil {
				done()
				c.failQueued(req.ID, e, "")
				return
			}
		}
		if planID != "" {
			req.PlanID = planID
		}
		instance, change, e := c.EnsureWorker(spec)
		if e != nil {
			done()
			// A local device grant held by another package is CAPACITY PRESSURE, not a
			// verdict on this request. The holder may still be executing (and must never
			// be preempted), or it may be between the terminal and the report that proves
			// it idle. Keep the durable request on the queue; a later idle-capacity report
			// re-enters select-or-start and the existing LRU safety fence decides whether
			// the holder can be reclaimed.
			if e.ErrName() == "device_envelope_held" {
				c.logf("%s remains QUEUED for the local device envelope: %s", req.ID, e.Message)
				return
			}
			if deferred, _ := c.deferUnavailable(req, e); deferred {
				return
			}
			c.failQueued(req.ID, autoRentalGate(req, e), "")
			return
		}
		c.logf("%s: %s is %s for the queued request", req.Package, instance, change)
		if e := c.EnsurePlacementReady(instance, req.PlanID, req.ID); e != nil {
			if deferred, _ := c.deferUnavailable(req, e); deferred {
				done()
				return
			}
			// AND THE WORKER GOES. A process that cannot make its binding resident still
			// holds a device grant, and `selectOrStart` returns early whenever a worker
			// for the package exists — so leaving it would hang the NEXT request behind a
			// worker that will never serve it, with nothing to start a replacement.
			if e.ErrName() == "worker_recycled" {
				// A COMPLETION, not a failure. The worker's own exit already asked the
				// queue for a replacement; failing the request here would settle a
				// requeued job after one of its budgeted attempts.
				c.logf("%s: the worker for %s recycled; the queue asks for the next one",
					req.Package, req.ID)
				done()
				return
			}
			done()
			c.failQueued(req.ID, autoRentalGate(req, e), instance)
			return
		}
		c.drain()
		done()
		// The launch is over and the queue's world has changed: whatever is at the head now
		// gets its own question asked, which is what closes the loop when this launch was
		// for a plan the head does not need.
		c.reviveQueue()
	}()
}

func (c *Orchestrator) deferUnavailable(req records.Request, problem *exit.Error) (deferred, news bool) {
	if !req.Rental || problem == nil || problem.Code != exit.Unavailable {
		return false, false
	}
	if c.QueuePosition(req.ID) == 0 {
		return false, false
	}
	return true, c.parkFor(req, problem.Message)
}

// parkFor parks a queued --rental request with the reason it waits, and says whether that
// reason is news.
func (c *Orchestrator) parkFor(req records.Request, reason string) bool {
	position := c.QueuePosition(req.ID)
	if position == 0 {
		return false
	}
	return c.park(req, position-1, waitFacts{}, reason)
}

func autoRentalGate(_ records.Request, cause *exit.Error) *exit.Error { return cause }

// requestSlot names what a request needs resident, before any pin: the package, an
// install for an editable one, a job function for a job.
func requestSlot(req records.Request) string {
	slot := req.Package
	if req.InstallID != "" {
		slot += "/install/" + req.InstallID
	}
	if req.IsJob() {
		slot += "/job/" + req.Entrypoint
	}
	return slot
}

// rentalHeld answers whether any attached rental has this request's plan staged under
// that rental's pinned package name — capacity `drain` routes onto the moment its lane
// has room, so the request waits unpinned. Callers hold c.mu.
func (c *Orchestrator) rentalHeld(req records.Request) bool {
	for _, w := range c.workers {
		if w.exited || w.stopping || !w.supportsCurrentProtocol() || w.spec.IsJob() != req.IsJob() ||
			retirementGround(w) != "" || w.spec.Connection == nil {
			continue
		}
		slot := pinnedPackage(req.Package, w.spec.Connection.RentalID)
		if req.IsJob() {
			if w.spec.Placement.Package == slot &&
				w.spec.Placement.Jobs[0].Function == req.Entrypoint && stagedFor(w, req) {
				return true
			}
			continue
		}
		if w.remoteStaged(slot, req.PlanID, req.Release, req.LocalPackageDigest, req.Models) {
			return true
		}
	}
	return false
}

// logPlacement is the decision log for the capacity half: no worker held the placement,
// so the fleet placed the run on an attached rental or a bought pod (cl-165), and this
// request is pinned there until its placement reports.
func (c *Orchestrator) logPlacement(req records.Request, decision PlacementDecision) {
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
	if req.Worker != "" && w.spec.Connection != nil && !req.IsJob() {
		return w.remoteStaged(pinnedPackage(req.Package, req.Worker), req.PlanID,
			req.Release, req.LocalPackageDigest, req.Models)
	}
	return staged(w, req.PlanID) && selectionServes(req.Models, w.spec.Placement.Models)
}

func exactJobSelection(placement DesiredPlacement, req records.Request) bool {
	if req.InstallID != "" && placement.InstallID != req.InstallID ||
		req.LocalPackageDigest != "" && placement.LocalRevisionDigest != req.LocalPackageDigest ||
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
		return true
	}
	holds := make(map[string]string, len(held))
	for _, m := range held {
		holds[m.Slot] = m.Manifest
	}
	for _, m := range requested {
		manifest, ok := holds[m.Slot]
		if !ok {
			continue
		}
		if _, fits := rungHolding(m, manifest); !fits {
			return false
		}
	}
	return true
}

// rungHolding answers whether a held manifest is one the request's ref accepts: its own
// pin, or — unpinned — any rung of its ladder (cl-166). The rung is what a dispatch onto
// that placement pins the request to.
func rungHolding(m ModelRef, manifest string) (records.ModelRung, bool) {
	if m.Pinned() {
		return records.ModelRung{Lane: m.Lane, Manifest: m.Manifest, Bytes: m.Bytes}, m.Manifest == manifest
	}
	for _, rung := range m.Ladder {
		if rung.Manifest == manifest {
			return rung, true
		}
	}
	return records.ModelRung{}, false
}

// pinToPlacement binds a request's unpinned refs to the manifests the placement that
// won routing already holds. Nil when nothing was unpinned.
func pinToPlacement(requested, held []ModelRef) []ModelRef {
	holds := make(map[string]string, len(held))
	for _, m := range held {
		holds[m.Slot] = m.Manifest
	}
	var out []ModelRef
	for i, m := range requested {
		manifest, ok := holds[m.Slot]
		if m.Pinned() || !ok {
			continue
		}
		rung, fits := rungHolding(m, manifest)
		if !fits {
			continue
		}
		if out == nil {
			out = append([]ModelRef(nil), requested...)
		}
		out[i] = m.Pin(rung)
	}
	return out
}

// settledState answers whether the authority has already recorded this request's outcome.
func settledState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// resolveFor keeps local package execution separate from generic rented capacity.
// A rented worker receives only Creator's logical package refs; the worker
// resolves and reports the exact binding it made dispatchable.
func (c *Orchestrator) resolveFor(req records.Request) (resolved WorkerLaunchSpec, planID string, problem *exit.Error) {
	defer func() {
		if problem == nil && resolved.IsJob() {
			resolved, problem = c.jobExecutionRole(req, resolved)
		}
	}()
	if req.Worker == "" {
		if c.opt.Packages == nil {
			return WorkerLaunchSpec{}, "", exit.Unavailablef("this host resolves no local packages")
		}
		if req.InstallID != "" {
			if req.IsJob() {
				spec, e := c.opt.Packages.ResolveJobInstall(req.InstallID, req.Entrypoint)
				return c.exactLocalTransferProducer(req, spec, e)
			}
			spec, e := c.opt.Packages.ResolveInstall(req.InstallID, req.Models)
			if e != nil {
				return WorkerLaunchSpec{}, "", e
			}
			if spec.Preparation != nil {
				return c.prepareLocalServing(req, spec)
			}
			if spec.Placement.Package != req.Package {
				return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict,
					"request_install_package_mismatch",
					"install %s serves %s, not request package %s",
					req.InstallID, spec.Placement.Package, req.Package)
			}
			return spec, req.PlanID, nil
		}
		spec, e := c.opt.Packages.Resolve(req.Package)
		if req.IsJob() {
			spec, e = c.opt.Packages.ResolveJob(req.Package, req.Entrypoint)
			if req.ModelTransfer != nil {
				return c.exactLocalTransferProducer(req, spec, e)
			}
		}
		return spec, req.PlanID, e
	}
	if c.opt.Rentals == nil {
		return WorkerLaunchSpec{}, "", exit.Unavailablef("this Cozy daemon attaches no remote workers")
	}
	remote, e := c.opt.Rentals(req.Worker)
	if e != nil {
		return WorkerLaunchSpec{}, "", e
	}
	if remote == nil || remote.Connection == nil {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Internal, "rental.target_incomplete",
			"rental %s resolved without a complete remote target", req.Worker)
	}
	if req.Release == "" ||
		(len(req.Models) == 0 && !validDigest(req.PlanID)) {
		return WorkerLaunchSpec{}, "", exit.Unavailablef(
			"remote package preparation requires one exact package revision")
	}
	logical := LogicalPackage{Package: req.Package, Release: req.Release,
		Function: req.Entrypoint,
		Outputs:  strings.FieldsFunc(req.Outputs, func(r rune) bool { return r == ',' }),
		PlanID:   req.PlanID, Models: append([]ModelRef(nil), req.Models...),
		NeedsAccelerator: req.NeedsAccelerator}
	instance, _, _, e := c.EnsureRental(req.Worker)
	if e != nil {
		return WorkerLaunchSpec{}, "", e
	}
	if req.RetainWork {
		if _, problem := c.rentalControl(req.Worker); problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		c.mu.Lock()
		var minor uint32
		if worker := c.workers[instance]; worker != nil {
			minor = worker.wireMinor
		}
		c.mu.Unlock()
		required, problem := c.requiredPrivateWire(req)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		if minor < required {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Structural, "request.retention_unsupported", "this private work requires worker wire %d; selected worker speaks %d", required, minor)
		}
	}
	var jobPrepared *pb.DesiredPlacementSet
	if req.InstallID != "" {
		if c.opt.Packages == nil || !validDigest(req.LocalPackageDigest) {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Structural,
				"local_package_request_incomplete",
				"editable rental request %s names no sealed local package revision", req.ID)
		}
		revision, problem := c.opt.Packages.LocalRevision(req.InstallID, req.LocalPackageDigest)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		if revision.Package != req.Package || revision.Release != req.Release ||
			revision.Digest != req.LocalPackageDigest {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict,
				"local_package_revision_changed",
				"request %s no longer matches its sealed local package revision", req.ID)
		}
		var parent *pb.JobDirective
		if req.ParentRequestID != "" {
			plan, problem := c.retainedOrchestrationParent(req)
			if problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			parent = c.jobDirective(plan)
		}
		c.mu.Lock()
		if worker := c.workers[instance]; worker != nil {
			worker.orchestrationParent = parent
		}
		c.mu.Unlock()
		if req.IsJob() {
			jobPrepared, problem = c.prepareLocalJob(instance, req, revision)
			if problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
		} else if e := c.ConvergeLocalPackage(instance, req.ID, revision,
			req.LocalPackageUploadedBootID, func(bootID string) *exit.Error {
				return c.opt.Store.MarkLocalPackageUploaded(req.ID, revision.Digest, bootID)
			}); e != nil {
			return WorkerLaunchSpec{}, "", e
		}
		if !req.IsJob() && len(logical.Models) > 0 && req.ParentRequestID != "" {
			native, problem := c.nativeServingModels(req)
			if problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			if problem := c.convergePrivateModels(instance, req.ID, req.LocalPackageDigest, nil, native); problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
		} else if !req.IsJob() && len(logical.Models) > 0 {
			models := downloadModelRefs(logical.Models)
			if len(models) != len(logical.Models) {
				return WorkerLaunchSpec{}, "", exit.Named(exit.Validation,
					"private_placement_model_unpublished",
					"private serving requires exact published model releases")
			}
			if e := c.ConvergePrivatePlacement(instance, req.ID,
				req.LocalPackageDigest, models); e != nil {
				return WorkerLaunchSpec{}, "", e
			}
		}
	} else {
		if c.opt.RentalPackageSet == nil || req.LocalPackageDigest != "" {
			return WorkerLaunchSpec{}, "", exit.Unavailablef(
				"published remote package preparation requires a package_set signer")
		}
		c.mu.Lock()
		held := preparedPlacementServes(c.workers[instance], logical)
		c.mu.Unlock()
		if held {
			c.logf("%s: rental %s already holds %s/%s under this selection; no package prepare",
				req.ID, req.Worker, logical.Package, logical.Function)
		} else if e := c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
			Package: logical.Package, Release: logical.Release,
		}}, downloadModelRefs(logical.Models)); e != nil {
			return WorkerLaunchSpec{}, "", e
		}
	}
	if req.IsJob() {
		preparedSet := ""
		var preparedBytes []byte
		if jobPrepared != nil {
			preparedSet = spellOf(jobPrepared.PlacementSetDigest)
			preparedBytes = jobPrepared.PlacementSetCanonicalBytes
		} else {
			if e := c.waitPackageStaged(instance); e != nil {
				return WorkerLaunchSpec{}, "", e
			}
			c.mu.Lock()
			if worker := c.workers[instance]; worker != nil {
				preparedSet = spellOf(worker.setDigest)
				preparedBytes = append([]byte(nil), worker.setBytes...)
			}
			c.mu.Unlock()
		}
		if !validDigest(preparedSet) {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Structural,
				"rental.package_preparation_identity_missing",
				"the rented worker staged %s without an exact PlacementSet", req.Package)
		}
		// The worker staged its own job plan records during that preparation, under its
		// own placement's identity. The directive names THAT, read back off the same
		// document — never this owner's set digest, which the worker never saw.
		buildID, e := JobBuildID(preparedBytes, req.Package)
		if e != nil {
			return WorkerLaunchSpec{}, "", e
		}
		weights, e := decodeWeightsOutputs(req.WeightsOutputs)
		if e != nil {
			return WorkerLaunchSpec{}, "", e
		}
		spec := WorkerLaunchSpec{Connection: remote.Connection, Devices: remote.Devices, Placement: DesiredPlacement{
			Package: pinnedPackage(req.Package, req.Worker), Release: req.Release,
			InstallID: req.InstallID, Models: append([]ModelRef(nil), logical.Models...),
			LocalRevisionDigest: req.LocalPackageDigest,
			PlacementSetDigest:  preparedSet,
			Jobs: []*JobPlan{{Function: req.Entrypoint, DescriptorID: req.PlanID,
				BuildID:        buildID,
				Outputs:        strings.FieldsFunc(req.Outputs, func(r rune) bool { return r == ',' }),
				WeightsOutputs: weights, RSSCap: DefaultJobRSSCap,
				NeedsAccelerator: req.NeedsAccelerator}},
		}}
		spec, e = c.jobExecutionRole(req, spec)
		if e != nil {
			return WorkerLaunchSpec{}, "", e
		}
		current, problem := c.opt.Store.RequestRow(req.ID)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		if current == nil || (current.State != "submitted" && current.State != "queued") {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "request.execution_stopped", "request stopped while its private job was being prepared")
		}
		if e := c.ConvergeRemoteJob(instance, spec); e != nil {
			return WorkerLaunchSpec{}, "", e
		}
		return spec, req.PlanID, nil
	}
	spec, planID, e := c.ensureLogicalPackageReady(instance, req.Worker, logical)
	if e != nil {
		return WorkerLaunchSpec{}, "", e
	}
	return spec, planID, nil
}

func (c *Orchestrator) exactLocalTransferProducer(req records.Request, spec WorkerLaunchSpec,
	problem *exit.Error,
) (WorkerLaunchSpec, string, *exit.Error) {
	if problem != nil {
		return spec, req.PlanID, problem
	}
	if req.Release != "" && spec.Placement.Release != req.Release {
		return WorkerLaunchSpec{}, req.PlanID, exit.Unavailablef(
			"local producer does not exactly match frozen %s@%s", req.Package, req.Release)
	}
	return spec, req.PlanID, nil
}

// failQueued settles a request that can never be placed. It is a request-level terminal:
// no offer crossed to a worker, so there is no worker terminal to replay and the request
// row is what settles. A closed dispatch_aborted row may remain as preparation history.
func (c *Orchestrator) failQueued(requestID string, cause *exit.Error, workerToStop string) {
	payload := map[string]any{"status": "FAILED", "cause": cause.ErrName(),
		"error_type": cause.ErrName(), "error": cause.Message,
		"outputs": []any{}, "requeuing": false}
	// The preparation waiter can outlive an accepted attempt or its successful
	// finalization. The store is the sole authority to fail queued work; neither
	// cleanup nor worker/provider teardown may run before that transaction wins.
	applied, problem := c.opt.Store.FailQueuedRequest(requestID, payload)
	if problem != nil {
		if problem.Code == exit.Conflict {
			c.logf("%s preparation failure no longer applies: %s", requestID, problem.Message)
			return
		}
		c.logf("%s could not be settled: %s", requestID, problem.Message)
		time.AfterFunc(2*time.Second, func() { c.failQueued(requestID, cause, workerToStop) })
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
	if workerToStop != "" {
		c.ShutdownWorker(workerToStop, StopGrace)
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
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return 0, problem
	}
	if current == nil || (current.State != "submitted" && current.State != "queued") {
		return 0, exit.Named(exit.Conflict, "request.execution_stopped", "request %s is not queued for execution", req.ID)
	}
	req = *current
	// PLACEMENT is the orchestrator's: the caller names the binding, and dispatch picks a
	// worker whose placement advertises it as DISPATCHABLE now and whose admission fence
	// is open. `pick` also returns the admission epoch it OBSERVED, which is what
	// makes a stale offer refuse deterministically rather than race.
	if hit, problem := c.lookupOperation(req); hit || problem != nil {
		return 0, problem
	}
	current, problem = c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return 0, problem
	}
	if current == nil || (current.State != "submitted" && current.State != "queued") {
		return 0, exit.Named(exit.Conflict, "request.execution_stopped", "request stopped while its operation lookup was in progress")
	}
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
	if req.RetainWork {
		required, problem := c.requiredPrivateWire(req)
		if problem != nil {
			return 0, problem
		}
		if w.wireMinor < required {
			return 0, exit.Named(exit.Structural, "request.retention_unsupported",
				"this private work requires worker wire %d; selected worker speaks %d", required, w.wireMinor)
		}
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
		// The placement that won routing holds one rung of the request's ladder; the
		// pin binds the request to that lane in the same write (cl-166).
		var models []ModelRef
		if placement, ok := w.remotePlacements[remotePlanKey(pinnedPackage(req.Package, rentalID), req.PlanID)]; ok {
			models = pinToPlacement(req.Models, placement.Models)
		}
		pinned, e := c.opt.Store.PinRental(req.ID, rentalID, models)
		if e != nil {
			return 0, e
		}
		if !pinned {
			return 0, exit.New(exit.Conflict, "request %s settled before it could be pinned to rental %s",
				req.ID, rentalID)
		}
		if models != nil {
			req.Models = models
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
	environmentDigest, e := c.invocationIdentity(w, req)
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
		if selected, ok := w.remotePlacements[remotePlanKey(pinnedPackage(req.Package, req.Worker), req.PlanID)]; ok {
			servingPlacement = selected
		}
		c.mu.Unlock()
		if servingPlacement.BindingsDigest == "" || len(servingPlacement.PlacementSetBytes) == 0 {
			return 0, exit.Named(exit.Conflict, "serving.placement_evidence_absent", "serving dispatch needs the exact prepared model bindings")
		}
	}
	spec := &pb.InvocationSpec{
		// `image_digest` is GONE, renamed to what it always meant (#483): "image" is wrong
		// for a native install with no OCI image at all. The value is the same one this
		// daemon was frozen with — a request cannot choose the environment it runs under.
		EnvironmentDigest: environmentDigest,
		PayloadDigest:     payloadDigest,
		Inputs:            inputBindings(req, payloadDigest),
		Outputs:           invocationOutputBindings(splitList(req.Outputs), weightsOutputs, outputLimit),
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
			BuildId:         w.spec.Placement.Jobs[0].BuildID,
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
	placementID := w.placementFor(pinnedPackage(req.Package, req.Worker), req.PlanID)
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

// invocationIdentity names the exact environment an
// invocation on w rides. Remote identity becomes durable here, after a dispatchable
// worker is selected and before the attempt ordinal is minted. Cold and warm requests
// therefore have one writer: neither package preparation nor rental selection needs a
// second identity path.
func (c *Orchestrator) invocationIdentity(w *worker,
	req records.Request) (environment string, e *exit.Error) {
	c.mu.Lock()
	placement, remote, instanceID := w.spec.Placement, w.spec.Connection != nil, w.instanceID
	if selected, ok := w.remotePlacements[remotePlanKey(
		pinnedPackage(req.Package, req.Worker), req.PlanID)]; remote && ok {
		placement = selected
	}
	c.mu.Unlock()
	if remote && (placement.Release != req.Release ||
		(req.LocalPackageDigest != "" && placement.LocalRevisionDigest != req.LocalPackageDigest)) {
		return "", exit.Named(exit.Conflict, "request_invocation_identity_changed",
			"worker %s no longer matches the release pinned to request %s", instanceID, req.ID)
	}
	if req.IsJob() && remote {
		return "", nil
	}
	environment = placement.EnvironmentDigest
	if environment == "" {
		if !remote && placement.SourceDigest != "" {
			return "", nil
		}
		return "", exit.Named(exit.Structural, "placement_identity_missing",
			"worker %s carries no selected environment digest", instanceID)
	}
	if remote {
		// A local revision is DEVELOPMENT execution on the pod, and development execution
		// has no published Environment identity: the worker refuses a spec that names one
		// (development_environment_present). The pod's prepared Environment is still bound
		// to the row, so a requeue derives the same identity; only the spec omits it.
		specEnvironment := environment
		if req.LocalPackageDigest != "" {
			specEnvironment = ""
		}
		if req.EnvironmentDigest == "" {
			e = c.opt.Store.BindRemoteInvocation(req.ID, req.PlanID, environment)
			if e != nil {
				return "", e
			}
			return specEnvironment, nil
		}
		if req.EnvironmentDigest != environment {
			return "", exit.Named(exit.Conflict,
				"request_invocation_identity_changed",
				"worker %s no longer matches the invocation identity pinned to request %s",
				instanceID, req.ID)
		}
		return specEnvironment, nil
	}
	return environment, nil
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
		path := model.Slot
		if model.BindingPath != "" {
			path = model.BindingPath
		}
		// ONE PLACEMENT PER CONSTRUCTION (h3a-018): the selection rides under every slot
		// that shares its bytes, so the pod binds each of those entrypoints in the one
		// placement it prepares.
		for _, slot := range append([]string{path}, model.SharedSlots...) {
			out = append(out, &pb.DownloadModelRef{Package: model.Package, Slot: slot,
				Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest})
		}
	}
	return out
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
	routed := c.route(req)
	pick := routed.pick()
	if pick == nil {
		return offerTarget{}, routed.noCapacity(req)
	}
	w := pick.worker
	target := offerTarget{worker: w, sess: c.sessions[w.bootID], admissionEpoch: w.admissionEpoch,
		routed: routed}
	c.overtaken(req.ID, laneKey{w.instanceID, pick.laneID})
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
	perOutput := c.maxOutputBytes()
	if len(outputIDs) > 0 && perOutput > uint64(math.MaxInt64)/uint64(len(outputIDs)) {
		return nil, exit.New(exit.Validation,
			"the output grant for %s#%d exceeds the media plane's byte range", req.ID, attempt)
	}
	reservedOutputBytes := int64(perOutput * uint64(len(outputIDs)))
	complete := false
	defer func() {
		if !complete {
			c.cleanupRemote(req.ID, attempt, w)
		}
	}()
	dir, e := w.media.ReserveOutputs(slot, reservedOutputBytes)
	if e != nil {
		return nil, e
	}
	path, e := w.media.PutInput(slot+"-payload", req.Payload)
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
		path, e := w.media.PutInputFile(slot+"-input-"+strconv.Itoa(index),
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
	for _, model := range req.Models {
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
