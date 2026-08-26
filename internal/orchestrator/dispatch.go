package orchestrator

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Submission is one local request. The orchestrator owns everything in it that decides
// WHAT runs; the runtime owns everything about HOW.
type Submission struct {
	IdemKey    string // the caller's idempotency key
	Endpoint   string // org/name
	Entrypoint string // the function
	PlanID     string // the entrypoint_binding_plan_id this attempt binds

	// Payload is the request body, verbatim. It rides the DeliveryGrant as the input
	// `payload` — a grant input, never a wire field, so refreshing the grant can never
	// substitute it.
	Payload []byte
	// Assets are immutable, service-staged files bound to exact payload field paths.
	// The request row keeps them so every requeue derives the same spec and grant.
	Assets []records.AssetBinding

	// Outputs is one destination per RESULT FIELD PATH (`image`, `detail.thumb`). Binding
	// by field path with exact set equality is what makes a two-output result
	// unswappable; a positional grant would silently cross them (decisions #248).
	Outputs []string

	// BodyDigest is the caller's own digest of the WHOLE submission it is making
	// idempotent, not merely of the payload. cl-006 supplies the digest of
	// (endpoint, function, input, outputs) so that one key naming a different ENDPOINT
	// conflicts as loudly as one naming different input — a digest over the payload
	// alone would let a key be reused across functions and mean two different things.
	// Empty falls back to the payload's digest.
	BodyDigest string

	// Kind is the ATTEMPT CLASS: "" or `serving`, or `job`. A job carries two more facts
	// a serving request has no version of.
	Kind string
	// Org is the publishing org whose scratch repo this job publishes into.
	Org string
	// Trees are the job's typed input TREES as `ref=dir`, one grant input each.
	Trees []string

	// Worker pins this request to an ATTACHED remote worker (a rental id resolved
	// through Options.Rentals). Empty = any local worker.
	Worker string
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
	bodyDigest := s.BodyDigest
	if bodyDigest == "" {
		spelled, err := canonical.Spell(canonical.Digest(s.Payload))
		if err != nil {
			return "", 0, false, exit.Internalf("cannot digest the request body: %s", err)
		}
		bodyDigest = spelled
	}
	id := records.NewID("req")
	if s.Kind == "job" {
		id = records.NewID("job")
	}
	req, fresh, e := c.opt.Store.Submit(records.Request{
		ID: id, IdemKey: s.IdemKey, BodyDigest: bodyDigest,
		Endpoint: s.Endpoint, Entrypoint: s.Entrypoint, PlanID: s.PlanID, Payload: s.Payload,
		Outputs: strings.Join(s.Outputs, ","),
		Assets:  s.Assets,
		Kind:    s.Kind, Org: s.Org, Trees: strings.Join(s.Trees, ","),
		Worker: s.Worker,
	})
	if e != nil {
		return "", 0, false, e
	}
	if !fresh {
		// The recorded answer. A settled request is settled; a live one is already
		// running the attempt this call would otherwise duplicate.
		c.logf("request %s is the recorded answer for idempotency key %s (state %s, attempt %d)",
			req.ID, s.IdemKey, req.State, req.Ordinal)
		return req.ID, uint64(req.Ordinal), false, nil
	}
	c.emit(req.ID, "request.submitted", 0, map[string]any{
		"endpoint": s.Endpoint, "function": s.Entrypoint,
		"body_digest": bodyDigest, "plan_id": s.PlanID, "outputs": s.Outputs,
	})
	// A SUBMISSION NEVER OVERTAKES WORK ALREADY WAITING. Dispatching straight from submit
	// is what keeps a warm request fast, and it is exactly what breaks FIFO when a queue
	// exists: a request arriving while six are parked would take the free slot the head of
	// the queue is waiting for. Found live by cl-004's depth pass — the LAST of six
	// submissions settled FIRST. So: with a queue, join it; the drain below still runs
	// immediately, so the head goes out now rather than at the next Report (cr-019).
	if c.queueDepth() > 0 {
		c.enqueue(req.ID)
		c.emit(req.ID, "request.queued", 0, map[string]any{
			"reason":   "the dispatch queue is not empty; this request joins it in submission order",
			"position": c.QueuePosition(req.ID),
		})
		c.logf("%s QUEUED behind %d waiting request(s)", req.ID, c.QueuePosition(req.ID)-1)
		c.selectOrStart(req)
		go c.drain()
		return req.ID, 0, true, nil
	}
	attempt, e := c.dispatch(req)
	if e != nil {
		// NO CAPACITY is a STATE, not a failure (cl-006): the request row is already
		// durable, so refusing it here would mean the client holds an id for something
		// that never runs. It waits for capacity exactly as a requeue does — one queue,
		// one projection. Every OTHER refusal (an open recovered obligation, a live
		// attempt) is a real conflict and still refuses.
		if e.Code != exit.Unavailable {
			c.failQueued(req.ID, e)
			return req.ID, 0, true, e
		}
		c.enqueue(req.ID)
		c.emit(req.ID, "request.queued", 0, map[string]any{"reason": e.Message})
		c.logf("%s QUEUED for capacity: %s", req.ID, e.Message)
		c.selectOrStart(req)
		return req.ID, 0, true, nil
	}
	return req.ID, attempt, true, nil
}

// Requeue is the orchestrator's PROJECTION over a neutral terminal: an ABANDONED attempt
// or an infra-class failure earns a NEW ordinal, a fresh grant and a fresh execution —
// never a patch to the one that died. It is charged against the request's durable budget.
func (c *Orchestrator) Requeue(requestID, why string) {
	n, started, e := c.opt.Store.BeginRequeue(requestID, MaxRequeues)
	if e != nil {
		// THE REQUEST ENDS HERE, and it has to SAY so. Settling the row without emitting a
		// terminal event left a client watching the durable stream with `attempt_failed
		// (requeuing: true)` as its last frame and nothing after it — the contract's
		// terminal-stop rule never fired, and `cozy run` waited on a request that had been
		// settled for ten minutes. Observed live, in cl-003's ARM 3.
		c.logf("%s NOT requeued (%s): %s", requestID, why, e.Message)
		c.forget(requestID)
		_ = c.opt.Store.SettleRequest(requestID, "failed")
		c.frames.forget(requestID)
		c.emit(requestID, "request.failed", 0, map[string]any{
			"status": "FAILED", "cause": "REQUEUE_BUDGET_EXHAUSTED",
			"error_type": e.ErrName(), "error": e.Message,
			"outputs": []any{}, "requeuing": false,
		})
		c.waitRequest(requestID).markClosed(e)
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
		c.emit(requestID, "request.queued", 0, map[string]any{"reason": e.Message, "requeues": n})
		c.logf("%s requeued %d/%d and QUEUED for capacity: %s", requestID, n, MaxRequeues, e.Message)
		c.selectOrStart(*req)
		return
	}
	c.logf("%s requeued as attempt %d (%d/%d of the budget, cause %s)",
		requestID, attempt, n, MaxRequeues, why)
}

// selectOrStart makes a queued request's endpoint resident. It is the half of `cozy run`
// that "cold and warm traverse the same states" rests on: SELECT the worker that already
// advertises the binding, or START one — never a second invocation mechanism, and never a
// client's job. `cozy start` is the same act made explicit for prewarming.
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
	if c.opt.Endpoints == nil {
		return
	}
	// A JOB names its own slot — one worker per (endpoint, job function) — so the
	// "already starting" and "already resident" questions are asked about that slot and
	// not about the endpoint. Without this, submitting a job while a serving worker of
	// the same endpoint is up would decide a job worker already existed.
	// A request PINNED to a rental asks its questions about the rental's own slot: the
	// attached worker's spec carries the pinned endpoint name, so comparing against the
	// bare one would decide no worker was resident and attach a second control stream to
	// the pod on every request.
	slot := pinnedEndpoint(req.Endpoint, req.Worker)
	if req.IsJob() {
		slot += "/job/" + req.Entrypoint
	}
	c.mu.Lock()
	if c.starting[slot] {
		c.mu.Unlock()
		return
	}
	stale := ""
	for _, w := range c.workers {
		if w.exited || w.spec.Placement.Endpoint != pinnedEndpoint(req.Endpoint, req.Worker) ||
			w.spec.IsJob() != req.IsJob() {
			continue
		}
		if !req.IsJob() || w.spec.Placement.Jobs[0].Function == req.Entrypoint {
			// RESIDENCY IS ABOUT THE PLAN, not about the slot. A worker that STAGED this
			// request's plan is already resident or loading, and its READY drains the
			// queue — which is exactly how several submitted jobs queue against ONE
			// worker (cr-019). A worker in the same slot that staged a DIFFERENT plan is
			// STALE: the generation's surface moved under it, and treating it as capacity
			// queues the request behind a worker that will never advertise what it needs.
			// Found live by cl-004's escape arm, which changes the descriptor and
			// therefore the job descriptor id: six queued jobs waited on a worker holding
			// the previous digest, forever.
			if staged(w, req.PlanID) {
				c.mu.Unlock()
				return
			}
			stale = w.instanceID
		}
	}
	c.starting[slot] = true
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
			delete(c.starting, slot)
			c.mu.Unlock()
		}
		spec, e := c.resolveFor(req)
		if e != nil {
			done()
			c.failQueued(req.ID, e)
			return
		}
		instance, change, e := c.EnsureWorker(spec)
		if e != nil {
			done()
			c.failQueued(req.ID, e)
			return
		}
		c.logf("%s: %s is %s for the queued request", req.Endpoint, instance, change)
		if e := c.EnsurePlacementReady(instance, req.PlanID); e != nil {
			// AND THE WORKER GOES. A process that cannot make its binding resident still
			// holds a device grant, and `selectOrStart` returns early whenever a worker
			// for the endpoint exists — so leaving it would hang the NEXT request behind a
			// worker that will never serve it, with nothing to start a replacement.
			if e.ErrName() == "worker_recycled" {
				// A COMPLETION, not a failure. The worker's own exit already asked the
				// queue for a replacement; failing the request here would settle a
				// requeued job after one of its budgeted attempts.
				c.logf("%s: the worker for %s recycled; the queue asks for the next one",
					req.Endpoint, req.ID)
				done()
				return
			}
			c.ShutdownWorker(instance, StopGrace)
			done()
			c.failQueued(req.ID, e)
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

// settledState answers whether the authority has already recorded this request's outcome.
func settledState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// resolveFor asks the launcher for the spec of the LANE this request runs in. A request
// pinned to a connected worker (cl-015) resolves the SAME spec — same bindings,
// same plan ids — and gains the dial triple: the pod installed the identical release, so
// the plan digests agree by construction or the worker refuses typed.
func (c *Orchestrator) resolveFor(req records.Request) (WorkerLaunchSpec, *exit.Error) {
	if req.Worker == "" {
		spec, e := c.opt.Endpoints.Resolve(req.Endpoint)
		if req.IsJob() {
			spec, e = c.opt.Endpoints.ResolveJob(req.Endpoint, req.Entrypoint)
		}
		return spec, e
	}
	// Serving placements are the first cl-020 consumer: their descriptor and binding
	// plans belong in platform-neutral control metadata. Jobs retain their existing local
	// materialized path until the job half of that manifest contract is defined.
	var placement DesiredPlacement
	var e *exit.Error
	if req.IsJob() {
		var spec WorkerLaunchSpec
		spec, e = c.opt.Endpoints.ResolveJob(req.Endpoint, req.Entrypoint)
		placement = spec.Placement
	} else {
		placement, e = c.opt.Endpoints.ResolvePlacement(req.Endpoint)
	}
	if e != nil {
		return WorkerLaunchSpec{}, e
	}
	if c.opt.Rentals == nil {
		return WorkerLaunchSpec{}, exit.Unavailablef("this LocalService attaches no remote workers")
	}
	remote, e := c.opt.Rentals(req.Worker)
	if e != nil {
		return WorkerLaunchSpec{}, e
	}
	spec := WorkerLaunchSpec{Placement: placement, Connection: remote}
	// The rental IS the slot: one connected worker per rental id, its own instance
	// namespace, and no local device envelope (the pod's card is the pod's).
	spec.Placement.Endpoint = pinnedEndpoint(spec.Placement.Endpoint, req.Worker)
	return spec, nil
}

// failQueued settles a request that can never be placed. It is a request-level terminal:
// no offer crossed to a worker, so there is no worker terminal to replay and the request
// row is what settles. A closed dispatch_aborted row may remain as preparation history.
func (c *Orchestrator) failQueued(requestID string, cause *exit.Error) {
	c.forget(requestID)
	// A REQUEST THAT ALREADY SETTLED IS NOT FAILED BY A LATER OBSERVATION. The launch
	// goroutine that made this request's worker resident OUTLIVES the request: a job
	// worker is terminal-and-reclaim, so it EXITS the moment its terminal is acknowledged,
	// and `EnsurePlacementReady` then answers "the endpoint worker exited before reporting ready" —
	// about a process whose exit was the successful end of the work.
	//
	// Observed live in cl-004's crash arm, and it is the worst failure class this system
	// has: `request.completed` followed by `request.failed` on ONE request, with the row
	// overwritten to `failed` after its publication had already committed. The request's
	// own settled state is the authority; nothing that happens to a process afterwards may
	// contradict it.
	if row, e := c.opt.Store.RequestRow(requestID); e == nil && row != nil && settledState(row.State) {
		c.logf("%s already settled %s — NOT failing it over: %s",
			requestID, row.State, cause.Message)
		return
	}
	if e := c.opt.Store.SettleRequest(requestID, "failed"); e != nil {
		c.logf("%s could not be settled: %s", requestID, e.Message)
	}
	if row, e := c.opt.Store.RequestRow(requestID); e == nil && row != nil {
		go c.cleanupRequestAssets(*row)
	}
	c.emit(requestID, "request.failed", 0, map[string]any{
		"status": "FAILED", "cause": cause.ErrName(),
		"error_type": cause.ErrName(), "error": cause.Message,
		"outputs": []any{}, "requeuing": false,
	})
	c.logf("%s FAILED before any offer: %s", requestID, cause.Message)
	c.waitRequest(requestID).markClosed(cause)
}

func (c *Orchestrator) dispatch(req records.Request) (uint64, *exit.Error) {
	// PLACEMENT is the orchestrator's: the caller names the binding, and dispatch picks a
	// worker whose placement advertises it as DISPATCHABLE now and whose admission fence
	// is open. `pick` also returns the admission generation it OBSERVED, which is what
	// makes a stale offer refuse deterministically rather than race.
	w, sess, admissionGen, reservation, e := c.pick(req)
	if e != nil {
		return 0, e
	}
	reserved := true
	defer func() {
		if reserved {
			c.releaseDispatch(reservation)
		}
	}()

	// The InvocationSpec DOCUMENT (#439): everything that gives the invocation meaning —
	// the payload digest, the ORDERED input identities, the output contracts, the
	// deadline — lives INSIDE the digest. Its key set is closed: no human model ref, no
	// service class, no local extension has a slot.
	spec := &pb.InvocationSpec{
		EndpointReleaseId: w.spec.Placement.ReleaseID,
		// `image_digest` is GONE, renamed to what it always meant (#483): "image" is wrong
		// for a native install with no OCI image at all. The value is the same one this
		// service was frozen with — a request cannot choose the environment it runs under.
		EnvironmentSpecDigest: c.opt.EnvironmentSpecDigest,
		ConfigDigest:          c.opt.ConfigDigest,
		PayloadDigest:         spellOf(canonical.Digest(req.Payload)),
		Inputs:                inputBindings(req),
		Outputs:               outputBindings(splitList(req.Outputs), c.maxOutputBytes()),
		Spec: &pb.InvocationSpec_Serving{Serving: &pb.ServingInvocationSpec{
			EntrypointBindingPlanId: req.PlanID,
			// With no adapters the binding IS the plan, so the two ids are equal by
			// construction rather than by copying a value around.
			AttemptBindingId: req.PlanID,
		}},
	}
	if req.IsJob() {
		// ONE mode names ONE spec. The per-attempt publication contract names THIS
		// request's scratch repo, which is why a queue-serving worker can hold one
		// directive and still publish each attempt into its own place.
		spec.Spec = &pb.InvocationSpec_Job{Job: &pb.JobInvocationSpec{
			BuildId:         w.spec.Placement.ReleaseID,
			JobDescriptorId: req.PlanID,
			PublicationContract: &pb.PublicationContract{
				GrantId: home.ScratchRepo(req.Org, req.ID),
				Outputs: outputBindings(splitList(req.Outputs), maxOutputBytes),
			},
		}}
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
	})
	if e != nil {
		return 0, e
	}
	attempt := uint64(ordinal)

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

	placementID := w.placementID
	if req.IsJob() {
		placementID = "" // job mode routes by the directive, not a placement (#446/#481)
	}
	// AN OFFER, NOT A START (#481): receiving one begins VALIDATION, not execution, which
	// is why refusing it is an ordinary journaled outcome rather than an exception. EVERY
	// offer gets AttemptAccepted or a journaled AttemptOutcome(REFUSED) — never silence.
	offer := &pb.AttemptOffer{
		RequestId: req.ID, AttemptOrdinal: attempt, InvocationSpecDigest: digest,
		Grant: grant, InvocationSpecCanonicalBytes: canonicalBytes, PlacementId: placementID,
		// THE GENERATION THIS OWNER OBSERVED WHEN IT DISPATCHED. The worker admits only if
		// this is still current; a stale echo refuses deterministically — same input, same
		// verdict, no race window — instead of running under capacity meaning that moved.
		AdmissionGeneration: admissionGen,
	}
	offer.RecordOwnerEpoch, offer.ControlStreamGeneration, offer.WorkerBootId =
		recordOwnerEpoch, sess.generation, sess.bootID
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
	c.logf("AttemptOffer %s#%d spec=%s (%d canonical bytes) placement=%s admission=%d outputs=%s on %s",
		req.ID, attempt, shortDigest(spelled), len(canonicalBytes), placementID,
		admissionGen, req.Outputs, w.instanceID)
	c.emit(req.ID, "request.dispatched", attempt, map[string]any{
		"instance_id": w.instanceID, "invocation_digest": spelled,
	})
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

func (c *Orchestrator) rollbackGrant(req records.Request, attempt uint64, w *worker) {
	if w.media != nil {
		c.cleanupRemote(req.ID, attempt, w)
		return
	}
	path := c.opt.Layout.AttemptDir(req.ID, attempt)
	if req.IsJob() {
		path = c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)
	}
	if err := os.RemoveAll(path); err != nil {
		c.logf("%s#%d local grant rollback failed: %s", req.ID, attempt, err)
	}
}

func (c *Orchestrator) maxOutputBytes() uint64 {
	maxBytes := c.opt.MaxOutputMiB
	if maxBytes <= 0 {
		maxBytes = 64
	}
	return uint64(maxBytes) << 20
}

func spellOf(raw []byte) string {
	s, err := canonical.Spell(raw)
	if err != nil {
		return ""
	}
	return s
}

// inputBindings is the spec's ORDERED input identity list (#439): the payload always,
// plus a job's materialized trees (their verification is the store's own, so no digest).
func inputBindings(req records.Request) []*pb.InputBinding {
	rows := []*pb.InputBinding{{
		InputId:  "payload",
		Digest:   spellOf(canonical.Digest(req.Payload)),
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
	order := uint32(len(req.Assets) + 1)
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

func outputBindings(ids []string, maxBytes uint64) []*pb.OutputBinding {
	out := make([]*pb.OutputBinding, 0, len(ids))
	for _, id := range ids {
		out = append(out, &pb.OutputBinding{OutputId: id, MaxBytes: maxBytes})
	}
	return out
}

// pick resolves a worker whose PLACEMENT is DISPATCHABLE for this binding and whose
// worker-level ADMISSION FENCE will take it. Compatibility and capacity matching stay
// here, in the orchestrator, exactly as they do in the cloud.
//
// TWO GATES, TWO OWNERS (#472e/#482). Dispatchability is a PLACEMENT property — the
// serving axis and `dispatchable_plan_ids`. Capacity is a WORKER property — the admission
// state, its generation, and the one shared seat window. Per-placement `attempt_credits`
// are DELETED because N counters over ONE serialized device advertise N times the real
// capacity, and that defect is arithmetic rather than a race.
//
// THE SLOT IS PART OF THE MATCH, not only the binding. Matching on the plan id alone sent
// a request pinned to rental B to rental A's worker — same endpoint, same plan digest, so
// it looked like capacity — which made the pin advisory and, worse, let a request run on a
// pod whose credential it never presented. It cuts the other way too: an UNPINNED request
// must never land on a rented worker, because someone is being billed for that card and
// nobody asked for it here.
type dispatchReservation struct {
	worker *worker
	job    bool
	planID string
}

func (c *Orchestrator) pick(req records.Request) (*worker, *session, uint64, *dispatchReservation, *exit.Error) {
	planID := req.PlanID
	slot := pinnedEndpoint(req.Endpoint, req.Worker)
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, w := range c.workers {
		if w.exited || w.spec.Placement.Endpoint != slot {
			continue
		}
		sess := c.sessions[w.bootID]
		if sess == nil {
			continue
		}
		if w.spec.IsJob() {
			// A JOB worker hosts no placement: its dispatchability IS its job capacity.
			if !w.dispatchable[planID] || w.jobsAvail <= 0 {
				continue
			}
			// RESERVE the seat this dispatch is about to consume. The worker's own next
			// observed state is still the authority — this only stops ONE drain pass from
			// handing two queued jobs to a one-attempt worker on one reading.
			w.reservedJobs++
			w.jobsAvail = max(0, w.reportedJobs-w.reservedJobs)
			if w.jobsAvail <= 0 {
				w.dispatchable[planID] = false
			}
			return w, sess, w.admissionGen,
				&dispatchReservation{worker: w, job: true, planID: planID}, nil
		}
		if !w.dispatchableFor(planID) {
			continue
		}
		if !w.admissible() {
			// CLOSED and OPEN-with-no-seat are BOTH "not now", and they are deliberately
			// distinguished on the worker facts rather than here: this side's answer is the
			// same either way — the request parks in the deep queue, which is this owner's
			// and never the worker's.
			continue
		}
		w.reservedSlots++
		w.slots = max(0, w.reportedSlots-w.reservedSlots)
		return w, sess, w.admissionGen, &dispatchReservation{worker: w}, nil
	}
	return nil, nil, 0, nil, exit.Unavailablef(
		"no claimed worker in %s has a DISPATCHABLE placement for %s with a free attempt slot",
		slot, planID)
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
	r.worker.reservedSlots = max(0, r.worker.reservedSlots-1)
	r.worker.observeSlots(r.worker.reportedSlots)
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
		}
		r.worker.observeJobs(r.worker.reportedJobs)
		if !consumed && r.worker.jobsAvail > 0 {
			r.worker.dispatchable[r.planID] = true
		}
		return
	}
	r.worker.reservedSlots = max(0, r.worker.reservedSlots-1)
	if consumed {
		r.worker.reportedSlots = max(0, r.worker.reportedSlots-1)
	}
	r.worker.observeSlots(r.worker.reportedSlots)
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
	if req.IsJob() {
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
		WithRemedy("use --worker with a remote pod; local file URL authorization is currently POSIX-only")
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
	assets := make([][]byte, len(req.Assets))
	for i, asset := range req.Assets {
		data, e := inputasset.Read(asset, inputasset.MaxBytes)
		if e != nil {
			return nil, e
		}
		assets[i] = data
	}
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
	for index, asset := range req.Assets {
		path, e := w.media.PutInput(slot+"-input-"+strconv.Itoa(index), assets[index])
		if e != nil {
			return nil, e
		}
		g.Inputs = append(g.Inputs, &pb.InputAccess{InputId: asset.FieldPath, Url: "file://" + path})
		c.logf("%s#%d: input asset %s (%d B, %s) crossed to %s at %s",
			req.ID, attempt, asset.FieldPath, asset.Length, asset.Digest, w.media.Addr(), path)
	}
	for _, id := range outputIDs {
		g.Outputs = append(g.Outputs, &pb.OutputAccess{
			OutputId: id, Url: "file://" + dir + "/" + id,
		})
	}
	complete = true
	return g, nil
}

// grant builds the LOCAL delivery grant: a payload input and one destination per result
// field path, under this attempt's own directory. There is no credential — a local grant
// is a CAS root plus an output dir, and a fabricated token would be a lie about
// authority nobody issued.
func (c *Orchestrator) grant(requestID string, attempt uint64, req records.Request) (*pb.DeliveryGrant, *exit.Error) {
	dir := c.opt.Layout.AttemptDir(requestID, attempt)
	inDir := filepath.Join(dir, "in")
	if err := os.MkdirAll(inDir, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the attempt directory %s: %s", dir, err)
	}
	payloadPath := filepath.Join(inDir, "payload")
	if err := os.WriteFile(payloadPath, req.Payload, 0o644); err != nil {
		return nil, exit.Internalf("cannot stage the request payload: %s", err)
	}
	g := &pb.DeliveryGrant{
		FileBaseUrl: "file://" + dir,
		// NO EXPIRY, because this host mints no deadline to derive one from. A grant lasts
		// as long as the attempt it was minted for (cr-009), and the attempt's bound is the
		// caller's deadline — which `dispatch` never sets, because there is no wire field
		// for it here and `--timeout` is enforced client-side by CANCELLING. 10 minutes was
		// therefore a ceiling on how long a local attempt could be, invented at the one
		// place nobody was asked. The worker reads 0 as "does not expire" (`grants.expired`).
		ExpiresAtUnix: 0,
		// ACCESS ONLY (#439): the identities (digest, length, media kind) live in the
		// spec's bindings, inside the invocation digest.
		Inputs: []*pb.InputAccess{{InputId: "payload", Url: "file://" + payloadPath}},
	}
	for _, asset := range req.Assets {
		if _, e := inputasset.Read(asset, inputasset.MaxBytes); e != nil {
			return nil, e
		}
		g.Inputs = append(g.Inputs, &pb.InputAccess{
			InputId: asset.FieldPath, Url: "file://" + asset.LocalPath,
		})
	}
	for _, id := range strings.Split(req.Outputs, ",") {
		if id == "" {
			continue
		}
		g.Outputs = append(g.Outputs, &pb.OutputAccess{
			OutputId: id,
			Url:      "file://" + filepath.Join(dir, id),
		})
	}
	return g, nil
}

// Await blocks until the attempt is closed — the terminal accepted, its outputs visible,
// and the ack sent. The error it returns is the orchestrator's PROJECTION of the neutral
// terminal, never a worker-authored retryability claim.
func (c *Orchestrator) Await(requestID string, attempt uint64, timeout time.Duration) (*Result, *exit.Error) {
	w := c.waitFor(key(requestID, attempt))
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
	w := c.waitRequest(requestID)
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
	w := c.waitFor(key(requestID, attempt))
	select {
	case <-w.accepted:
		return nil
	case <-w.closed:
		return w.err
	case <-time.After(timeout):
		return exit.New(exit.Deadline, "%s#%d was not accepted in %s", requestID, attempt, timeout)
	}
}

// CancelClient is the client-reason cancel, for the callers that have no business
// naming a protocol enum.
func (c *Orchestrator) CancelClient(requestID string, attempt, graceMS uint64) *exit.Error {
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
	cancel.RecordOwnerEpoch, cancel.ControlStreamGeneration, cancel.WorkerBootId =
		recordOwnerEpoch, sess.generation, sess.bootID
	sess.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_CancelAttempt{CancelAttempt: cancel}})
	c.logf("CancelAttempt %s#%d reason=%s", requestID, attempt,
		pb.CancelReason_name[int32(reason)])
	return nil
}
