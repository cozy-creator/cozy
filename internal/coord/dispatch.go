package coord

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Submission is one local request. The coordinator owns everything in it that decides
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
// coordinator's requeue PROJECTION over a terminal, never a client repeating itself.
func (c *Coordinator) Submit(s Submission) (string, uint64, *exit.Error) {
	id, attempt, _, e := c.SubmitDetail(s)
	return id, attempt, e
}

// SubmitDetail is Submit plus the one fact an HTTP host must not guess: whether THIS
// call started the work. A client that retried a timed-out POST needs "202, I started it"
// and "200, this key was already yours" to be different answers, and inferring it from
// equal request ids is a race.
func (c *Coordinator) SubmitDetail(s Submission) (string, uint64, bool, *exit.Error) {
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
		Kind:    s.Kind, Org: s.Org, Trees: strings.Join(s.Trees, ","),
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

// Requeue is the coordinator's PROJECTION over a neutral terminal: an ABANDONED attempt
// or an infra-class failure earns a NEW ordinal, a fresh grant and a fresh execution —
// never a patch to the one that died. It is charged against the request's durable budget.
func (c *Coordinator) Requeue(requestID, why string) {
	n, e := c.opt.Store.ChargeRequeue(requestID, MaxRequeues)
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
		return
	}
	req, e := c.opt.Store.RequestRow(requestID)
	if e != nil || req == nil {
		return
	}
	// THE REQUEUE IS A FACT THE MOMENT THE BUDGET IS CHARGED, and it is announced here —
	// before dispatch, which may or may not find capacity. Announcing it only on the
	// successful branch lost the fact exactly when it mattered most: the coordinator-kill
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
func (c *Coordinator) selectOrStart(req records.Request) {
	if c.opt.Endpoints == nil {
		return
	}
	// A JOB names its own slot — one worker per (endpoint, job function) — so the
	// "already starting" and "already resident" questions are asked about that slot and
	// not about the endpoint. Without this, submitting a job while a serving worker of
	// the same endpoint is up would decide a job worker already existed.
	slot := req.Endpoint
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
		if w.exited || w.spec.Endpoint != req.Endpoint || w.spec.IsJob() != req.IsJob() {
			continue
		}
		if !req.IsJob() || w.spec.Jobs[0].Function == req.Entrypoint {
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
		c.StopWorker(stale, 20*time.Second)
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
		instance, e := c.StartWorker(spec)
		if e != nil {
			done()
			c.failQueued(req.ID, e)
			return
		}
		c.logf("%s: started %s for the queued request", req.Endpoint, instance)
		if e := c.WaitReady(instance, req.PlanID, 30*time.Minute); e != nil {
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
			c.StopWorker(instance, 20*time.Second)
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

// resolveFor asks the launcher for the spec of the LANE this request runs in.
func (c *Coordinator) resolveFor(req records.Request) (EndpointSpec, *exit.Error) {
	if req.IsJob() {
		return c.opt.Endpoints.ResolveJob(req.Endpoint, req.Entrypoint)
	}
	return c.opt.Endpoints.Resolve(req.Endpoint)
}

// failQueued settles a request that can never be placed. It is a request-level terminal:
// no attempt was ever dispatched, so there is no attempt terminal to replay and the
// request row is what settles.
func (c *Coordinator) failQueued(requestID string, cause *exit.Error) {
	c.forget(requestID)
	// A REQUEST THAT ALREADY SETTLED IS NOT FAILED BY A LATER OBSERVATION. The launch
	// goroutine that made this request's worker resident OUTLIVES the request: a job
	// worker is terminal-and-reclaim, so it EXITS the moment its terminal is acknowledged,
	// and `WaitReady` then answers "the endpoint worker exited before reporting ready" —
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
	c.emit(requestID, "request.failed", 0, map[string]any{
		"status": "FAILED", "cause": cause.ErrName(),
		"error_type": cause.ErrName(), "error": cause.Message,
		"outputs": []any{}, "requeuing": false,
	})
	c.logf("%s FAILED before any attempt: %s", requestID, cause.Message)
	c.waitRequest(requestID).markClosed(cause)
}

func (c *Coordinator) dispatch(req records.Request) (uint64, *exit.Error) {
	// PLACEMENT is the coordinator's: the caller names the binding, and dispatch picks a
	// worker that advertises it as dispatchable NOW.
	w, sess, e := c.pick(req.PlanID)
	if e != nil {
		return 0, e
	}

	// The ExecutionSpec DOCUMENT. Its key set is closed and is exactly this message's
	// fields: no human model ref, no service class, no local extension has a slot.
	spec := &pb.ExecutionSpec{
		EndpointReleaseId: w.spec.ReleaseID,
		ImageDigest:       c.opt.ImageDigest,
		ConfigDigest:      c.opt.ConfigDigest,
		Spec: &pb.ExecutionSpec_Serving{Serving: &pb.ServingExecutionSpec{
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
		spec.Spec = &pb.ExecutionSpec_Job{Job: &pb.JobExecutionSpec{
			BuildId:         w.spec.ReleaseID,
			JobDescriptorId: req.PlanID,
			PublicationContract: &pb.PublicationContract{
				GrantId: home.ScratchRepo(req.Org, req.ID),
				Outputs: outputSpecs(splitList(req.Outputs)),
			},
		}}
	}
	canonicalBytes, digest, err := canonical.Identity(spec)
	if err != nil {
		return 0, exit.Internalf("cannot mint the ExecutionSpec document: %s", err)
	}
	spelled, _ := canonical.Spell(digest)

	// THE LAW AND THE ORDINAL, in ONE transaction that also journals the assignment: a
	// terminal crossing a restart is authorized by the persisted assignment, and an ordinal
	// that is minted outside the write that records it is a race (cl-003's finding — two
	// `drain()` goroutines minted 1 and 2 for one request across the old split call).
	ordinal, e := c.opt.Store.Dispatch(records.Attempt{
		RequestID: req.ID, InstanceID: w.instanceID,
		SessionID: w.sessionID, ExecSpecDigest: spelled, ExecSpec: canonicalBytes,
	})
	if e != nil {
		return 0, e
	}
	attempt := uint64(ordinal)

	// The grant names this attempt's own directory, so it is built after the ordinal is
	// real. A failure here is a broken local filesystem under our own root: the journaled
	// row stands as the record of a dispatch that could not be granted, and the request
	// answers with the error rather than silently retrying into the same disk.
	//
	// A JOB's grant names the DURABLE PUBLICATION ROOT instead, and the destination fence
	// runs HERE — before StartAttempt, so an escaping destination is never a capability
	// anybody held.
	grant, e := c.grantFor(req, attempt)
	if e != nil {
		return 0, e
	}

	sess.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_StartAttempt{
		StartAttempt: &pb.StartAttempt{
			SessionId: w.sessionID, ExecutorIncarnation: w.incarnation,
			RequestId: req.ID, Attempt: attempt, ExecSpecDigest: digest,
			Grant: grant, ExecSpecCanonical: canonicalBytes,
		}}})
	c.logf("StartAttempt %s#%d spec=%s (%d canonical bytes) outputs=%s on %s",
		req.ID, attempt, shortDigest(spelled), len(canonicalBytes), req.Outputs, w.instanceID)
	c.emit(req.ID, "request.dispatched", attempt, map[string]any{
		"instance_id": w.instanceID, "exec_spec_digest": spelled,
	})
	return attempt, nil
}

// pick resolves a worker that advertises this binding as READY. Compatibility and
// capacity matching stay here, in the coordinator, exactly as they do in the cloud.
func (c *Coordinator) pick(planID string) (*worker, *session, *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, w := range c.workers {
		if w.exited || w.intake != pb.IntakeState_INTAKE_STATE_READY || !w.ready[planID] {
			continue
		}
		if sess := c.sessions[w.sessionID]; sess != nil {
			if w.spec.IsJob() {
				// RESERVE the slot this dispatch is about to consume. The worker's own
				// next Report is still the authority — this only stops ONE drain pass
				// from handing two queued jobs to a one-attempt worker on one reading.
				w.jobsAvail--
				if w.jobsAvail <= 0 {
					w.ready[planID] = false
				}
			}
			return w, sess, nil
		}
	}
	return nil, nil, exit.Unavailablef("no registered worker advertises %s as ready", planID)
}

// grantFor picks the lane's grant. The two differ in exactly one thing that matters —
// WHERE the destinations are — and that difference is the whole point of the job lane's:
// a bounded job's writes must outlive the reclaim that ends it.
func (c *Coordinator) grantFor(req records.Request, attempt uint64) (*pb.DeliveryGrant, *exit.Error) {
	if req.IsJob() {
		g, _, e := c.jobGrant(req, attempt)
		return g, e
	}
	return c.grant(req.ID, attempt, req)
}

// grant builds the LOCAL delivery grant: a payload input and one destination per result
// field path, under this attempt's own directory. There is no credential — a local grant
// is a CAS root plus an output dir, and a fabricated token would be a lie about
// authority nobody issued.
func (c *Coordinator) grant(requestID string, attempt uint64, req records.Request) (*pb.DeliveryGrant, *exit.Error) {
	dir := c.opt.Layout.AttemptDir(requestID, attempt)
	inDir := filepath.Join(dir, "in")
	if err := os.MkdirAll(inDir, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the attempt directory %s: %s", dir, err)
	}
	payloadPath := filepath.Join(inDir, "payload")
	if err := os.WriteFile(payloadPath, req.Payload, 0o644); err != nil {
		return nil, exit.Internalf("cannot stage the request payload: %s", err)
	}
	ttl := c.opt.GrantTTL
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}
	maxBytes := c.opt.MaxOutputMiB
	if maxBytes <= 0 {
		maxBytes = 64
	}
	g := &pb.DeliveryGrant{
		FileBaseUrl:   "file://" + dir,
		ExpiresAtUnix: uint64(time.Now().Add(ttl).Unix()),
		Inputs: []*pb.InputLocation{{
			Digest:   canonical.Digest(req.Payload),
			Url:      "file://" + payloadPath,
			Length:   uint64(len(req.Payload)),
			InputId:  "payload",
			KindMime: "application/json",
		}},
	}
	for _, id := range strings.Split(req.Outputs, ",") {
		if id == "" {
			continue
		}
		g.Outputs = append(g.Outputs, &pb.OutputDestination{
			OutputId: id,
			Url:      "file://" + filepath.Join(dir, id),
			MaxBytes: uint64(maxBytes) << 20,
		})
	}
	return g, nil
}

// Await blocks until the attempt is closed — the terminal accepted, its outputs visible,
// and the ack sent. The error it returns is the coordinator's PROJECTION of the neutral
// terminal, never a worker-authored retryability claim.
func (c *Coordinator) Await(requestID string, attempt uint64, timeout time.Duration) (*Result, *exit.Error) {
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
// ordinals the coordinator's projection minted. This is what a caller waits on; an
// individual attempt is the coordinator's business.
func (c *Coordinator) AwaitSettled(requestID string, timeout time.Duration) (*Result, *exit.Error) {
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
func (c *Coordinator) AwaitAccepted(requestID string, attempt uint64, timeout time.Duration) *exit.Error {
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
func (c *Coordinator) CancelClient(requestID string, attempt, graceMS uint64) *exit.Error {
	return c.Cancel(requestID, attempt, pb.CancelReason_CANCEL_REASON_CLIENT, graceMS)
}

// Cancel is the only way to supersede a live attempt: explicit, digest-fenced, and
// followed by a journaled terminal. Silent supersession does not exist in this protocol,
// and an attempt is never killed to improve queue latency.
func (c *Coordinator) Cancel(requestID string, attempt uint64, reason pb.CancelReason, graceMS uint64) *exit.Error {
	row, e := c.opt.Store.AttemptRow(requestID, int64(attempt))
	if e != nil {
		return e
	}
	if row == nil {
		return exit.New(exit.NotFound, "no attempt %s#%d to cancel", requestID, attempt)
	}
	raw, err := canonical.Raw(row.ExecSpecDigest)
	if err != nil {
		return exit.Internalf("the journaled spec digest is unreadable: %s", err)
	}
	c.mu.Lock()
	sess := c.sessions[row.SessionID]
	w := c.workers[row.InstanceID]
	c.mu.Unlock()
	if sess == nil || w == nil {
		return exit.Unavailablef("the session that holds %s#%d is gone", requestID, attempt)
	}
	sess.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_CancelAttempt{
		CancelAttempt: &pb.CancelAttempt{
			SessionId: row.SessionID, ExecutorIncarnation: w.incarnation,
			RequestId: requestID, Attempt: attempt, Reason: reason,
			GraceMs: graceMS, ExecSpecDigest: raw,
		}}})
	c.logf("CancelAttempt %s#%d reason=%s", requestID, attempt,
		pb.CancelReason_name[int32(reason)])
	return nil
}
