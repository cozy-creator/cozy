package coord

import (
	"os"
	"path/filepath"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Submission is one local request. The coordinator owns everything in it that decides
// WHAT runs; the runtime owns everything about HOW.
type Submission struct {
	IdemKey    string // the caller's idempotency key
	Endpoint   string // org/name
	Entrypoint string // the function
	InstanceID string // the worker this request is placed on
	PlanID     string // the entrypoint_binding_plan_id this attempt binds

	// Payload is the request body, verbatim. It rides the DeliveryGrant as the input
	// `payload` — a grant input, never a wire field, so refreshing the grant can never
	// substitute it.
	Payload []byte

	// Outputs is one destination per RESULT FIELD PATH (`image`, `detail.thumb`). Binding
	// by field path with exact set equality is what makes a two-output result
	// unswappable; a positional grant would silently cross them (decisions #248).
	Outputs []string

	ImageDigest  string // class (b): the execution environment's content digest
	ConfigDigest string // class (a): the evaluated-config document identity (cr-003's)
	DeadlineMS   uint64
	MaxOutputMiB int64
	GrantTTL     time.Duration
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

// Submit records the request and dispatches its next attempt. Cold and warm runs
// traverse the same states: a warm worker changes latency, never the path.
func (c *Coordinator) Submit(s Submission) (string, uint64, *exit.Error) {
	bodyDigest, err := canonical.Spell(canonical.Digest(s.Payload))
	if err != nil {
		return "", 0, exit.Internalf("cannot digest the request body: %s", err)
	}
	req, fresh, e := c.opt.Store.Submit(records.Request{
		ID: records.NewID("req"), IdemKey: s.IdemKey, BodyDigest: bodyDigest,
		Endpoint: s.Endpoint, Entrypoint: s.Entrypoint, PlanID: s.PlanID, Payload: s.Payload,
	})
	if e != nil {
		return "", 0, e
	}
	if !fresh {
		c.logf("request %s is the recorded answer for idempotency key %s", req.ID, s.IdemKey)
	}
	attempt, e := c.dispatch(req, s)
	if e != nil {
		return req.ID, 0, e
	}
	return req.ID, attempt, nil
}

func (c *Coordinator) dispatch(req records.Request, s Submission) (uint64, *exit.Error) {
	// THE LAW, enforced here: NextOrdinal refuses while any recovered attempt for this
	// request id is still an open obligation.
	ordinal, e := c.opt.Store.NextOrdinal(req.ID)
	if e != nil {
		return 0, e
	}
	attempt := uint64(ordinal)

	c.mu.Lock()
	w := c.workers[s.InstanceID]
	sess := (*session)(nil)
	if w != nil {
		sess = c.sessions[w.sessionID]
	}
	c.mu.Unlock()
	if w == nil || sess == nil {
		return 0, exit.Unavailablef("no registered worker %s to dispatch %s to", s.InstanceID, req.ID)
	}
	if w.intake != pb.IntakeState_INTAKE_STATE_READY || !w.ready[s.PlanID] {
		return 0, exit.Unavailablef("worker %s does not advertise %s as ready (intake %s)",
			s.InstanceID, s.PlanID, pb.IntakeState_name[int32(w.intake)])
	}

	// The ExecutionSpec DOCUMENT. Its key set is closed and is exactly this message's
	// fields: no human model ref, no service class, no local extension has a slot.
	spec := &pb.ExecutionSpec{
		EndpointReleaseId: w.spec.ReleaseID,
		ImageDigest:       s.ImageDigest,
		ConfigDigest:      s.ConfigDigest,
		DeadlineUnixMs:    s.DeadlineMS,
		Spec: &pb.ExecutionSpec_Serving{Serving: &pb.ServingExecutionSpec{
			EntrypointBindingPlanId: s.PlanID,
			// With no adapters the binding IS the plan, so the two ids are equal by
			// construction rather than by copying a value around.
			AttemptBindingId: s.PlanID,
		}},
	}
	canonicalBytes, digest, err := canonical.Identity(spec)
	if err != nil {
		return 0, exit.Internalf("cannot mint the ExecutionSpec document: %s", err)
	}
	spelled, _ := canonical.Spell(digest)

	grant, e := c.grant(req.ID, attempt, s)
	if e != nil {
		return 0, e
	}

	// Journal the assignment BEFORE StartAttempt: a terminal crossing a restart is
	// authorized by the persisted assignment, and nothing else.
	if e := c.opt.Store.Dispatch(records.Attempt{
		RequestID: req.ID, Attempt: ordinal, InstanceID: s.InstanceID,
		SessionID: w.sessionID, ExecSpecDigest: spelled, ExecSpec: canonicalBytes,
	}); e != nil {
		return 0, e
	}

	sess.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_StartAttempt{
		StartAttempt: &pb.StartAttempt{
			SessionId: w.sessionID, ExecutorIncarnation: w.incarnation,
			RequestId: req.ID, Attempt: attempt, ExecSpecDigest: digest,
			Grant: grant, ExecSpecCanonical: canonicalBytes,
		}}})
	c.logf("StartAttempt %s#%d spec=%s (%d canonical bytes) outputs=%v",
		req.ID, attempt, shortDigest(spelled), len(canonicalBytes), s.Outputs)
	return attempt, nil
}

// grant builds the LOCAL delivery grant: a payload input and one destination per result
// field path, under this attempt's own directory. There is no credential — a local grant
// is a CAS root plus an output dir, and a fabricated token would be a lie about
// authority nobody issued.
func (c *Coordinator) grant(requestID string, attempt uint64, s Submission) (*pb.DeliveryGrant, *exit.Error) {
	dir := c.opt.Layout.AttemptDir(requestID, attempt)
	inDir := filepath.Join(dir, "in")
	if err := os.MkdirAll(inDir, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the attempt directory %s: %s", dir, err)
	}
	payloadPath := filepath.Join(inDir, "payload")
	if err := os.WriteFile(payloadPath, s.Payload, 0o644); err != nil {
		return nil, exit.Internalf("cannot stage the request payload: %s", err)
	}
	ttl := s.GrantTTL
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}
	maxBytes := s.MaxOutputMiB
	if maxBytes <= 0 {
		maxBytes = 64
	}
	g := &pb.DeliveryGrant{
		FileBaseUrl:   "file://" + dir,
		ExpiresAtUnix: uint64(time.Now().Add(ttl).Unix()),
		Inputs: []*pb.InputLocation{{
			Digest:   canonical.Digest(s.Payload),
			Url:      "file://" + payloadPath,
			Length:   uint64(len(s.Payload)),
			InputId:  "payload",
			KindMime: "application/json",
		}},
	}
	for _, id := range s.Outputs {
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
