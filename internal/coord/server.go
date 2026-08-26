package coord

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// hub is the coordinator side of the `cozy.worker.v1` service. The worker is the gRPC
// CLIENT and dials this socket; terminal authority lives only on Control.
type hub struct {
	pb.UnimplementedWorkerServer
	c *Coordinator
}

type session struct {
	id         string
	instanceID string
	out        chan *pb.CoordinatorMessage
}

func (s *session) send(m *pb.CoordinatorMessage) {
	defer func() { _ = recover() }() // a closed stream is not an error worth a panic
	s.out <- m
}

// Control is the durable bidi stream: one causal order, one fence. Durable messages are
// never shed to backpressure.
func (h *hub) Control(stream pb.Worker_ControlServer) error {
	var s *session
	defer func() {
		if s != nil {
			h.c.dropSession(s)
		}
	}()
	for {
		msg, err := stream.Recv()
		if err != nil {
			return nil
		}
		switch m := msg.Msg.(type) {
		case *pb.WorkerMessage_Register:
			if s != nil {
				h.c.logf("REFUSED: a second Register on a bound stream (session %s)", s.id)
				continue
			}
			s = h.c.onRegister(stream, m.Register, peerPID(stream.Context()))
		case *pb.WorkerMessage_BootFailure:
			h.c.logf("BOOT FAILURE from %s: %s (%s)", m.BootFailure.InstanceId,
				pb.BootFailureReason_name[int32(m.BootFailure.Reason)], m.BootFailure.Detail)
			return nil
		case *pb.WorkerMessage_Report:
			if h.c.fenced(s, m.Report.SessionId, m.Report.ExecutorIncarnation, true) {
				continue
			}
			h.c.onReport(s, m.Report)
		case *pb.WorkerMessage_AttemptAccepted:
			if h.c.fenced(s, m.AttemptAccepted.SessionId, m.AttemptAccepted.ExecutorIncarnation, false) {
				continue
			}
			h.c.onAccepted(s, m.AttemptAccepted)
		case *pb.WorkerMessage_AttemptTerminal:
			if h.c.fenced(s, m.AttemptTerminal.SessionId, m.AttemptTerminal.ExecutorIncarnation, false) {
				continue
			}
			h.c.onTerminal(s, m.AttemptTerminal)
		case *pb.WorkerMessage_CheckpointRequest:
			r := m.CheckpointRequest
			if h.c.fenced(s, r.SessionId, r.ExecutorIncarnation, false) {
				continue
			}
			// A SERVING attempt has no checkpoint lane, and the refusal is typed rather
			// than silence. The mode is the worker's own, read off the spec this service
			// launched it under.
			if !h.c.jobMode(s) {
				s.send(checkpointReceipt(r, "", pb.CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED,
					pb.CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE,
					"this worker is in serving mode; the checkpoint lane is the job lane's"))
				continue
			}
			h.c.onCheckpoint(s, r)
		}
	}
}

// StreamProgress is the one LOSSY lane: sequence-numbered, gaps visible, and structurally
// incapable of blocking Control or carrying terminal authority.
func (h *hub) StreamProgress(stream pb.Worker_StreamProgressServer) error {
	for {
		p, err := stream.Recv()
		if err != nil {
			return stream.SendAndClose(&emptypb.Empty{})
		}
		// The lossy lane's ONE consumer: the live fanout cl-006 serves as
		// `request.progress`. It never touches the authority and never blocks this
		// reader — a slow SSE client sheds frames, it does not slow the protocol.
		h.c.frames.publish(frameOf(p))
	}
}

// fenced evaluates the fence tuple in its fixed order and refuses on the first mismatch,
// BEFORE any body field is interpreted (02 §0).
func (c *Coordinator) fenced(s *session, sessionID string, incarnation uint64, advanceOK bool) bool {
	if s == nil {
		c.logf("DROPPED: a message arrived on a stream that never registered")
		return true
	}
	if sessionID != s.id {
		c.logf("DROPPED: session %q is not this stream's bound session %q", sessionID, s.id)
		return true
	}
	c.mu.Lock()
	w := c.workers[s.instanceID]
	current := uint64(0)
	if w != nil {
		current = w.incarnation
	}
	c.mu.Unlock()
	if incarnation < current {
		c.logf("DROPPED: stale executor_incarnation %d (current %d)", incarnation, current)
		return true
	}
	if incarnation > current && !advanceOK {
		c.logf("DROPPED: future executor_incarnation %d on a non-Report message", incarnation)
		return true
	}
	return false
}

func (c *Coordinator) dropSession(s *session) {
	c.mu.Lock()
	if c.sessions[s.id] == s {
		delete(c.sessions, s.id)
	}
	c.mu.Unlock()
	close(s.out)
	c.logf("control stream for session %s closed", s.id)
}

// --------------------------------------------------------------------------- register

func (c *Coordinator) onRegister(stream pb.Worker_ControlServer, r *pb.Register, pid int32) *session {
	s := &session{id: r.SessionId, instanceID: r.InstanceId, out: make(chan *pb.CoordinatorMessage, 32)}
	go func() {
		for m := range s.out {
			if err := stream.Send(m); err != nil {
				return
			}
		}
	}()

	c.mu.Lock()
	w := c.workers[r.InstanceId]
	c.mu.Unlock()
	reject := func(reason pb.RegisterRejection, why string) *session {
		c.logf("Register REFUSED (%s): %s", pb.RegisterRejection_name[int32(reason)], why)
		s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_RegisterAck{
			RegisterAck: &pb.RegisterAck{
				SessionId: r.SessionId, ExecutorIncarnation: r.ExecutorIncarnation,
				WireMinor: pb.WireMinor, Accepted: false, Rejection: reason,
			}}})
		return s
	}
	if w == nil {
		return reject(pb.RegisterRejection_REGISTER_REJECTION_WORKER_ID_MISMATCH,
			"instance "+r.InstanceId+" was never spawned by this LocalService")
	}
	if w.spec.ReleaseID != "" && r.ReleaseId != w.spec.ReleaseID {
		return reject(pb.RegisterRejection_REGISTER_REJECTION_RELEASE_ID_MISMATCH,
			"release "+r.ReleaseId+" is not the pinned "+w.spec.ReleaseID)
	}
	// The kernel-attested pid. A supervisor from a PREVIOUS LocalService, still
	// reconnecting on its backoff, dials the same socket path and presents a perfectly
	// valid instance_id — its own. Only the process this coordinator started may bind
	// this slot, and only the kernel can say which one that is.
	c.mu.Lock()
	want, bound := w.pid, w.sessionID
	c.mu.Unlock()
	if pid != 0 && want != 0 && int(pid) != want {
		return reject(pb.RegisterRejection_REGISTER_REJECTION_STALE_SESSION,
			fmt.Sprintf("pid %d is not the process this LocalService started for %s (pid %d)",
				pid, r.InstanceId, want))
	}
	// Where the transport carries no kernel identity (Windows loopback), the per-spawn
	// bootstrap credential is the substitute: only the child this launcher handed it to
	// can echo it. The value never appears in a log — a mismatch is reported as a fact.
	if w.bootstrap.Present() {
		got := ""
		if md, ok := metadata.FromIncomingContext(stream.Context()); ok {
			if vals := md.Get("cozy-bootstrap"); len(vals) > 0 {
				got = vals[0]
			}
		}
		if !w.bootstrap.Equal(got) {
			return reject(pb.RegisterRejection_REGISTER_REJECTION_STALE_SESSION,
				"the bootstrap credential for "+r.InstanceId+" was absent or wrong")
		}
	}
	if bound != "" && bound != r.SessionId {
		return reject(pb.RegisterRejection_REGISTER_REJECTION_SESSION_COLLISION,
			"instance "+r.InstanceId+" already has the live session "+bound)
	}
	if e := c.opt.Store.BindSession(r.InstanceId, r.SessionId, int64(r.ExecutorIncarnation)); e != nil {
		return reject(pb.RegisterRejection_REGISTER_REJECTION_SESSION_COLLISION, e.Message)
	}

	c.mu.Lock()
	c.sessions[r.SessionId] = s
	w.sessionID = r.SessionId
	w.incarnation = r.ExecutorIncarnation
	c.mu.Unlock()

	c.logf("Register session=%s instance=%s incarnation=%d minor=%d gpu=%q recovered=%d",
		r.SessionId, r.InstanceId, r.ExecutorIncarnation, r.WireMinor,
		r.Resources.GetGpuName(), len(r.RecoveredAttempts))

	// THE LAW (02 §6.2). The recovered journal is reported FIRST and is an OPEN
	// OBLIGATION from this moment: records.Recover puts each attempt in `recovered_open`,
	// and NextOrdinal refuses for those request ids until each is closed by its own
	// journaled terminal. Absence is never manufactured by a fresh session_id.
	for _, ra := range r.RecoveredAttempts {
		if e := c.opt.Store.Recover(ra.RequestId, int64(ra.Attempt), r.SessionId); e != nil {
			c.logf("recovered attempt %s#%d REFUSED: %s", ra.RequestId, ra.Attempt, e.Message)
			continue
		}
		// Assert the obligation at the moment it is taken: ask this coordinator's own
		// ordinal gate what it now answers for that request id. The refusal is the law
		// as an OBSERVED fact rather than an inference from the row's state, and it is
		// recorded here because the window it holds for is milliseconds wide.
		_, blocked := c.opt.Store.NextOrdinal(ra.RequestId)
		c.logf("recovered attempt %s#%d (%s) is an OPEN OBLIGATION — the ordinal gate now "+
			"refuses: %s", ra.RequestId, ra.Attempt,
			pb.AttemptState_name[int32(ra.State)], briefOf(blocked))
	}

	s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_RegisterAck{
		RegisterAck: &pb.RegisterAck{
			SessionId: r.SessionId, ExecutorIncarnation: r.ExecutorIncarnation,
			WireMinor: pb.WireMinor, Accepted: true,
		}}})
	// ONE MODE, chosen from the spec this service launched. Sending the mode the worker
	// is in — rather than setting both members of the Directive's oneof — is the same
	// lesson cr-009's `fabd6fc` recorded on the Report side.
	if w.spec.IsJob() {
		c.sendJobDirective(s, w)
	} else {
		c.sendDirective(s, w)
	}
	return s
}

// jobMode answers whether the worker behind this session was launched in job mode.
func (c *Coordinator) jobMode(s *session) bool {
	if s == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[s.instanceID]
	return w != nil && w.spec.IsJob()
}

// sendDirective issues the full-replace Directive. It carries NO credential: a local
// grant is a CAS root and an output directory, and there is no token to mint.
func (c *Coordinator) sendDirective(s *session, w *worker) {
	rev := c.nextRevision()
	c.mu.Lock()
	w.revision = rev
	c.mu.Unlock()
	s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_Directive{
		Directive: &pb.Directive{
			SessionId: s.id, Revision: rev, WireMinor: pb.WireMinor,
			Posture: pb.Posture_POSTURE_ACCEPTING,
			Mode: &pb.Directive_Serving{Serving: &pb.ServingDirective{
				EndpointReleaseId:        w.spec.ReleaseID,
				EntrypointBindingPlanIds: w.planIDs,
			}},
		}}})
	c.logf("Directive revision=%d posture=accepting plans=%d -> %s", rev, len(w.planIDs), s.id)
}

// --------------------------------------------------------------------------- report

func (c *Coordinator) onReport(s *session, r *pb.Report) {
	c.mu.Lock()
	w := c.workers[s.instanceID]
	if w != nil && w.sessionID != s.id {
		// A superseded session's Report is a fact about a worker that no longer exists.
		c.mu.Unlock()
		c.logf("DROPPED: Report from superseded session %s (the live one is %s)", s.id, w.sessionID)
		return
	}
	if w != nil {
		w.lastReport = time.Now()
		w.incarnation = r.ExecutorIncarnation
		w.epoch = r.ReadinessEpoch
		w.intake = r.IntakeState
		w.revision = r.AppliedRevision
		ready := map[string]bool{}
		if w.spec.IsJob() {
			// THE JOB LANE READS JOB CAPACITY. `jobs_available` is the worker's own
			// answer to "can I take an attempt right now" — law 8 in the job lane, so a
			// worker with a live attempt reports zero and its queue simply waits. The
			// coordinator dispatches by the SAME map the serving lane uses, keyed on the
			// job descriptor id, so there is one placement path and not two.
			avail := r.GetJobCapacity().GetJobsAvailable()
			w.jobsAvail = int(avail)
			for _, p := range w.spec.Jobs {
				ready[p.DescriptorID] = avail > 0
			}
		} else {
			for _, id := range r.GetServingCapacity().GetReadyEntrypointBindingPlanIds() {
				ready[id] = true
			}
		}
		w.ready = ready
		// A worker that says ERROR is timed from the FIRST time it said so, and the clock
		// resets the moment it stops. Its reason is the worker's own — this side echoes
		// the fault it reported and never composes one.
		if r.IntakeState == pb.IntakeState_INTAKE_STATE_ERROR {
			if w.errorSince.IsZero() {
				w.errorSince = time.Now()
			}
		} else {
			w.errorSince = time.Time{}
		}
		for _, f := range r.Faults {
			w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
		}
	}
	c.mu.Unlock()
	_ = c.opt.Store.ReportWorker(s.id, pb.IntakeState_name[int32(r.IntakeState)],
		int64(r.AppliedRevision), int64(r.ReadinessEpoch), int64(r.ExecutorIncarnation))
	for _, f := range r.Faults {
		c.logf("worker fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
			f.Subject, f.Reason, f.Detail)
	}
	for _, a := range r.Activity {
		c.logf("activity seq=%d %s: %s", a.Seq, a.Kind, a.Step)
	}
	if r.IntakeState == pb.IntakeState_INTAKE_STATE_READY {
		go c.drain()
	}
	// A Report is the only thing a STUCK worker keeps producing, so it is where the stall
	// watchdog runs.
	go c.checkStall()
	if r.GetJobCapacity() != nil {
		c.logf("Report intake=%s revision=%d epoch=%d jobs_available=%d jobs_in_flight=%d",
			pb.IntakeState_name[int32(r.IntakeState)], r.AppliedRevision, r.ReadinessEpoch,
			r.GetJobCapacity().GetJobsAvailable(), r.GetJobCapacity().GetJobsInFlight())
		return
	}
	c.logf("Report intake=%s revision=%d epoch=%d ready=%d in_flight=%d",
		pb.IntakeState_name[int32(r.IntakeState)], r.AppliedRevision, r.ReadinessEpoch,
		len(r.GetServingCapacity().GetReadyEntrypointBindingPlanIds()), len(r.ActiveAttempts))
}

// --------------------------------------------------------------------------- accepted

func (c *Coordinator) onAccepted(s *session, a *pb.AttemptAccepted) {
	row, e := c.opt.Store.AttemptRow(a.RequestId, int64(a.Attempt))
	if e != nil || row == nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: no such assigned attempt", a.RequestId, a.Attempt)
		return
	}
	spelled, err := canonical.Spell(a.ExecSpecDigest)
	if err != nil || spelled != row.ExecSpecDigest {
		c.logf("AttemptAccepted for %s#%d REFUSED: it echoes %v, the assignment is %s",
			a.RequestId, a.Attempt, spelled, row.ExecSpecDigest)
		return
	}
	planDigest, _ := canonical.Spell(a.PlanDigest)
	construction, _ := canonical.Spell(a.ModelConstructionDigest)
	summary := planSummary(a.Plan)
	if e := c.opt.Store.Accepted(a.RequestId, int64(a.Attempt), s.id, planDigest, construction, summary); e != nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: %s", a.RequestId, a.Attempt, e.Message)
		return
	}
	c.logf("AttemptAccepted %s#%d plan=%s construction=%s [%s]",
		a.RequestId, a.Attempt, shortDigest(planDigest), shortDigest(construction), summary)
	c.emit(a.RequestId, "request.accepted", a.Attempt, map[string]any{
		"plan_digest": planDigest, "construction_digest": construction, "plan": summary,
	})
	c.waitFor(key(a.RequestId, a.Attempt)).markAccepted()
}

// planSummary renders the CLOSED observable projection of the chosen plan. The
// coordinator sees WHAT was chosen, not only that something was — and it renders those
// facts without ever choosing one.
func planSummary(p *pb.AttemptPlanSummary) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("%s/%s %s %s vram=%dB host=%dB",
		p.Delivery, p.Materialization, p.ComputeDtype, p.Placement, //cozy:allow the protocol's own AttemptPlanSummary field, transported and rendered — never chosen here
		p.ReservedVramBytes, p.ReservedHostBytes)
}

// --------------------------------------------------------------------------- terminal

func (c *Coordinator) onTerminal(s *session, t *pb.AttemptTerminal) {
	refuse := func(format string, args ...any) {
		c.logf("AttemptTerminal %s#%d REFUSED: "+format,
			append([]any{t.RequestId, t.Attempt}, args...)...)
	}

	// 1. The digest is RECOMPUTED over the resident bytes. A digest never bypasses the
	//    lower check, and a mismatched one is not acked — the worker keeps replaying.
	computed := canonical.Digest(t.TerminalCanonical)
	if !bytes.Equal(computed, t.TerminalDigest) {
		refuse("terminal_digest %x does not hash the %d resident bytes (%x)",
			t.TerminalDigest, len(t.TerminalCanonical), computed)
		return
	}
	// 2. The document is parsed under UNKNOWN-FIELD REFUSAL and the re-emit law.
	doc, err := canonical.Read(t.TerminalCanonical, &pb.TerminalBody{})
	if err != nil {
		refuse("the terminal document is inadmissible (%s)", err)
		return
	}
	// 3. The routing copies must agree with the DOCUMENT, which is authoritative.
	spelledSpec, _ := canonical.Spell(t.ExecSpecDigest)
	if doc.Str("request_id") != t.RequestId || uint64(doc.Int("attempt")) != t.Attempt ||
		doc.Str("exec_spec_digest") != spelledSpec {
		refuse("envelope/document divergence: envelope %s#%d/%s, document %s#%d/%s",
			t.RequestId, t.Attempt, shortDigest(spelledSpec),
			doc.Str("request_id"), doc.Int("attempt"), shortDigest(doc.Str("exec_spec_digest")))
		return
	}

	status := terminalStatus(doc.Int("status"))
	cause := causeCode(doc.Sub("cause").Int("code"))
	req, e := c.opt.Store.RequestRow(t.RequestId)
	if e != nil || req == nil {
		refuse("no request row to settle")
		return
	}
	outputs := c.outputsOf(*req, t.Attempt, doc)
	// cl-006 OWNS TRIAGE PERSISTENCE (cr-011's seam): the bundle lives in the worker's
	// own root, and a worker root does not outlive its worker. Copy it out and verify it
	// against the terminal document's OWN TriageBundleRef before the transaction runs.
	triage := c.captureTriage(s, t.RequestId, t.Attempt, doc.Sub("triage_bundle"))
	// WHETHER THIS ATTEMPT ENDS THE REQUEST is decided BEFORE the event is written, not
	// after. An ABANDONED attempt that the requeue projection will re-dispatch has ended
	// an ATTEMPT, not a REQUEST — and the contract's terminal-stop rule means a client
	// that saw `request.failed` would close its stream and report a failure for a request
	// that goes on to succeed. Found live by the coordinator-kill arm: the killed
	// attempt's ABANDONED terminal stopped the client's stream while attempt 2 was still
	// being minted.
	requeuing := requeueable(status, cause)
	kept := triage.keep(status, cause, doc.Str("safe_message"), outputs, requeuing)

	// THE PUBLICATION, for a job. The attempt's staged writes are PROMOTED into the
	// addressable publication root first, and the row that makes the publication exist
	// rides the very transaction that accepts the terminal. A kill before that commit
	// therefore exposes no publication at all — not in the authority and not at the
	// addressable path — and a kill after it has already preserved one.
	//
	// A REQUEUEING attempt writes NO publication. Its terminal ended an attempt, not the
	// request, so a row stamped ABANDONED with zero entries would be a claim about a
	// publication that does not exist — and it would be overwritten by the ordinal that
	// goes on to succeed. Observed live: attempt 1 of a killed job committed an empty
	// `local/_job-…` publication seconds before attempt 2 published the real one.
	var publication *records.Publication
	if req.IsJob() && !requeuing {
		if e := c.promote(*req, t.Attempt, outputs); e != nil {
			refuse("%s", e.Message)
			return
		}
		publication = c.publicationOf(*req, t.Attempt, status, cause, outputs)
	}

	began := time.Now()
	applied, e := c.opt.Store.AcceptTerminal(records.Terminal{
		RequestID: t.RequestId, Attempt: int64(t.Attempt), SessionID: s.id,
		ExecSpecDigest: spelledSpec, TerminalID: t.TerminalId,
		TerminalDigest: shortNone(t.TerminalDigest), Status: status, Cause: cause,
		SafeMessage:   doc.Str("safe_message"),
		TriageSubject: triage.Subject, TriageDigest: triage.Digest,
		TriageLength: triage.Length, TriagePath: triage.Path,
		Body: t.TerminalCanonical, Outputs: outputs,
		EventType: kept.Type, EventPayload: kept.Payload,
		// A requeueing request is QUEUED for its next ordinal, not failed. Writing the
		// attempt's own status onto the request row would make the status document say
		// `failed` for a request that is still going.
		RequestState: requeueState(status, requeuing),
		Publication:  publication,
	})
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	if applied {
		c.logf("AttemptTerminal %s#%d %s/%s applied in %.2f ms: %d output(s) became visible "+
			"in the SAME transaction", t.RequestId, t.Attempt, status, cause,
			float64(time.Since(began).Microseconds())/1000, len(outputs))
		if publication != nil {
			c.logf("publication %s committed: %d entr(y|ies), %d B, root %s",
				publication.Repo, publication.Entries, publication.Bytes, publication.Root)
		}
	} else {
		c.logf("AttemptTerminal %s#%d is an exact replay of a closed terminal: re-acked, "+
			"nothing applied twice", t.RequestId, t.Attempt)
	}

	// The ack follows the COMMIT. A crash before this line replays; a crash after it is
	// a closed attempt either way.
	s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_TerminalAck{
		TerminalAck: &pb.TerminalAck{
			SessionId: s.id, ExecutorIncarnation: t.ExecutorIncarnation,
			RequestId: t.RequestId, Attempt: t.Attempt, ExecSpecDigest: t.ExecSpecDigest,
			TerminalId: t.TerminalId, TerminalDigest: t.TerminalDigest,
		}}})
	_ = c.opt.Store.Closed(t.RequestId, int64(t.Attempt))
	outcome := terminalError(status, cause, doc.Str("safe_message"))
	c.waitFor(key(t.RequestId, t.Attempt)).markClosed(outcome)
	if !applied {
		return // a replay settles nothing twice and requeues nothing twice
	}

	// The requeue PROJECTION over (status, cause). Retryability is never a wire
	// observation: this is the only place the neutral facts become a decision.
	if requeueable(status, cause) {
		c.Requeue(t.RequestId, status+"/"+cause)
		return
	}
	c.frames.forget(t.RequestId)
	_ = c.opt.Store.SettleRequest(t.RequestId, strings.ToLower(status))
	c.waitRequest(t.RequestId).markClosed(outcome)
}

// Triage is what cl-006 persisted for one attempt. An empty Subject means the terminal
// named no bundle, which is a fact, not a failure.
type Triage struct {
	Subject string
	Digest  string
	Length  int64
	Path    string
	Fault   string // why the bytes were not kept, when a subject was named and they were not
}

// terminalEvent is the durable lifecycle event that rides the terminal transaction.
type terminalEvent struct {
	Type    string
	Payload map[string]any
}

// requeueState is the REQUEST's state after one attempt ended. A requeueing request is
// queued for its next ordinal; anything else takes the attempt's own status.
func requeueState(status string, requeuing bool) string {
	if requeuing {
		return "queued"
	}
	return strings.ToLower(status)
}

// keep renders the attempt-end event's closed payload. It carries an OPAQUE handle for
// every asset (media id) and for triage (subject) — never a path, never bytes, and never
// base64. That is what makes "no client-supplied path exists" a property of the schema
// rather than of a validator.
//
// `requeuing` decides whether this is a TERMINAL event at all. `request.attempt_failed`
// is deliberately not in the terminal set: it says an attempt ended and the request did
// not, which is precisely the state the recovered-attempts law creates.
func (tr Triage) keep(status, cause, safeMessage string, outputs []records.Output, requeuing bool) terminalEvent {
	eventType := "request.failed"
	switch {
	case requeuing:
		eventType = "request.attempt_failed"
	case status == "SUCCEEDED":
		eventType = "request.completed"
	case status == "CANCELED":
		eventType = "request.canceled"
	}
	media := make([]map[string]any, 0, len(outputs))
	for _, o := range outputs {
		media = append(media, map[string]any{
			"output_id": o.OutputID, "media_id": o.MediaID,
			"mime_type": o.MimeType, "length": o.Length, "digest": o.Digest,
		})
	}
	payload := map[string]any{
		"status": status, "cause": cause, "outputs": media, "requeuing": requeuing,
	}
	if status != "SUCCEEDED" {
		payload["error_type"] = cause
		payload["error"] = safeMessage
	}
	if tr.Subject != "" {
		payload["triage_subject"] = tr.Subject
	}
	if tr.Fault != "" {
		payload["triage_fault"] = tr.Fault
	}
	return terminalEvent{Type: eventType, Payload: payload}
}

// captureTriage copies the worker's bundle into cozy-creator's own store, verified
// against the TERMINAL DOCUMENT's reference rather than against the worker's journal.
//
// That choice is deliberate and stronger than cr-011's own reader: this coordinator has
// already recomputed the terminal digest over the resident bytes and parsed the document
// under unknown-field refusal, so `triage_bundle.write_receipt_digest` is a fact it
// ACCEPTED. Reading the worker's journal instead would be trusting a file the worker can
// still write. A mismatch is recorded as a fault on the attempt and the bytes are not
// kept — a bundle that does not hash to what the terminal claimed is not evidence.
func (c *Coordinator) captureTriage(s *session, requestID string, attempt uint64, ref canonical.Doc) Triage {
	tr := Triage{Subject: ref.Str("subject_id")}
	if tr.Subject == "" {
		return tr
	}
	tr.Digest, tr.Length = ref.Str("write_receipt_digest"), ref.Int("length")

	c.mu.Lock()
	w := c.workers[s.instanceID]
	c.mu.Unlock()
	if w == nil {
		tr.Fault = "the worker that wrote it is gone before its bundle could be copied"
		return tr
	}
	source := filepath.Join(c.opt.Layout.WorkerDir(w.instanceID), "run", "triage", tr.Subject+".json")
	data, err := os.ReadFile(source)
	if err != nil {
		tr.Fault = "bundle_absent: the terminal names a bundle that is not on disk"
		c.logf("triage %s for %s#%d NOT kept: %s", tr.Subject, requestID, attempt, err)
		return tr
	}
	spelled, _ := canonical.Spell(canonical.Digest(data))
	if int64(len(data)) != tr.Length || spelled != tr.Digest {
		tr.Fault = fmt.Sprintf("bundle_corrupt: %d B hashing to %s does not match the terminal's %d B / %s",
			len(data), shortDigest(spelled), tr.Length, shortDigest(tr.Digest))
		c.logf("triage %s for %s#%d REFUSED: %s", tr.Subject, requestID, attempt, tr.Fault)
		return tr
	}
	dest := c.opt.Layout.TriageFile(tr.Subject)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		tr.Fault = "the local triage store is unwritable: " + err.Error()
		return tr
	}
	if err := os.WriteFile(dest, data, 0o600); err != nil {
		tr.Fault = "the bundle could not be kept: " + err.Error()
		return tr
	}
	tr.Path = dest
	c.logf("triage %s kept for %s#%d (%d B, %s)", tr.Subject, requestID, attempt,
		tr.Length, shortDigest(tr.Digest))
	return tr
}

// requeueable is the coordinator's projection: an accepted-but-incomplete attempt and
// the infra-class failures earn a new ordinal. A deterministic body failure and every
// refusal are terminal — re-running them would only fail again.
func requeueable(status, cause string) bool {
	if status == "ABANDONED" {
		return true
	}
	if status != "FAILED" {
		return false
	}
	switch cause {
	case "EXECUTOR_FAULT", "GRANT_EXPIRED", "ARTIFACT_UNFETCHABLE", "CAPABILITY_UNAVAILABLE":
		return true
	}
	return false
}

// outputsOf joins the manifest's entries to the destinations THIS coordinator granted.
// The runtime names what it wrote; the coordinator names where it was allowed to write
// and decides that the result is visible. Neither half can do the other's job.
func (c *Coordinator) outputsOf(req records.Request, attempt uint64, doc canonical.Doc) []records.Output {
	manifest := doc.Sub("output_manifest")
	list, _ := manifest["outputs"].([]canonical.Value)
	// WHERE THE COORDINATOR GRANTED. A serving attempt writes into its own disposable
	// attempt directory; a job writes into the durable publication root, which is the
	// same fact stated at the other end of the same grant.
	dir := c.opt.Layout.AttemptDir(req.ID, attempt)
	if req.IsJob() {
		// Where the coordinator GRANTED: an attempt in flight writes into its stage, and
		// `promote` rewrites these paths when the bundle crosses into the publication.
		dir = c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)
	}
	out := make([]records.Output, 0, len(list))
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		e := canonical.Doc(entry)
		id := e.Str("output_id")
		out = append(out, records.Output{
			OutputID: id,
			// The OPAQUE media id is minted HERE, before the terminal transaction, so
			// the terminal EVENT can announce it in the same commit that publishes the
			// row. Minting it inside the transaction instead left the announcement with
			// an empty handle — found live by the arm that reads the event's own
			// `outputs`, which is exactly the field a UI would render from.
			MediaID:  records.NewID("med"),
			Path:     filepath.Join(dir, id),
			Digest:   e.Str("digest"),
			Length:   e.Int("length"),
			MimeType: e.Str("mime_type"),
		})
	}
	return out
}

// brief keeps a worker's own words readable in one line without editing them.
func brief(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func terminalStatus(n int64) string {
	name := pb.TerminalStatus_name[int32(n)]
	return strings.TrimPrefix(name, "TERMINAL_STATUS_")
}

func causeCode(n int64) string {
	return strings.TrimPrefix(pb.CauseCode_name[int32(n)], "CAUSE_CODE_")
}

// terminalError is the coordinator PROJECTION over (status, cause). Retryability is
// never a wire observation; this is the only place the neutral facts become an outcome.
func terminalError(status, cause, message string) *exit.Error {
	switch status {
	case "SUCCEEDED":
		return nil
	case "REFUSED":
		return exit.New(exit.Validation, "the attempt was refused (%s): %s", cause, message)
	case "CANCELED":
		if cause == "DEADLINE_EXPIRED" {
			return exit.New(exit.Deadline, "the attempt hit its deadline: %s", message)
		}
		return exit.New(exit.Canceled, "the attempt was canceled (%s): %s", cause, message)
	case "ABANDONED":
		return exit.New(exit.Failed, "the attempt was abandoned (%s): %s", cause, message).
			WithRemedy("an abandoned attempt is requeued as a NEW ordinal, never re-executed")
	default:
		return exit.New(exit.Failed, "the attempt failed (%s): %s", cause, message)
	}
}

func briefOf(e *exit.Error) string {
	if e == nil {
		return "NOT REFUSED — the gate is open, which would be the law failing"
	}
	return e.Message
}

func shortDigest(s string) string {
	s = strings.TrimPrefix(s, "sha256:")
	if len(s) > 16 {
		return s[:16]
	}
	return s
}

func shortNone(raw []byte) string {
	s, err := canonical.Spell(raw)
	if err != nil {
		return ""
	}
	return s
}
