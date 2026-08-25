package coord

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"

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
			s = h.c.onRegister(stream, m.Register)
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
			// Serving attempts have no checkpoint lane; the refusal is typed, not silence.
			r := m.CheckpointRequest
			s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_CheckpointReceipt{
				CheckpointReceipt: &pb.JobCheckpointReceipt{
					SessionId: r.SessionId, ExecutorIncarnation: r.ExecutorIncarnation,
					RequestId: r.RequestId, Attempt: r.Attempt, OperationKey: r.OperationKey,
					LogicalKey: r.LogicalKey, ContentDigest: r.ContentDigest,
					Outcome: pb.CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED,
					Fault: &pb.CheckpointFault{
						Code:   pb.CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE,
						Detail: "this coordinator dispatches serving attempts; job mode is cl-004",
					},
				}}})
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
		h.c.logf("progress %s#%d seq=%d %s %dB", p.RequestId, p.Attempt, p.Seq,
			p.ContentType, len(p.Data))
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

func (c *Coordinator) onRegister(stream pb.Worker_ControlServer, r *pb.Register) *session {
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
		c.logf("recovered attempt %s#%d (%s) is an OPEN OBLIGATION: no next ordinal for %s "+
			"until its terminal closes it", ra.RequestId, ra.Attempt,
			pb.AttemptState_name[int32(ra.State)], ra.RequestId)
	}

	s.send(&pb.CoordinatorMessage{Msg: &pb.CoordinatorMessage_RegisterAck{
		RegisterAck: &pb.RegisterAck{
			SessionId: r.SessionId, ExecutorIncarnation: r.ExecutorIncarnation,
			WireMinor: pb.WireMinor, Accepted: true,
		}}})
	c.sendDirective(s, w)
	return s
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
	if w != nil {
		w.incarnation = r.ExecutorIncarnation
		w.epoch = r.ReadinessEpoch
		w.intake = r.IntakeState
		w.revision = r.AppliedRevision
		ready := map[string]bool{}
		for _, id := range r.GetServingCapacity().GetReadyEntrypointBindingPlanIds() {
			ready[id] = true
		}
		w.ready = ready
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
	outputs := c.outputsOf(t.RequestId, t.Attempt, doc)

	applied, e := c.opt.Store.AcceptTerminal(records.Terminal{
		RequestID: t.RequestId, Attempt: int64(t.Attempt), SessionID: s.id,
		ExecSpecDigest: spelledSpec, TerminalID: t.TerminalId,
		TerminalDigest: shortNone(t.TerminalDigest), Status: status, Cause: cause,
		SafeMessage:   doc.Str("safe_message"),
		TriageSubject: doc.Sub("triage_bundle").Str("subject_id"),
		Body:          t.TerminalCanonical, Outputs: outputs,
	})
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	if applied {
		c.logf("AttemptTerminal %s#%d %s/%s applied: %d output(s) became visible in the "+
			"SAME transaction", t.RequestId, t.Attempt, status, cause, len(outputs))
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
	c.waitFor(key(t.RequestId, t.Attempt)).markClosed(terminalError(status, cause, doc.Str("safe_message")))
}

// outputsOf joins the manifest's entries to the destinations THIS coordinator granted.
// The runtime names what it wrote; the coordinator names where it was allowed to write
// and decides that the result is visible. Neither half can do the other's job.
func (c *Coordinator) outputsOf(requestID string, attempt uint64, doc canonical.Doc) []records.Output {
	manifest := doc.Sub("output_manifest")
	list, _ := manifest["outputs"].([]canonical.Value)
	dir := c.opt.Layout.AttemptDir(requestID, attempt)
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
			Path:     filepath.Join(dir, id),
			Digest:   e.Str("digest"),
			Length:   e.Int("length"),
			MimeType: e.Str("mime_type"),
		})
	}
	return out
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
