package coord

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// The owner-side FRAME HANDLERS for one claimed stream (owner.go runs the conversation;
// #436 flipped the dial direction, so the old worker-dials hub/Register machinery is
// gone — ClaimAck/snapshot are its successors, handled in owner.go).

func (c *Coordinator) dropSession(s *session) {
	c.mu.Lock()
	if c.sessions[s.bootID] == s {
		delete(c.sessions, s.bootID)
	}
	c.mu.Unlock()
	// The out channel is closed by `converse`'s own defer — one owner, one close.
	c.logf("control stream for boot %s closed", s.bootID)
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

// sendDirective issues the full-replace Directive: the digest-addressed DeploymentSet
// (#446), one Deployment at launch (the len<=1 clamp is the worker's).
func (c *Coordinator) sendDirective(s *session, w *worker) {
	rev := c.nextRevision()
	c.mu.Lock()
	w.revision = rev
	deploymentID := w.deploymentID
	c.mu.Unlock()
	set := &pb.DeploymentSet{Deployments: []*pb.Deployment{{
		DeploymentId:             deploymentID,
		EndpointReleaseId:        w.spec.ReleaseID,
		EntrypointBindingPlanIds: w.planIDs,
	}}}
	// THE ONE-DIGESTER RULE (#454): this owner authors BOTH the structured set and its
	// canonical document digest; the worker echoes the digest as the set's NAME.
	_, setDigest, err := canonical.Identity(set)
	if err != nil {
		c.logf("cannot mint the DeploymentSet document for %s: %s", w.instanceID, err)
		return
	}
	d := &pb.Directive{
		Revision: rev, WireMinor: pb.WireMinor,
		Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.Directive_DeploymentSet{DeploymentSet: &pb.DeploymentSetDirective{
			SetDigest: setDigest, Set: set,
		}},
	}
	d.OwnerEpoch, d.ControlGeneration, d.WorkerBootId = ownerEpoch, s.generation, s.bootID
	s.send(&pb.OwnerFrame{Msg: &pb.OwnerFrame_Directive{Directive: d}})
	c.logf("Directive revision=%d posture=accepting deployment=%s plans=%d -> %s",
		rev, deploymentID, len(w.planIDs), s.bootID)
}

// --------------------------------------------------------------------------- report

func (c *Coordinator) onReport(s *session, r *pb.Report) {
	// Serving truth is PER-DEPLOYMENT (#446): the one hosted deployment's status carries
	// readiness, the executor generation and the ADMISSION CREDITS this owner dispatches
	// against (#441 — the deep queue stays here; the worker holds a small window).
	var status *pb.DeploymentStatus
	c.mu.Lock()
	w := c.workers[s.instanceID]
	if w != nil && w.bootID != s.bootID {
		// A superseded stream's Report is a fact about a worker that no longer exists.
		c.mu.Unlock()
		c.logf("DROPPED: Report from superseded boot %s (the live one is %s)", s.bootID, w.bootID)
		return
	}
	if w != nil {
		w.lastReport = time.Now()
		w.revision = r.AppliedRevision
		ready := map[string]bool{}
		if w.spec.IsJob() {
			// THE JOB LANE READS JOB CAPACITY. `jobs_available` IS the job credit (#441).
			w.intake = r.IntakeState
			avail := r.GetJobCapacity().GetJobsAvailable()
			w.jobsAvail = int(avail)
			for _, p := range w.spec.Jobs {
				ready[p.DescriptorID] = avail > 0
			}
		} else {
			for _, d := range r.Deployments {
				if d.DeploymentId == w.deploymentID {
					status = d
				}
			}
			if status != nil {
				w.intake = status.IntakeState
				w.generation = status.ExecutorGeneration
				w.epoch = status.ReadinessEpoch
				w.credits = int(status.AttemptCredits)
				for _, id := range status.ReadyEntrypointBindingPlanIds {
					ready[id] = true
				}
				for _, f := range status.Faults {
					w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
				}
			} else {
				// No deployment yet (the directive has not applied): machine intake is
				// what there is, and nothing is ready.
				w.intake = r.IntakeState
			}
		}
		w.ready = ready
		// A worker that says ERROR is timed from the FIRST time it said so, and the clock
		// resets the moment it stops. Its reason is the worker's own — this side echoes
		// the fault it reported and never composes one.
		if w.intake == pb.IntakeState_INTAKE_STATE_ERROR {
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
	generation := int64(0)
	if w != nil {
		generation = int64(w.generation)
	}
	c.mu.Unlock()
	_ = c.opt.Store.ReportWorker(s.bootID, pb.IntakeState_name[int32(r.IntakeState)],
		int64(r.AppliedRevision), int64(0), generation)
	for _, f := range r.Faults {
		c.logf("worker fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
			f.Subject, f.Reason, f.Detail)
	}
	for _, a := range r.Activity {
		c.logf("activity seq=%d %s: %s", a.Seq, a.Kind, a.Step)
	}
	ready := r.IntakeState == pb.IntakeState_INTAKE_STATE_READY
	if status != nil {
		ready = status.IntakeState == pb.IntakeState_INTAKE_STATE_READY
	}
	if ready {
		go c.drain()
	}
	// A Report is the only thing a STUCK worker keeps producing, so it is where the stall
	// watchdog runs.
	go c.checkStall()
	if r.GetJobCapacity() != nil {
		c.logf("Report intake=%s revision=%d jobs_available=%d jobs_in_flight=%d",
			pb.IntakeState_name[int32(r.IntakeState)], r.AppliedRevision,
			r.GetJobCapacity().GetJobsAvailable(), r.GetJobCapacity().GetJobsInFlight())
		return
	}
	if status != nil {
		c.logf("Report intake=%s revision=%d deployment=%s epoch=%d generation=%d credits=%d "+
			"ready=%d in_flight=%d",
			pb.IntakeState_name[int32(status.IntakeState)], r.AppliedRevision,
			status.DeploymentId, status.ReadinessEpoch, status.ExecutorGeneration,
			status.AttemptCredits, len(status.ReadyEntrypointBindingPlanIds),
			len(r.ActiveAttempts))
		return
	}
	c.logf("Report intake=%s revision=%d (no deployment applied yet)",
		pb.IntakeState_name[int32(r.IntakeState)], r.AppliedRevision)
}

// --------------------------------------------------------------------------- accepted

func (c *Coordinator) onAccepted(s *session, a *pb.AttemptAccepted) {
	row, e := c.opt.Store.AttemptRow(a.RequestId, int64(a.Attempt))
	if e != nil || row == nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: no such assigned attempt", a.RequestId, a.Attempt)
		return
	}
	spelled, err := canonical.Spell(a.InvocationDigest)
	if err != nil || spelled != row.InvocationDigest {
		c.logf("AttemptAccepted for %s#%d REFUSED: it echoes %v, the assignment is %s",
			a.RequestId, a.Attempt, spelled, row.InvocationDigest)
		return
	}
	planDigest, _ := canonical.Spell(a.PlanDigest)
	construction, _ := canonical.Spell(a.ModelConstructionDigest)
	summary := planSummary(a.Plan)
	if e := c.opt.Store.Accepted(a.RequestId, int64(a.Attempt), s.bootID, planDigest, construction, summary); e != nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: %s", a.RequestId, a.Attempt, e.Message)
		return
	}
	c.logf("AttemptAccepted %s#%d deployment=%s generation=%d plan=%s construction=%s [%s]",
		a.RequestId, a.Attempt, a.DeploymentId, a.ExecutorGeneration,
		shortDigest(planDigest), shortDigest(construction), summary)
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
	return fmt.Sprintf("%s/%s %s %s device=%dB host=%dB",
		p.Delivery, p.Materialization, p.ComputeDtype, p.Placement, //cozy:allow the protocol's own AttemptPlanSummary field, transported and rendered — never chosen here
		p.ReservedDeviceMemoryBytes, p.ReservedHostBytes)
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
	spelledSpec, _ := canonical.Spell(t.InvocationDigest)
	if doc.Str("request_id") != t.RequestId || uint64(doc.Int("attempt")) != t.Attempt ||
		doc.Str("invocation_digest") != spelledSpec {
		refuse("envelope/document divergence: envelope %s#%d/%s, document %s#%d/%s",
			t.RequestId, t.Attempt, shortDigest(spelledSpec),
			doc.Str("request_id"), doc.Int("attempt"), shortDigest(doc.Str("invocation_digest")))
		return
	}

	status := terminalStatus(doc.Int("status"))
	cause := causeCode(doc.Sub("cause").Int("code"))
	req, e := c.opt.Store.RequestRow(t.RequestId)
	if e != nil || req == nil {
		refuse("no request row to settle")
		return
	}
	// THE MIRROR RUNS BEFORE ANYTHING IS ACCEPTED OR ACKED. An output that is not here,
	// or is not what the manifest says it is, means this terminal cannot be honoured: the
	// worker's journaled terminal stays owed and replayable, exactly as a failed job
	// promotion does below, rather than being acked with a row that points at nothing.
	c.mu.Lock()
	holder := c.workers[s.instanceID]
	c.mu.Unlock()
	outputs, e := c.mirrorOutputs(*req, t.Attempt, doc, holder)
	if e != nil {
		refuse("%s", e.Message)
		return
	}
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
		RequestID: t.RequestId, Attempt: int64(t.Attempt), SessionID: s.bootID,
		InvocationDigest: spelledSpec, TerminalID: t.TerminalId,
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
	ack := &pb.TerminalAck{
		RequestId: t.RequestId, Attempt: t.Attempt, InvocationDigest: t.InvocationDigest,
		TerminalId: t.TerminalId, TerminalDigest: t.TerminalDigest,
	}
	ack.OwnerEpoch, ack.ControlGeneration, ack.WorkerBootId = ownerEpoch, s.generation, s.bootID
	s.send(&pb.OwnerFrame{Msg: &pb.OwnerFrame_TerminalAck{TerminalAck: ack}})
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

// mirrorOutputs joins the manifest's entries to the destinations THIS coordinator granted
// and PROVES the bytes are there before any of them becomes visible. The runtime names
// what it wrote; the coordinator names where it was allowed to write and decides that the
// result is visible. Neither half can do the other's job.
//
// THE PROOF IS THE POINT, and it is why this is not a join any more. Composing a local
// path out of the grant and stamping the manifest's own digest onto it made the record a
// RESTATEMENT of the worker's claim rather than an observation: for a local worker the
// two happen to agree, and for a REMOTE one the bytes are on the pod and the row pointed
// at a path with nothing in it. `--out` then wrote a file the client had never received.
// So: the destination is read, its length and its digest are recomputed here, and an
// output that cannot be shown is never acked as one that can.
func (c *Coordinator) mirrorOutputs(req records.Request, attempt uint64, doc canonical.Doc,
	holder *worker) ([]records.Output, *exit.Error) {
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
	// AND WHERE THE BYTES ACTUALLY ARE. For a pod attempt the granted destination is a
	// directory on the pod, so "mirror" stops being a figure of speech: each declared
	// output is FETCHED over the pod's media plane and landed here first. Everything after
	// that is the local law unchanged — the bytes are at the path, their length and digest
	// are recomputed here, and an output that cannot be shown is never acked.
	if holder != nil && holder.media != nil {
		if e := c.fetchOutputs(req, attempt, list, dir, holder); e != nil {
			return nil, e
		}
	}
	out := make([]records.Output, 0, len(list))
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		e := canonical.Doc(entry)
		id := e.Str("output_id")
		path := filepath.Join(dir, id)
		if err := verifyBytes(path, e.Str("digest"), e.Int("length")); err != nil {
			return nil, err.WithRemedy(
				"a remote worker writes on its own machine; its outputs are MIRRORED here " +
					"before the outcome is acked, and this attempt has nothing to show")
		}
		out = append(out, records.Output{
			OutputID: id,
			// The OPAQUE media id is minted HERE, before the terminal transaction, so
			// the terminal EVENT can announce it in the same commit that publishes the
			// row. Minting it inside the transaction instead left the announcement with
			// an empty handle — found live by the arm that reads the event's own
			// `outputs`, which is exactly the field a UI would render from.
			MediaID:  records.NewID("med"),
			Path:     path,
			Digest:   e.Str("digest"),
			Length:   e.Int("length"),
			MimeType: e.Str("mime_type"),
		})
	}
	return out, nil
}

// fetchOutputs pulls one remote attempt's declared outputs across the pod's media plane
// and lands them where this coordinator granted, so the verification below runs against
// bytes THIS host holds.
//
// It is deliberately not a verification step: the media client checks only the pod's own
// declared digest against what arrived (a transport's business), and whether an output may
// become visible is decided one function up, against the TERMINAL's manifest. Two checks
// of two different claims, and neither substitutes for the other.
//
// RE-MIRRORING CONVERGES. A coordinator killed between fetch and commit re-runs this on
// the worker's replayed terminal: the slot name is derived from the attempt's identity
// rather than remembered, the fetch is idempotent, and the write is atomic.
func (c *Coordinator) fetchOutputs(req records.Request, attempt uint64,
	list []canonical.Value, dir string, holder *worker) *exit.Error {
	slot := media.Slot(req.ID, attempt)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return exit.Internalf("cannot create the local mirror directory %s: %s", dir, err)
	}
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		id := canonical.Doc(entry).Str("output_id")
		if id == "" {
			continue
		}
		data, e := holder.media.GetOutput(slot, id)
		if e != nil {
			return e.WithRemedy("the pod declared this output in its terminal and this host "+
				"cannot fetch it from the pod's media plane, so the terminal stays OWED and "+
				"unacked rather than being accepted with a row that points at nothing (%s)",
				e.Remedy)
		}
		staging := filepath.Join(dir, id+".mirroring")
		if err := os.WriteFile(staging, data, 0o644); err != nil {
			return exit.Internalf("cannot land the mirrored output %s: %s", id, err)
		}
		if err := os.Rename(staging, filepath.Join(dir, id)); err != nil {
			return exit.Internalf("cannot commit the mirrored output %s: %s", id, err)
		}
		c.logf("mirrored %s#%d/%s: %d B from %s", req.ID, attempt, id, len(data),
			holder.media.Addr())
	}
	return nil
}

// verifyBytes is the mirror's proof: the declared identity, recomputed over the bytes
// that are actually at the granted destination. It reads the file once — the grant's own
// per-output ceiling already bounds how large one can be — because a length that agrees
// with a digest that does not is the interesting failure, not the cheap one.
func verifyBytes(path, digest string, length int64) *exit.Error {
	data, err := os.ReadFile(path)
	if err != nil {
		return exit.New(exit.Failed,
			"the declared output at %s cannot be read here: %s", path, err)
	}
	if int64(len(data)) != length {
		return exit.New(exit.Failed,
			"the output at %s is %d B and its manifest declares %d B", path, len(data), length)
	}
	sum := sha256.Sum256(data)
	spelled, serr := canonical.Spell(sum[:])
	if serr != nil {
		return exit.Internalf("cannot spell the mirrored output's digest: %s", serr)
	}
	if spelled != digest {
		return exit.New(exit.Failed,
			"the output at %s hashes to %s and its manifest declares %s",
			path, shortDigest(spelled), shortDigest(digest))
	}
	return nil
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
