package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/inputasset"
	"github.com/cozy-creator/cozy-creator/internal/media"
	"github.com/cozy-creator/cozy-creator/internal/records"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// The owner-side FRAME HANDLERS for one claimed stream (owner.go runs the conversation;
// #436 flipped the dial direction, so the old worker-dials hub/Register machinery is
// gone — ClaimAck/snapshot are its successors, handled in owner.go).

func (c *Orchestrator) dropSession(s *session) {
	c.mu.Lock()
	if c.sessions[s.bootID] == s {
		delete(c.sessions, s.bootID)
	}
	c.mu.Unlock()
	// The out channel is closed by `converse`'s own defer — one owner, one close.
	c.logf("control stream for boot %s closed", s.bootID)
}

// jobMode answers whether the worker behind this session was launched in job mode.
func (c *Orchestrator) jobMode(s *session) bool {
	if s == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[s.instanceID]
	return w != nil && w.spec.IsJob()
}

// ConvergePlacementSet is the ONE issuer of a serving-mode desired state: the full-replace
// DesiredWorkerState carrying a GENUINELY content-addressed DesiredPlacementSet (§4).
//
// The frozen wire carried both a structured set and a digest this owner authored for it,
// and the worker echoed the digest as the set's NAME — one digester, two representations,
// and nothing that could ever disagree out loud. rev-2 deletes the structured copy: the
// set travels as its exact canonical bytes, the worker RECOMPUTES sha256 over them before
// parsing a single field, and journals the bytes it accepted. So this side authors bytes
// once and keeps them; there is no second place for the set to disagree with itself.
func (c *Orchestrator) ConvergePlacementSet(instanceID string, placements []DesiredPlacement) *exit.Error {
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil {
		s = c.sessions[w.bootID]
	}
	c.mu.Unlock()
	if w == nil {
		return exit.New(exit.NotFound, "no worker %s on this host", instanceID)
	}
	if s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream to converge", instanceID)
	}
	return c.converge(s, w, placements)
}

func (c *Orchestrator) converge(s *session, w *worker, placements []DesiredPlacement) *exit.Error {
	var setBytes, digest []byte
	if len(placements) == 1 && len(placements[0].ExactPlacementSetBytes) > 0 {
		// REMOTE: relay Tensorhub's exact acquisition-attempt bytes. Parsing is
		// validation only; these bytes are never marshaled again.
		p := placements[0]
		declared, err := canonical.Raw(p.ExactPlacementSetDigest)
		if err != nil || !bytes.Equal(canonical.Digest(p.ExactPlacementSetBytes), declared) {
			return exit.Named(exit.Conflict, "placement_set_identity_mismatch",
				"the persisted remote PlacementSet bytes do not match %s", p.ExactPlacementSetDigest)
		}
		doc, err := canonical.Read(p.ExactPlacementSetBytes, &pb.PlacementSet{})
		if err != nil || len(doc.List("placements")) != 1 ||
			doc.List("placements")[0].Str("placement_id") != p.PlacementID() {
			return exit.Named(exit.Conflict, "placement_set_closure_mismatch",
				"the persisted remote PlacementSet does not name placement %s: %v", p.PlacementID(), err)
		}
		setBytes, digest = append([]byte(nil), p.ExactPlacementSetBytes...), append([]byte(nil), declared...)
	} else {
		// LOCAL: retain the independent install-derived authoring path. An attached
		// worker never gets this host's identity digests substituted for its own.
		set := &pb.PlacementSet{}
		for _, p := range placements {
			if w.spec.Connection != nil && p.EnvironmentSpecDigest == "" {
				return exit.Named(exit.Structural, "remote_placement_identity_missing",
					"attached worker %s placement %s carries no frozen environment digest",
					w.instanceID, p.PlacementID())
			}
			set.Placements = append(set.Placements, &pb.Placement{
				PlacementId: p.PlacementID(),
				Spec:        c.placementSpec(w, p),
			})
		}
		var err error
		setBytes, digest, err = canonical.Identity(set)
		if err != nil {
			return exit.Internalf("cannot mint the PlacementSet document for %s: %s", w.instanceID, err)
		}
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.revision, w.setDigest, w.setBytes = rev, digest, setBytes
	c.mu.Unlock()

	d := &pb.DesiredWorkerState{
		Revision: rev, WireMinor: pb.WireMinor,
		Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
			PlacementSetDigest:         digest,
			PlacementSetCanonicalBytes: setBytes,
		}},
	}
	if len(placements) == 0 {
		// An empty set is a worker asked to host nothing. It DRAINS: the posture is the
		// only thing that says "finish what you hold and take no more", and a retirement
		// that left the posture ACCEPTING would be asking for work it has nowhere to run.
		d.Posture = pb.Posture_POSTURE_DRAINING
	}
	d.RecordOwnerEpoch, d.ControlStreamGeneration, d.WorkerBootId = recordOwnerEpoch, s.generation, s.bootID
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}})
	c.logf("DesiredWorkerState revision=%d posture=%s placements=%d set=%s (%d canonical bytes) -> %s",
		rev, trimEnum(pb.Posture_name[int32(d.Posture)], "POSTURE_"), len(placements),
		shortDigest(shortNone(digest)), len(setBytes), s.bootID)
	return nil
}

// placementSpec mints the immutable PlacementSpec document for one desired placement. THE
// DESIRED SET NAMES BYTES, NEVER A POINTER (§1): `endpoint_release_id` survives only as
// PROVENANCE and every other field is an immutable digest, so two workers handed the same
// set converge to the same bytes or fault typed.
//
// A local install leaves environment/receipt empty unless its own launcher supplied
// those facts; it never manufactures Tensorhub documents. A remote placement normally
// bypasses this author entirely because converge relays Tensorhub's exact PlacementSet.
func (c *Orchestrator) placementSpec(w *worker, p DesiredPlacement) *pb.PlacementSpec {
	spec := &pb.PlacementSpec{
		EndpointReleaseId: p.ReleaseID,
		BindingPlans:      w.subjects,
		ModelObjectSet:    modelObjectSetSubject(p),
	}
	environmentDigest := p.EnvironmentSpecDigest
	if environmentDigest == "" && w.spec.Connection == nil {
		environmentDigest = c.opt.EnvironmentSpecDigest
	}
	if raw, err := canonical.Raw(environmentDigest); err == nil {
		spec.EnvironmentSpecDigest = raw
	}
	if raw, err := canonical.Raw(p.InstalledEnvironmentReceiptDigest); err == nil {
		spec.InstalledEnvironmentReceiptDigest = raw
	}
	if raw, err := canonical.Raw(p.DescriptorDigest); err == nil {
		spec.DescriptorDigest = raw
	}
	return spec
}

const emptyModelObjectSet = `{"kind":"tensorhub.resolved_object_set/1","roots":[]}`

// modelObjectSetSubject makes the required PlacementSpec/2 closure explicit even for a
// weightless local endpoint. The empty object-set document is a real canonical subject,
// not absence; remote placements carry Tensorhub's exact non-empty subject and normally
// bypass local authoring by relaying the complete PlacementSet bytes.
func modelObjectSetSubject(p DesiredPlacement) *pb.ArtifactSubject {
	digest, length := p.ModelObjectSetDigest, p.ModelObjectSetLength
	var raw []byte
	if digest == "" {
		raw = canonical.Digest([]byte(emptyModelObjectSet))
		digest, _ = canonical.Spell(raw)
		length = uint64(len(emptyModelObjectSet))
	} else {
		var err error
		raw, err = canonical.Raw(digest)
		if err != nil {
			return nil
		}
	}
	if length == 0 {
		return nil
	}
	return &pb.ArtifactSubject{Digest: raw, SubjectId: digest, Kind: "model_object_set", Length: length}
}

// ------------------------------------------------------------------- observed state

// onObserved applies one ObservedWorkerState. Nothing here infers: every field is the
// worker's own last word, and the two facts the frozen wire could not tell apart —
// "your message arrived" and "your intent is satisfied" — are now two separate readable
// numbers (#473).
func (c *Orchestrator) onObserved(s *session, r *pb.ObservedWorkerState) {
	defer c.wakeWorkflows()
	var status *pb.PlacementStatus
	var acquisition *records.PlacementAcquisition
	c.mu.Lock()
	w := c.workers[s.instanceID]
	if w != nil && w.bootID != s.bootID {
		// A superseded stream's report is a fact about a worker that no longer exists.
		c.mu.Unlock()
		c.logf("DROPPED: observed state from superseded boot %s (the live one is %s)",
			s.bootID, w.bootID)
		return
	}
	if w != nil {
		w.lastReport = time.Now()
		w.phase = r.WorkerPhase
		// THE ONE ADMISSION FENCE, worker-level. Per-placement credits are gone: the seats
		// are a property of the machine, and `available_attempt_slots` already counts both
		// running attempts and outcomes this owner has not acked (#480d).
		w.admission, w.admissionGen = r.AdmissionState, r.AdmissionGeneration
		w.observeSlots(int(r.AvailableAttemptSlots))
		w.acceptedRevision = r.AcceptedDesiredStateRevision
		w.convergedRevision = r.ConvergedRevision
		w.acceptedSetDigest = r.AcceptedPlacementSetDigest
		w.appliedGrantRevision, w.appliedGrantID = r.AppliedGrantRevision, r.AppliedArtifactGrantId
		dispatchable, materializable := map[string]bool{}, map[string]bool{}
		if w.spec.IsJob() {
			// THE JOB LANE READS JOB CAPACITY: a job worker is in JobDirective mode and
			// hosts no placement at all, so it has no serving axis. `jobs_available` IS
			// the job credit, and this lane's dispatchability is that number.
			avail := r.GetJobCapacity().GetJobsAvailable()
			w.observeJobs(int(avail))
			for _, p := range w.spec.Placement.Jobs {
				dispatchable[p.DescriptorID] = avail > 0
			}
			// A job worker's own admission fence is its job capacity: the placement-set
			// machinery does not run, so the seats are what it says they are.
			if w.admission == pb.AdmissionState_ADMISSION_STATE_UNSPECIFIED {
				w.admission = pb.AdmissionState_ADMISSION_STATE_OPEN
			}
		} else {
			for _, p := range r.Placements {
				if p.PlacementId == w.placementID {
					status = p
				}
			}
			if status != nil {
				w.materialization, w.serving = status.Materialization, status.Serving
				w.generation = status.ExecutorGeneration
				w.specDigest, w.fallbackPin = status.PlacementSpecDigest, status.RetainedFallbackSpecDigest
				observed, facts := placementAcquisitionOf(w.instanceID, s.bootID, status)
				w.acquisition = facts
				if observed != nil {
					acquisition = observed
				}
				for _, id := range status.DispatchablePlanIds {
					dispatchable[id] = true
				}
				for _, id := range status.MaterializablePlanIds {
					materializable[id] = true
				}
				for _, f := range status.Faults {
					w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
				}
			} else {
				// No placement yet — the desired state has not been accepted, or it has and
				// nothing has been materialized. Both axes stay UNSPECIFIED and nothing is
				// dispatchable, which is exactly true.
				w.materialization = pb.MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED
				w.serving = pb.ServingState_SERVING_STATE_UNSPECIFIED
			}
		}
		w.dispatchable, w.materializable = dispatchable, materializable
		// Fault rows explain state; FAILED axes decide terminality. In particular,
		// BINDING_DEGRADED explicitly means "the worker still serves" and must never become
		// kill authority merely because it shares the diagnostic list with fatal faults.
		w.faulted = faulted(status, r)
		if w.faulted {
			if w.errorSince.IsZero() {
				w.errorSince = time.Now()
			}
		} else {
			w.errorSince = time.Time{}
			if status != nil && len(status.Faults) == 0 && len(r.Faults) == 0 {
				w.fault = ""
			}
		}
		for _, f := range r.Faults {
			w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
		}
		if len(r.Faults) == 0 && status != nil {
			for _, f := range status.Faults {
				w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
			}
		}
		// THE NO-PROGRESS GROUND'S BOOKKEEPING (cl-025). Movement is a changed signature
		// between two of the worker's own reports; the wedge verdict is the worker's own
		// liveness monitor speaking on the activity lane. The orchestrator only counts.
		sig := progressSignature(r, status)
		moved := sig != w.progressSig
		w.progressSig = sig
		w.wedgedSubjects = wedgeDeclared(r)
		w.wedged = len(w.wedgedSubjects) > 0
		switch {
		case moved, !w.wedged:
			w.noProgress = 0
		default:
			w.noProgress++
		}
	}
	generation, accepted := int64(0), int64(0)
	phase := trimEnum(pb.WorkerPhase_name[int32(r.WorkerPhase)], "WORKER_PHASE_")
	if w != nil {
		generation, accepted = int64(w.generation), int64(w.acceptedRevision)
	}
	c.mu.Unlock()
	_ = c.opt.Store.ReportWorker(s.bootID, phase, accepted, int64(r.ConvergedRevision), generation)
	if acquisition != nil {
		if problem := c.opt.Store.ObservePlacementAcquisition(*acquisition); problem != nil {
			c.logf("placement acquisition observation REFUSED: %s", problem.Message)
		}
	}
	if w != nil && w.media != nil {
		go c.retryMediaCleanup(w)
	}
	for _, f := range r.Faults {
		c.logf("worker fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
			f.Subject, f.Reason, f.Detail)
	}
	for _, a := range r.Activity {
		c.logf("activity seq=%d %s: %s", a.Seq, a.Kind, a.Step)
	}
	if status != nil {
		for _, f := range status.Faults {
			c.logf("placement fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
				f.Subject, f.Reason, f.Detail)
		}
	}
	// DISPATCHABLE is the only state that can change the queue's answer.
	if status != nil && status.Serving == pb.ServingState_SERVING_STATE_DISPATCHABLE {
		go c.drain()
	} else if w != nil && w.spec.IsJob() && r.GetJobCapacity().GetJobsAvailable() > 0 {
		go c.drain()
	}
	if r.GetJobCapacity() != nil {
		c.logf("observed phase=%s accepted=%d converged=%d jobs_available=%d jobs_in_flight=%d",
			phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision,
			r.GetJobCapacity().GetJobsAvailable(), r.GetJobCapacity().GetJobsInFlight())
		return
	}
	if status != nil {
		c.logf("observed phase=%s accepted=%d converged=%d placement=%s %s/%s generation=%d "+
			"admission=%s/%d slots=%d dispatchable=%d held=%d",
			phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision, status.PlacementId,
			trimEnum(pb.MaterializationState_name[int32(status.Materialization)], "MATERIALIZATION_STATE_"),
			trimEnum(pb.ServingState_name[int32(status.Serving)], "SERVING_STATE_"),
			status.ExecutorGeneration,
			trimEnum(pb.AdmissionState_name[int32(r.AdmissionState)], "ADMISSION_STATE_"),
			r.AdmissionGeneration, r.AvailableAttemptSlots,
			len(status.DispatchablePlanIds), len(r.HeldAttempts))
		return
	}
	c.logf("observed phase=%s accepted=%d converged=%d (no placement applied yet)",
		phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision)
}

func placementAcquisitionOf(instanceID, bootID string,
	status *pb.PlacementStatus) (*records.PlacementAcquisition, PlacementAcquisitionFacts) {
	var facts PlacementAcquisitionFacts
	if status == nil || status.Acquisition == nil {
		return nil, facts
	}
	specDigest, err := canonical.Spell(status.PlacementSpecDigest)
	if err != nil {
		return nil, facts
	}
	leg := func(in *pb.AcquisitionLegObservation) (records.AcquisitionLeg, AcquisitionLegFacts) {
		if in == nil {
			return records.AcquisitionLeg{}, AcquisitionLegFacts{}
		}
		stored := records.AcquisitionLeg{
			StartedNS: in.StartedMonotonicNs, EndedNS: in.EndedMonotonicNs,
			DownloadedBytes: in.DownloadedBytes, ReusedBytes: in.ReusedBytes,
		}
		return stored, AcquisitionLegFacts{
			StartedNS: stored.StartedNS, EndedNS: stored.EndedNS,
			DownloadedBytes: stored.DownloadedBytes, ReusedBytes: stored.ReusedBytes,
		}
	}
	endpoint, endpointFacts := leg(status.Acquisition.Endpoint)
	model, modelFacts := leg(status.Acquisition.Model)
	facts.Endpoint, facts.Model = endpointFacts, modelFacts
	return &records.PlacementAcquisition{
		InstanceID: instanceID, WorkerBootID: bootID, PlacementID: status.PlacementId,
		PlacementSpecDigest: specDigest, Endpoint: endpoint, Model: model,
	}, facts
}

// progressSignature renders every axis one ObservedWorkerState reports into one
// comparable string, so "no axis moved since the last report" is a comparison of two
// worker reports and nothing else (cl-025). The activity lane's high-water sequence is
// the important member: the worker's liveness notes, boot steps and admission changes all
// bump it, so a worker doing anything at all shows movement here even while its placement
// axes hold still.
func progressSignature(r *pb.ObservedWorkerState, status *pb.PlacementStatus) string {
	maxSeq := uint64(0)
	for _, a := range r.Activity {
		if a.Seq > maxSeq {
			maxSeq = a.Seq
		}
	}
	sig := fmt.Sprintf("phase=%d adm=%d/%d slots=%d acc=%d conv=%d held=%d faults=%d seq=%d",
		r.WorkerPhase, r.AdmissionState, r.AdmissionGeneration, r.AvailableAttemptSlots,
		r.AcceptedDesiredStateRevision, r.ConvergedRevision, len(r.HeldAttempts),
		len(r.Faults), maxSeq)
	if jc := r.GetJobCapacity(); jc != nil {
		sig += fmt.Sprintf(" jobs=%d/%d", jc.GetJobsAvailable(), jc.GetJobsInFlight())
	}
	if status != nil {
		sig += fmt.Sprintf(" mat=%d srv=%d gen=%d disp=%s matz=%s spec=%x",
			status.Materialization, status.Serving, status.ExecutorGeneration,
			strings.Join(status.DispatchablePlanIds, ","),
			strings.Join(status.MaterializablePlanIds, ","), status.PlacementSpecDigest)
		if acquisition := status.GetAcquisition(); acquisition != nil {
			for _, observed := range []struct {
				name string
				leg  *pb.AcquisitionLegObservation
			}{{"endpoint", acquisition.Endpoint}, {"model", acquisition.Model}} {
				name, leg := observed.name, observed.leg
				if leg != nil {
					sig += fmt.Sprintf(" %s=%d/%d/%d/%d", name, leg.StartedMonotonicNs,
						leg.EndedMonotonicNs, leg.DownloadedBytes, leg.ReusedBytes)
				}
			}
		}
	}
	return sig
}

// wedgeDeclared reads the worker's OWN no-progress verdict off the activity lane: its
// liveness monitor emits a `liveness` note naming the WEDGED subject when consecutive
// observations find a monotone position unmoved (a count of observations, clock-free on
// the worker too). The orchestrator never diagnoses a wedge itself — it acts on this
// report, which is the whole of decisions #613's rule.
func wedgeDeclared(r *pb.ObservedWorkerState) map[string]bool {
	type verdict struct {
		seq    uint64
		wedged bool
	}
	latest := map[string]verdict{}
	for _, activity := range r.Activity {
		if activity.Kind != "liveness" {
			continue
		}
		subject, wedged, ok := livenessVerdict(activity.Step)
		if !ok || latest[subject].seq > activity.Seq {
			continue
		}
		latest[subject] = verdict{seq: activity.Seq, wedged: wedged}
	}
	out := map[string]bool{}
	for subject, current := range latest {
		if current.wedged {
			out[subject] = true
		}
	}
	return out
}

// livenessVerdict reads only the two exact Runtime spellings. In particular, the
// recovery note contains the historical word WEDGED while explicitly retracting it;
// substring matching turned that retraction into a fresh wedge.
func livenessVerdict(step string) (subject string, wedged bool, ok bool) {
	if subject, _, ok = strings.Cut(step, " is WEDGED by silence"); ok && subject != "" {
		return subject, true, true
	}
	if subject, _, ok = strings.Cut(step, " resumed ("); ok && subject != "" {
		return subject, false, true
	}
	return "", false, false
}

// faulted reads only the protocol's terminal axes. Fault rows are explanations and may
// coexist with a serving placement (BINDING_DEGRADED and a rejected replacement both do);
// treating their mere presence as terminal killed healthy active attempts.
func faulted(status *pb.PlacementStatus, r *pb.ObservedWorkerState) bool {
	if r.WorkerPhase == pb.WorkerPhase_WORKER_PHASE_FAILED {
		return true
	}
	if status == nil {
		return false
	}
	return status.Materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED
}

// --------------------------------------------------------------------------- accepted

func (c *Orchestrator) onAccepted(s *session, a *pb.AttemptAccepted) {
	ordinal := a.AttemptOrdinal
	row, e := c.opt.Store.AttemptRow(a.RequestId, int64(ordinal))
	if e != nil || row == nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: no such assigned attempt", a.RequestId, ordinal)
		return
	}
	spelled, err := canonical.Spell(a.InvocationSpecDigest)
	if err != nil || spelled != row.InvocationDigest {
		c.logf("AttemptAccepted for %s#%d REFUSED: it echoes %v, the assignment is %s",
			a.RequestId, ordinal, spelled, row.InvocationDigest)
		return
	}
	planDigest, _ := canonical.Spell(a.PlanDigest)
	construction, _ := canonical.Spell(a.ModelConstructionDigest)
	summary := planSummary(a.Plan)
	if e := c.opt.Store.Accepted(a.RequestId, int64(ordinal), s.bootID, planDigest, construction, summary); e != nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: %s", a.RequestId, ordinal, e.Message)
		return
	}
	c.settleDispatch(a.RequestId, ordinal, true)
	c.logf("AttemptAccepted %s#%d placement=%s generation=%d plan=%s construction=%s [%s]",
		a.RequestId, ordinal, a.PlacementId, a.ExecutorGeneration,
		shortDigest(planDigest), shortDigest(construction), summary)
	c.emit(a.RequestId, "request.accepted", ordinal, map[string]any{
		"plan_digest": planDigest, "construction_digest": construction, "plan": summary,
	})
	c.waitFor(key(a.RequestId, ordinal)).markAccepted()
}

// planSummary renders the CLOSED observable projection of the chosen plan. The
// orchestrator sees WHAT was chosen, not only that something was — and it renders those
// facts without ever choosing one.
func planSummary(p *pb.AttemptPlanSummary) string {
	if p == nil {
		return ""
	}
	// `p.Placement` is TENSOR RESIDENCY and has nothing to do with a placement_id — the
	// protocol's own field name, transported and rendered, never chosen here. #510i renames
	// it to `residency` on the protocol's next touch; until then the comment is the fence.
	return fmt.Sprintf("%s/%s %s %s device=%dB host=%dB",
		p.Delivery, p.Materialization, p.ComputeDtype, p.Placement,
		p.ReservedDeviceMemoryBytes, p.ReservedHostBytes)
}

// --------------------------------------------------------------------------- outcome

// onOutcome applies one AttemptOutcome — was AttemptTerminal, and the rename is the point
// (#481): "terminal" reads as a device and the thing is an OUTCOME. It is also no longer
// only the end of execution, because a pre-execution REFUSAL is a journaled outcome too.
func (c *Orchestrator) onOutcome(s *session, t *pb.AttemptOutcome) {
	ordinal := t.AttemptOrdinal
	refuse := func(format string, args ...any) {
		c.logf("AttemptOutcome %s#%d REFUSED: "+format,
			append([]any{t.RequestId, ordinal}, args...)...)
	}

	// 1. The digest is RECOMPUTED over the resident bytes. A digest never bypasses the
	//    lower check, and a mismatched one is not acked — the worker keeps replaying.
	computed := canonical.Digest(t.OutcomeCanonicalBytes)
	if !bytes.Equal(computed, t.OutcomeDigest) {
		refuse("outcome_digest %x does not hash the %d resident bytes (%x)",
			t.OutcomeDigest, len(t.OutcomeCanonicalBytes), computed)
		return
	}
	// 2. The document is parsed under UNKNOWN-FIELD REFUSAL and the re-emit law. It is an
	//    AttemptOutcomeBody/3 — a NEW document version, because artifact receipts are a
	//    new key and a digest-fenced document is not additively versioned (th-049).
	doc, err := canonical.Read(t.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
	if err != nil {
		refuse("the outcome document is inadmissible (%s)", err)
		return
	}
	// 3. The routing copies must agree with the DOCUMENT, which is authoritative.
	spelledSpec, _ := canonical.Spell(t.InvocationSpecDigest)
	if doc.Str("request_id") != t.RequestId || uint64(doc.Int("attempt_ordinal")) != ordinal ||
		doc.Str("invocation_spec_digest") != spelledSpec {
		refuse("envelope/document divergence: envelope %s#%d/%s, document %s#%d/%s",
			t.RequestId, ordinal, shortDigest(spelledSpec),
			doc.Str("request_id"), doc.Int("attempt_ordinal"),
			shortDigest(doc.Str("invocation_spec_digest")))
		return
	}

	// THE OUTCOME IS HELD FROM HERE UNTIL THE ACK, and the worker counts it against its own
	// free seats while this owner holds it (#480d). Tracking it here is what makes an owner
	// that stops acking VISIBLE as an owner starving its own admission, rather than as a
	// worker that mysteriously stopped taking work.
	c.mu.Lock()
	if holder := c.workers[s.instanceID]; holder != nil {
		holder.unacked++
	}
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		if holder := c.workers[s.instanceID]; holder != nil && holder.unacked > 0 {
			holder.unacked--
		}
		c.mu.Unlock()
	}()

	status := outcomeStatus(doc.Int("status"))
	cause := causeCode(doc.Sub("cause").Int("code"))
	origin := causeOrigin(doc.Sub("cause").Int("origin"))
	executionStarted, _ := doc["execution_started"].(bool)
	req, e := c.opt.Store.RequestRow(t.RequestId)
	if e != nil || req == nil {
		refuse("no request row to settle")
		return
	}
	attemptRow, e := c.opt.Store.AttemptRow(t.RequestId, int64(ordinal))
	if e != nil || attemptRow == nil {
		refuse("no assigned attempt row to settle")
		return
	}
	declaredArtifactOutputs, e := decodeArtifactOutputs(attemptRow.ArtifactOutputs)
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	if len(declaredArtifactOutputs) > 0 && len(doc.Sub("output_manifest").List("outputs")) > 0 {
		refuse("artifact-only job also returned ordinary output-manifest entries")
		return
	}
	artifactReceipts, receiptsBySlot, e := artifactReceiptsFromOutcome(*req, *attemptRow, doc)
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	// An exact replay of a terminal already in the authority needs only another ack. In
	// particular it must not fetch pod outputs again: after the first mirror and ack this
	// owner is allowed to delete the remote attempt subtree, while the durable local output
	// rows and bytes remain the accepted answer.
	knownReplay := false
	if attemptRow.State == "terminal" || attemptRow.State == "closed" {
		if attemptRow.TerminalDigest != shortNone(t.OutcomeDigest) {
			refuse("the attempt already closed with terminal %s, not %s",
				shortDigest(attemptRow.TerminalDigest), shortDigest(shortNone(t.OutcomeDigest)))
			return
		}
		knownReplay = true
	}
	// THE MIRROR RUNS BEFORE ANYTHING IS ACCEPTED OR ACKED. An output that is not here,
	// or is not what the manifest says it is, means this outcome cannot be honoured: the
	// worker's journaled outcome stays owed and replayable, exactly as a failed job
	// promotion does below, rather than being acked with a row that points at nothing.
	c.mu.Lock()
	holder := c.workers[s.instanceID]
	c.mu.Unlock()
	var outputs []records.Output
	if !knownReplay {
		outputs, e = c.mirrorOutputs(*req, ordinal, doc, holder)
		if e != nil {
			refuse("%s", e.Message)
			return
		}
	}
	// cl-006 OWNS TRIAGE PERSISTENCE (cr-011's seam): the bundle lives in the worker's
	// own root, and a worker root does not outlive its worker. Copy it out and verify it
	// against the outcome document's OWN TriageBundleRef before the transaction runs.
	triage := Triage{}
	if !knownReplay {
		triage = c.captureTriage(s, t.RequestId, ordinal, doc.Sub("triage_bundle"))
	}
	// WHETHER THIS ATTEMPT ENDS THE REQUEST is decided BEFORE the event is written, not
	// after. An ABANDONED attempt that the requeue projection will re-dispatch has ended
	// an ATTEMPT, not a REQUEST — and the contract's terminal-stop rule means a client
	// that saw `request.failed` would close its stream and report a failure for a request
	// that goes on to succeed. Found live by the orchestrator-kill arm: the killed
	// attempt's ABANDONED terminal stopped the client's stream while attempt 2 was still
	// being minted.
	requeuing := requeueable(status, cause, origin, executionStarted)
	kept := triage.keep(status, cause, doc.Str("safe_message"), outputs, requeuing)
	artifactFinalizations, e := artifactFinalizationIntents(
		*req, *attemptRow, status, requeuing, receiptsBySlot)
	if e != nil {
		refuse("%s", e.Message)
		return
	}

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
	if req.IsJob() && len(declaredArtifactOutputs) == 0 && !requeuing && !knownReplay {
		if e := c.promote(*req, ordinal, outputs); e != nil {
			refuse("%s", e.Message)
			return
		}
		publication = c.publicationOf(*req, ordinal, status, cause, outputs)
	}

	began := time.Now()
	applied, e := c.opt.Store.AcceptTerminal(records.Terminal{
		RequestID: t.RequestId, Attempt: int64(ordinal), SessionID: s.bootID,
		InvocationDigest: spelledSpec, TerminalID: t.OutcomeId,
		TerminalDigest: shortNone(t.OutcomeDigest), Status: status, Cause: cause,
		SafeMessage:   doc.Str("safe_message"),
		TriageSubject: triage.Subject, TriageDigest: triage.Digest,
		TriageLength: triage.Length, TriagePath: triage.Path,
		Body: t.OutcomeCanonicalBytes, Outputs: outputs,
		ArtifactReceipts: artifactReceipts, ArtifactFinalizations: artifactFinalizations,
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
	// CAPACITY MOVES ONLY AFTER AUTHORITY ACCEPTS THE OUTCOME. In particular, a frame from
	// the wrong session or carrying the wrong InvocationSpec digest must not be able to
	// return another worker's reservation and reopen its seat before AcceptTerminal rejects
	// it. The durable attempt row authenticates the causal answer; the frame cannot do so by
	// naming a request id that happens to hold an offer.
	c.settleDispatch(t.RequestId, ordinal, status != "REFUSED" || executionStarted)
	if applied {
		c.logf("AttemptOutcome %s#%d %s/%s(%s) started=%v applied in %.2f ms: %d output(s) "+
			"became visible in the SAME transaction", t.RequestId, ordinal, status, cause,
			origin, executionStarted, float64(time.Since(began).Microseconds())/1000,
			len(outputs))
		if len(artifactReceipts) > 0 {
			c.logf("AttemptOutcome %s#%d durably recorded %s before acknowledgement",
				t.RequestId, ordinal, artifactReceiptSummary(artifactReceipts))
		}
		if publication != nil {
			c.logf("publication %s committed: %d entr(y|ies), %d B, root %s",
				publication.Repo, publication.Entries, publication.Bytes, publication.Root)
		}
	} else {
		c.logf("AttemptOutcome %s#%d is an exact replay of a closed outcome: re-acked, "+
			"nothing applied twice", t.RequestId, ordinal)
	}

	// Artifact decisions are independent frames, but the worker may reclaim after Ack. Send
	// every persisted first-wins intent now and withhold Ack until their exact results land.
	pending, e := c.sendPendingArtifactFinalizations(s, t.RequestId, int64(ordinal))
	if e != nil {
		refuse("artifact finalization remains pending: %s", e.Message)
		return
	}
	if pending > 0 {
		c.logf("AttemptOutcome %s#%d remains unacked behind %d artifact finalization(s)",
			t.RequestId, ordinal, pending)
		return
	}
	c.ackSettledOutcome(s, t.RequestId, ordinal)
}

// afterAck is the one post-terminal continuation, shared by the live frame and snapshot
// recovery. It is intentionally idempotent: cleanup and BeginRequeue both have durable
// guards, so a replay cannot spend twice or delete a still-owned asset.
func (c *Orchestrator) afterAck(req records.Request, attempt records.Attempt, holder *worker) {
	defer c.wakeWorkflows()
	requeue := req.State == "requeue_pending"
	c.cleanupAttempt(req, uint64(attempt.Attempt), holder, !requeue)
	verdict := outcomeError(attempt.TerminalStatus, attempt.TerminalCause, attempt.SafeMessage)
	c.waitFor(key(req.ID, uint64(attempt.Attempt))).markClosed(verdict)
	if requeue {
		c.Requeue(req.ID, attempt.TerminalStatus+"/"+attempt.TerminalCause)
		return
	}
	c.frames.forget(req.ID)
	c.waitRequest(req.ID).markClosed(verdict)
}

// cleanupAttempt runs only after the outcome's bytes were mirrored, its terminal commit
// succeeded, and OutcomeAck was sent. Per-attempt inputs are always disposable here;
// request assets are dropped only after the final attempt and only when no other live
// request owns the same content digest. Locally mirrored outputs remain addressable.
func (c *Orchestrator) cleanupAttempt(req records.Request, attempt uint64, holder *worker, final bool) {
	if holder != nil && holder.media != nil {
		c.cleanupRemote(req.ID, attempt, holder)
	} else if !req.IsJob() {
		if err := os.RemoveAll(filepath.Join(c.opt.Layout.AttemptDir(req.ID, attempt), "in")); err != nil {
			c.logf("%s#%d local attempt input cleanup failed: %s", req.ID, attempt, err)
		}
	}
	if !final {
		return
	}
	c.cleanupRequestAssets(req)
}

func (c *Orchestrator) cleanupRequestAssets(req records.Request) {
	unlock := inputasset.Guard()
	defer unlock()
	if e := inputasset.DropUnowned(c.opt.Layout, c.opt.Store, req.Assets); e != nil {
		c.logf("request %s input asset cleanup deferred: %s", req.ID, e.Message)
	}
}

func (c *Orchestrator) cleanupRemote(requestID string, attempt uint64, holder *worker) {
	k := key(requestID, attempt)
	c.mu.Lock()
	if c.mediaCleaning[k] {
		c.mu.Unlock()
		return
	}
	c.mediaCleaning[k] = true
	c.mu.Unlock()

	e := holder.media.DropAttempt(media.Slot(requestID, attempt))
	if e == nil {
		e = c.opt.Store.MarkMediaCleaned(requestID, int64(attempt))
	}
	c.mu.Lock()
	delete(c.mediaCleaning, k)
	c.mu.Unlock()
	if e != nil {
		c.logf("%s#%d pod media cleanup deferred: %s", requestID, attempt, e.Message)
	}
}

// Every remote Report retries the durable obligations still assigned to that worker.
// DELETE is idempotent, and media_cleaned stops successful rows from being revisited.
func (c *Orchestrator) retryMediaCleanup(holder *worker) {
	// Local workers have no pod media plane. Their attempt directories are removed by
	// cleanupAttempt; a recovered local snapshot must never try to call a nil client.
	if holder == nil || holder.media == nil {
		return
	}
	owed, e := c.opt.Store.MediaCleanupOwed(holder.instanceID)
	if e != nil {
		c.logf("cannot read media cleanup obligations of %s: %s", holder.instanceID, e.Message)
		return
	}
	for _, attempt := range owed {
		c.cleanupRemote(attempt.RequestID, uint64(attempt.Attempt), holder)
	}
}

// Triage is what cl-006 persisted for one attempt. An empty Subject means the outcome
// named no bundle, which is a fact, not a failure.
type Triage struct {
	Subject string
	Digest  string
	Length  int64
	Path    string
	Fault   string // why the bytes were not kept, when a subject was named and they were not
}

// outcomeEvent is the durable lifecycle event that rides the outcome transaction.
type outcomeEvent struct {
	Type    string
	Payload map[string]any
}

// requeueState is the REQUEST's state after one attempt ended. A requeueing request is
// queued for its next ordinal; anything else takes the attempt's own status.
func requeueState(status string, requeuing bool) string {
	if requeuing {
		return "requeue_pending"
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
func (tr Triage) keep(status, cause, safeMessage string, outputs []records.Output, requeuing bool) outcomeEvent {
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
	return outcomeEvent{Type: eventType, Payload: payload}
}

// captureTriage copies the worker's bundle into cozy-creator's own store, verified
// against the TERMINAL DOCUMENT's reference rather than against the worker's journal.
//
// That choice is deliberate and stronger than cr-011's own reader: this orchestrator has
// already recomputed the terminal digest over the resident bytes and parsed the document
// under unknown-field refusal, so `triage_bundle.write_receipt_digest` is a fact it
// ACCEPTED. Reading the worker's journal instead would be trusting a file the worker can
// still write. A mismatch is recorded as a fault on the attempt and the bytes are not
// kept — a bundle that does not hash to what the terminal claimed is not evidence.
func (c *Orchestrator) captureTriage(s *session, requestID string, attempt uint64, ref canonical.Doc) Triage {
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

// requeueable is the record owner's projection: an accepted-but-incomplete attempt, the
// infra-class failures, and a WORKER pre-execution refusal earn a new ordinal. A
// deterministic body failure and an author/runtime refusal settle — re-running them would
// only fail again.
//
// THE REFUSED PROJECTION SPLITS BY (cause, origin) — #480b, and it is the reason
// retryability cannot be a wire fact. "Refused" used to mean one thing: a judgment about
// the WORK, settle it. rev-2 routes a pre-execution capacity decline through the same
// status, because every offer must get a JOURNALED outcome and there is no
// AttemptDeclined message to add. Those consume the attempt ordinal and ZERO billed
// execution budget, so the next ordinal may go out immediately and elsewhere.
//
// `execution_started` is the structural check, not a cause-code allowlist (#480c): the
// author having run is a bit the worker sets, and inferring it from a code list is exactly
// the fragility the bit exists to remove. A "worker refusal" that claims execution
// started is a contradiction, and it settles rather than being re-dispatched.
//
// The re-dispatch still charges the request's DURABLE requeue budget. Zero BILLED budget
// is a statement about money; the bound on how many times this record owner will try is its own,
// and a worker refusing forever must still terminate.
func requeueable(status, cause, origin string, executionStarted bool) bool {
	if status == "ABANDONED" {
		return true
	}
	if status == "REFUSED" {
		return origin == "WORKER" && !executionStarted && preExecution(cause)
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

// preExecution names the four causes rev-2 §6/§7 defines as worker pre-execution
// refusals. It is a CLOSED list because the protocol's is closed: a cause outside it that
// arrives with origin WORKER is a peer saying something this contract does not define,
// and settling is the conservative answer.
func preExecution(cause string) bool {
	switch cause {
	case "NO_CAPACITY", "ADMISSION_GENERATION_STALE", "UNKNOWN_PLACEMENT",
		"PLACEMENT_NOT_DISPATCHABLE":
		return true
	}
	return false
}

// mirrorOutputs joins the manifest's entries to the destinations THIS record owner granted
// and PROVES the bytes are there before any of them becomes visible. The runtime names
// what it wrote; the record owner names where it was allowed to write and decides that the
// result is visible. Neither half can do the other's job.
//
// THE PROOF IS THE POINT, and it is why this is not a join any more. Composing a local
// path out of the grant and stamping the manifest's own digest onto it made the record a
// RESTATEMENT of the worker's claim rather than an observation: for a local worker the
// two happen to agree, and for a REMOTE one the bytes are on the pod and the row pointed
// at a path with nothing in it. `--out` then wrote a file the client had never received.
// So: the destination is read, its length and its digest are recomputed here, and an
// output that cannot be shown is never acked as one that can.
func (c *Orchestrator) mirrorOutputs(req records.Request, attempt uint64, doc canonical.Doc,
	holder *worker) ([]records.Output, *exit.Error) {
	manifest := doc.Sub("output_manifest")
	list, _ := manifest["outputs"].([]canonical.Value)
	// WHERE THE RECORD OWNER GRANTED. A serving attempt writes into its own disposable
	// attempt directory; a job writes into the durable publication root, which is the
	// same fact stated at the other end of the same grant.
	dir := c.opt.Layout.AttemptDir(req.ID, attempt)
	if req.IsJob() {
		// Where the orchestrator GRANTED: an attempt in flight writes into its stage, and
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
	declared := map[string]bool{}
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		e := canonical.Doc(entry)
		id := e.Str("output_id")
		path, problem := outputDest(req, dir, id)
		if problem != nil {
			return nil, problem
		}
		declared[id] = true
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
	if !req.IsJob() {
		// A serving manifest may name only granted ids, and a SUCCEEDED one names all of them.
		granted := map[string]bool{}
		for _, id := range splitList(req.Outputs) {
			granted[id] = true
			if !declared[id] && holder.media != nil && outcomeStatus(doc.Int("status")) == "SUCCEEDED" {
				return nil, exit.Named(exit.Validation, "output_set_mismatch",
					"the terminal omits granted output %q", id)
			}
		}
		for id := range declared {
			if !granted[id] {
				return nil, exit.Named(exit.Validation, "output_set_mismatch",
					"the terminal names output %q, which was never granted", id)
			}
		}
	}
	return out, nil
}

// outputDest resolves where one declared output may land: a job's publication fence, or
// a serving attempt's single-element id under its attempt directory.
func outputDest(req records.Request, dir, id string) (string, *exit.Error) {
	if req.IsJob() {
		return publicationDest(dir, id)
	}
	if e := FenceOutputID(id); e != nil {
		return "", e
	}
	return filepath.Join(dir, id), nil
}

// fetchOutputs pulls one remote attempt's declared outputs across the pod's media plane
// and lands them where this orchestrator granted, so the verification below runs against
// bytes THIS host holds.
//
// It is deliberately not a verification step: the media client checks only the pod's own
// declared digest against what arrived (a transport's business), and whether an output may
// become visible is decided one function up, against the TERMINAL's manifest. Two checks
// of two different claims, and neither substitutes for the other.
//
// RE-MIRRORING CONVERGES. A orchestrator killed between fetch and commit re-runs this on
// the worker's replayed terminal: the slot name is derived from the attempt's identity
// rather than remembered, the fetch is idempotent, and the write is atomic.
func (c *Orchestrator) fetchOutputs(req records.Request, attempt uint64,
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
		destination, e := outputDest(req, dir, id)
		if e != nil {
			return e
		}
		length := canonical.Doc(entry).Int("length")
		digest := canonical.Doc(entry).Str("digest")
		written, e := holder.media.GetOutputTo(slot, id, destination, digest, length)
		if e != nil {
			return e.WithRemedy("the pod declared this output in its terminal and this host "+
				"cannot fetch it from the pod's media plane, so the terminal stays OWED and "+
				"unacked rather than being accepted with a row that points at nothing (%s)",
				e.Remedy)
		}
		c.logf("mirrored %s#%d/%s: %d B from %s", req.ID, attempt, id, written,
			holder.media.Addr())
	}
	return nil
}

// verifyBytes is the mirror's proof: the declared identity, recomputed over the bytes
// that are actually at the granted destination. It reads the file once — the grant's own
// per-output ceiling already bounds how large one can be — because a length that agrees
// with a digest that does not is the interesting failure, not the cheap one.
func verifyBytes(path, digest string, length int64) *exit.Error {
	file, err := os.Open(path)
	if err != nil {
		return exit.New(exit.Failed,
			"the declared output at %s cannot be read here: %s", path, err)
	}
	defer file.Close()
	hash := sha256.New()
	read, err := io.Copy(hash, io.LimitReader(file, length+1))
	if err != nil {
		return exit.New(exit.Failed,
			"the declared output at %s cannot be read here: %s", path, err)
	}
	if read != length {
		return exit.New(exit.Failed,
			"the output at %s is %d B and its manifest declares %d B", path, read, length)
	}
	spelled, serr := canonical.Spell(hash.Sum(nil))
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

func outcomeStatus(n int64) string {
	return trimEnum(pb.OutcomeStatus_name[int32(n)], "OUTCOME_STATUS_")
}

func causeCode(n int64) string {
	return trimEnum(pb.CauseCode_name[int32(n)], "CAUSE_CODE_")
}

func causeOrigin(n int64) string {
	return trimEnum(pb.CauseOrigin_name[int32(n)], "CAUSE_ORIGIN_")
}

// outcomeError is the record owner's PROJECTION over (status, cause). Retryability is
// never a wire observation; this is the only place the neutral facts become an outcome.
func outcomeError(status, cause, message string) *exit.Error {
	switch status {
	case "SUCCEEDED":
		return nil
	case "REFUSED":
		if preExecution(cause) {
			// A PRE-EXECUTION refusal is not a judgment about the work: nothing about the
			// request was wrong, the worker simply could not take it. Rendering it as a
			// validation failure would tell a user to change a payload that is fine.
			return exit.Named(exit.Unavailable, "attempt_not_admitted",
				"the worker did not admit this attempt (%s): %s", cause, message).
				WithRemedy("the ordinal is consumed and no execution budget was spent; the " +
					"next ordinal may be dispatched immediately, here or elsewhere")
		}
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
