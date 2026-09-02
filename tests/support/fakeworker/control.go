package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type fakeControl struct {
	pb.UnimplementedWorkerControlServer
	say       func(string, ...any)
	arm       string
	bootID    string
	instance  string
	releaseID string
	root      string // this worker's OWN filesystem root
	verify    func(string) bool

	epoch        uint64
	snapshotSent atomic.Bool
}

func (f *fakeControl) WatchProgress(_ *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}

func (f *fakeControl) Control(stream pb.WorkerControl_ControlServer) error {
	frame, err := stream.Recv()
	if err != nil {
		return nil
	}
	claim := frame.GetClaim()
	if claim == nil {
		f.say("first frame was not a Claim; closing")
		return nil
	}
	f.epoch++
	env := func(set func(owner, control uint64, boot string)) {
		set(claim.RecordOwnerEpoch, f.epoch, f.bootID)
	}
	// The worker-level admission fence. It is a CONSTANT here on purpose: this adversary
	// never respawns an executor, so an offer echoing anything else is an owner dispatching
	// against capacity it did not observe.
	const admissionEpoch = 1
	var sendMu sync.Mutex
	send := func(m *pb.WorkerFrame) {
		sendMu.Lock()
		defer sendMu.Unlock()
		if err := stream.Send(m); err != nil {
			f.say("send failed: %v", err)
		}
	}
	refuseClaim := func(reason pb.ClaimRejection, why string) {
		ack := &pb.ClaimAck{Accepted: false, Rejection: reason, WireMinor: pb.WireMinor}
		env(func(e, g uint64, b string) {
			ack.RecordOwnerEpoch, ack.ControlStreamEpoch, ack.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
		f.say("Claim REFUSED (%s): %s", pb.ClaimRejection_name[int32(reason)], why)
	}
	if !f.verify(string(claim.Proof)) {
		refuseClaim(pb.ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED,
			"the presented proof is not the provisioned credential")
		return nil
	}
	ack := &pb.ClaimAck{
		Accepted: true, WireMinor: pb.WireMinor, WorkerId: "local",
		WorkerInstanceId: f.instance, WorkerReleaseId: f.releaseID,
		Resources: &pb.WorkerResources{Platform: "fake"},
	}
	env(func(e, g uint64, b string) {
		ack.RecordOwnerEpoch, ack.ControlStreamEpoch, ack.WorkerBootId = e, g, b
	})
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
	f.say("ClaimAck sent: boot=%s instance=%s release=%s minor=%d",
		f.bootID, f.instance, f.releaseID, pb.WireMinor)
	// ONE BOUNDED, DIGEST-ACKED SNAPSHOT (§5). The body is a real canonical document and
	// the digest is over exactly its bytes, so a truncated snapshot cannot match one.
	// Admission stays CLOSED until the ack: that barrier is stated on the wire.
	snapshotID := "snp-fake-" + randomHex(6)
	emptySet, emptySetDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		f.say("cannot mint the empty placement set: %v", err)
		return nil
	}
	bodyBytes, bodyDigest, err := canonical.Identity(&pb.WorkerSnapshotBody{
		WorkerPhase:                pb.WorkerPhase_WORKER_PHASE_ONLINE,
		AdmissionState:             pb.AdmissionState_ADMISSION_STATE_CLOSED,
		AdmissionEpoch:             1,
		AcceptedPlacementSetDigest: emptySetDigest,
	})
	if err != nil {
		f.say("cannot mint the snapshot body: %v", err)
		return nil
	}
	snap := &pb.WorkerSnapshot{SnapshotId: snapshotID, SnapshotDigest: bodyDigest,
		SnapshotCanonicalBytes: bodyBytes, AcceptedPlacementSetCanonicalBytes: emptySet}
	env(func(e, g uint64, b string) {
		snap.RecordOwnerEpoch, snap.ControlStreamEpoch, snap.WorkerBootId = e, g, b
	})
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: snap}})
	f.snapshotSent.Store(true)
	f.say("WorkerSnapshot %s sent (%d B); admission is CLOSED until the ack", snapshotID, len(bodyBytes))

	// THE LANES ARM (proto-024). Two device lanes over ordinals 0 and 1, ONE seat each,
	// and every placement on lane-0 — so the worker-level count is 2 (the sum, exactly as
	// the wire promises) while a placement can draw only ONE seat. An owner that reads the
	// sum offers two attempts to one placement and gets the second refused
	// CAUSE_CODE_NO_CAPACITY, journaled; an owner that reads lanes never offers it.
	lanes := f.arm == "lanes"
	var laneMu sync.Mutex
	laneHeld := map[string]int{}  // lane id -> seats held by a live or unacked attempt
	seatOf := map[string]string{} // "request#attempt" -> the lane whose seat it holds
	laneOf := func(string) string { return "lane-0" }
	laneRows := func() []*pb.DeviceLane {
		laneMu.Lock()
		defer laneMu.Unlock()
		return []*pb.DeviceLane{
			{LaneId: "lane-0", DeviceOrdinals: []uint32{0},
				AvailableAttemptSlots: uint32(max(0, 1-laneHeld["lane-0"]))},
			{LaneId: "lane-1", DeviceOrdinals: []uint32{1},
				AvailableAttemptSlots: uint32(max(0, 1-laneHeld["lane-1"]))},
		}
	}
	var lastRevision uint64
	var lastSetDigest []byte
	var lastPlacements []*pb.PlacementStatus
	observedMany := func(revision uint64, setDigest []byte, placements []*pb.PlacementStatus) {
		availableSlots := uint32(2)
		r := &pb.ObservedWorkerState{
			AcceptedDesiredStateRevision: revision, ConvergedRevision: revision,
			AcceptedPlacementSetDigest: setDigest,
			WorkerPhase:                pb.WorkerPhase_WORKER_PHASE_ONLINE,
			AppliedWireMinor:           pb.WireMinor,
			AdmissionState:             pb.AdmissionState_ADMISSION_STATE_OPEN,
			AdmissionEpoch:             admissionEpoch, AvailableAttemptSlots: availableSlots,
		}
		if lanes {
			r.Lanes = laneRows()
			r.AvailableAttemptSlots = 0
			for _, l := range r.Lanes {
				r.AvailableAttemptSlots += l.AvailableAttemptSlots
				for _, p := range placements {
					if laneOf(p.PlacementId) == l.LaneId {
						l.PlacementIds = append(l.PlacementIds, p.PlacementId)
					}
				}
			}
			for _, p := range placements {
				p.DeviceLaneId = laneOf(p.PlacementId)
			}
			lastRevision, lastSetDigest, lastPlacements = revision, setDigest, placements
		}
		r.Placements = placements
		env(func(e, g uint64, b string) {
			r.RecordOwnerEpoch, r.ControlStreamEpoch, r.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: r}})
		if lanes {
			f.say("ObservedWorkerState slots=%d lanes=%d", r.AvailableAttemptSlots, len(r.Lanes))
		}
	}
	placementStatus := func(placementID string, setDigest []byte, planIDs []string,
		packageRevision, environmentDigest, configDigest string,
	) *pb.PlacementStatus {
		var bindingDigests [][]byte
		for _, planID := range planIDs {
			digest, err := canonical.Raw(planID)
			if err == nil {
				bindingDigests = append(bindingDigests, digest)
			}
		}
		return &pb.PlacementStatus{
			PlacementId: placementID, Materialization: pb.MaterializationState_MATERIALIZATION_STATE_STAGED,
			Serving: pb.ServingState_SERVING_STATE_DISPATCHABLE, ExecutorEpoch: 1,
			DispatchableBindingDigests: bindingDigests, PlacementSetDigest: setDigest,
			PackageRevisionDigest: packageRevision, EnvironmentDigest: environmentDigest,
			ConfigDigest: configDigest,
		}
	}
	outcome := func(t *pb.AttemptOutcome) {
		env(func(e, g uint64, b string) {
			t.RecordOwnerEpoch, t.ControlStreamEpoch, t.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: t}})
	}

	var dropAck *pb.AttemptOutcome
	for {
		frame, err := stream.Recv()
		if err != nil {
			f.say("stream closed: %v", err)
			return nil
		}
		switch m := frame.Msg.(type) {
		case *pb.RecordOwnerFrame_SnapshotAck:
			// The ack is COMPARED, never recomputed: an ack naming a different id or digest
			// is not an ack, and the barrier stays closed.
			a := m.SnapshotAck
			if a.SnapshotId != snapshotID || !bytes.Equal(a.SnapshotDigest, bodyDigest) {
				f.say("SnapshotAck names %s, which is not this snapshot — the barrier stays CLOSED",
					a.SnapshotId)
				continue
			}
			f.say("SnapshotAck for %s matches the digest; dispatch open", a.SnapshotId)
			if f.arm == "steal" {
				f.stealOutcome(outcome)
				time.Sleep(2 * time.Second)
				return nil
			}
		case *pb.RecordOwnerFrame_DesiredState:
			d := m.DesiredState
			setDigest := []byte(nil)
			placements := []*pb.PlacementStatus(nil)
			if ds := d.GetPlacementSet(); ds != nil {
				// THE BYTES ARE THE SET. Recompute BEFORE parsing a single field — a
				// mismatch is a typed refusal with the desired state UNAPPLIED.
				if !bytes.Equal(canonical.Digest(ds.PlacementSetCanonicalBytes), ds.PlacementSetDigest) {
					f.say("ARM: placement_set_digest does not hash the %d resident bytes — UNAPPLIED",
						len(ds.PlacementSetCanonicalBytes))
					continue
				}
				setDigest = ds.PlacementSetDigest
				doc, err := canonical.Read(ds.PlacementSetCanonicalBytes, &pb.PlacementSet{})
				if err != nil {
					f.say("ARM: the placement set document is inadmissible: %v", err)
					continue
				}
				for _, p := range doc.List("placements") {
					planIDs := []string(nil)
					emptyConfig, _ := canonical.Write(map[string]canonical.Value{})
					configDigest, _ := canonical.Spell(canonical.Digest(emptyConfig))
					for _, entrypoint := range p.List("entrypoints") {
						planIDs = append(planIDs, entrypoint.Str("entrypoint_binding_digest"))
					}
					placements = append(placements, placementStatus(p.Str("placement_id"), setDigest,
						planIDs, p.Sub("package").Str("release_digest"),
						p.Str("environment_digest"), configDigest))
				}
			}
			plans := 0
			for _, placement := range placements {
				plans += len(placement.DispatchableBindingDigests)
			}
			f.say("DesiredWorkerState revision=%d placements=%d plans=%d", d.Revision,
				len(placements), plans)
			observedMany(d.Revision, setDigest, placements)
		case *pb.RecordOwnerFrame_AttemptOffer:
			offer := m.AttemptOffer
			f.say("AttemptOffer %s#%d placement=%s admission=%d", offer.RequestId,
				offer.AttemptOrdinal, offer.PlacementId, offer.AdmissionEpoch)
			// EVERY OFFER GETS A JOURNALED ANSWER (#472f/#480b). A stale admission
			// epoch is refused with an OUTCOME, never with silence — and the outcome
			// says `execution_started: false`, which is the STRUCTURAL billing fact.
			if offer.AdmissionEpoch != admissionEpoch {
				t, _ := outcomeFor(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
					pb.OutcomeStatus_OUTCOME_STATUS_REFUSED,
					"the echoed admission epoch is not current",
					pb.CauseCode_CAUSE_CODE_ADMISSION_EPOCH_STALE,
					pb.CauseOrigin_CAUSE_ORIGIN_WORKER, false)
				t.PlacementId = offer.PlacementId
				outcome(t)
				continue
			}
			if lanes {
				// THE SEAT IS THE LANE'S. A second offer against a placement whose lane is
				// full is refused exactly as a saturated window is: a journaled outcome,
				// pre-execution, never silence and never a hold.
				laneID := laneOf(offer.PlacementId)
				laneMu.Lock()
				full := laneHeld[laneID] >= 1
				if !full {
					laneHeld[laneID]++
					seatOf[fmt.Sprintf("%s#%d", offer.RequestId, offer.AttemptOrdinal)] = laneID
				}
				laneMu.Unlock()
				if full {
					f.say("ARM: lane %s has no free seat for %s#%d — REFUSED NO_CAPACITY (journaled)",
						laneID, offer.RequestId, offer.AttemptOrdinal)
					t, _ := outcomeFor(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
						pb.OutcomeStatus_OUTCOME_STATUS_REFUSED,
						"lane "+laneID+" has no free seat",
						pb.CauseCode_CAUSE_CODE_NO_CAPACITY,
						pb.CauseOrigin_CAUSE_ORIGIN_WORKER, false)
					t.PlacementId = offer.PlacementId
					outcome(t)
					continue
				}
				f.say("lane %s seat taken by %s#%d", laneID, offer.RequestId, offer.AttemptOrdinal)
			}
			accepted := &pb.AttemptAccepted{
				RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
				InvocationSpecDigest:    offer.InvocationSpecDigest,
				PlanDigest:              canonical.Digest([]byte("fake-plan")),
				ModelConstructionDigest: canonical.Digest([]byte("fake-construction")),
				Plan:                    &pb.AttemptPlanSummary{Delivery: "native", Placement: "all_resident"},
				PlacementId:             offer.PlacementId, ExecutorEpoch: 1,
			}
			env(func(e, g uint64, b string) {
				accepted.RecordOwnerEpoch, accepted.ControlStreamEpoch, accepted.WorkerBootId = e, g, b
			})
			send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: accepted}})
			switch f.arm {
			case "lanes":
				// The attempt runs for a moment and ends; the seat stays held until the
				// ack, which is when the worker's next observed state frees it.
				go func(offer *pb.AttemptOffer) {
					time.Sleep(1500 * time.Millisecond)
					t, _ := authorOutcome(offer.RequestId, offer.AttemptOrdinal,
						offer.InvocationSpecDigest, pb.OutcomeStatus_OUTCOME_STATUS_FAILED,
						"the fake worker has no GPU")
					t.PlacementId = offer.PlacementId
					outcome(t)
				}(offer)
			case "badterminal":
				f.badOutcomes(outcome, offer)
			case "dropack":
				dropAck = f.outcomeWithOutput(outcome, offer)
			case "missing-output":
				t, _ := authorOutcome(offer.RequestId, offer.AttemptOrdinal,
					offer.InvocationSpecDigest, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
					"success that omits its granted output")
				f.say("ARM: SUCCEEDED outcome omits the granted output")
				outcome(t)
			}
		case *pb.RecordOwnerFrame_OutcomeAck:
			a := m.OutcomeAck
			f.say("AttemptOutcomeAck for %s#%d", a.RequestId, a.AttemptOrdinal)
			if lanes {
				// Only an attempt that HELD a seat returns one: a refused offer took none.
				laneMu.Lock()
				seat := fmt.Sprintf("%s#%d", a.RequestId, a.AttemptOrdinal)
				if laneID, held := seatOf[seat]; held {
					laneHeld[laneID] = max(0, laneHeld[laneID]-1)
					delete(seatOf, seat)
				}
				laneMu.Unlock()
				observedMany(lastRevision, lastSetDigest, lastPlacements)
			}
			if f.arm == "badterminal" {
				time.Sleep(500 * time.Millisecond)
				return nil
			}
			if f.arm == "dropack" && dropAck != nil {
				// THE DROP. A worker whose ack never arrived keeps replaying its journaled
				// outcome — byte for byte. Ignoring the ack here is what a lost one looks
				// like from the orchestrator's side.
				f.say("ARM: the AttemptOutcomeAck is DROPPED, and the journaled outcome is replayed")
				outcome(dropAck)
				dropAck = nil
				go func() { time.Sleep(3 * time.Second); os.Exit(0) }()
			}
		case *pb.RecordOwnerFrame_CancelAttempt:
			f.say("CancelAttempt %s#%d", m.CancelAttempt.RequestId, m.CancelAttempt.AttemptOrdinal)
		}
	}
}

// outcomeFor builds one journaled AttemptOutcomeBody/1 and its envelope, the way a worker
// does: the document is canonicalized once, the digest is over exactly those bytes, and the
// envelope's routing copies are copies of the document's own fields.
//
// `executionStarted` is the STRUCTURAL billing fact (#480c) and it is a parameter rather
// than something derived from the status, because that is exactly the derivation the bit
// exists to delete: a pre-execution refusal and an author exception are both "not
// succeeded" and only one of them cost anything.
func outcomeFor(requestID string, ordinal uint64, spec []byte, status pb.OutcomeStatus,
	message string, code pb.CauseCode, origin pb.CauseOrigin,
	executionStarted bool) (*pb.AttemptOutcome, []byte) {
	spelled, _ := canonical.Spell(spec)
	data, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: spelled,
		Status: status, SafeMessage: message, ExecutionStarted: executionStarted,
		Cause: &pb.OutcomeCause{Code: code, Origin: origin, Detail: "a fake worker's outcome"},
	})
	if err != nil {
		panic(err)
	}
	return &pb.AttemptOutcome{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: spec,
		OutcomeId: "out-" + randomHex(8), OutcomeDigest: digest, OutcomeCanonicalBytes: data,
	}, data
}

func authorOutcome(requestID string, ordinal uint64, spec []byte, status pb.OutcomeStatus,
	message string) (*pb.AttemptOutcome, []byte) {
	if status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		return outcomeFor(requestID, ordinal, spec, status, message,
			pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME, true)
	}
	return outcomeFor(requestID, ordinal, spec, status, message,
		pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR, true)
}

// badOutcomes is the terminal refusal matrix, sent in order against ONE open attempt. Only
// the LAST one is admissible, and the ack that follows it is the orchestrator saying so.
func (f *fakeControl) badOutcomes(emit func(*pb.AttemptOutcome), offer *pb.AttemptOffer) {
	send := func(t *pb.AttemptOutcome) {
		emit(t)
		time.Sleep(400 * time.Millisecond)
	}
	succeeded := pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED

	// 1. An outcome_digest that does not hash the resident bytes. The receiver
	//    RECOMPUTES; a digest never bypasses the lower check.
	t, _ := authorOutcome(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		succeeded, "planted digest")
	t.OutcomeDigest = canonical.Digest([]byte("not the body"))
	f.say("ARM 1: outcome_digest planted")
	send(t)

	// 2. The envelope's routing copies disagree with the document. The DOCUMENT is
	//    authoritative, so divergence refuses rather than picking a winner.
	t, _ = authorOutcome(offer.RequestId, offer.AttemptOrdinal+7, offer.InvocationSpecDigest,
		succeeded, "divergent envelope")
	t.AttemptOrdinal = offer.AttemptOrdinal
	f.say("ARM 2: envelope/document divergence")
	send(t)

	// 3. A key the closed document has no slot for, written BY HAND because the schema
	//    cannot express it — which is the point of the arm.
	t, data := authorOutcome(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		succeeded, "planted key")
	if doc, err := canonical.Read(data, &pb.AttemptOutcomeBody{}); err == nil {
		raw := map[string]canonical.Value(doc)
		raw["service_class"] = "priority"
		if planted, werr := canonical.Write(raw); werr == nil {
			t.OutcomeCanonicalBytes, t.OutcomeDigest = planted, canonical.Digest(planted)
		}
	}
	f.say("ARM 3: a planted key in the outcome document")
	send(t)

	// 4/5. An admissible outcome, then the exact same one again: a replay must re-ack and
	// apply nothing twice.
	t, _ = authorOutcome(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "the fake worker has no GPU")
	f.say("ARM 4: an admissible outcome")
	send(t)
	f.say("ARM 5: the same outcome replayed")
	send(t)
}

// outcomeWithOutput writes ONE real PNG under the attempt's granted directory and sends a
// SUCCEEDED terminal declaring exactly those bytes. It returns the identical envelope,
// which the `dropack` arm replays when the ack arrives.
func (f *fakeControl) outcomeWithOutput(emit func(*pb.AttemptOutcome),
	offer *pb.AttemptOffer) *pb.AttemptOutcome {
	layout, e := home.Open(f.root)
	if e != nil {
		f.say("no layout: %s", e.Message)
		return nil
	}
	dest := filepath.Join(layout.AttemptDir(offer.RequestId, offer.AttemptOrdinal), "image")
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		f.say("cannot write under the grant: %v", err)
		return nil
	}
	body, err := hex.DecodeString(onePixelPNG)
	if err != nil || os.WriteFile(dest, body, 0o644) != nil {
		f.say("cannot write the output")
		return nil
	}
	sum := sha256.Sum256(body)
	t, _ := authorOutcome(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "one output, written where the grant said")
	doc, err := canonical.Read(t.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
	if err != nil {
		f.say("cannot read back the outcome: %v", err)
		return nil
	}
	raw := map[string]canonical.Value(doc)
	raw["output_manifest"] = map[string]canonical.Value{
		"publication_receipt_digest": "sha256:" + hex.EncodeToString(sum[:]),
		"outputs": []canonical.Value{map[string]canonical.Value{
			"output_id": "image", "digest": "sha256:" + hex.EncodeToString(sum[:]),
			"length": int64(len(body)), "mime_type": "image/png",
		}},
	}
	written, err := canonical.Write(raw)
	if err != nil {
		f.say("cannot canonicalize the manifest: %v", err)
		return nil
	}
	t.OutcomeCanonicalBytes, t.OutcomeDigest = written, canonical.Digest(written)
	f.say("outcome with ONE %d B output at %s", len(body), dest)
	emit(t)
	return t
}

// stealOutcome is a worker writing an attempt row it was never assigned.
func (f *fakeControl) stealOutcome(emit func(*pb.AttemptOutcome)) {
	spec, _ := hex.DecodeString(*stealSpec)
	t, _ := authorOutcome(*stealRequest, *stealAttempt, spec,
		pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "an outcome from a worker that does not own it")
	f.say("ARM: boot %s claims %s#%d, which it was never assigned", f.bootID, *stealRequest, *stealAttempt)
	emit(t)
}

// onePixelPNG is a 1x1 PNG, hex-encoded: the smallest thing that is really an image.
const onePixelPNG = "89504e470d0a1a0a0000000d4948445200000001000000010806000000" +
	"1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("no randomness for a fake boot id: " + err.Error())
	}
	return hex.EncodeToString(b)
}
