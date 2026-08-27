package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// fakeWorker is the ADVERSARY: a second, independent implementation of the WORKER side of
// `cozy.worker.v1` REV-2, in Go, that HOSTS `WorkerControl` exactly as the real worker does
// (#436 — the RecordOwner dials). It exists so the orchestrator's refusal arms have someone
// real to refuse — a real peer serving real bytes, never a plant inside the orchestrator.
//
// It also proves something the SDXL run cannot: the orchestrator interoperates with a
// worker it did not co-develop against, over the committed contract alone.
//
// REV-2 makes it host the new frames and, crucially, AUTHOR THE NEW DOCUMENTS: it mints a
// WorkerSnapshotBody and digests it, recomputes the owner's DesiredPlacementSet digest over
// the resident bytes before parsing them, and spells its outcomes as AttemptOutcomeBody/2.
// worker-protocol's frozen canonical corpus is the reference for every one of those bytes.
func fakeWorker() int {
	// The serve grammar the orchestrator speaks: `--socket` is the LISTEN grant, `--out`
	// the root whose `control.addr` publishes the bound address (the discovery contract).
	listen := flag("socket", "")
	out := flag("out", "")
	// --fake-instance wins over the --instance-id StartWorker appends, so an arm can
	// report an instance identity this orchestrator never spawned.
	instance := flag("fake-instance", flag("instance-id", ""))
	releaseID := flag("release-id", "")
	arm := flag("arm", "idle")
	bootID := flag("session", "boot-fake-"+randomHex(8))

	say := func(format string, args ...any) {
		fmt.Printf("[fake %s] %s\n", arm, fmt.Sprintf(format, args...))
		os.Stdout.Sync()
	}
	if arm == "badrelease" {
		releaseID = "cozy/not-the-pinned-release@v0"
	}

	// Bind exactly as the real worker does: a unix path, or host:port loopback.
	network, address := "unix", listen
	if strings.Contains(listen, ":") && !strings.ContainsAny(listen, `/\`) {
		network, address = "tcp", listen
	}
	if network == "unix" {
		_ = os.MkdirAll(filepath.Dir(address), 0o755)
	}
	_ = os.Remove(address)
	ln, err := net.Listen(network, address)
	if err != nil {
		say("cannot bind %s: %v", listen, err)
		return 1
	}
	bound := address
	if network == "tcp" {
		bound = ln.Addr().String()
	}
	if out != "" {
		if err := os.MkdirAll(out, 0o755); err == nil {
			staged := filepath.Join(out, "control.addr.staging")
			_ = os.WriteFile(staged, []byte(bound+"\n"), 0o644)
			_ = os.Rename(staged, filepath.Join(out, "control.addr"))
		}
	}
	say("hosting WorkerControl at %s (boot %s)", bound, bootID)

	// THE CLAIM CREDENTIAL, in whichever of the two launch modes this pod was provisioned
	// in (#560g). `--tokens` is the RENTED shape: a file of `sha256:<64 hex>` lines, which
	// is everything the provisioner was ever given, re-read per claim so a rotation
	// converges. COZY_BOOTSTRAP_CREDENTIAL is the SELF-SPAWNED shape, where the launcher
	// minted the credential and legitimately holds it. Both compare in constant time and
	// neither reads a raw value out of the Value that holds it.
	//
	// The `badcred` arm refuses EVERY proof, so the real owner's correct one is refused —
	// which is the arm.
	verify := func(string) bool { return true }
	if tokens := flag("tokens", ""); tokens != "" {
		verify = func(presented string) bool {
			data, err := os.ReadFile(tokens)
			if err != nil {
				return false // fail closed: a set that cannot be read admits nobody
			}
			for _, line := range strings.Split(string(data), "\n") {
				if secret.MatchesHash(presented, line) {
					return true
				}
			}
			return false
		}
	} else if cfg, e := config.Load(); e == nil && cfg.Bootstrap.Present() {
		bootstrap := cfg.Bootstrap
		verify = bootstrap.Equal
	}
	if arm == "badcred" {
		verify = func(string) bool { return false }
	}

	// The POD's spelling of the same listener (#445): the same `WorkerControl`, behind TLS
	// with the certificate the rental pins. `cozy-runtime serve --tls-cert/--tls-key` is
	// the real worker's flag pair and this is the adversary's, so the owner's dial is
	// exercised against a peer it did not co-develop with.
	var opts []grpc.ServerOption
	if cert, key := flag("tls-cert", ""), flag("tls-key", ""); cert != "" && key != "" {
		creds, err := credentials.NewServerTLSFromFile(cert, key)
		if err != nil {
			say("cannot host TLS from %s/%s: %v", cert, key, err)
			return 1
		}
		opts = append(opts, grpc.Creds(creds))
		say("hosting behind TLS with the certificate the owner pins")
	}
	server := grpc.NewServer(opts...)
	pb.RegisterWorkerControlServer(server, &fakeControl{
		say: say, arm: arm, bootID: bootID, instance: instance, releaseID: releaseID,
		root: flag("cozy-home", ""), verify: verify,
	})
	if err := server.Serve(ln); err != nil {
		say("serve ended: %v", err)
	}
	return 0
}

type fakeControl struct {
	pb.UnimplementedWorkerControlServer
	say        func(string, ...any)
	arm        string
	bootID     string
	instance   string
	releaseID  string
	root       string // this worker's OWN filesystem root; nothing outside it is writable
	verify     func(string) bool
	generation uint64
}

func (f *fakeControl) WatchProgress(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
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
	f.generation++
	generation := f.generation
	env := func(set func(epoch uint64, gen uint64, boot string)) {
		set(claim.RecordOwnerEpoch, generation, f.bootID)
	}
	// The worker-level admission fence. It is a CONSTANT here on purpose: this adversary
	// never respawns an executor or resizes its window, so the generation never moves, and
	// an offer echoing anything else is a RecordOwner dispatching against capacity meaning
	// it did not observe.
	const admissionGeneration = 1
	// One mutex over Send: the stall arms report on a cadence from their own goroutine
	// while the main loop answers frames, and a grpc stream tolerates one sender at a time.
	var sendMu sync.Mutex
	send := func(m *pb.WorkerFrame) {
		sendMu.Lock()
		defer sendMu.Unlock()
		if err := stream.Send(m); err != nil {
			f.say("send failed: %v", err)
		}
	}
	refuseClaim := func(reason pb.ClaimRejection, why string) {
		ack := &pb.ClaimAck{Accepted: false, Rejection: reason, WireMinor: pb.WireMinor,
			WireSchemaDigest: mySchemaDigest()}
		env(func(e, g uint64, b string) {
			ack.RecordOwnerEpoch, ack.ControlStreamGeneration, ack.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
		f.say("Claim REFUSED (%s): %s", pb.ClaimRejection_name[int32(reason)], why)
	}
	// THE SCHEMA FENCE IS CHECKED BEFORE ANY OTHER BODY FIELD (#530-A1). A worker that
	// authenticated a peer it cannot parse would be trusting a shape, not a credential —
	// so this runs ahead of the proof check, not after it.
	if !bytes.Equal(claim.WireSchemaDigest, mySchemaDigest()) {
		refuseClaim(pb.ClaimRejection_CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH,
			"the RecordOwner declares a wire schema this worker does not speak")
		return nil
	}
	if !f.verify(string(claim.Proof)) {
		refuseClaim(pb.ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED,
			"the presented proof is not the provisioned credential")
		return nil
	}
	ack := &pb.ClaimAck{
		Accepted: true, WireMinor: pb.WireMinor, WorkerId: "local",
		WorkerInstanceId: f.instance, WorkerReleaseId: f.releaseID,
		WireSchemaDigest: mySchemaDigest(),
		Resources:        &pb.WorkerResources{Platform: "fake", Backend: ""},
	}
	env(func(e, g uint64, b string) {
		ack.RecordOwnerEpoch, ack.ControlStreamGeneration, ack.WorkerBootId = e, g, b
	})
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
	f.say("ClaimAck sent: boot=%s instance=%s release=%s schema=rev%d",
		f.bootID, f.instance, f.releaseID, pb.WireSchemaRev)

	// ONE BOUNDED, DIGEST-ACKED SNAPSHOT (§5). The body is a real canonical document and
	// the digest is over exactly its bytes — a truncated snapshot cannot match one, which
	// is what replaces the old counted-entries claim.
	snapshotID := "snp-fake-" + randomHex(6)
	emptySet, emptySetDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		f.say("cannot mint the empty placement set: %v", err)
		return nil
	}
	spelledSet, _ := canonical.Spell(emptySetDigest)
	bodyBytes, bodyDigest, err := canonical.Identity(&pb.WorkerSnapshotBody{
		WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE,
		// ADMISSION IS CLOSED UNTIL THE ACK. That is not politeness: it is the barrier
		// stated on the wire, and an owner reading OPEN before it acked would be reading a
		// worker in breach.
		AdmissionState:             pb.AdmissionState_ADMISSION_STATE_CLOSED,
		AdmissionGeneration:        1,
		AvailableAttemptSlots:      0,
		AcceptedPlacementSetDigest: emptySetDigest,
	})
	if err != nil {
		f.say("cannot mint the snapshot body: %v", err)
		return nil
	}
	_ = spelledSet
	snap := &pb.WorkerSnapshot{
		SnapshotId: snapshotID, SnapshotDigest: bodyDigest,
		SnapshotCanonicalBytes:             bodyBytes,
		AcceptedPlacementSetCanonicalBytes: emptySet,
	}
	env(func(e, g uint64, b string) {
		snap.RecordOwnerEpoch, snap.ControlStreamGeneration, snap.WorkerBootId = e, g, b
	})
	send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: snap}})
	f.say("WorkerSnapshot %s sent (%d B, digest %s); admission is CLOSED until the ack",
		snapshotID, len(bodyBytes), hex.EncodeToString(bodyDigest)[:16])

	// observed reports the state on rev-2's own axes: a machine phase, a placement on TWO
	// axes, and the ONE worker-level admission fence. `intake_state` and `attempt_credits`
	// are gone, and nothing here spells a substitute for them.
	observed := func(revision uint64, placementID string, setDigest []byte, planIDs []string) {
		r := &pb.ObservedWorkerState{
			AcceptedDesiredStateRevision: revision,
			ConvergedRevision:            revision,
			AcceptedPlacementSetDigest:   setDigest,
			WorkerPhase:                  pb.WorkerPhase_WORKER_PHASE_ONLINE,
			AppliedWireMinor:             pb.WireMinor,
			AdmissionState:               pb.AdmissionState_ADMISSION_STATE_OPEN,
			AdmissionGeneration:          admissionGeneration,
			AvailableAttemptSlots:        2,
		}
		if placementID != "" {
			r.Placements = []*pb.PlacementStatus{{
				PlacementId:         placementID,
				Materialization:     pb.MaterializationState_MATERIALIZATION_STATE_STAGED,
				Serving:             pb.ServingState_SERVING_STATE_DISPATCHABLE,
				ExecutorGeneration:  1,
				DispatchablePlanIds: planIDs,
				PlacementSpecDigest: setDigest,
			}}
		}
		env(func(e, g uint64, b string) {
			r.RecordOwnerEpoch, r.ControlStreamGeneration, r.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: r}})
	}

	outcome := func(t *pb.AttemptOutcome) {
		env(func(e, g uint64, b string) {
			t.RecordOwnerEpoch, t.ControlStreamGeneration, t.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: t}})
	}

	var dropAck *pb.AttemptOutcome
	reporting := false
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
				f.say("SnapshotAck names %s/%s, which is not this snapshot — the barrier "+
					"stays CLOSED", a.SnapshotId, hex.EncodeToString(a.SnapshotDigest))
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
			placementID, planIDs, setDigest := "", []string(nil), []byte(nil)
			if ds := d.GetPlacementSet(); ds != nil {
				// THE BYTES ARE THE SET. Recompute BEFORE parsing a single field — a
				// mismatch is a typed refusal with the desired state UNAPPLIED, which is
				// the whole reason the structured copy was deleted.
				computed := canonical.Digest(ds.PlacementSetCanonicalBytes)
				if !bytes.Equal(computed, ds.PlacementSetDigest) {
					f.say("ARM: placement_set_digest %s does not hash the %d resident bytes "+
						"— UNAPPLIED (FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH)",
						hex.EncodeToString(ds.PlacementSetDigest)[:16],
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
					placementID = p.Str("placement_id")
					for _, sub := range p.Sub("spec").List("binding_plans") {
						planIDs = append(planIDs, sub.Str("subject_id"))
					}
				}
			}
			f.say("DesiredWorkerState revision=%d placement=%s plans=%d", d.Revision,
				placementID, len(planIDs))
			if placementID != "" && !reporting &&
				(f.arm == "slowfill" || f.arm == "wedged" || f.arm == "silent") {
				// The STALL ARMS (cl-025): a worker that is not yet DISPATCHABLE and says
				// so on the report cadence, from its own goroutine, exactly as the real
				// worker's reporter thread does.
				reporting = true
				go f.stallReports(stream.Context(), send, env, observed, d.Revision,
					placementID, setDigest, planIDs)
				continue
			}
			observed(d.Revision, placementID, setDigest, planIDs)
			f.say("observed DISPATCHABLE for %d plan(s)", len(planIDs))
		case *pb.RecordOwnerFrame_AttemptOffer:
			offer := m.AttemptOffer
			f.say("AttemptOffer %s#%d placement=%s admission=%d spec=%s", offer.RequestId,
				offer.AttemptOrdinal, offer.PlacementId, offer.AdmissionGeneration,
				hex.EncodeToString(offer.InvocationSpecDigest)[:16])
			// EVERY OFFER GETS A JOURNALED ANSWER (#472f/#480b). A stale admission
			// generation is refused with an OUTCOME, never with silence — and the outcome
			// says `execution_started: false`, which is the STRUCTURAL billing fact.
			if offer.AdmissionGeneration != admissionGeneration {
				f.say("ARM: admission generation %d is stale (current %d) — a journaled "+
					"REFUSED outcome, zero budget", offer.AdmissionGeneration, admissionGeneration)
				t, _ := outcomeFor(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
					pb.OutcomeStatus_OUTCOME_STATUS_REFUSED,
					"the echoed admission generation is not current",
					pb.CauseCode_CAUSE_CODE_ADMISSION_GENERATION_STALE,
					pb.CauseOrigin_CAUSE_ORIGIN_WORKER, false)
				t.PlacementId = offer.PlacementId
				outcome(t)
				continue
			}
			if f.arm == "remote" || f.arm == "remotelie" {
				if why := f.validateRemoteOffer(offer); why != nil {
					f.say("ARM: the remote offer is not an exact invocation/grant pair: %v", why)
					t, _ := outcomeFor(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
						pb.OutcomeStatus_OUTCOME_STATUS_REFUSED, why.Error(),
						pb.CauseCode_CAUSE_CODE_PROTOCOL, pb.CauseOrigin_CAUSE_ORIGIN_WORKER, false)
					t.PlacementId = offer.PlacementId
					outcome(t)
					continue
				}
			}
			accepted := &pb.AttemptAccepted{
				RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
				InvocationSpecDigest:    offer.InvocationSpecDigest,
				PlanDigest:              canonical.Digest([]byte("fake-plan")),
				ModelConstructionDigest: canonical.Digest([]byte("fake-construction")),
				Plan:                    &pb.AttemptPlanSummary{Delivery: "native", Placement: "all_resident"},
				PlacementId:             offer.PlacementId, ExecutorGeneration: 1,
			}
			env(func(e, g uint64, b string) {
				accepted.RecordOwnerEpoch, accepted.ControlStreamGeneration, accepted.WorkerBootId = e, g, b
			})
			send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: accepted}})
			if f.arm == "badterminal" {
				f.badOutcomes(outcome, offer)
			}
			if f.arm == "slowfill" {
				// The fill completed and the attempt was dispatched: settle it, which is
				// what "observed completing warm-up and serving" means for this arm.
				t, _ := outcomeFor(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
					pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
					"served after the deliberately throttled multi-minute fill",
					pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME, true)
				t.PlacementId = offer.PlacementId
				outcome(t)
			}
			if f.arm == "dropack" {
				dropAck = f.outcomeWithOutput(outcome, offer)
			}
			if f.arm == "remote" || f.arm == "remotelie" {
				f.remoteOutcome(outcome, offer)
			}
		case *pb.RecordOwnerFrame_OutcomeAck:
			f.say("AttemptOutcomeAck for %s#%d digest=%s", m.OutcomeAck.RequestId,
				m.OutcomeAck.AttemptOrdinal, hex.EncodeToString(m.OutcomeAck.OutcomeDigest)[:16])
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
			cancel := m.CancelAttempt
			f.say("CancelAttempt %s#%d", cancel.RequestId, cancel.AttemptOrdinal)
			if f.arm == "remote" || f.arm == "remotelie" {
				// A cancel is an ASK and the attempt's own journaled outcome is what
				// settles it. A worker that never answers leaves the owner watching
				// forever — a worker defect, not a protocol one — so the pod side answers
				// here the way a real worker does.
				t, _ := outcomeFor(cancel.RequestId, cancel.AttemptOrdinal, cancel.InvocationSpecDigest,
					pb.OutcomeStatus_OUTCOME_STATUS_CANCELED, "canceled by the RecordOwner",
					pb.CauseCode_CAUSE_CODE_CLIENT_CANCEL, pb.CauseOrigin_CAUSE_ORIGIN_CLIENT, true)
				outcome(t)
			}
		}
	}
}

// stallReports is the cl-025 arm family: a worker whose placement is NOT yet
// dispatchable, reporting on the real worker's cadence from its own goroutine.
//
//	slowfill  a THROTTLED MULTI-MINUTE fill: every report moves the activity sequence
//	          (as a real filling worker's progress notes do), the placement stays
//	          MATERIALIZING/ACTIVATING for --fill-ms, and only then does it report
//	          DISPATCHABLE. Under the deleted 90 s StallGrace this worker was killed
//	          mid-fill forever; under cl-025 nothing may touch it.
//	wedged    a worker that is ALIVE and heartbeating but whose own liveness monitor
//	          declared a subject WEDGED: the same two activity events ride every report
//	          and no axis ever moves. The orchestrator retires it on the worker's own
//	          no-progress report after NoProgressReports successive still reports.
//	silent    a worker that stops heartbeating entirely (the stream stays open, the
//	          process stays alive): dead to the record plane, retired via liveness.
func (f *fakeControl) stallReports(ctx context.Context, send func(*pb.WorkerFrame),
	env func(func(uint64, uint64, string)), ready func(uint64, string, []byte, []string),
	revision uint64, placementID string, setDigest []byte, planIDs []string) {
	fillMS := 130000
	if v := flag("fill-ms", ""); v != "" {
		fmt.Sscanf(v, "%d", &fillMS)
	}
	report := func(activity []*pb.ActivityEvent) {
		r := &pb.ObservedWorkerState{
			AcceptedDesiredStateRevision: revision,
			ConvergedRevision:            revision - 1,
			AcceptedPlacementSetDigest:   setDigest,
			WorkerPhase:                  pb.WorkerPhase_WORKER_PHASE_ONLINE,
			AppliedWireMinor:             pb.WireMinor,
			AdmissionState:               pb.AdmissionState_ADMISSION_STATE_OPEN,
			AdmissionGeneration:          1,
			AvailableAttemptSlots:        2,
			Activity:                     activity,
			Placements: []*pb.PlacementStatus{{
				PlacementId:           placementID,
				Materialization:       pb.MaterializationState_MATERIALIZATION_STATE_MATERIALIZING,
				Serving:               pb.ServingState_SERVING_STATE_ACTIVATING,
				ExecutorGeneration:    1,
				MaterializablePlanIds: planIDs,
				PlacementSpecDigest:   setDigest,
			}},
		}
		env(func(e, g uint64, b string) {
			r.RecordOwnerEpoch, r.ControlStreamGeneration, r.WorkerBootId = e, g, b
		})
		send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: r}})
	}
	event := func(seq uint64, kind, step string) *pb.ActivityEvent {
		return &pb.ActivityEvent{Seq: seq, Kind: kind, Step: step,
			AtUnixMs: uint64(time.Now().UnixMilli())}
	}
	began, n := time.Now(), uint64(0)
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		n++
		switch f.arm {
		case "slowfill":
			if time.Since(began) >= time.Duration(fillMS)*time.Millisecond {
				ready(revision, placementID, setDigest, planIDs)
				f.say("fill done after %s: observed DISPATCHABLE for %d plan(s)",
					time.Since(began).Round(time.Second), len(planIDs))
				return
			}
			// Every report MOVES: a fresh activity sequence, the way a real fill's
			// progress notes advance. Throttled, not wedged.
			report([]*pb.ActivityEvent{event(n, "download",
				fmt.Sprintf("filled unit %d of a deliberately throttled materialization", n))})
		case "wedged":
			activity := []*pb.ActivityEvent{event(1, "boot", "prepare started")}
			if n >= 3 {
				activity = append(activity, event(2, "liveness",
					"prepare:unet is WEDGED by silence (no clock involved)"))
			}
			report(activity)
		case "silent":
			if n > 2 {
				f.say("going SILENT: the stream stays open and nothing more is reported")
				return
			}
			report([]*pb.ActivityEvent{event(1, "boot", "prepare started")})
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

// outcomeFor builds one journaled AttemptOutcomeBody/2 and its envelope, the way a worker
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
	body := &pb.AttemptOutcomeBody{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: spelled,
		Status: status, SafeMessage: message, ExecutionStarted: executionStarted,
		Cause: &pb.OutcomeCause{
			Code: code, Origin: origin, Detail: "a fake worker's outcome",
		},
	}
	data, digest, err := canonical.Identity(body)
	if err != nil {
		panic(err)
	}
	return &pb.AttemptOutcome{
		RequestId: requestID, AttemptOrdinal: ordinal,
		InvocationSpecDigest: spec, OutcomeId: "out-" + randomHex(8), OutcomeDigest: digest,
		OutcomeCanonicalBytes: data,
	}, data
}

// authorFailure and runtimeSuccess are the two shapes every arm below wants, so the cause
// pair is stated once instead of at each call site.
func authorFailure(requestID string, ordinal uint64, spec []byte, status pb.OutcomeStatus,
	message string) (*pb.AttemptOutcome, []byte) {
	if status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		return outcomeFor(requestID, ordinal, spec, status, message,
			pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME, true)
	}
	return outcomeFor(requestID, ordinal, spec, status, message,
		pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR, true)
}

// badOutcomes is the refusal matrix, sent in order. Only the LAST one is admissible,
// and the ack that follows it is the orchestrator saying so.
func (f *fakeControl) badOutcomes(emit func(*pb.AttemptOutcome), offer *pb.AttemptOffer) {
	send := func(t *pb.AttemptOutcome) {
		emit(t)
		time.Sleep(400 * time.Millisecond)
	}

	// 1. An outcome_digest that does not hash the resident bytes. The receiver
	//    RECOMPUTES; a digest never bypasses the lower check.
	t, _ := authorFailure(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "planted digest")
	t.OutcomeDigest = canonical.Digest([]byte("not the body"))
	f.say("ARM 1: outcome_digest planted")
	send(t)

	// 2. The envelope's routing copies disagree with the document. The DOCUMENT is
	//    authoritative, so divergence refuses rather than picking a winner.
	t, _ = authorFailure(offer.RequestId, offer.AttemptOrdinal+7, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "divergent envelope")
	t.AttemptOrdinal = offer.AttemptOrdinal // the envelope says N, the document says N+7
	f.say("ARM 2: envelope/document divergence")
	send(t)

	// 3. A key the closed document has no slot for, written BY HAND because the schema
	//    cannot express it — which is the point of the arm.
	t, data := authorFailure(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "planted key")
	doc, err := canonical.Read(data, &pb.AttemptOutcomeBody{})
	if err == nil {
		raw := map[string]canonical.Value(doc)
		raw["service_class"] = "priority"
		planted, werr := canonical.Write(raw)
		if werr == nil {
			t.OutcomeCanonicalBytes = planted
			t.OutcomeDigest = canonical.Digest(planted)
		}
	}
	f.say("ARM 3: a planted key in the outcome document")
	send(t)

	// 4. An admissible outcome for an attempt this worker does hold.
	t, _ = authorFailure(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "the fake worker has no GPU")
	f.say("ARM 4: an admissible outcome")
	send(t)
	// 5. The exact same outcome again: a replay must re-ack and apply nothing twice.
	f.say("ARM 5: the same outcome replayed")
	send(t)
}

// remoteOutcome is the POD's side of one attempt, and the arm that makes the byte
// boundary REAL. A worker writes where the GRANT says and nowhere else — and it READS
// where the grant says too, which is the half that had no implementation until the pod
// grew a media server to put the bytes there (#506b).
//
//	remote     the honest pod: it reads the granted input off ITS OWN disk (the owner
//	           uploaded it through the media server), writes the output into the granted
//	           directory, and declares exactly those bytes. A granted destination that is
//	           NOT on this machine is refused by name — the client-local `file://` law,
//	           still enforced, now as the case that should never arise.
//	remotelie  does the work, writes the bytes somewhere else on its own disk, and declares
//	           them anyway: the manifest is true about what exists on the pod and false
//	           about what the owner can fetch, which is exactly what an UNMIRRORED output is
func (f *fakeControl) remoteOutcome(emit func(*pb.AttemptOutcome), offer *pb.AttemptOffer) {
	// validateRemoteOffer already consumed every input and compared the bytes with the
	// InvocationSpec before this attempt was accepted. Execution chooses only among output
	// ids that same validation proved are an exact grant/spec set.
	granted, outputID := "", ""
	for _, o := range offer.Grant.GetOutputs() {
		outputID = o.OutputId
		granted = strings.TrimPrefix(o.Url, "file://")
		break
	}
	if granted == "" {
		f.say("the grant names no output destination; nothing to write")
		return
	}
	if !underRoot(granted, f.root) {
		f.say("ARM: the granted destination %s is not on this machine (my root is %s)", granted, f.root)
		t, _ := authorFailure(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
			pb.OutcomeStatus_OUTCOME_STATUS_FAILED,
			"the granted output destination is on the owner's filesystem, not this worker's")
		emit(t)
		return
	}
	if f.arm == "remotelie" {
		// The bytes go where this worker CAN write and where the OWNER was never told to
		// look. Nothing here lies about the digest — the manifest describes real bytes on a
		// real disk. What is false is only that the owner can fetch them.
		granted = filepath.Join(f.root, "elsewhere", filepath.Base(granted))
	}
	f.writeGrantedOutput(emit, offer, outputID, granted)
}

// validateRemoteOffer is the fake pod's independent grant reader. It recomputes the
// InvocationSpec identity, proves the grant names exactly its input/output sets, consumes
// every granted input from THIS filesystem, and compares byte length, digest, and asset
// media type before AttemptAccepted can be sent. A URL alone is never an input identity.
func (f *fakeControl) validateRemoteOffer(offer *pb.AttemptOffer) error {
	if offer.Grant == nil {
		return fmt.Errorf("the offer carries no DeliveryGrant")
	}
	computed := canonical.Digest(offer.InvocationSpecCanonicalBytes)
	if !bytes.Equal(computed, offer.InvocationSpecDigest) {
		return fmt.Errorf("the invocation digest does not hash its canonical bytes")
	}
	if !bytes.Equal(offer.Grant.InvocationSpecDigest, offer.InvocationSpecDigest) {
		return fmt.Errorf("the grant names a different invocation digest")
	}
	doc, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	if err != nil {
		return fmt.Errorf("the invocation document is inadmissible: %w", err)
	}

	access := make(map[string]*pb.InputAccess, len(offer.Grant.Inputs))
	for _, in := range offer.Grant.Inputs {
		if in == nil || in.InputId == "" {
			return fmt.Errorf("the grant contains an unnamed input")
		}
		if access[in.InputId] != nil {
			return fmt.Errorf("the grant names input %s twice", in.InputId)
		}
		access[in.InputId] = in
	}
	bindings := doc.List("inputs")
	if len(access) != len(bindings) {
		return fmt.Errorf("the grant has %d inputs for the spec's %d", len(access), len(bindings))
	}
	for _, binding := range bindings {
		id := binding.Str("input_id")
		in := access[id]
		if in == nil {
			return fmt.Errorf("the grant has no access for input %s", id)
		}
		if !strings.HasPrefix(in.Url, "file://") {
			return fmt.Errorf("input %s is not a file grant", id)
		}
		path := strings.TrimPrefix(in.Url, "file://")
		if !underRoot(path, f.root) {
			return fmt.Errorf("input %s is outside this worker's filesystem", id)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("input %s cannot be read here: %w", id, err)
		}
		if int64(len(data)) != binding.Int("length") {
			return fmt.Errorf("input %s is %d B and its binding declares %d B",
				id, len(data), binding.Int("length"))
		}
		sum := sha256.Sum256(data)
		actual := "sha256:" + hex.EncodeToString(sum[:])
		if actual != binding.Str("digest") {
			return fmt.Errorf("input %s hashes to %s and its binding declares %s",
				id, actual, binding.Str("digest"))
		}
		kind := binding.Str("kind_mime")
		if id != "payload" && kind != "" && http.DetectContentType(data) != kind {
			return fmt.Errorf("input %s sniffs as %s and its binding declares %s",
				id, http.DetectContentType(data), kind)
		}
		if id == "payload" {
			f.say("granted input read from the pod's own disk: %s is %d B hashing to %s",
				path, len(data), shortSHA(actual))
		} else {
			f.say("validated granted input %s against its invocation binding: %d B, %s, %s",
				id, len(data), kind, actual)
		}
	}

	outputs := make(map[string]bool, len(offer.Grant.Outputs))
	for _, out := range offer.Grant.Outputs {
		if out == nil || out.OutputId == "" || outputs[out.OutputId] {
			return fmt.Errorf("the grant contains an unnamed or duplicate output")
		}
		if !strings.HasPrefix(out.Url, "file://") ||
			!underRoot(strings.TrimPrefix(out.Url, "file://"), f.root) {
			return fmt.Errorf("output %s is not a file grant on this worker", out.OutputId)
		}
		outputs[out.OutputId] = true
	}
	outputBindings := doc.List("outputs")
	if len(outputs) != len(outputBindings) {
		return fmt.Errorf("the grant has %d outputs for the spec's %d", len(outputs), len(outputBindings))
	}
	for _, binding := range outputBindings {
		if !outputs[binding.Str("output_id")] {
			return fmt.Errorf("the grant has no destination for output %s", binding.Str("output_id"))
		}
	}
	return nil
}

func shortSHA(digest string) string {
	if len(digest) > len("sha256:")+16 {
		return digest[:len("sha256:")+16]
	}
	return digest
}

// underRoot answers whether a granted path is on this worker's own filesystem root. It
// is the whole boundary check, and it is a string test on purpose: the interesting case
// is a path that does not exist HERE at all, which no stat can tell apart from a typo.
func underRoot(path, root string) bool {
	if root == "" {
		return false
	}
	rel, err := filepath.Rel(root, path)
	return err == nil && !strings.HasPrefix(rel, "..")
}

// outcomeWithOutput writes ONE real output under the attempt's granted directory and
// sends a SUCCEEDED terminal that declares it. It returns the identical envelope, which
// the `dropack` arm replays when the ack arrives — the orchestrator half of a lost
// TerminalAck.
func (f *fakeControl) outcomeWithOutput(emit func(*pb.AttemptOutcome),
	offer *pb.AttemptOffer) *pb.AttemptOutcome {
	layout, e := home.Open(flag("cozy-home", ""))
	if e != nil {
		f.say("no layout: %s", e.Message)
		return nil
	}
	return f.writeGrantedOutput(emit, offer,
		"image", filepath.Join(layout.AttemptDir(offer.RequestId, offer.AttemptOrdinal), "image"))
}

// writeGrantedOutput puts one transport fixture at `dest` and declares exactly those
// bytes. Images remain real PNGs. The video fixture is intentionally only MP4-shaped: this
// driver proves byte transport and never pretends that its adversarial worker ran H3.
func (f *fakeControl) writeGrantedOutput(emit func(*pb.AttemptOutcome),
	offer *pb.AttemptOffer, outputID, dest string) *pb.AttemptOutcome {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		f.say("cannot write under the grant: %v", err)
		return nil
	}
	bodyHex, mimeType := onePixelPNG, "image/png"
	if outputID == "video" {
		bodyHex, mimeType = transportMP4, "video/mp4"
	}
	body, err := hex.DecodeString(bodyHex)
	if err != nil {
		f.say("bad fixture: %v", err)
		return nil
	}
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		f.say("cannot write the output: %v", err)
		return nil
	}
	sum := sha256.Sum256(body)
	t, _ := authorFailure(offer.RequestId, offer.AttemptOrdinal, offer.InvocationSpecDigest,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "one transport output, written where the grant said")
	doc, err := canonical.Read(t.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
	if err != nil {
		f.say("cannot read back the outcome: %v", err)
		return nil
	}
	raw := map[string]canonical.Value(doc)
	// `manifest_id` is GONE: the field's own comment always said what it carries and the
	// NAME was the lie (#481/#486). The reader's key set is closed, so spelling the retired
	// name here would be a refusal rather than a silently ignored field.
	raw["output_manifest"] = map[string]canonical.Value{
		"publication_receipt_digest": "sha256:" + hex.EncodeToString(sum[:]),
		"outputs": []canonical.Value{map[string]canonical.Value{
			"output_id": outputID, "digest": "sha256:" + hex.EncodeToString(sum[:]),
			"length": int64(len(body)), "mime_type": mimeType,
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

// onePixelPNG is a 1x1 PNG, hex-encoded: the smallest thing that is really an image.
const onePixelPNG = "89504e470d0a1a0a0000000d4948445200000001000000010806000000" +
	"1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"

// transportMP4 is an ISO-BMFF ftyp box followed by empty free and mdat boxes. It has the
// shape and declared media type the last mile must preserve, but no encoded frame and no
// inference claim.
const transportMP4 = "000000186674797069736f6d0000020069736f6d69736f32" +
	"0000000866726565000000086d646174"

// stealOutcome is a worker trying to write an attempt row it was never assigned.
func (f *fakeControl) stealOutcome(emit func(*pb.AttemptOutcome)) {
	requestID := flag("request", "")
	var ordinal uint64
	fmt.Sscanf(flag("attempt", "1"), "%d", &ordinal)
	spec, _ := hex.DecodeString(flag("spec", ""))
	t, _ := authorFailure(requestID, ordinal, spec,
		pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "an outcome from a worker that does not own it")
	f.say("ARM: boot %s claims %s#%d, which it was never assigned", f.bootID, requestID, ordinal)
	emit(t)
}

// mySchemaDigest is THE FENCE as the raw bytes the wire carries — derived from the schema
// this adversary's own bindings were generated against. It is deliberately read from the
// same constant the RecordOwner reads: two peers built from one worker-protocol revision
// agree, and any other pair refuses at the handshake.
func mySchemaDigest() []byte {
	raw, err := canonical.Raw(pb.SchemaDigest)
	if err != nil {
		panic("the vendored wire_identity.go carries an unspellable schema digest: " + err.Error())
	}
	return raw
}

// randomHex uses crypto/rand, which speaks for every OS — the /dev/urandom spelling
// silently produced all-zero session ids on Windows, colliding every fake worker on one
// session (found by the windows-proc run: the "second writer" and "worker B" arms were
// really SESSION_COLLISION refusals).
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("no randomness for a fake boot id: " + err.Error())
	}
	return hex.EncodeToString(b)
}
