package orchestrator

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/workertls"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// THE RECORDOWNER SIDE of th-024's orientation (#436/#446/#454, renamed by #481 — bare
// "owner" is retired and the formal role is RecordOwner): the WORKER hosts `WorkerControl`
// and THIS side dials it, claims it, reconciles its ONE snapshot, and only then
// dispatches. One goroutine per worker owns the whole conversation; the `session` is that
// stream's sender half, fenced by the control stream generation the worker minted.
//
// LAUNCH TIER (#437): record_owner_epoch is the constant 1 — this service is the one
// RecordOwner of every worker it spawns or connects to; the machinery that MINTS competing
// epochs is the hub's Wave-2 lease. The bearer credential (#445/#449/#463) rides
// `Claim.proof`: for a spawned worker it is the per-spawn bootstrap credential this
// launcher minted; for a connected worker it is the provisioned renter access token.

const recordOwnerEpoch = 1

// recordOwnerID names this owner on Claim. Stable per service run is enough at launch:
// equal-epoch claims from the SAME RecordOwner are reconnects, anything else is refused.
const recordOwnerID = "cozy-local-client"

type session struct {
	ctx                context.Context
	bootID             string
	generation         uint64
	instanceID         string
	out                chan *pb.RecordOwnerFrame
	remoteGrantStarted bool
	claimAck           []byte
	snapshot           []byte
	relay              chan RentalSessionEvidence
}

func (s *session) send(m *pb.RecordOwnerFrame) (sent bool) {
	defer func() {
		if recover() != nil { // a closed stream is an ordinary unavailable answer
			sent = false
		}
	}()
	select {
	case s.out <- m:
		return true
	case <-s.ctx.Done():
		return false
	}
}

func (s *session) trySend(m *pb.RecordOwnerFrame) (sent bool) {
	defer func() {
		if recover() != nil {
			sent = false
		}
	}()
	select {
	case s.out <- m:
		return true
	default:
		return false
	}
}

// attach owns one worker's control conversation for the life of its process: read the
// published address, dial, claim, reconcile, direct, then pump frames. On a stream drop
// with the process still alive it re-dials and RE-CLAIMS the same boot (the worker mints
// a fresh control generation and resends its snapshot; replay covers the durables).
//
// A RECORDED REFUSAL ENDS THE LOOP. The verdicts this owner reaches at claim time — a
// foreign instance identity or an unpinned release — are facts about the THING AT THE
// OTHER END, and redialing cannot change any of them. Left
// running, the loop burns one of the worker's control generations every 200 ms forever;
// observed at 1,111 generations against a pre-rev-2 worker while a waiter sat on a
// readiness poll that was never going to end. The refusal is already the waiter's answer
// (`EnsurePlacementReady` reads it first) — this stops the conversation from outliving it.
func (c *Orchestrator) attach(w *worker) {
	defer close(w.attachDone)
	for {
		c.mu.Lock()
		current, live := c.workers[w.instanceID]
		closing, refused, exited, stopping := c.closing, w.refusal, w.exited, w.stopping
		c.mu.Unlock()
		if closing || !live || current != w || exited || stopping {
			return
		}
		if refused != nil {
			c.logf("worker %s: not re-claiming — this owner has refused it (%s)",
				w.instanceID, refused.ErrName())
			return
		}
		addr, e := c.workerAddr(w)
		if e != nil {
			c.logf("worker %s: %s", w.instanceID, e.Message)
			return
		}
		err := c.converse(w, addr)
		if err != nil {
			c.logf("worker %s: control stream ended: %s", w.instanceID, err)
		}
		c.mu.Lock()
		if w.spawned.IsZero() && w.lastReport.IsZero() {
			// An attached worker whose dial FAILED before any claim: that failure is
			// measured non-progress, and the silence budget counts from its first one.
			w.spawned = time.Now()
		}
		gone := w.exited || w.stopping || c.closing
		c.mu.Unlock()
		if gone {
			return
		}
		// The process is alive and the stream is not: redial. The cadence is a sampling
		// resolution (the worker republishes nothing; its listener persists), not a bound.
		time.Sleep(200 * time.Millisecond)
	}
}

// workerAddr resolves the dialable address: the remote spec's own, or the file-handoff
// `control.addr` the spawned worker publishes after binding (#436 discovery contract).
// The wait is bounded by the PROCESS: a worker that dies before binding is the answer.
func (c *Orchestrator) workerAddr(w *worker) (string, *exit.Error) {
	if w.spec.Connection != nil {
		return w.spec.Connection.Addr, nil
	}
	addrFile := filepath.Join(c.opt.Layout.WorkerDir(w.instanceID), "run", "control.addr")
	for {
		data, err := os.ReadFile(addrFile)
		if err == nil && len(data) > 0 {
			return strings.TrimSpace(string(data)), nil
		}
		c.mu.Lock()
		dead := w.exited || w.stopping
		c.mu.Unlock()
		if dead {
			return "", exit.New(exit.Failed,
				"the worker exited before publishing its control address").
				WithRemedy("its log is %s", w.logPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// dialWorker opens the channel: insecure over the unix socket / loopback a spawned worker
// binds; TLS with the PINNED cert for a remote one (#445 — the presented leaf must equal
// that PEM byte for byte; never mTLS).
func dialWorker(addr string, remote *WorkerConnection) (*grpc.ClientConn, error) {
	if remote != nil && remote.CACert != "" {
		pin, err := workertls.LoadPin(remote.CACert)
		if err != nil {
			return nil, err
		}
		creds := credentials.NewTLS(pin.TLSConfig())
		return grpc.NewClient(addr, grpc.WithTransportCredentials(creds))
	}
	target := addr
	if strings.HasPrefix(addr, "unix:") || strings.HasPrefix(addr, "/") {
		target = "unix:" + strings.TrimPrefix(addr, "unix:")
	}
	return grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
}

// converse runs ONE claimed stream to its end: Claim -> ClaimAck -> WorkerSnapshot ->
// SnapshotAck -> DesiredWorkerState -> frames. Returns when the stream closes.
func (c *Orchestrator) converse(w *worker, addr string) error {
	conn, err := dialWorker(addr, w.spec.Connection)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pb.NewWorkerControlClient(conn)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c.mu.Lock()
	if current := c.workers[w.instanceID]; current != w || w.exited || w.stopping || c.closing {
		c.mu.Unlock()
		return fmt.Errorf("worker %s is stopping", w.instanceID)
	}
	w.cancelControl = cancel
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		if current := c.workers[w.instanceID]; current == w {
			w.cancelControl = nil
		}
		c.mu.Unlock()
	}()
	stream, err := client.Control(ctx)
	if err != nil {
		return err
	}
	c.mu.Lock()
	if w.spawned.IsZero() && w.lastReport.IsZero() {
		w.spawned = time.Now() // the claim is out; silence from here is the pod's
	}
	c.mu.Unlock()
	s := &session{ctx: ctx, instanceID: w.instanceID, out: make(chan *pb.RecordOwnerFrame, 32)}
	if w.spec.Connection != nil && c.opt.RelayRentalSession != nil {
		s.relay = make(chan RentalSessionEvidence, 1)
		go c.runRentalSessionRelay(s, w.spec.Connection)
	}
	go func() {
		for m := range s.out {
			if err := stream.Send(m); err != nil {
				return
			}
		}
	}()
	defer close(s.out)

	proof := w.bootstrap.Reveal()
	if w.spec.Connection != nil {
		proof = w.spec.Connection.Token.Reveal()
	}
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: &pb.Claim{
		RecordOwnerEpoch: recordOwnerEpoch,
		RecordOwnerId:    recordOwnerID,
		WireMinor:        pb.WireMinor,
		Proof:            []byte(proof),
	}}}) //cozy:allow-reveal the one place the credential leaves this process: Claim.proof over the worker's own channel

	// WatchProgress rides a PHYSICALLY separate connection (01) once the claim lands;
	// opened after ClaimAck below.
	var watchCancel context.CancelFunc
	defer func() {
		if watchCancel != nil {
			watchCancel()
		}
	}()

	for {
		frame, err := stream.Recv()
		if err != nil {
			c.dropSession(s)
			return nil
		}
		switch m := frame.Msg.(type) {
		case *pb.WorkerFrame_ClaimAck:
			ack := m.ClaimAck
			if !ack.Accepted {
				c.logf("Claim REFUSED by %s: %s", w.instanceID,
					pb.ClaimRejection_name[int32(ack.Rejection)])
				return fmt.Errorf("claim refused: %s", pb.ClaimRejection_name[int32(ack.Rejection)])
			}
			s.bootID, s.generation = ack.WorkerBootId, ack.ControlStreamGeneration
			if e := c.onClaimAck(w, s, ack); e != nil {
				return fmt.Errorf("%s", e.Message)
			}
			raw, e := deterministicWorkerFrame(frame)
			if e != nil {
				c.refuseClaim(w, e)
				return fmt.Errorf("%s", e.Message)
			}
			s.claimAck = raw
			watchCancel = c.openWatch(addr, w, s)
		case *pb.WorkerFrame_BootFailure:
			c.logf("BOOT FAILURE from %s: %s (%s)", m.BootFailure.WorkerInstanceId,
				pb.BootFailureReason_name[int32(m.BootFailure.Reason)], m.BootFailure.Detail)
			c.relayBootFailure(s, w, frame, m.BootFailure)
			return nil
		case *pb.WorkerFrame_Snapshot:
			if c.fenced(s, m.Snapshot.RecordOwnerEpoch, m.Snapshot.ControlStreamGeneration,
				m.Snapshot.WorkerBootId) {
				continue
			}
			if c.onSnapshot(w, s, m.Snapshot) {
				raw, e := deterministicWorkerFrame(frame)
				if e != nil {
					c.logf("WorkerSnapshot %s could not be retained for rental evidence: %s",
						m.Snapshot.SnapshotId, e.Message)
					continue
				}
				s.snapshot = raw
			}
		case *pb.WorkerFrame_ObservedState:
			r := m.ObservedState
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamGeneration, r.WorkerBootId) {
				continue
			}
			raw, e := deterministicWorkerFrame(frame)
			if e != nil {
				c.logf("observed state could not be retained for rental evidence: %s", e.Message)
				continue
			}
			c.onObserved(s, r, raw)
		case *pb.WorkerFrame_AttemptAccepted:
			a := m.AttemptAccepted
			if c.fenced(s, a.RecordOwnerEpoch, a.ControlStreamGeneration, a.WorkerBootId) {
				continue
			}
			c.onAccepted(s, a)
		case *pb.WorkerFrame_AttemptOutcome:
			t := m.AttemptOutcome
			if c.fenced(s, t.RecordOwnerEpoch, t.ControlStreamGeneration, t.WorkerBootId) {
				continue
			}
			c.onOutcome(s, t)
		case *pb.WorkerFrame_CheckpointRequest:
			r := m.CheckpointRequest
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamGeneration, r.WorkerBootId) {
				continue
			}
			if !c.jobMode(s) {
				s.send(checkpointReceipt(s, r, "",
					pb.CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED,
					pb.CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE,
					"this worker is in serving mode; the checkpoint lane is the job lane's"))
				continue
			}
			c.onCheckpoint(s, r)
		case *pb.WorkerFrame_CheckpointAck:
			// the worker's echo of a receipt already durable here; nothing to apply
		case *pb.WorkerFrame_ArtifactFinalizeResult:
			r := m.ArtifactFinalizeResult
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamGeneration, r.WorkerBootId) {
				continue
			}
			c.onArtifactFinalizeResult(s, r)
		}
	}
}

func deterministicWorkerFrame(frame *pb.WorkerFrame) ([]byte, *exit.Error) {
	raw, err := (proto.MarshalOptions{Deterministic: true}).Marshal(frame)
	if err != nil {
		return nil, exit.Internalf("cannot deterministically encode WorkerFrame evidence: %s", err)
	}
	return raw, nil
}

// relayBootFailure preserves the protocol's closed boot-fatal alternative. It is
// accepted only from this claim's epoch and a named stream/boot, then sent through the
// renter-authenticated relay. Tensorhub verifies provider OCI and Runtime provenance;
// Creator neither turns this into a synthetic ObservedWorkerState nor gives Hub a way to
// dial the worker itself.
func (c *Orchestrator) relayBootFailure(s *session, w *worker, frame *pb.WorkerFrame,
	failure *pb.BootFailure) {
	if w.spec.Connection == nil || c.opt.RelayRentalSession == nil ||
		failure.RecordOwnerEpoch != recordOwnerEpoch || failure.ControlStreamGeneration == 0 ||
		failure.WorkerBootId == "" {
		return
	}
	if s.generation != 0 && c.fenced(s, failure.RecordOwnerEpoch,
		failure.ControlStreamGeneration, failure.WorkerBootId) {
		return
	}
	if e := instancePin(w, failure.WorkerInstanceId); e != nil {
		c.refuseClaim(w, e)
		return
	}
	if e := releasePin(w, failure.WorkerReleaseId); e != nil {
		c.refuseClaim(w, e)
		return
	}
	raw, e := deterministicWorkerFrame(frame)
	if e != nil {
		c.logf("rental %s boot failure could not be encoded: %s",
			w.spec.Connection.RentalID, e.Message)
		return
	}
	if e := c.opt.RelayRentalSession(s.ctx, w.spec.Connection,
		RentalSessionEvidence{BootFailure: raw}); e != nil {
		c.logf("rental %s boot failure evidence was not accepted: %s",
			w.spec.Connection.RentalID, e.Message)
	}
}

func (c *Orchestrator) queueRentalSessionEvidence(s *session, desired uint64, observed []byte) {
	if s.relay == nil || desired == 0 || len(s.claimAck) == 0 || len(s.snapshot) == 0 {
		return
	}
	evidence := RentalSessionEvidence{
		ClaimAck: append([]byte(nil), s.claimAck...), Snapshot: append([]byte(nil), s.snapshot...),
		ObservedState: append([]byte(nil), observed...), DesiredRevision: desired,
	}
	select {
	case s.relay <- evidence:
		return
	default:
	}
	// Only the newest observation matters while one HTTP call is in flight. The Hub
	// persists every accepted digest idempotently; replacing an unsent progress sample
	// cannot erase a frame it has already learned.
	select {
	case <-s.relay:
	default:
	}
	select {
	case s.relay <- evidence:
	case <-s.ctx.Done():
	}
}

func (c *Orchestrator) runRentalSessionRelay(s *session, connection *WorkerConnection) {
	for {
		select {
		case <-s.ctx.Done():
			return
		case evidence := <-s.relay:
			for {
				problem := c.opt.RelayRentalSession(s.ctx, connection, evidence)
				if problem == nil {
					break
				}
				c.logf("rental %s convergence evidence was not accepted: %s",
					connection.RentalID, problem.Message)
				if problem.Code != exit.Unavailable && problem.Code != exit.Deadline {
					break
				}
				timer := time.NewTimer(ReportCadence)
				select {
				case <-s.ctx.Done():
					timer.Stop()
					return
				case newer := <-s.relay:
					timer.Stop()
					evidence = newer
				case <-timer.C:
				}
			}
		}
	}
}

// fenced evaluates the three-field envelope in its fixed order, BEFORE any body field is
// interpreted (02 §0).
func (c *Orchestrator) fenced(s *session, epoch, generation uint64, bootID string) bool {
	if epoch != recordOwnerEpoch {
		c.logf("DROPPED: epoch %d is not this owner's %d", epoch, recordOwnerEpoch)
		return true
	}
	if generation != s.generation {
		c.logf("DROPPED: superseded control generation %d (live %d)", generation, s.generation)
		return true
	}
	if bootID != s.bootID {
		c.logf("DROPPED: boot %q is not this stream's %q", bootID, s.bootID)
		return true
	}
	return false
}

// onClaimAck binds the claimed boot to the worker slot: identity checks, the durable
// binding, and the session registry (the ClaimAck is the flip's Register successor).
func (c *Orchestrator) onClaimAck(w *worker, s *session, ack *pb.ClaimAck) *exit.Error {
	c.logf("ClaimAck boot=%s generation=%d instance=%s minor=%d backend=%q device=%q",
		ack.WorkerBootId, ack.ControlStreamGeneration, ack.WorkerInstanceId, ack.WireMinor,
		ack.Resources.GetBackend(), ack.Resources.GetDeviceName())
	if e := instancePin(w, ack.WorkerInstanceId); e != nil {
		// Nothing is dispatched to it; the stream ends on the next recv when we stop
		// talking.
		c.refuseClaim(w, e)
		return e
	}
	if e := releasePin(w, ack.WorkerReleaseId); e != nil {
		c.refuseClaim(w, e)
		return e
	}
	if w.spec.Connection != nil {
		if c.opt.ObserveRental == nil || c.opt.RelayRentalSession == nil ||
			w.spec.Connection.RentalID == "" {
			e := exit.Named(exit.Structural, "rental.worker_readback_unowned",
				"the attached worker has no durable rental observation and convergence-relay owner")
			c.refuseClaim(w, e)
			return e
		}
		resources := ack.GetResources()
		if resources == nil {
			e := exit.Named(exit.Conflict, "rental.worker_readback_missing",
				"rental %s ClaimAck carries no worker resources", w.spec.Connection.RentalID)
			c.refuseClaim(w, e)
			return e
		}
		if e := c.opt.ObserveRental(RentalObservation{
			RentalID: w.spec.Connection.RentalID, Accelerator: resources.GetDeviceName(),
			DeviceCount: int(resources.GetDeviceCount()), Backend: resources.GetBackend(),
			DriverVersion: resources.GetDriverVersion(), BackendVersion: resources.GetBackendVersion(),
			DeviceMemoryTotalBytes: resources.GetDeviceMemoryTotalBytes(),
			WorkerInstance:         ack.WorkerInstanceId, WorkerBootID: ack.WorkerBootId,
		}); e != nil {
			c.refuseClaim(w, e)
			return e
		}
	}
	if e := c.opt.Store.BindSession(w.instanceID, ack.WorkerBootId,
		int64(ack.ControlStreamGeneration)); e != nil {
		c.logf("boot binding for %s REFUSED: %s", w.instanceID, e.Message)
		c.refuseClaim(w, e)
		return e
	}
	c.mu.Lock()
	if w.spec.Connection != nil {
		w.remoteInstance = ack.WorkerInstanceId
	}
	// The ClaimAck is the first thing this worker said; the silence clock runs from here.
	w.lastReport = time.Now()
	if w.bootID != "" && w.bootID != ack.WorkerBootId {
		delete(c.sessions, w.bootID)
	}
	c.sessions[ack.WorkerBootId] = s
	w.bootID = ack.WorkerBootId
	c.mu.Unlock()
	c.wakeWorkflows()
	return nil
}

// instancePin is the IDENTITY FENCE on a claim, and #505's carried-not-verified gap in its
// second place.
//
// The check used to be one comparison against `w.instanceID` — the name of the SLOT — and
// that is the right question for exactly one lane. A worker this host SPAWNED was handed
// `--instance-id`, so a different name at that address is a stranger and refuses.
//
// A rented pod is not that. This host never spawned it, never named it, and could not
// have: the hub provisioned the worker before any owner attached, so the pod names its own
// instance exactly as it mints its own boot id. Demanding it answer to the slot's name was
// this host asking a machine it does not own to have been called something else — and it
// only ever passed because the stand-in pod declared NO instance at all, which took the
// empty-string branch. Carried and verified were the same branch again.
//
// So the rule splits the same way the release pin's does. SPAWNED: a declared identity that
// is not ours is a stranger. ATTACHED: silence cannot be attributed — a terminal from an
// unnamed instance belongs to nobody — and what this host holds a pod to is STABILITY: the
// identity recorded at the first claim is the identity every later claim must carry, so a
// DIFFERENT worker arriving at the same address is caught, which is the thing the old check
// was actually protecting.
func instancePin(w *worker, declared string) *exit.Error {
	if w.spec.Connection == nil {
		if declared != "" && declared != w.instanceID {
			return exit.Named(exit.Conflict, "worker_instance_mismatch",
				"the worker at this address answers as instance %q and this slot is %q",
				declared, w.instanceID)
		}
		return nil
	}
	if declared == "" {
		return exit.Named(exit.Conflict, "worker_instance_undeclared",
			"this pod's ClaimAck declares no worker instance, so nothing it settles could be "+
				"attributed to a worker at all").
			WithRemedy("a rented pod's worker is started with an instance identity; one that " +
				"will not say which worker it is cannot be dispatched to")
	}
	if w.remoteInstance != "" && w.remoteInstance != declared {
		return exit.Named(exit.Conflict, "worker_instance_changed",
			"this pod answered as instance %q and now answers as %q: a DIFFERENT worker is at "+
				"the address this rental pinned", w.remoteInstance, declared).
			WithRemedy("release the rental; a pod whose worker identity moved under this host " +
				"is not the machine its attempts were dispatched to")
	}
	return nil
}

// releasePin is the RELEASE FENCE on a claim, and #505's carried-not-verified gap closed.
//
// A base worker image has no endpoint release at ClaimAck time: exact endpoint identity
// arrives later in the digest-fenced PlacementSet. Silence is therefore the only truthful
// answer from a freshly attached pod and is accepted in both launch lanes. A worker that
// does claim a release is still held to the host's pin; a stale preloaded endpoint must not
// be mistaken for the dynamic placement this owner is about to converge.
func releasePin(w *worker, declared string) *exit.Error {
	pinned := w.spec.Placement.ReleaseID
	if pinned == "" {
		return nil // nothing to pin against — an uninstalled dev spec names no release
	}
	if declared == "" {
		return nil
	}
	if declared != pinned {
		return exit.Named(exit.Conflict, "release_mismatch",
			"this worker serves release %q and this host pinned %q", declared, pinned).
			WithRemedy("the pod installed a different release; its binding plan ids are "+
				"digests of ITS release and not of %q, so nothing this host dispatches "+
				"would resolve there", pinned)
	}
	return nil
}

// refuseClaim records this owner's verdict on the thing at the other end and logs it. The
// verdict is kept on the worker so a WAITER gets the answer: before this, a refused claim
// simply stopped the conversation and the request waited out the silence window to be told
// the worker was "stalled" — a network sentence for an identity fact.
func (c *Orchestrator) refuseClaim(w *worker, e *exit.Error) {
	c.mu.Lock()
	w.refusal = e
	c.mu.Unlock()
	c.logf("REFUSING the claimed worker %s (%s): %s", w.instanceID, e.ErrName(), e.Message)
}

// onSnapshot is THE ONE DIGEST-ACKED BARRIER (§5). The three-message
// SnapshotBegin/Entry/End form could only be bounded by a counted-entries check, which is
// a claim; a single canonical document is bounded by its own digest, and a truncated
// snapshot cannot match one. So the count check is GONE and a stronger thing replaces it.
//
// The order is not negotiable: recompute over the resident bytes, parse under
// unknown-field refusal, reconcile every held attempt DURABLY, and only then ack the exact
// (snapshot_id, snapshot_digest). Dispatch stays CLOSED until the worker sees that ack, so
// a snapshot this owner could not read is one nothing is ever dispatched against.
func (c *Orchestrator) onSnapshot(w *worker, s *session, snap *pb.WorkerSnapshot) bool {
	refuse := func(format string, args ...any) {
		c.logf("WorkerSnapshot %s from %s NOT acknowledged: "+format,
			append([]any{snap.SnapshotId, w.instanceID}, args...)...)
	}
	computed := canonical.Digest(snap.SnapshotCanonicalBytes)
	if !bytes.Equal(computed, snap.SnapshotDigest) {
		refuse("snapshot_digest %x does not hash the %d resident bytes (%x)",
			snap.SnapshotDigest, len(snap.SnapshotCanonicalBytes), computed)
		return false
	}
	doc, err := canonical.Read(snap.SnapshotCanonicalBytes, &pb.WorkerSnapshotBody{})
	if err != nil {
		refuse("the snapshot document is inadmissible (%s)", err)
		return false
	}
	// The accepted set travels beside the body as the EXACT bytes the worker journaled,
	// never a re-serialization, and its digest lives INSIDE the body. Checking one against
	// the other is what makes "the set it is serving" a fact rather than a claim.
	setDigest, _ := canonical.Spell(canonical.Digest(snap.AcceptedPlacementSetCanonicalBytes))
	if declared := doc.Str("accepted_placement_set_digest"); declared != "" && declared != setDigest {
		refuse("the body names accepted set %s and the %d bytes beside it hash to %s",
			shortDigest(declared), len(snap.AcceptedPlacementSetCanonicalBytes),
			shortDigest(setDigest))
		return false
	}

	held := doc.List("held_attempts")
	heldSet := make(map[string]bool, len(held))
	for _, ha := range held {
		requestID, ordinal := ha.Str("request_id"), uint64(ha.Int("attempt_ordinal"))
		heldSet[key(requestID, ordinal)] = true
		if e := c.opt.Store.Recover(requestID, int64(ordinal), s.bootID); e != nil {
			c.logf("held attempt %s#%d REFUSED: %s", requestID, ordinal, e.Message)
			continue
		}
		_, blocked := c.opt.Store.NextOrdinal(requestID)
		c.logf("held attempt %s#%d (%s) is an OPEN OBLIGATION — the ordinal gate now "+
			"refuses: %s", requestID, ordinal,
			trimEnum(pb.AttemptState_name[int32(ha.Int("state"))], "ATTEMPT_STATE_"),
			briefOf(blocked))
	}
	continuations := c.reconcileSnapshotAbsence(w, heldSet)
	// The worker's own admission facts arrive with the snapshot, so the barrier's other
	// side is readable before the first observed state: admission reports CLOSED until the
	// ack lands, which is exactly what "dispatch stays closed" looks like on the wire.
	c.mu.Lock()
	w.acceptedRevision = uint64(doc.Int("accepted_desired_state_revision"))
	w.convergedRevision = uint64(doc.Int("converged_revision"))
	w.admissionGen = uint64(doc.Int("admission_generation"))
	w.admission = pb.AdmissionState(doc.Int("admission_state"))
	w.observeSlots(int(doc.Int("available_attempt_slots")))
	w.phase = pb.WorkerPhase(doc.Int("worker_phase"))
	c.mu.Unlock()

	ackMsg := &pb.SnapshotAck{SnapshotId: snap.SnapshotId, SnapshotDigest: snap.SnapshotDigest}
	ackMsg.RecordOwnerEpoch, ackMsg.ControlStreamGeneration, ackMsg.WorkerBootId =
		recordOwnerEpoch, s.generation, s.bootID
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_SnapshotAck{SnapshotAck: ackMsg}})
	for _, continuation := range continuations {
		c.afterAck(continuation.request, continuation.attempt, w)
	}
	c.retryMediaCleanup(w)
	c.logf("snapshot %s (%s, %d B) acknowledged: %d held attempt(s), accepted revision %d, "+
		"converged %d; dispatch is open", snap.SnapshotId,
		shortDigest(shortNone(snap.SnapshotDigest)), len(snap.SnapshotCanonicalBytes),
		len(held), doc.Int("accepted_desired_state_revision"), doc.Int("converged_revision"))
	if w.spec.IsJob() {
		c.sendJobDirective(s, w)
		return true
	}
	if w.spec.Connection != nil {
		if s.remoteGrantStarted {
			return true
		}
		s.remoteGrantStarted = true
		go c.runRemoteGrantLoop(s, w)
		return true
	}
	if e := c.converge(s, w, []DesiredPlacement{w.spec.Placement}); e != nil {
		c.logf("the desired placement set for %s could not be issued: %s", w.instanceID, e.Message)
	}
	return true
}

// runRemoteGrantLoop owns the standing access lane for one claimed stream. The first
// successful update is queued before DesiredPlacementSet; later updates touch only
// grant_revision, so expiring URLs cannot manufacture a new desired revision.
func (c *Orchestrator) runRemoteGrantLoop(s *session, w *worker) {
	if c.opt.ArtifactGrants == nil {
		c.logf("rental %s has no artifact-grant source; desired placement was not issued",
			w.spec.Connection.RentalID)
		return
	}
	initial := true
	for {
		w.grantMu.Lock()
		c.mu.Lock()
		placement := w.spec.Placement
		c.mu.Unlock()
		grant, problem := c.issueRemoteGrant(s.ctx, s, w, placement, initial)
		w.grantMu.Unlock()
		if problem != nil {
			c.logf("rental %s artifact grant was not refreshed: %s",
				w.spec.Connection.RentalID, problem.Message)
			if !waitContext(s.ctx, ReportCadence) {
				return
			}
			continue
		}
		initial = false
		if !waitContext(s.ctx, grantRefreshDelay(grant.ExpiresAtUnix)) {
			return
		}
	}
}

func (c *Orchestrator) issueRemoteGrant(ctx context.Context, s *session, w *worker,
	placement DesiredPlacement, issueDesired bool) (*pb.ArtifactGrant, *exit.Error) {
	revision, grant, problem := c.opt.ArtifactGrants(ctx, w.spec.Connection)
	if problem != nil {
		return nil, problem
	}
	if problem := grantCovers(placement, grant); problem != nil {
		return nil, problem
	}
	update := &pb.ArtifactGrantUpdate{GrantRevision: revision, Grant: grant}
	update.RecordOwnerEpoch, update.ControlStreamGeneration, update.WorkerBootId =
		recordOwnerEpoch, s.generation, s.bootID
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ArtifactGrantUpdate{
		ArtifactGrantUpdate: update,
	}}) {
		return nil, exit.Unavailablef("worker %s control stream closed before its artifact grant", w.instanceID)
	}
	c.mu.Lock()
	w.grantRevision, w.grantID = revision, grant.GrantId
	w.grantSubjects = grantSubjectFacts(grant.Subjects)
	c.mu.Unlock()
	c.logf("ArtifactGrantUpdate revision=%d grant=%s subjects=%d expires=%d -> %s",
		revision, grant.GrantId, len(grant.Subjects), grant.ExpiresAtUnix, s.bootID)
	if issueDesired {
		if problem := c.converge(s, w, []DesiredPlacement{placement}); problem != nil {
			return nil, problem
		}
	}
	return grant, nil
}

func grantSubjectFacts(subjects []*pb.ArtifactSubject) []ArtifactSubjectFacts {
	out := make([]ArtifactSubjectFacts, 0, len(subjects))
	for _, subject := range subjects {
		digest, err := canonical.Spell(subject.Digest)
		if err != nil {
			continue
		}
		out = append(out, ArtifactSubjectFacts{Digest: digest, SubjectID: subject.SubjectId,
			Kind: subject.Kind, Length: subject.Length})
	}
	return out
}

// ReviseRental changes the desired endpoint on one already-claimed pod. Tensorhub first
// durably authors the exact revision; this owner then sends the matching active grant and
// only after it the set, on the same stream and worker identity.
func (c *Orchestrator) ReviseRental(ctx context.Context, rentalID, endpointRef,
	idempotencyKey, reason string) (WorkerFacts, uint64, *exit.Error) {
	if c.opt.PlacementRevisions == nil || c.opt.ArtifactGrants == nil {
		return WorkerFacts{}, 0, exit.Unavailablef("this LocalService consumes no live rental revisions")
	}
	c.mu.Lock()
	var w *worker
	for _, candidate := range c.workers {
		if candidate.spec.Connection != nil && candidate.spec.Connection.RentalID == rentalID &&
			!candidate.exited && !candidate.stopping {
			w = candidate
			break
		}
	}
	c.mu.Unlock()
	if w == nil {
		return WorkerFacts{}, 0, exit.New(exit.NotFound,
			"rental %s has no claimed worker in this service", rentalID).
			WithRemedy("claim it first with `cozy rent probe %s`", rentalID)
	}
	placement, placementRevision, problem := c.opt.PlacementRevisions(ctx,
		w.spec.Connection, endpointRef, idempotencyKey, reason)
	if problem != nil {
		return WorkerFacts{}, 0, problem
	}
	placement.Endpoint = pinnedEndpoint(placement.Endpoint, rentalID)
	if placement.PlacementID() != w.placementID {
		return WorkerFacts{}, 0, exit.Named(exit.Conflict, "rental.placement_revision_attempt_changed",
			"revision %d names placement %s, live worker owns %s",
			placementRevision, placement.PlacementID(), w.placementID)
	}
	c.mu.Lock()
	if w.spec.Placement.PlacementRevision == placementRevision {
		if w.spec.Placement.ExactPlacementSetDigest != placement.ExactPlacementSetDigest {
			c.mu.Unlock()
			return WorkerFacts{}, 0, exit.Named(exit.Conflict, "rental.placement_revision_conflict",
				"Tensorhub revision %d changed its exact PlacementSet", placementRevision)
		}
		facts := factsOf(w)
		c.mu.Unlock()
		return facts, placementRevision, nil
	}
	if w.spec.Placement.PlacementRevision > placementRevision {
		prior := w.spec.Placement.PlacementRevision
		c.mu.Unlock()
		return WorkerFacts{}, 0, exit.Named(exit.Conflict, "rental.placement_revision_regressed",
			"live rental revision %d cannot move backward to %d", prior, placementRevision)
	}
	c.mu.Unlock()
	planIDs, subjects, problem := remoteBindingSubjects(placement)
	if problem != nil {
		return WorkerFacts{}, 0, problem
	}
	if problem := c.opt.Store.ReviseAttachedWorker(w.instanceID, placement.Endpoint, placement.ReleaseID); problem != nil {
		return WorkerFacts{}, 0, problem
	}
	c.mu.Lock()
	if c.workers[w.instanceID] != w || w.exited || w.stopping {
		c.mu.Unlock()
		return WorkerFacts{}, 0, exit.Unavailablef("worker %s changed while revision %d was authored",
			w.instanceID, placementRevision)
	}
	w.spec.Placement, w.planIDs, w.subjects = placement, planIDs, subjects
	w.acquisition = PlacementAcquisitionFacts{}
	s := c.sessions[w.bootID]
	c.mu.Unlock()
	if s == nil {
		return WorkerFacts{}, 0, exit.Unavailablef("worker %s has no claimed stream for revision %d",
			w.instanceID, placementRevision)
	}
	w.grantMu.Lock()
	_, problem = c.issueRemoteGrant(s.ctx, s, w, placement, true)
	w.grantMu.Unlock()
	if problem != nil {
		return WorkerFacts{}, 0, problem
	}
	c.mu.Lock()
	facts := factsOf(w)
	c.mu.Unlock()
	return facts, placementRevision, nil
}

func grantCovers(placement DesiredPlacement, grant *pb.ArtifactGrant) *exit.Error {
	if grant == nil || grant.GrantId == "" || grant.ExpiresAtUnix <= uint64(time.Now().Unix()) {
		return exit.Named(exit.Conflict, "rental.artifact_grant_invalid",
			"the rental artifact grant is absent, unnamed, or expired")
	}
	type requiredSubject struct {
		kind, id string
		length   uint64
	}
	want := map[string]requiredSubject{}
	for _, binding := range placement.Bindings {
		if binding.RuntimePlan == nil {
			return exit.Named(exit.Conflict, "rental.artifact_grant_closure_mismatch",
				"remote binding %s has no exact plan subject", binding.Entrypoint)
		}
		want[binding.RuntimePlan.Digest] = requiredSubject{
			kind: "plan", id: binding.RuntimePlan.SubjectID, length: binding.RuntimePlan.Length,
		}
	}
	if placement.ModelObjectSetDigest == "" || placement.ModelObjectSetLength == 0 {
		return exit.Named(exit.Conflict, "rental.model_object_set_missing",
			"the remote placement carries no exact model-object-set subject")
	}
	want[placement.ModelObjectSetDigest] = requiredSubject{
		kind: "model_object_set", id: placement.ModelObjectSetDigest,
		length: placement.ModelObjectSetLength,
	}
	for _, subject := range grant.Subjects {
		spelled, err := canonical.Spell(subject.Digest)
		if err != nil {
			continue
		}
		if required, exists := want[spelled]; exists {
			if required.length != subject.Length || required.kind != subject.Kind ||
				required.id != subject.SubjectId {
				return exit.Named(exit.Conflict, "rental.artifact_grant_closure_mismatch",
					"grant subject %s does not equal desired kind=%s id=%s length=%d",
					spelled, required.kind, required.id, required.length)
			}
			delete(want, spelled)
		}
	}
	if len(want) != 0 {
		return exit.Named(exit.Conflict, "rental.artifact_grant_closure_mismatch",
			"the complete rental grant omits %d subject(s) required by desired state", len(want))
	}
	return nil
}

func grantRefreshDelay(expiresAt uint64) time.Duration {
	remaining := time.Until(time.Unix(int64(expiresAt), 0))
	if remaining <= 0 {
		return 0
	}
	lead := remaining / 3
	if lead > 5*time.Minute {
		lead = 5 * time.Minute
	}
	if delay := remaining - lead; delay > 0 {
		return delay
	}
	return 0
}

func waitContext(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

type snapshotContinuation struct {
	request records.Request
	attempt records.Attempt
}

// reconcileSnapshotAbsence closes the two facts a crash can leave ambiguous. A prepared
// or offered assignment absent from the worker's durable snapshot never crossed the
// worker boundary and is aborted. A committed terminal absent from that snapshot has
// already been compacted, which the protocol permits only after its ack, so local closure
// and post-ack cleanup may resume. Nothing accepted-but-nonterminal is inferred absent.
func (c *Orchestrator) reconcileSnapshotAbsence(w *worker, held map[string]bool) []snapshotContinuation {
	attempts, e := c.opt.Store.OpenAttemptsOf(w.instanceID)
	if e != nil {
		c.logf("cannot reconcile attempts of %s: %s", w.instanceID, e.Message)
		return nil
	}
	var continuations []snapshotContinuation
	for _, attempt := range attempts {
		if held[key(attempt.RequestID, uint64(attempt.Attempt))] {
			continue
		}
		switch attempt.State {
		case "preparing", "offered":
			c.settleDispatch(attempt.RequestID, uint64(attempt.Attempt), false)
			if e := c.opt.Store.AbortDispatch(attempt.RequestID, attempt.Attempt,
				attempt.SessionID, "absent from the worker's reconciled snapshot"); e != nil {
				c.logf("snapshot could not abort absent %s#%d: %s",
					attempt.RequestID, attempt.Attempt, e.Message)
				continue
			}
			req, e := c.opt.Store.RequestRow(attempt.RequestID)
			if e != nil || req == nil {
				continue
			}
			if w.media == nil {
				c.rollbackGrant(*req, uint64(attempt.Attempt), w)
			}
			c.enqueue(req.ID)
			c.logf("%s#%d was absent from the worker snapshot; preparation aborted and request requeued",
				attempt.RequestID, attempt.Attempt)
		case "terminal":
			if e := c.opt.Store.Closed(attempt.RequestID, attempt.Attempt); e != nil {
				c.logf("snapshot could not close compacted terminal %s#%d: %s",
					attempt.RequestID, attempt.Attempt, e.Message)
				continue
			}
			req, e := c.opt.Store.RequestRow(attempt.RequestID)
			if e == nil && req != nil {
				continuations = append(continuations, snapshotContinuation{*req, attempt})
			}
		}
	}
	return continuations
}

// openWatch opens the LOSSY progress lane on its own connection, bound to the claimed
// generation. Its death is invisible to control; the redial cycle reopens it.
func (c *Orchestrator) openWatch(addr string, w *worker, s *session) context.CancelFunc {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		conn, err := dialWorker(addr, w.spec.Connection)
		if err != nil {
			return
		}
		defer conn.Close()
		open := &pb.ProgressOpen{}
		open.RecordOwnerEpoch, open.ControlStreamGeneration, open.WorkerBootId =
			recordOwnerEpoch, s.generation, s.bootID
		watch, err := pb.NewWorkerControlClient(conn).WatchProgress(ctx, open)
		if err != nil {
			return
		}
		for {
			p, err := watch.Recv()
			if err != nil {
				return
			}
			c.frames.publish(frameOf(p))
		}
	}()
	return cancel
}
