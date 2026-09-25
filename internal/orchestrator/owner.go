package orchestrator

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// THE RECORDOWNER SIDE of th-024's orientation (#436/#446/#454, renamed by #481 — bare
// "owner" is retired and the formal role is RecordOwner): the WORKER hosts `WorkerControl`
// and THIS side dials it, claims it, reconciles its ONE snapshot, and only then
// dispatches. One goroutine per worker owns the whole conversation; the `session` is that
// stream's sender half, fenced by the control stream epoch the worker minted.
//
// LAUNCH TIER (#437): record_owner_epoch is the constant 1 — this daemon is the one
// RecordOwner of every worker it spawns or connects to; the machinery that MINTS competing
// epochs is the hub's Wave-2 lease. Spawned workers authenticate Claim with their local
// bootstrap; rented workers authenticate the channel with the provisioned Creator mTLS key.

const recordOwnerEpoch = 1

// recordOwnerID names this owner on Claim. Stable per daemon run is enough at launch:
// equal-epoch claims from the SAME RecordOwner are reconnects, anything else is refused.
const recordOwnerID = "cozy-local-client"

type session struct {
	ctx        context.Context
	bootID     string
	epoch      uint64
	instanceID string
	out        chan *pb.RecordOwnerFrame
	// host is the pod's PodHost lane (proto-025), on the same pinned connection as the
	// control stream; nil for a local worker, whose host is this daemon in-process. claim is
	// the exact Claim this session presented, re-presented on every host call.
	host             pb.PodHostClient
	preparation      pb.RuntimePreparationClient
	claim            *pb.Claim
	outcomes         chan *pb.AttemptOutcome
	completions      chan snapshotContinuation
	completionMu     sync.RWMutex
	completionClosed bool
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
// a fresh control epoch and resends its snapshot; replay covers the durables).
//
// A RECORDED REFUSAL ENDS THE LOOP. The verdicts this owner reaches at claim time — a
// foreign instance identity or an unpinned release — are facts about the THING AT THE
// OTHER END, and redialing cannot change any of them. Left
// running, the loop burns one of the worker's control epochs every 200 ms forever;
// observed at 1,111 epochs against a pre-rev-2 worker while a waiter sat on a
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
			if c.refuseUnimplemented(w, err) || c.refusePendingDesiredState(w, err) {
				return
			}
		}
		c.mu.Lock()
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

// refuseUnimplemented ends the conversation on codes.Unimplemented, which is the ONE
// status a redial can never turn into a success: it does not say the callee failed, it
// says the callee HAS NO SUCH CALL. Nothing about reconnecting installs one.
//
// This is deliberately NOT part of refusePendingDesiredState, and the difference is the
// whole defect. That function answers a PENDING desired revision, so it returns false
// whenever `acceptedRevision >= revision` — and the lane that produced this hazard, a
// model-source frame refused by a pod that deleted the lane (tensorhub th-124), is sent
// long AFTER the pod converged its desired state. Making Unimplemented terminal only
// there would have left the loop running: 200 ms redial, forever, burning one control
// epoch each time on a rented pod at roughly a dollar an hour. A sibling session measured
// the shape at 2,434 revisions in 100 minutes from a different cause.
//
// The verdict is recorded as this owner's refusal of the worker (the same field a foreign
// instance identity or an unpinned release lands in) rather than as a desired-state
// refusal, because it is a fact about the thing at the other end and not about one
// revision: a waiter must be answered whether or not a revision was in flight.
//
// It stops at Unimplemented on purpose. Unauthenticated and PermissionDenied look similar
// but a reconnect legitimately re-presents a fresh Claim against a newer epoch, and
// InvalidArgument depends on which frame was in flight; only "the callee does not
// implement this at all" is unconditionally beyond retry.
func (c *Orchestrator) refuseUnimplemented(w *worker, err error) bool {
	if status.Code(err) != codes.Unimplemented {
		return false
	}
	c.refuseClaim(w, exit.Named(exit.Structural, "worker.call_unimplemented",
		"this worker does not implement a call this owner made: %s", refusalDetail(err)).
		WithRemedy("the two sides disagree about the wire — the worker image is older or "+
			"newer than this daemon; redialing cannot install the missing call"))
	return true
}

// refusalDetail bounds what a worker's status message is quoted as.
func refusalDetail(err error) string {
	detail := status.Convert(err).Message()
	if len(detail) > 1024 {
		detail = detail[:1024] + "…"
	}
	return detail
}

// FailedPrecondition is the worker host's final answer when it could not apply the
// desired revision. Redialing cannot make that exact revision acceptable, and hiding the
// status behind reconnects leaves the request waiting for a state the worker rejected.
//
// It used to carry ONE exception (xs-007 row 5): the signed download delegation this
// owner minted could age out while the pod was still downloading against it -- a 100 GB
// materialization at 50 MB/s runs 33 minutes -- so every large rental download failed as
// a credential error, and the answer was to re-issue under a freshly signed one. The
// credential is deleted (owner ruling 2026-09-03): a download can no longer become
// unauthorized by running long, so there is nothing left to except and every
// FailedPrecondition is the worker's final word. Lack of PROGRESS remains the pod's own
// verdict and arrives here as an ordinary permanent refusal, as it always did.
func (c *Orchestrator) refusePendingDesiredState(w *worker, err error) bool {
	if status.Code(err) != codes.FailedPrecondition {
		return false
	}
	c.mu.Lock()
	if w.revision == 0 || w.acceptedRevision >= w.revision {
		c.mu.Unlock()
		return false
	}
	detail := refusalDetail(err)
	w.desiredRefusal = exit.Named(exit.Structural, "worker.desired_state_refused",
		"%s (the worker refused desired revision %d before applying it)", detail, w.revision)
	revision := w.revision
	c.mu.Unlock()
	c.logf("worker %s: desired revision %d REFUSED before it was applied: %s",
		w.instanceID, revision, detail)
	return true
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

// dialWorker opens the channel: local sockets use the per-spawn proof; rented workers
// pin the exact readiness certificate and authenticate Claim with Creator's signature.
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
	if problem := probeWorkerProtocol(ctx, conn, w.spec.Connection != nil); problem != nil {
		if problem.Code != exit.Unavailable && problem.Code != exit.Deadline {
			c.refuseClaim(w, problem)
		}
		return fmt.Errorf("%s", problem.Message)
	}
	stream, err := client.Control(ctx)
	if err != nil {
		return err
	}
	s := &session{ctx: ctx, instanceID: w.instanceID, out: make(chan *pb.RecordOwnerFrame, 32)}
	s.preparation = pb.NewRuntimePreparationClient(conn)
	go func() {
		for m := range s.out {
			if err := stream.Send(m); err != nil {
				return
			}
		}
	}()
	defer close(s.out)

	var proof []byte
	claimWorkerID, claimBootID := "", ""
	if w.spec.Connection != nil {
		if c.opt.RentalClaimProof == nil {
			return fmt.Errorf("the rental has no ClaimProof signer")
		}
		signed, problem := c.opt.RentalClaimProof(w.spec.Connection, recordOwnerEpoch)
		if problem != nil {
			c.refuseClaim(w, problem)
			return fmt.Errorf("%s", problem.Message)
		}
		proof = signed
		claimWorkerID, claimBootID = w.spec.Connection.WorkerID, w.spec.Connection.WorkerBootID
		s.host = pb.NewPodHostClient(conn)
	} else {
		proof = []byte(w.bootstrap.Reveal())
	}
	s.claim = &pb.Claim{
		RecordOwnerEpoch: recordOwnerEpoch,
		RecordOwnerId:    recordOwnerID,
		WorkerId:         claimWorkerID,
		WorkerBootId:     claimBootID,
		WireMinor:        pb.WireMinor,
		Proof:            proof,
	}
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{
		Claim: proto.Clone(s.claim).(*pb.Claim),
	}}) //cozy:allow-reveal local bootstrap or signed rental ClaimProof crosses only on Claim

	// WatchProgress rides a PHYSICALLY separate connection (01), opened at the snapshot
	// barrier below -- not at ClaimAck, which does not by itself complete the control
	// transition the lossy lane must not race.
	var watchCancel context.CancelFunc
	defer func() {
		if watchCancel != nil {
			watchCancel()
		}
	}()
	var settleDone func()
	defer func() {
		cancel()
		if settleDone != nil {
			settleDone()
		}
	}()

	for {
		frame, err := stream.Recv()
		if err != nil {
			c.dropSession(s)
			return err
		}
		switch m := frame.Msg.(type) {
		case *pb.WorkerFrame_ClaimAck:
			ack := m.ClaimAck
			if settleDone != nil {
				return fmt.Errorf("worker repeated its claim acknowledgment")
			}
			if !ack.Accepted {
				name := pb.ClaimRejection_name[int32(ack.Rejection)]
				problem := exit.Named(exit.Conflict, "rental.worker_claim_refused",
					"worker %s refused this owner's claim: %s", w.instanceID, name).
					WithRemedy("release the rental; a worker that rejects its renter's owner credential cannot converge")
				c.logf("Claim REFUSED by %s: %s", w.instanceID, name)
				c.refuseClaim(w, problem)
				return fmt.Errorf("%s", problem.Message)
			}
			s.bootID, s.epoch = ack.WorkerBootId, ack.ControlStreamEpoch
			// Set up owned work before onClaimAck publishes this session to callers.
			settleDone = c.startOutcomeSettlement(s, w, int(ack.GetResources().GetDeviceCount()))
			if e := c.onClaimAck(w, s, ack); e != nil {
				return fmt.Errorf("%s", e.Message)
			}
		case *pb.WorkerFrame_NativeSourceStatus:
			c.onNativeSourceStatus(s, m.NativeSourceStatus)
		case *pb.WorkerFrame_NativeArtifactTransferStatus:
			c.onNativeArtifactTransfer(s, m.NativeArtifactTransferStatus)
		case *pb.WorkerFrame_ChildCallRequest:
			c.onChildCall(s, m.ChildCallRequest)
		case *pb.WorkerFrame_ChildCallCancel:
			c.onChildCancel(s, m.ChildCallCancel)
		case *pb.WorkerFrame_BootFailure:
			c.logf("BOOT FAILURE from %s: %s (%s)", m.BootFailure.WorkerInstanceId,
				pb.BootFailureReason_name[int32(m.BootFailure.Reason)], m.BootFailure.Detail)
			return nil
		case *pb.WorkerFrame_Snapshot:
			if c.fenced(s, m.Snapshot.RecordOwnerEpoch, m.Snapshot.ControlStreamEpoch,
				m.Snapshot.WorkerBootId) {
				continue
			}
			if c.onSnapshot(w, s, m.Snapshot) {
				// Open the lossy progress lane only after the durable recovery barrier is
				// acknowledged. ClaimAck and WorkerSnapshot are one control transition;
				// a second connection before that pair completes adds no useful progress
				// visibility and must not perturb bootstrap on provider TCP relays.
				if watchCancel == nil {
					watchCancel = c.openWatch(addr, w, s)
				}
			}
		case *pb.WorkerFrame_ObservedState:
			r := m.ObservedState
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamEpoch, r.WorkerBootId) {
				continue
			}
			c.onObserved(s, r)
		case *pb.WorkerFrame_AttemptAccepted:
			a := m.AttemptAccepted
			if c.fenced(s, a.RecordOwnerEpoch, a.ControlStreamEpoch, a.WorkerBootId) {
				continue
			}
			c.onAccepted(s, a)
		case *pb.WorkerFrame_AttemptOutcome:
			t := m.AttemptOutcome
			if s.outcomes == nil {
				return fmt.Errorf("worker sent an outcome before its claim acknowledgment")
			}
			if c.fenced(s, t.RecordOwnerEpoch, t.ControlStreamEpoch, t.WorkerBootId) {
				continue
			}
			select {
			case s.outcomes <- t:
			case <-ctx.Done():
				return ctx.Err()
			}
		case *pb.WorkerFrame_CheckpointRequest:
			r := m.CheckpointRequest
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamEpoch, r.WorkerBootId) {
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
		case *pb.WorkerFrame_ModelSourceFileStatus:
			status := m.ModelSourceFileStatus
			if c.fenced(s, status.RecordOwnerEpoch, status.ControlStreamEpoch,
				status.WorkerBootId) {
				continue
			}
			c.onModelSourceFileStatus(s, status)
		case *pb.WorkerFrame_ModelSourcePrepared:
			prepared := m.ModelSourcePrepared
			if c.fenced(s, prepared.RecordOwnerEpoch, prepared.ControlStreamEpoch,
				prepared.WorkerBootId) {
				continue
			}
			c.onModelSourcePrepared(s, prepared)
		case *pb.WorkerFrame_LocalPackageFileStatus:
			status := m.LocalPackageFileStatus
			if c.fenced(s, status.RecordOwnerEpoch, status.ControlStreamEpoch,
				status.WorkerBootId) {
				continue
			}
			c.onLocalPackageFileStatus(s, status)
		case *pb.WorkerFrame_LocalPackageAbortStatus:
			status := m.LocalPackageAbortStatus
			if c.fenced(s, status.RecordOwnerEpoch, status.ControlStreamEpoch,
				status.WorkerBootId) {
				continue
			}
			c.onLocalPackageAbortStatus(s, status)
		case *pb.WorkerFrame_WeightsTransaction:
			c.onWeightsTransaction(s, m.WeightsTransaction)
		case *pb.WorkerFrame_WeightsCheckpoint:
			c.onWeightsCheckpoint(s, m.WeightsCheckpoint)
		case *pb.WorkerFrame_WeightsReceipt:
			receipt := m.WeightsReceipt
			if c.fenced(s, receipt.RecordOwnerEpoch, receipt.ControlStreamEpoch,
				receipt.WorkerBootId) {
				continue
			}
			c.onModelTransferWeightsReceipt(s, receipt)
		case *pb.WorkerFrame_WeightsTransferStatus:
			status := m.WeightsTransferStatus
			if c.fenced(s, status.RecordOwnerEpoch, status.ControlStreamEpoch,
				status.WorkerBootId) {
				continue
			}
			c.onModelTransferWeightsStatus(s, status)
		case *pb.WorkerFrame_CheckpointAck:
			// the worker's echo of a receipt already durable here; nothing to apply
		case *pb.WorkerFrame_WeightsFinalizeResult:
			r := m.WeightsFinalizeResult
			if c.fenced(s, r.RecordOwnerEpoch, r.ControlStreamEpoch, r.WorkerBootId) {
				continue
			}
			c.onWeightsFinalizeResult(s, r)
		}
	}
}

// fenced evaluates the three-field envelope in its fixed order, BEFORE any body field is
// interpreted (02 §0).
func (c *Orchestrator) fenced(s *session, owner, controlStream uint64, bootID string) bool {
	if owner != recordOwnerEpoch {
		c.logf("DROPPED: epoch %d is not this owner's %d", owner, recordOwnerEpoch)
		return true
	}
	if controlStream != s.epoch {
		c.logf("DROPPED: superseded control epoch %d (live %d)", controlStream, s.epoch)
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
	c.logf("ClaimAck boot=%s epoch=%d instance=%s minor=%d backend=%q device=%q",
		ack.WorkerBootId, ack.ControlStreamEpoch, ack.WorkerInstanceId, ack.WireMinor,
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
		if c.opt.ObserveRental == nil || w.spec.Connection.RentalID == "" {
			e := exit.Named(exit.Structural, "rental.worker_readback_unowned",
				"the attached worker has no durable rental observation owner")
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
		if ack.WorkerId != w.spec.Connection.WorkerID ||
			ack.WorkerBootId != w.spec.Connection.WorkerBootID ||
			w.remoteWorkerID != "" && w.remoteWorkerID != ack.WorkerId {
			e := exit.Named(exit.Conflict, "rental.worker_id_mismatch",
				"rental %s declared an absent or changed worker id", w.spec.Connection.RentalID)
			c.refuseClaim(w, e)
			return e
		}
		if e := c.opt.ObserveRental(RentalObservation{
			RentalID: w.spec.Connection.RentalID, Accelerator: resources.GetDeviceName(),
			DeviceCount: int(resources.GetDeviceCount()), Backend: resources.GetBackend(),
			DriverVersion: resources.GetDriverVersion(), BackendVersion: resources.GetBackendVersion(),
			DeviceMemoryTotalBytes: resources.GetDeviceMemoryTotalBytes(),
			WorkerInstance:         ack.WorkerInstanceId, WorkerID: ack.WorkerId,
			WorkerBootID: ack.WorkerBootId,
		}); e != nil {
			c.refuseClaim(w, e)
			return e
		}
	}
	if e := c.opt.Store.BindSession(w.instanceID, ack.WorkerBootId); e != nil {
		c.logf("boot binding for %s REFUSED: %s", w.instanceID, e.Message)
		c.refuseClaim(w, e)
		return e
	}
	c.mu.Lock()
	w.declaredInstance = ack.WorkerInstanceId
	w.wireMinor = ack.WireMinor
	if w.refusal != nil && w.refusal.ErrName() == "worker.protocol_incompatible" {
		w.refusal = nil
	}
	if w.spec.Connection != nil {
		w.remoteWorkerID = ack.WorkerId
	}
	// The ClaimAck is the first observed worker identity on this stream.
	w.lastReport = time.Now()
	w.snapshotAcknowledged = false
	if w.bootID != "" && w.bootID != ack.WorkerBootId {
		delete(c.sessions, w.bootID)
	}
	c.sessions[ack.WorkerBootId] = s
	w.bootID = ack.WorkerBootId
	c.mu.Unlock()
	return nil
}

// instancePin is the IDENTITY FENCE on a claim, and #505's carried-not-verified gap in its
// second place.
//
// Creator names a stable worker slot, not the Runtime process incarnation. Runtime mints
// the latter just as it mints its boot id. The claim must name an incarnation, and every
// reconnect to this slot must keep it stable.
func instancePin(w *worker, declared string) *exit.Error {
	if declared == "" {
		return exit.Named(exit.Conflict, "worker_instance_undeclared",
			"this worker's ClaimAck declares no worker instance, so nothing it settles could be "+
				"attributed to a worker at all").
			WithRemedy("a worker mints an instance identity; one that " +
				"will not say which worker it is cannot be dispatched to")
	}
	if w.declaredInstance != "" && w.declaredInstance != declared {
		return exit.Named(exit.Conflict, "worker_instance_changed",
			"this worker answered as instance %q and now answers as %q: a different process holds the slot",
			w.declaredInstance, declared)
	}
	return nil
}

// releasePin is the RELEASE FENCE on a claim, and #505's carried-not-verified gap closed.
//
// A base worker image has no package release at ClaimAck time: exact package identity
// arrives later in the digest-fenced PlacementSet. Silence is therefore the only truthful
// answer from a freshly attached pod and is accepted in both launch lanes. A worker that
// does claim a release is still held to the host's pin; a stale preloaded package must not
// be mistaken for the dynamic placement this owner is about to converge.
func releasePin(w *worker, declared string) *exit.Error {
	pinned := w.spec.Placement.Release
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
	reconcileHeld := func(rows []canonical.Doc, holder string) {
		for _, ha := range rows {
			requestID, ordinal := ha.Str("request_id"), uint64(ha.Int("attempt_ordinal"))
			heldSet[key(requestID, ordinal)] = true
			if e := c.opt.Store.Recover(requestID, int64(ordinal), s.bootID); e != nil {
				c.logf("%s attempt %s#%d REFUSED: %s", holder, requestID, ordinal, e.Message)
				continue
			}
			_, blocked := c.opt.Store.NextOrdinal(requestID)
			c.logf("%s attempt %s#%d (%s) is an OPEN OBLIGATION — the ordinal gate now "+
				"refuses: %s", holder, requestID, ordinal,
				trimEnum(pb.AttemptState_name[int32(ha.Int("state"))], "ATTEMPT_STATE_"),
				briefOf(blocked))
		}
	}
	reconcileHeld(held, "held")
	// THE HOST'S OWN DOCUMENT (proto-025). A pod host between this owner and the worker
	// holds outcomes the live child may not name (a replaced child's survivors) and announces
	// the receipts it replays after this ack. It is fenced exactly like the worker's document
	// and reconciled before the same ack; the worker's bytes above are the worker's own.
	var hostHeld []canonical.Doc
	var hostWeights []*pb.WeightsTransactionStatus
	var hostRetainedRevision uint64
	if len(snap.HostSnapshotCanonicalBytes) > 0 || len(snap.HostSnapshotDigest) > 0 {
		computed := canonical.Digest(snap.HostSnapshotCanonicalBytes)
		if !bytes.Equal(computed, snap.HostSnapshotDigest) {
			refuse("host_snapshot_digest %x does not hash the %d resident host bytes (%x)",
				snap.HostSnapshotDigest, len(snap.HostSnapshotCanonicalBytes), computed)
			return false
		}
		hostDoc, err := canonical.Read(snap.HostSnapshotCanonicalBytes, &pb.HostSnapshotBody{})
		if err != nil {
			refuse("the host snapshot document is inadmissible (%s)", err)
			return false
		}
		hostHeld = hostDoc.List("held_outcomes")
		typed := new(pb.HostSnapshotBody)
		if err := canonical.Unmarshal(snap.HostSnapshotCanonicalBytes, typed); err != nil {
			refuse("host snapshot typed conversion failed: %s", err)
			return false
		}
		hostWeights = typed.WeightsTransactions
		hostRetainedRevision = typed.RetainedDesiredRevision
		reconcileHeld(hostHeld, "host-held")
	}
	continuations := c.reconcileSnapshotAbsence(w, heldSet)
	// The worker's own admission facts arrive with the snapshot, so the barrier's other
	// side is readable before the first observed state: admission reports CLOSED until the
	// ack lands, which is exactly what "dispatch stays closed" looks like on the wire.
	c.mu.Lock()
	restateRevision := w.revision
	w.acceptedRevision = uint64(doc.Int("accepted_desired_state_revision"))
	// The surviving Host may have received a revision the replacement Runtime
	// has never accepted. Seed only the owner's existing outgoing sequence;
	// acceptance and convergence remain the Runtime's independent observations.
	c.revision = max(c.revision, w.acceptedRevision, hostRetainedRevision)
	w.convergedRevision = uint64(doc.Int("converged_revision"))
	w.admissionEpoch = uint64(doc.Int("admission_epoch"))
	w.admission = pb.AdmissionState(doc.Int("admission_state"))
	w.observeSlots(int(doc.Int("available_attempt_slots")))
	// The per-lane seats a reconnecting owner reconciles from (proto-024) ride the same
	// digest-fenced document, beside the worker-level sum.
	laneBreaches := w.lanes.observe(laneReportsOfDoc(doc.List("lanes")), w.spec.Devices)
	for _, p := range doc.List("placements") {
		w.lanes.route(p.Str("placement_id"), p.Str("device_lane_id"))
	}
	heldPlacements := make([]string, 0, len(held))
	for _, ha := range held {
		heldPlacements = append(heldPlacements, ha.Str("placement_id"))
	}
	w.observeHeld(heldPlacements)
	w.heldManifests = setOf(doc.Strs("held_manifests"))
	w.phase = pb.WorkerPhase(doc.Int("worker_phase"))
	c.mu.Unlock()
	for _, breach := range laneBreaches {
		c.logf("worker breach on %s: %s", w.instanceID, breach)
	}

	ackMsg := &pb.SnapshotAck{SnapshotId: snap.SnapshotId, SnapshotDigest: snap.SnapshotDigest,
		HostSnapshotDigest: append([]byte(nil), snap.HostSnapshotDigest...)}
	ackMsg.RecordOwnerEpoch, ackMsg.ControlStreamEpoch, ackMsg.WorkerBootId =
		recordOwnerEpoch, s.epoch, s.bootID
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_SnapshotAck{SnapshotAck: ackMsg}}) {
		return false
	}
	c.mu.Lock()
	w.snapshotAcknowledged = true
	c.mu.Unlock()
	for _, continuation := range continuations {
		c.afterAck(continuation.request, continuation.attempt, w)
	}
	c.retryMediaCleanup(w)
	for _, row := range hostWeights {
		c.onWeightsTransaction(s, row)
	}
	_ = c.restoreRetainedWork()
	c.logf("snapshot %s (%s, %d B) acknowledged: %d held attempt(s), %d host-held outcome(s), "+
		"accepted revision %d, converged %d; dispatch is open", snap.SnapshotId,
		shortDigest(shortNone(snap.SnapshotDigest)), len(snap.SnapshotCanonicalBytes),
		len(held), len(hostHeld), doc.Int("accepted_desired_state_revision"),
		doc.Int("converged_revision"))
	c.mu.Lock()
	// Native absence invalidates the existing prepared plan. Reissue its saved
	// inputs before JOB mode so the queued retry can become dispatchable.
	replayJob := w.spec.IsJob() && (w.spec.Connection == nil || staged(w, w.spec.Placement.Jobs[0].DescriptorID))
	c.mu.Unlock()
	if replayJob {
		c.signalAllTransfers()
		_ = c.sendJobDirective(s, w, nil)
		return true
	}
	if w.spec.Connection != nil {
		c.signalAllTransfers()
		c.replayLocalAborts(s, w.spec.Connection.RentalID)
		c.mu.Lock()
		private := cloneLocalPackageSet(w.desiredLocal)
		privatePlacement := cloneUnpublishedPlacementSet(w.desiredUnpublishedPlacement)
		packages := clonePackageRefs(w.desiredPackages)
		models := cloneModelRefs(w.desiredModels)
		c.mu.Unlock()
		if private != nil {
			if e := c.issueLocalPackageSet(s, w, private); e != nil {
				c.logf("rental %s local_package_set could not be issued: %s",
					w.spec.Connection.RentalID, e.Message)
			}
			return true
		}
		if privatePlacement != nil {
			if e := c.issueUnpublishedPlacementSet(s, w, privatePlacement); e != nil {
				c.logf("rental %s private_placement_set could not be issued: %s",
					w.spec.Connection.RentalID, e.Message)
			}
			return true
		}
		if len(packages) == 0 && len(models) == 0 {
			return true // generic capacity stays empty until Creator actually selects work
		}
		w.desiredMu.Lock()
		defer w.desiredMu.Unlock()
		c.mu.Lock()
		changed := w.revision != restateRevision
		c.mu.Unlock()
		if changed {
			return true // a concurrent caller already issued a newer selection
		}
		if e := c.issuePackageSet(s, w, packages, models, issueRestated); e != nil {
			c.logf("rental %s package_set could not be issued: %s",
				w.spec.Connection.RentalID, e.Message)
		}
		return true
	}
	placements := []DesiredPlacement(nil)
	if w.spec.Preparation != nil && w.spec.Placement.PlacementSetDigest == "" {
		return true // the owner activates only the worker's completed preparation
	}
	if w.spec.Placement.Package == "" && w.spec.Placement.PlacementSetDigest == "" {
		return true // empty local Runtime serves workspace RPCs without an executor
	}
	if w.spec.Placement.PlacementSetDigest != "" {
		placements = []DesiredPlacement{w.spec.Placement}
	}
	if e := c.converge(s, w, placements); e != nil {
		c.logf("the desired placement set for %s could not be issued: %s", w.instanceID, e.Message)
	}
	return true
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

// openWatch owns the independent lossy connection for this control session. A
// dropped watch must reconnect even while the healthy control lane keeps running.
func (c *Orchestrator) openWatch(addr string, w *worker, s *session) context.CancelFunc {
	ctx, cancel := context.WithCancel(s.ctx)
	go func() {
		backoff := 250 * time.Millisecond
		for ctx.Err() == nil {
			received, err := c.watchProgress(ctx, addr, w, s)
			if ctx.Err() != nil {
				return
			}
			if received {
				backoff = 250 * time.Millisecond
			}
			// gRPC details may contain transport addresses; log only the safe status class.
			c.logf("progress stream for %s disconnected (%s); reconnecting in %s", w.instanceID, status.Code(err), backoff)
			timer := time.NewTimer(backoff)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
			backoff = min(backoff*2, 5*time.Second)
		}
	}()
	return cancel
}

func (c *Orchestrator) watchProgress(ctx context.Context, addr string, w *worker, s *session) (bool, error) {
	conn, err := dialWorker(addr, w.spec.Connection)
	if err != nil {
		return false, err
	}
	defer conn.Close()
	open := &pb.ProgressOpen{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID}
	watch, err := pb.NewWorkerControlClient(conn).WatchProgress(ctx, open)
	if err != nil {
		return false, err
	}
	received := false
	for {
		p, err := watch.Recv()
		if err != nil {
			return received, err
		}
		if p.RecordOwnerEpoch != recordOwnerEpoch || p.ControlStreamEpoch != s.epoch || p.WorkerBootId != s.bootID {
			continue
		}
		received = true
		c.frames.publish(frameOf(p))
	}
}
