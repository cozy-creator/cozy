package coord

import (
	"context"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// The owner side of th-024's re-landed orientation (#436/#446/#454): the WORKER hosts
// `WorkerControl` and THIS side dials it, claims it, reconciles its snapshot, and only
// then dispatches. One goroutine per worker owns the whole conversation; the `session`
// is that stream's sender half, fenced by the control generation the worker minted.
//
// LAUNCH TIER (#437): owner_epoch is the constant 1 — this service is the one owner of
// every worker it spawns or attaches; the machinery that MINTS competing epochs is the
// hub's Wave-2 lease. The bearer credential (#445/#449/#463) rides `Claim.proof`: for a
// spawned worker it is the per-spawn bootstrap credential this launcher minted; for a
// remote worker it is the provisioned owner token.

const ownerEpoch = 1

// controllerID names this owner on Claim. Stable per service run is enough at launch:
// equal-epoch claims from the SAME controller are reconnects, anything else is refused.
const controllerID = "cozy-local-client"

type session struct {
	bootID     string
	generation uint64
	instanceID string
	out        chan *pb.OwnerFrame
}

func (s *session) send(m *pb.OwnerFrame) {
	defer func() { _ = recover() }() // a closed stream is not an error worth a panic
	s.out <- m
}

// attach owns one worker's control conversation for the life of its process: read the
// published address, dial, claim, reconcile, direct, then pump frames. On a stream drop
// with the process still alive it re-dials and RE-CLAIMS the same boot (the worker mints
// a fresh control generation and resends its snapshot; replay covers the durables).
func (c *Coordinator) attach(w *worker) {
	for {
		c.mu.Lock()
		current, live := c.workers[w.instanceID]
		closing := c.closing
		c.mu.Unlock()
		if closing || !live || current != w || w.exited {
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
		gone := w.exited || c.closing
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
func (c *Coordinator) workerAddr(w *worker) (string, *exit.Error) {
	if w.spec.Remote != nil {
		return w.spec.Remote.Addr, nil
	}
	addrFile := filepath.Join(c.opt.Layout.WorkerDir(w.instanceID), "run", "control.addr")
	for {
		data, err := os.ReadFile(addrFile)
		if err == nil && len(data) > 0 {
			return strings.TrimSpace(string(data)), nil
		}
		c.mu.Lock()
		dead := w.exited
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
// binds; TLS with the PINNED cert for a remote one (#445 — trusting exactly that PEM is
// the fingerprint pin; never mTLS).
func dialWorker(addr string, remote *RemoteSpec) (*grpc.ClientConn, error) {
	if remote != nil && remote.CACert != "" {
		pem, err := os.ReadFile(remote.CACert)
		if err != nil {
			return nil, fmt.Errorf("reading the pinned worker cert: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("%s holds no usable certificate", remote.CACert)
		}
		creds := credentials.NewClientTLSFromCert(pool, "")
		return grpc.NewClient(addr, grpc.WithTransportCredentials(creds))
	}
	target := addr
	if strings.HasPrefix(addr, "unix:") || strings.HasPrefix(addr, "/") {
		target = "unix:" + strings.TrimPrefix(addr, "unix:")
	}
	return grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
}

// converse runs ONE claimed stream to its end: Claim -> ClaimAck -> snapshot ->
// SnapshotAck -> Directive -> frames. Returns when the stream closes.
func (c *Coordinator) converse(w *worker, addr string) error {
	conn, err := dialWorker(addr, w.spec.Remote)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pb.NewWorkerControlClient(conn)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := client.Control(ctx)
	if err != nil {
		return err
	}
	s := &session{instanceID: w.instanceID, out: make(chan *pb.OwnerFrame, 32)}
	go func() {
		for m := range s.out {
			if err := stream.Send(m); err != nil {
				return
			}
		}
	}()
	defer close(s.out)

	proof := w.bootstrap.Reveal()
	if w.spec.Remote != nil {
		proof = w.spec.Remote.Token.Reveal()
	}
	s.send(&pb.OwnerFrame{Msg: &pb.OwnerFrame_Claim{Claim: &pb.Claim{
		OwnerEpoch:   ownerEpoch,
		ControllerId: controllerID,
		WorkerId:     w.spec.WorkerID(),
		WireMinor:    pb.WireMinor,
		Proof:        []byte(proof),
	}}}) //cozy:allow-reveal the one place the credential leaves this process: Claim.proof over the worker's own channel

	// WatchProgress rides a PHYSICALLY separate connection (01) once the claim lands;
	// opened after ClaimAck below.
	var watchCancel context.CancelFunc
	defer func() {
		if watchCancel != nil {
			watchCancel()
		}
	}()

	var snapshot []*pb.ActiveAttempt
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
			s.bootID, s.generation = ack.WorkerBootId, ack.ControlGeneration
			c.onClaimAck(w, s, ack)
			watchCancel = c.openWatch(addr, w, s)
		case *pb.WorkerFrame_BootFailure:
			c.logf("BOOT FAILURE from %s: %s (%s)", m.BootFailure.InstanceId,
				pb.BootFailureReason_name[int32(m.BootFailure.Reason)], m.BootFailure.Detail)
			return nil
		case *pb.WorkerFrame_SnapshotBegin:
			snapshot = nil
		case *pb.WorkerFrame_SnapshotEntry:
			snapshot = append(snapshot, m.SnapshotEntry.Attempt)
		case *pb.WorkerFrame_SnapshotEnd:
			c.onSnapshot(w, s, snapshot, m.SnapshotEnd)
		case *pb.WorkerFrame_Report:
			if c.fenced(s, m.Report.OwnerEpoch, m.Report.ControlGeneration, m.Report.WorkerBootId) {
				continue
			}
			c.onReport(s, m.Report)
		case *pb.WorkerFrame_AttemptAccepted:
			a := m.AttemptAccepted
			if c.fenced(s, a.OwnerEpoch, a.ControlGeneration, a.WorkerBootId) {
				continue
			}
			c.onAccepted(s, a)
		case *pb.WorkerFrame_AttemptTerminal:
			t := m.AttemptTerminal
			if c.fenced(s, t.OwnerEpoch, t.ControlGeneration, t.WorkerBootId) {
				continue
			}
			c.onTerminal(s, t)
		case *pb.WorkerFrame_CheckpointRequest:
			r := m.CheckpointRequest
			if c.fenced(s, r.OwnerEpoch, r.ControlGeneration, r.WorkerBootId) {
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
		}
	}
}

// fenced evaluates the three-field envelope in its fixed order, BEFORE any body field is
// interpreted (02 §0).
func (c *Coordinator) fenced(s *session, epoch, generation uint64, bootID string) bool {
	if epoch != ownerEpoch {
		c.logf("DROPPED: epoch %d is not this owner's %d", epoch, ownerEpoch)
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
func (c *Coordinator) onClaimAck(w *worker, s *session, ack *pb.ClaimAck) {
	c.logf("ClaimAck boot=%s generation=%d instance=%s minor=%d backend=%q device=%q",
		ack.WorkerBootId, ack.ControlGeneration, ack.InstanceId, ack.WireMinor,
		ack.Resources.GetBackend(), ack.Resources.GetDeviceName())
	if ack.InstanceId != "" && ack.InstanceId != w.instanceID {
		// The thing at this address is not the worker this slot spawned. Nothing is
		// dispatched to it; the stream ends on the next recv when we stop talking.
		c.logf("REFUSING the claimed worker: instance %q is not this slot's %q",
			ack.InstanceId, w.instanceID)
		return
	}
	if w.spec.ReleaseID != "" && ack.ReleaseId != "" && ack.ReleaseId != w.spec.ReleaseID {
		c.logf("REFUSING the claimed worker (RELEASE_ID_MISMATCH): release %q is not the "+
			"pinned %q", ack.ReleaseId, w.spec.ReleaseID)
		return
	}
	if e := c.opt.Store.BindSession(w.instanceID, ack.WorkerBootId,
		int64(ack.ControlGeneration)); e != nil {
		c.logf("boot binding for %s REFUSED: %s", w.instanceID, e.Message)
		return
	}
	c.mu.Lock()
	if w.bootID != "" && w.bootID != ack.WorkerBootId {
		delete(c.sessions, w.bootID)
	}
	c.sessions[ack.WorkerBootId] = s
	w.bootID = ack.WorkerBootId
	c.mu.Unlock()
}

// onSnapshot reconciles the worker's recovered attempts DURABLY, acks the exact
// snapshot, and only then issues the directive — dispatch stays closed until the worker
// sees the ack (02 §6).
func (c *Coordinator) onSnapshot(w *worker, s *session, entries []*pb.ActiveAttempt, end *pb.SnapshotEnd) {
	if len(entries) != int(end.EntryCount) {
		c.logf("SNAPSHOT GAP from %s: %d entries, end says %d — not acknowledged",
			w.instanceID, len(entries), end.EntryCount)
		return
	}
	for _, ra := range entries {
		if e := c.opt.Store.Recover(ra.RequestId, int64(ra.Attempt), s.bootID); e != nil {
			c.logf("recovered attempt %s#%d REFUSED: %s", ra.RequestId, ra.Attempt, e.Message)
			continue
		}
		_, blocked := c.opt.Store.NextOrdinal(ra.RequestId)
		c.logf("recovered attempt %s#%d (%s) is an OPEN OBLIGATION — the ordinal gate now "+
			"refuses: %s", ra.RequestId, ra.Attempt,
			pb.AttemptState_name[int32(ra.State)], briefOf(blocked))
	}
	ackMsg := &pb.SnapshotAck{SnapshotId: end.SnapshotId}
	ackMsg.OwnerEpoch, ackMsg.ControlGeneration, ackMsg.WorkerBootId = ownerEpoch, s.generation, s.bootID
	s.send(&pb.OwnerFrame{Msg: &pb.OwnerFrame_SnapshotAck{SnapshotAck: ackMsg}})
	c.logf("snapshot %s acknowledged (%d entr%s); dispatch is open", end.SnapshotId,
		len(entries), map[bool]string{true: "y", false: "ies"}[len(entries) == 1])
	if w.spec.IsJob() {
		c.sendJobDirective(s, w)
	} else {
		c.sendDirective(s, w)
	}
}

// openWatch opens the LOSSY progress lane on its own connection, bound to the claimed
// generation. Its death is invisible to control; the redial cycle reopens it.
func (c *Coordinator) openWatch(addr string, w *worker, s *session) context.CancelFunc {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		conn, err := dialWorker(addr, w.spec.Remote)
		if err != nil {
			return
		}
		defer conn.Close()
		open := &pb.ProgressOpen{}
		open.OwnerEpoch, open.ControlGeneration, open.WorkerBootId = ownerEpoch, s.generation, s.bootID
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
