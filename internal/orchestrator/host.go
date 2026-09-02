package orchestrator

// THE HOST LANE (proto-025). A rented pod's supervisor is the pod-resident half of the host
// role this daemon performs in-process for a local worker: it downloads, verifies, and obtains
// PlacementSet bytes from the Runtime's loopback preparation. This owner asks it to PREPARE
// over PodHost, off the control stream, and then sends the prepared placement_set bytes
// itself as DesiredWorkerState on WorkerControl -- the identical three steps `converge` runs
// for a local install (prepare -> placement_set bytes -> desired_state). Before minor 21 the
// supervisor did all of this inline on the control stream's read loop, and every offer, ack
// and cancel for the package already serving waited behind package B's download.

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type prepareOpener func(context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error)

// issueThroughHost starts one preparation and returns. The caller has already journaled the
// logical desire on the worker (so a reconnect re-issues it); the prepare runs on its own
// goroutine because its callers include the control stream's own receive loop, and a
// multi-GiB materialization must never sit on that loop on either end of the wire.
//
// hostPrepareSeq numbers the desire. A prepare that finishes after a newer desire was issued
// sends nothing: its bytes describe a set this owner no longer wants, and a full-replace
// desired state carrying them would unload the newer one. The desired revision is minted
// HERE, not at the send, so every wait on `acceptedRevision >= revision` stays closed for the
// whole preparation rather than passing on the previous set's acceptance.
func (c *Orchestrator) issueThroughHost(s *session, w *worker, label string, open prepareOpener) *exit.Error {
	if s.host == nil {
		return exit.Internalf("worker %s has no PodHost lane to prepare %s on", w.instanceID, label)
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.hostPrepareSeq++
	seq := w.hostPrepareSeq
	w.revision, w.desiredRefusal = rev, nil
	c.mu.Unlock()
	c.logf("PodHost prepare %s#%d for revision %d -> %s", label, seq, rev, s.bootID)
	go c.prepareThroughHost(s, w, seq, rev, label, open)
	return nil
}

func (c *Orchestrator) prepareThroughHost(s *session, w *worker, seq, rev uint64, label string, open prepareOpener) {
	stream, err := open(s.ctx)
	if err != nil {
		c.hostPrepareFailed(s, w, seq, label, err)
		return
	}
	var stage pb.PrepareStage
	for {
		event, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				err = status.Error(codes.FailedPrecondition, "the host closed the prepare stream without a terminal event")
			}
			c.hostPrepareFailed(s, w, seq, label, err)
			return
		}
		if event.Stage != stage {
			stage = event.Stage
			c.logf("PodHost prepare %s#%d %s (%d/%d B)", label, seq,
				trimEnum(pb.PrepareStage_name[int32(stage)], "PREPARE_STAGE_"),
				event.TransferredBytes, event.TotalBytes)
		}
		switch event.Stage {
		case pb.PrepareStage_PREPARE_STAGE_REFUSED:
			c.hostPrepareRefused(s, w, seq, label, event.SafeCode+": "+event.SafeDetail)
			return
		case pb.PrepareStage_PREPARE_STAGE_PREPARED:
			prepared := event.PlacementSet
			if prepared == nil || !bytes.Equal(canonical.Digest(prepared.PlacementSetCanonicalBytes),
				prepared.PlacementSetDigest) {
				c.setDesiredRefusal(w, seq, exit.Named(exit.Structural, "worker.prepare_identity_mismatch",
					"the pod host's prepared PlacementSet for %s does not hash to its digest", label))
				return
			}
			if _, err := canonical.Read(prepared.PlacementSetCanonicalBytes, &pb.PlacementSet{}); err != nil {
				c.setDesiredRefusal(w, seq, exit.Named(exit.Structural, "worker.prepare_document_invalid",
					"the pod host's prepared PlacementSet for %s is inadmissible: %s", label, err))
				return
			}
			c.convergePrepared(s, w, seq, rev, label, prepared)
			return
		}
	}
}

// hostPrepareFailed classifies a prepare stream's end. A transport failure is not a verdict:
// the control stream's reconnect re-issues the journaled desire and the host answers from its
// ledger. A typed refusal from the host is the worker's final word on this desire, with the
// one exception hostPrepareRefused names.
func (c *Orchestrator) hostPrepareFailed(s *session, w *worker, seq uint64, label string, err error) {
	switch status.Code(err) {
	case codes.FailedPrecondition, codes.InvalidArgument, codes.PermissionDenied,
		codes.Unauthenticated, codes.Unimplemented:
		c.hostPrepareRefused(s, w, seq, label, status.Convert(err).Message())
	default:
		c.logf("PodHost prepare %s#%d ended without a verdict: %v", label, seq, err)
	}
}

// hostPrepareRefused is refusePendingDesiredState's rule for the host lane (xs-007 row 5):
// a refusal that names THIS OWNER'S lapsed download delegation, when the credential has
// aged out by this owner's clock too, is answered by re-issuing the same logical desire
// under a freshly signed one -- over the bytes already verified on the pod's disk. Every
// other refusal is the worker's final word on this desire. The pod's own progress verdict
// (a refreshed plan that landed no byte) arrives here as a different text and stays
// permanent; a lapse reported against a delegation still live here is clock skew, and
// re-signing would answer it forever, so it stays permanent too.
func (c *Orchestrator) hostPrepareRefused(s *session, w *worker, seq uint64, label, detail string) {
	if len(detail) > 1024 {
		detail = detail[:1024] + "…"
	}
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		c.mu.Unlock()
		return
	}
	expiry, revision := w.delegationExpiry, w.revision
	lapsed := lapsedDownloadDelegation(detail) && !expiry.IsZero() && !time.Now().Before(expiry)
	if !lapsed {
		w.desiredRefusal = exit.Named(exit.Structural, "worker.desired_state_refused",
			"worker rejected desired revision %d before applying it: %s", revision, detail)
	}
	private := cloneLocalPackageSet(w.desiredLocal)
	privatePlacement := clonePrivatePlacementSet(w.desiredPrivatePlacement)
	packages := clonePackageRefs(w.desiredPackages)
	models := cloneModelRefs(w.desiredModels)
	c.mu.Unlock()
	if !lapsed {
		c.logf("worker %s: desired revision %d REFUSED before it was applied: %s",
			w.instanceID, revision, detail)
		return
	}
	c.logf("worker %s: the download delegation this owner signed expired at %s while the "+
		"pod was still resolving; re-issuing the package set under a fresh one",
		w.instanceID, expiry.UTC().Format(time.RFC3339))
	var e *exit.Error
	switch {
	case private != nil:
		e = c.issueLocalPackageSet(s, w, private)
	case privatePlacement != nil:
		e = c.issuePrivatePlacementSet(s, w, privatePlacement)
	default:
		w.desiredMu.Lock()
		e = c.issuePackageSet(s, w, packages, models)
		w.desiredMu.Unlock()
	}
	if e != nil {
		c.logf("worker %s: %s could not be re-issued: %s", w.instanceID, label, e.Message)
	}
}

func (c *Orchestrator) setDesiredRefusal(w *worker, seq uint64, e *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.hostPrepareSeq != seq {
		return
	}
	w.desiredRefusal = e
	c.logf("worker %s: desired revision %d REFUSED before it was applied: %s",
		w.instanceID, w.revision, e.Message)
}

// convergePrepared is step three: the exact bytes this owner has verified become the desired
// state, over the same frame a local install sends.
func (c *Orchestrator) convergePrepared(s *session, w *worker, seq, rev uint64, label string, prepared *pb.DesiredPlacementSet) {
	setBytes := append([]byte(nil), prepared.PlacementSetCanonicalBytes...)
	digest := append([]byte(nil), prepared.PlacementSetDigest...)
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		c.mu.Unlock()
		c.logf("PodHost prepare %s#%d superseded by #%d before its bytes were sent", label, seq, w.hostPrepareSeq)
		return
	}
	w.setDigest, w.setBytes = digest, setBytes
	c.mu.Unlock()
	d := &pb.DesiredWorkerState{
		RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID,
		Revision: rev, WireMinor: pb.WireMinor, Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
			PlacementSetDigest: digest, PlacementSetCanonicalBytes: setBytes}},
	}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}}) {
		c.logf("PodHost prepare %s#%d: control stream closed before the placement_set send", label, seq)
		return
	}
	c.logf("DesiredWorkerState revision=%d placement_set=%s (%d canonical bytes, prepared by the host as %s) -> %s",
		rev, shortDigest(shortNone(digest)), len(setBytes), label, s.bootID)
}

func hostLabel(kind, id string) string { return fmt.Sprintf("%s(%s)", kind, id) }
