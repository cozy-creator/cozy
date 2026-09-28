package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (p *fakePod) RetainDerivedResult(ctx context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
	if p.derivedRetain == nil {
		return nil, status.Error(codes.Unimplemented, "derived retain not configured")
	}
	return p.derivedRetain(ctx, call)
}

func (p *fakePod) NativeArtifactTransfer(ctx context.Context, call *pb.NativeArtifactTransferCall) (*pb.NativeArtifactTransferStatus, error) {
	if p.artifactTransfer == nil {
		return nil, status.Error(codes.Unimplemented, "artifact transfer not configured")
	}
	return p.artifactTransfer(ctx, call)
}

func (p *fakePod) ReleaseDerivedRetention(ctx context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
	if p.derivedRelease == nil {
		return nil, status.Error(codes.Unimplemented, "derived release not configured")
	}
	return p.derivedRelease(ctx, call)
}

func (p *fakePod) ReleaseDerivedResult(ctx context.Context, call *pb.DerivedResultReleaseCall) (*pb.DerivedResultReleaseResult, error) {
	if p.resultRelease == nil {
		return nil, status.Error(codes.Unimplemented, "result release not configured")
	}
	return p.resultRelease(ctx, call)
}

// The exact native release is an independent TLS peer. Lose its first reply,
// then prove startup recovery retries custody without changing successful work.
func TestCancelledParentReleasesCompletedChildrenWithoutRewritingHistory(t *testing.T) {
	for _, memo := range []bool{false, true} {
		name := "executed"
		if memo {
			name = "zero_attempt_memo_hit"
		}
		t.Run(name, func(t *testing.T) {
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer func() { store.Close() }()
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "cancel-root", podRental))
			sourceParent := parent
			if memo {
				sourceParent = offerChildParent(t, store, recordPrivateTransaction(t, store, "independent-root", podRental))
			}
			source := offerChildParent(t, store, operationHistory(t, store, "completed-producer", sourceParent, true))
			nativeDigest, _ := canonical.Spell(canonical.Digest([]byte(`{}`)))
			receipt, _, err := canonical.Identity(&pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: source.ID,
				InvocationSpecDigest: childDigest("1"), OutputSlot: "weights", WeightsTransactionId: childDigest("5"),
				TensorfsReceiptDigest: nativeDigest, TensorfsReceiptCanonicalBytes: []byte(`{}`)})
			must(t, err)
			_, err = canonical.Read(receipt, &pb.WeightsReceipt{})
			must(t, err)
			receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
			fatal(t, store.RecordModelTransferWeights(records.ModelTransferWeights{RequestID: source.ID, Attempt: 1,
				OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 161,
				InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt}))
			closeChild(t, store, source, "SUCCEEDED", "succeeded")
			original, problem := store.AttemptRow(source.ID, 1)
			fatal(t, problem)
			child := source
			if memo {
				closeChild(t, store, sourceParent, "SUCCEEDED", "succeeded")
				child = operationHistory(t, store, "cached-recipient", parent, true)
				key, problem := records.QualifiedOperationKey(child, childDigest("f"))
				fatal(t, problem)
				fatal(t, store.BeginOperationLookup(child.ID, key))
				hold := records.WeightsRetention{RequestID: child.ID, Kind: "result", Slot: "weights/weights",
					ProducerRequestID: source.ID, ProducerAttempt: 1, ProducerOutputSlot: "weights",
					RetentionID: childDigest("d"), InstanceID: "private-worker", WorkerBootID: "private-boot", State: "held"}
				fatal(t, store.AdoptCachedOperation(child.ID, records.CachedOperation{Key: key, SourceRequestID: source.ID,
					SourceAttempt: 1, InvocationDigest: original.InvocationDigest, OutcomeID: original.TerminalID,
					OutcomeDigest: original.TerminalDigest, OutcomeBody: original.TerminalBody, Retentions: []records.WeightsRetention{hold}}))
			}
			closeChild(t, store, parent, "FAILED", "blocked")
			ready, problem := store.ReleaseCompletedChildWork(parent.ID, child.ID)
			fatal(t, problem)
			if ready {
				t.Fatal("completed custody released without a cancellation intent")
			}
			fatal(t, store.RequestRetainedCancellation(parent.ID, "test parent cancellation"))
			// The scheduler's earlier child value is unfinished, but the successful
			// execution/cache adoption committed before its cancellation SQL reached it.
			if records.Settled(child.State) {
				t.Fatal("fixture does not carry a stale unfinished child observation")
			}
			changed, problem := store.RequestDescendantCancellation(parent.ID, child.ID, "stale child observation")
			fatal(t, problem)
			current, problem := store.RequestRow(child.ID)
			fatal(t, problem)
			if changed || current.State != "succeeded" {
				t.Fatal("descendant cancellation overwrote a completion that won the race")
			}
			ready, problem = store.ReleaseRetainedWork(parent.ID)
			fatal(t, problem)
			if ready {
				t.Fatal("parent cleanup overtook retained successful descendant")
			}
			// The restart owns only the same recorded parent cancellation and native
			// retentions; no new completed-child cleanup record has been introduced.
			store.Close()
			store, problem = records.Open(layout.DB)
			fatal(t, problem)
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public}
			var releases, childACKs atomic.Int32
			lostReply := make(chan struct{})
			checkRelease := func(claim *pb.Claim) error {
				if err := pod.verifyClaim(claim, false); err != nil {
					return err
				}
				row, problem := store.RequestRow(child.ID)
				if problem != nil || row.State != "succeeded" || !row.RetainWork {
					t.Error("native release ran after success or custody was overwritten")
				}
				if releases.Add(1) == 1 {
					close(lostReply)
					return status.Error(codes.Unavailable, "native release applied but reply lost")
				}
				return nil
			}
			pod.resultRelease = func(_ context.Context, call *pb.DerivedResultReleaseCall) (*pb.DerivedResultReleaseResult, error) {
				if memo {
					t.Error("memo recipient cleanup disposed the independent original producer")
				}
				if err := checkRelease(call.Claim); err != nil {
					return nil, err
				}
				return &pb.DerivedResultReleaseResult{WeightsTransactionId: call.Request.WeightsTransactionId, TensorfsReceiptDigest: call.Request.TensorfsReceiptDigest, Released: true}, nil
			}
			pod.derivedRelease = func(_ context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
				if !memo || call.Request.RetentionId != childDigest("d") {
					t.Error("cleanup released another request's native hold")
				}
				if err := checkRelease(call.Claim); err != nil {
					return nil, err
				}
				return &pb.DerivedRetentionResult{WeightsTransactionId: call.Request.WeightsTransactionId, TensorfsReceiptDigest: call.Request.TensorfsReceiptDigest, RetentionId: call.Request.RetentionId, Released: true}, nil
			}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
				if ack := frame.GetOutcomeAck(); ack != nil && ack.RequestId == child.ID {
					childACKs.Add(1)
					row, problem := store.RequestRow(child.ID)
					if problem != nil || row.RetainWork || row.State != "succeeded" || ack.RetainWork || ack.OutcomeId != original.TerminalID {
						t.Error("final ACK preceded durable release or rewrote original success")
					}
				}
				return false, nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			fatal(t, store.RecordRental(records.Rental{ID: podRental, MachineName: "retention", AcceleratorCount: 1,
				State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100000, Address: connection.Addr, CertPath: connection.CACert,
				ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
			options := orchestrator.Options{Cfg: config.Config{Home: layout.Root}, Layout: layout, Store: store, Log: io.Discard}
			rentalWiring(connection, private)(&options)
			owner, problem := orchestrator.Open(options)
			fatal(t, problem)
			defer owner.Close(time.Second)
			go func() { _ = owner.Serve() }()
			_, _, problem = owner.Reconcile()
			fatal(t, problem)
			select {
			case <-lostReply:
			case <-time.After(10 * time.Second):
				rootNow, _ := store.RequestRow(parent.ID)
				childNow, _ := store.RequestRow(child.ID)
				t.Fatalf("restart did not resume native release; root=%s/%t child=%s/%t: %v", rootNow.State, rootNow.RetainWork, childNow.State, childNow.RetainWork, owner.Events())
			}
			waitUntil(t, "parent cancellation after native retry", func() bool {
				row, problem := store.RequestRow(parent.ID)
				return problem == nil && row.State == "canceled"
			})
			row, problem := store.RequestRow(child.ID)
			fatal(t, problem)
			if row.State != "succeeded" || row.RetainWork || (memo && (row.Ordinal != 0 || row.ReusedFrom != source.ID)) {
				t.Fatalf("custody cleanup rewrote completed child history: %+v", row)
			}
			if releases.Load() < 2 || (memo && childACKs.Load() != 0) || (!memo && childACKs.Load() == 0) {
				t.Fatalf("release/ACK replay changed execution: releases=%d childACKs=%d", releases.Load(), childACKs.Load())
			}
			last, problem := store.AttemptRow(source.ID, 1)
			fatal(t, problem)
			if last.TerminalStatus != "SUCCEEDED" || last.TerminalID != original.TerminalID || last.TerminalDigest != original.TerminalDigest || last.ClosedAt != original.ClosedAt {
				t.Fatal("cleanup changed the original successful execution")
			}
		})
	}
}

func TestCancelledAncestorWaitsForNestedCompletedCustody(t *testing.T) {
	store := successReleaseStore(t)
	root := offerChildParent(t, store, recordPrivateTransaction(t, store, "nested-root", ""))
	middle := offerChildParent(t, store, operationHistory(t, store, "nested-middle", root))
	leaf := operationHistory(t, store, "nested-leaf", middle, true)
	closeChild(t, store, leaf, "SUCCEEDED", "succeeded")
	closeChild(t, store, middle, "SUCCEEDED", "succeeded")
	closeChild(t, store, root, "FAILED", "blocked")
	fatal(t, store.RequestRetainedCancellation(root.ID, "test"))
	ready, problem := store.ReleaseCompletedChildWork(root.ID, middle.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("drained completed child was not released")
	}
	ready, problem = store.ReleaseRetainedWork(root.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("parent completed while its grandchild retained work")
	}
	ready, problem = store.ReleaseCompletedChildWork(middle.ID, leaf.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("success alone authorized a descendant's custody release")
	}
	ready, problem = store.ReleaseCompletedChildWork(root.ID, leaf.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("cancelled ancestor could not release a completed grandchild")
	}
	ready, problem = store.ReleaseRetainedWork(root.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("fully drained cancellation did not advance")
	}
}

func TestCancelledAncestorAuthorizesOnlyItsSuccessfulChildFinalization(t *testing.T) {
	store := successReleaseStore(t)
	root := offerChildParent(t, store, recordPrivateTransaction(t, store, "finalization-root", ""))
	child := operationHistory(t, store, "finalization-child", root, true)
	closeChild(t, store, child, "SUCCEEDED", "succeeded")
	otherRoot := offerChildParent(t, store, recordPrivateTransaction(t, store, "other-root", ""))
	other := operationHistory(t, store, "other-child", otherRoot, true)
	closeChild(t, store, other, "SUCCEEDED", "succeeded")
	closeChild(t, store, root, "FAILED", "blocked")
	intent := func(id string) records.WeightsFinalization {
		return records.WeightsFinalization{RequestID: id, Attempt: 1, InstanceID: "private-worker",
			OwnerScope: "owner", InvocationDigest: childDigest("1"), OutputSlot: "weights"}
	}
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(child.ID)))
	pending, problem := store.PendingWeightsFinalizations(child.ID, 1)
	fatal(t, problem)
	if len(pending) != 0 {
		t.Fatal("native finalization preceded ancestor cancellation authority")
	}
	fatal(t, store.RequestRetainedCancellation(root.ID, "test"))
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(child.ID)))
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(other.ID)))
	pending, problem = store.PendingWeightsFinalizations(child.ID, 1)
	fatal(t, problem)
	if len(pending) != 1 {
		t.Fatal("successful child's native finalization was not recorded")
	}
	foreign, problem := store.PendingWeightsFinalizations(other.ID, 1)
	fatal(t, problem)
	if len(foreign) != 0 {
		t.Fatal("ancestor cancellation escaped its family")
	}
	ready, problem := store.ReleaseCompletedChildWork(root.ID, child.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("completed custody cleared before native finalization confirmation")
	}
}
