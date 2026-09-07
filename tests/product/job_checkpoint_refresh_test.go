package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// The same callable descriptor can consume different immutable checkpoints. A warm
// job's old source and admission report must not stand in for the next preparation.
func TestRemoteJobRefreshesCheckpointBeforeUsingNewCredit(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	const pkg = "cozy/h3-package"
	const plan = "sha256:3535353535353535353535353535353535353535353535353535353535353535"
	a, b := "sha256:"+strings.Repeat("a", 64), "sha256:"+strings.Repeat("b", 64)
	entered, release := make(chan struct{}), make(chan struct{})
	directed, admit := make(chan struct{}), make(chan struct{})
	var releaseOnce, admitOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }); admitOnce.Do(func() { close(admit) }) }
	defer unblock()
	var preparedB atomic.Bool
	var acquisitions atomic.Int64
	var mu sync.Mutex
	var bad string
	var oldReady, latestReady *pb.WorkerFrame
	var directives int
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	pod.preparedPlacement = func(raw []byte, name, version string) *pb.Placement {
		doc, err := canonical.Read(raw, &pb.DownloadDelegation{})
		must(t, err)
		models := doc.List("models")
		if len(models) != 1 {
			t.Errorf("prepare has %d model inputs", len(models))
		}
		if len(models) == 1 && models[0].Str("manifest") == b {
			close(entered)
			<-release
			preparedB.Store(true)
		}
		return podPlacement(raw, name, version, "h3-package")
	}
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		directives++
		state := frame.GetObservedState()
		state.ConvergedRevision = 0 // real Runtime job mode has no serving-convergence axis
		state.AdmissionEpoch = uint64(7 + directives)
		latestReady = proto.Clone(frame).(*pb.WorkerFrame)
		if directives == 1 {
			oldReady = proto.Clone(frame).(*pb.WorkerFrame)
		}
		if directives == 2 {
			close(directed)
			// A delayed report still offers A's credits while B's directive is pending.
			if err := send(oldReady); err != nil {
				return err
			}
			<-admit
		}
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil && latestReady != nil {
			return true, send(proto.Clone(latestReady).(*pb.WorkerFrame))
		}
		return false, nil
	}
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		if err != nil {
			return nil, err
		}
		manifest := ""
		for _, input := range spec.List("inputs") {
			if input.Str("input_id") == "model:source" {
				manifest = input.Str("digest")
			}
		}
		if manifest == b && (!preparedB.Load() || offer.AdmissionEpoch != 9) {
			mu.Lock()
			bad = fmt.Sprintf("B offered with prepared=%t epoch=%d", preparedB.Load(), offer.AdmissionEpoch)
			mu.Unlock()
		}
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return nil, err
		}
		body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: identity,
			Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true,
			Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME},
		})
		return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
			WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + offer.RequestId,
			OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "job-checkpoint-refresh", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "one existing rental", nil }
		options.AcquireManagedRental = func(req records.Request) (orchestrator.RentalDecision, string, *exit.Error) {
			acquisitions.Add(1)
			_, problem := options.Store.PinRental(req.ID, podRental)
			return orchestrator.RentalDecision{RentalID: podRental}, "", problem
		}
	})
	submit := func(id, manifest, worker string) string {
		t.Helper()
		result, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: id, Package: pkg, Release: "1.0.7", Entrypoint: "four-lane", PlanID: plan,
			Kind: "job", Org: "paul", Payload: []byte("{}"), Worker: worker, Rental: true, RentalRequired: true,
			Models: []orchestrator.ModelRef{{Package: pkg, Slot: "source", BindingPath: "four-lane.models.source", Model: "proof/model", Release: "1.0.0", Lane: "native", Manifest: manifest, ManifestLength: 164, Bytes: 4096}},
		})
		fatal(t, problem)
		return result
	}
	first := submit("checkpoint-a", a, podRental)
	waitUntil(t, "first checkpoint job closed", func() bool {
		attempts, problem := o.store.Attempts(first)
		fatal(t, problem)
		return len(attempts) == 1 && attempts[0].State == "closed"
	})
	firstAttempt := attemptRow(t, o.store, first, 1)
	// Initial snapshot reconciliation can restate A's preparation. The input
	// cutover must add exactly one preparation after that completed startup.
	pod.mu.Lock()
	preparesAfterA := len(pod.prepares)
	pod.mu.Unlock()
	second := submit("checkpoint-b", b, "")
	waitUntil(t, "B starts preparing instead of borrowing A", func() bool {
		mu.Lock()
		failure := bad
		mu.Unlock()
		if failure != "" {
			t.Fatal(failure)
		}
		select {
		case <-entered:
			return true
		default:
			return false
		}
	})
	attempts, problem := o.store.Attempts(second)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("B was offered before its source preparation completed")
	}
	releaseOnce.Do(func() { close(release) })
	waitUntil(t, "new job directive waits for its own credit", func() bool {
		select {
		case <-directed:
			return true
		default:
			return false
		}
	})
	attempts, problem = o.store.Attempts(second)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("B used A's old admission before accepting its directive")
	}
	admitOnce.Do(func() { close(admit) })
	waitUntil(t, "second checkpoint job closed", func() bool {
		rows, p := o.store.Attempts(second)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	third := submit("checkpoint-b-again", b, podRental)
	waitUntil(t, "same checkpoint reuses preparation", func() bool {
		rows, p := o.store.Attempts(third)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	mu.Lock()
	failure := bad
	mu.Unlock()
	if failure != "" {
		t.Fatal(failure)
	}
	pod.mu.Lock()
	prepares, offers := len(pod.prepares), append([]*pb.AttemptOffer(nil), pod.offers...)
	pod.mu.Unlock()
	if prepares != preparesAfterA+1 || acquisitions.Load() != 1 {
		t.Fatalf("prepares=%d after startup=%d fleet selections=%d", prepares, preparesAfterA, acquisitions.Load())
	}
	if len(offers) != 3 {
		t.Fatalf("offers=%d", len(offers))
	}
	for index, offer := range offers {
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		must(t, err)
		want := b
		if index == 0 {
			want = a
		}
		found := false
		for _, input := range spec.List("inputs") {
			if input.Str("input_id") == "model:source" {
				found = input.Str("digest") == want
			}
		}
		if !found || offer.AttemptOrdinal != 1 {
			t.Fatal("a request borrowed another input or attempt ordinal")
		}
	}
	after := attemptRow(t, o.store, first, 1)
	if !bytes.Equal(after.InvocationCanonical, firstAttempt.InvocationCanonical) || after.TerminalDigest != firstAttempt.TerminalDigest {
		t.Fatal("first checkpoint history changed")
	}
}
