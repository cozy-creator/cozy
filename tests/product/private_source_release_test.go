package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (p *fakePod) ModelSourceRelease(ctx context.Context, call *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.sourceRelease == nil {
		return nil, status.Error(codes.Unimplemented, "source release not provided")
	}
	return p.sourceRelease(ctx, call)
}

func TestUnpublishedSourceCancellationWaitsForOriginalHostRelease(t *testing.T) {
	for _, cancelDrain := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel_drain=%t", cancelDrain), func(t *testing.T) { privateSourceCancellationDrain(t, cancelDrain) })
	}
}

func privateSourceCancellationDrain(t *testing.T, cancelDrain bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	entered, finish := make(chan struct{}), make(chan struct{})
	peerCanceled := make(chan struct{})
	var peerCanceledOnce sync.Once
	var finishOnce sync.Once
	release := func() { finishOnce.Do(func() { close(finish) }) }
	var calls atomic.Int64
	pod.sourceRelease = func(ctx context.Context, call *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error) {
		if call.OperationId != "req-source-release" || !bytes.Equal(call.SourceSelectionDigest, bytes.Repeat([]byte{0x22}, 32)) {
			t.Error("release changed the captured operation or source selection")
		}
		if calls.Add(1) == 1 {
			close(entered)
		}
		select {
		case <-finish:
		case <-ctx.Done():
			peerCanceledOnce.Do(func() { close(peerCanceled) })
			return nil, status.FromContextError(ctx.Err()).Err()
		}
		return &pb.ReleaseModelSourceResult{OperationId: call.OperationId, Released: true}, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "private-source-release", rentalWiring(connection, private))
	// Release blocked RPCs before hostOwner's cleanup waits for sessions to end.
	t.Cleanup(release)
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	request, _, problem := o.store.Submit(records.Request{
		ID: "req-source-release", IdemKey: "source-release", BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package: "local/source", Entrypoint: "prepare", Kind: "job", RetainWork: true,
		Worker: podRental, Rental: true, RentalRequired: true,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		ModelTransfer: &records.ModelTransferIntent{
			Kind: "model-upload", Source: "hf://proof/source@" + strings.Repeat("4", 40),
			SourceSelection: "sha256:" + strings.Repeat("2", 64),
			SourceProfiles:  map[string]string{"model": "hf/minimax-h3/shared-bf16/1"},
			SourceFiles:     []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json", SHA256: strings.Repeat("1", 64), Length: 2, Header: []byte("{}")}},
			Outputs:         []records.ModelTransferOutput{{Name: "model"}},
		},
	})
	fatal(t, problem)
	_, problem = o.store.BlockRetainedWork(request.ID, "author_exception", "step B failed")
	fatal(t, problem)
	_, _, _, problem = o.c.EnsureRental(podRental)
	fatal(t, problem)
	fatal(t, o.c.CancelRetainedRequest(request.ID, "abandon"))
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation never reached the original signed Host")
	}
	row, problem := o.store.RequestRow(request.ID)
	fatal(t, problem)
	retained, problem := o.store.RentalRetainsWork(podRental)
	fatal(t, problem)
	if row.State != "canceling" || !retained {
		t.Fatalf("source cleanup acknowledged while native work was live: state=%s retained=%t", row.State, retained)
	}
	// A repeat arriving while another cleanup owns the registry must remain
	// harmless; a delayed continuation cannot duplicate the confirmed native effect.
	fatal(t, o.c.CancelRetainedRequest(request.ID, "repeat while native release is active"))

	drain, cancel := context.WithCancel(t.Context())
	defer cancel()
	drained := make(chan struct{})
	go func() { o.c.WaitRetainedCancellations(drain); close(drained) }()
	// This deliberately outlasts down's 100ms poll. No second cancellation was
	// requested, but the accepted native disposal is still in flight.
	select {
	case <-drained:
		t.Fatal("shutdown called an active cleanup pass a fixed point")
	case <-time.After(150 * time.Millisecond):
	}
	if cancelDrain {
		cancel()
		select {
		case <-drained:
		case <-time.After(5 * time.Second):
			t.Fatal("canceled shutdown drain did not return")
		}
		select {
		case <-peerCanceled:
		case <-time.After(5 * time.Second):
			t.Fatal("shutdown context did not reach native disposal RPC")
		}
		row, problem := o.store.RequestRow(request.ID)
		fatal(t, problem)
		if row.State != "canceling" {
			t.Fatalf("unconfirmed native disposal was erased: %s", row.State)
		}
		return
	}
	release()
	select {
	case <-drained:
	case <-time.After(5 * time.Second):
		t.Fatal("completed cleanup did not wake shutdown")
	}
	waitUntil(t, "cancellation settles after native source release", func() bool {
		row, problem := o.store.RequestRow(request.ID)
		fatal(t, problem)
		return row.State == "canceled"
	})
	rental, problem := o.store.RentalRow(podRental)
	fatal(t, problem)
	if rental.State != "ready" {
		t.Fatal("canceling one source ended the manually reserved rental")
	}
	if calls.Load() != 1 {
		t.Fatalf("duplicate cleanup repeated an already-confirmed native release: %d", calls.Load())
	}
}
