package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
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

func (p *fakePod) ModelSourceRelease(_ context.Context, call *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.sourceRelease == nil {
		return nil, status.Error(codes.Unimplemented, "source release not provided")
	}
	return p.sourceRelease(call)
}

func TestPrivateSourceCancellationWaitsForOriginalHostRelease(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	entered, finish := make(chan struct{}), make(chan struct{})
	var finishOnce sync.Once
	release := func() { finishOnce.Do(func() { close(finish) }) }
	var calls atomic.Int64
	pod.sourceRelease = func(call *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error) {
		if call.OperationId != "req-source-release" || !bytes.Equal(call.SourceSelectionDigest, bytes.Repeat([]byte{0x22}, 32)) {
			t.Error("release changed the captured operation or source selection")
		}
		if calls.Add(1) == 1 {
			close(entered)
		}
		<-finish
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
	release()
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
}
