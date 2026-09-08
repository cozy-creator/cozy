package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (p *fakePod) ModelSourceControl(_ context.Context, call *pb.ModelSourceControlCall) (*pb.ModelSourceControlResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.sourceControl == nil {
		return nil, status.Error(codes.Unimplemented, "source control unavailable")
	}
	return p.sourceControl(call)
}

func TestSourcePauseRecoversOrphanedMaterializationOnlyAfterHostDrain(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	entered, drained := make(chan struct{}), make(chan struct{})
	var once, finishOnce sync.Once
	finish := func() { finishOnce.Do(func() { close(drained) }) }
	pod.sourceControl = func(call *pb.ModelSourceControlCall) (*pb.ModelSourceControlResult, error) {
		if call.OperationId != "req-paused-source" || !call.Paused || call.ControlRevision != 1 {
			t.Error("source control changed its immutable paused owner")
		}
		once.Do(func() { close(entered) })
		<-drained
		return &pb.ModelSourceControlResult{OperationId: call.OperationId, SourceSelectionDigest: call.SourceSelectionDigest, ControlRevision: call.ControlRevision, Paused: true}, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	var options orchestrator.Options
	o := hostOwner(t, "source-pause-recovery", rentalWiring(connection, private), func(opt *orchestrator.Options) { options = *opt })
	t.Cleanup(finish)
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: connection.Addr, CertPath: connection.CACert, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	request, _, problem := o.store.Submit(records.Request{ID: "req-paused-source", IdemKey: "paused-source", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "local/source", Entrypoint: "prepare", Kind: "job", RetainWork: true, Worker: podRental, Rental: true, RentalRequired: true, Payload: []byte("{}"),
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Source: "hf://proof/source@" + strings.Repeat("4", 40), SourceSelection: "sha256:" + strings.Repeat("2", 64), SourceProfiles: map[string]string{"model": "hf/minimax-h3/shared-bf16/1"}, SourceFiles: []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json", SHA256: strings.Repeat("1", 64), Length: 2, Header: []byte("{}")}}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
	fatal(t, o.store.BeginModelTransferMaterialization(request.ID))
	_, problem = o.store.RequestPause(request.ID, "pause just before owner death")
	fatal(t, problem)
	o.c.Close(20 * time.Second)
	restarted, problem := orchestrator.Open(options)
	fatal(t, problem)
	t.Cleanup(func() { finish(); restarted.Close(20 * time.Second) })
	go func() { _ = restarted.Serve() }()
	_, _, problem = restarted.Reconcile()
	fatal(t, problem)
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("restart did not reconcile source pause")
	}
	row, problem := o.store.RequestRow(request.ID)
	fatal(t, problem)
	transfer, problem := o.store.ModelTransferOf(request.ID)
	fatal(t, problem)
	if row.State != "pausing" || transfer.State != "materializing" {
		t.Fatalf("pause acknowledged before native drain: %s/%s", row.State, transfer.State)
	}
	finish()
	waitUntil(t, "source pause becomes resumable after Host drainage", func() bool {
		row, problem := o.store.RequestRow(request.ID)
		fatal(t, problem)
		return row.State == "paused"
	})
	transfer, problem = o.store.ModelTransferOf(request.ID)
	fatal(t, problem)
	if transfer.State != "pending" {
		t.Fatalf("orphaned materialization remains stuck: %s", transfer.State)
	}
	attempts, problem := o.store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("source pause manufactured execution")
	}
}
