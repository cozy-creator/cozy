package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Finding #5, rented half: canceling a transfer whose source the pod is fetching must tell
// the pod. Its Host stops fetching on ModelSourceRelease; a Creator that only forgets the
// request leaves the pod downloading and converting for nobody.
func TestCanceledRemoteSourceStopsThePod(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	selection := bytes.Repeat([]byte{0x22}, 32)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var mu sync.Mutex
	var requestID string
	declared := 0
	var released []*pb.ModelSourceReleaseCall
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if request := frame.GetModelSourceFileRequest(); request != nil {
			mu.Lock()
			if request.OperationId == requestID {
				declared++
			}
			mu.Unlock()
			return true, nil
		}
		return false, nil
	}
	pod.sourceRelease = func(_ context.Context, call *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error) {
		mu.Lock()
		released = append(released, call)
		mu.Unlock()
		return &pb.ReleaseModelSourceResult{OperationId: call.OperationId, Released: true}, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	files := []*pb.LocalModelSourceFile{{Member: "source.index.json", ObjectId: "sha256:" + verdictGoodSHA, Length: 2}}
	o := hostOwner(t, "remote-source-cancel", rentalWiring(connection, private), func(options *orchestrator.Options) {
		owner := cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, nil)
		options.ModelTransfers = sourceFixtureAccess{ModelTransferOwner: owner, files: files}
	})
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter",
		State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: "paul/minimax-h3",
		Source: "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40), SourceSelection: verdictSelection,
		SourceProfiles: map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		SourceFiles: []records.ModelTransferSourceFile{{Member: "source.index.json", SHA256: verdictGoodSHA,
			Length: 2, Header: []byte("{}")}}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}
	mu.Lock()
	id, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "remote-source-cancel", Package: "cozy/h3-package", Entrypoint: "four-lane",
		PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7", Kind: "job", Org: "paul",
		Payload: []byte("{}"), Outputs: []string{"model"}, Worker: podRental, Rental: true,
		RentalRequired: true, ModelTransfer: intent,
	})
	requestID = id
	mu.Unlock()
	fatal(t, problem)
	waitUntil(t, "the pod is given the source to fetch", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return declared > 0
	})

	// `cozy run cancel` of a run with no attempt yet reaches exactly this.
	fatal(t, o.c.CancelQueued(requestID, "cozy run cancel"))
	mu.Lock()
	sent := declared
	mu.Unlock()
	waitUntil(t, "the pod is told to stop the canceled source", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(released) > 0
	})
	mu.Lock()
	defer mu.Unlock()
	if call := released[0]; call.OperationId != requestID || !bytes.Equal(call.SourceSelectionDigest, selection) {
		t.Fatalf("release named another operation: %s", call.OperationId)
	}
	if declared != sent {
		t.Fatalf("Creator kept declaring the canceled source to the pod: %d, then %d", sent, declared)
	}
	row, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if row.State != "canceled" {
		t.Fatalf("canceled transfer reports %s", row.State)
	}
	rental, problem := o.store.RentalRow(podRental)
	fatal(t, problem)
	if rental.State != "ready" {
		t.Fatalf("canceling one transfer ended the named rental: %s", rental.State)
	}
}

// Provider access is fixture metadata. Custody, recovery and native conversion are the
// real production owners; no provider body is reachable at these deliberately inert URLs.
type sourceFixtureAccess struct {
	orchestrator.ModelTransferOwner
	files []*pb.LocalModelSourceFile
}

func (s sourceFixtureAccess) RefreshRemoteSource(context.Context, records.ModelTransferIntent) ([]orchestrator.ModelSourceCapability, *exit.Error) {
	var rows []orchestrator.ModelSourceCapability
	for _, file := range s.files {
		rows = append(rows, orchestrator.ModelSourceCapability{Member: file.Member, ObjectID: file.ObjectId, Length: int64(file.Length),
			Provider: pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE, URL: "https://source-body.invalid/" + file.Member})
	}
	return rows, nil
}
