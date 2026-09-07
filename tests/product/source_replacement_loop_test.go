package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

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

// This extends the existing live-publication proof through Creator's actual recovery
// loop. The independent claimed peer translates native spent facts into wire statuses;
// Tensorhub's real-Host source proof separately covers that protocol implementation.
func proveSourceReplacementLoop(t *testing.T, ctx context.Context, root string, store *records.Store,
	auth *accountauth.Manager, requestID string, original *pb.PrepareModelSourceRequest,
	acknowledged *pb.ModelSourceCheckpoint, address string) {
	t.Helper()
	connection, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	must(t, err)
	defer connection.Close()
	runtime := pb.NewRuntimePreparationClient(connection)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var mu sync.Mutex
	declared := map[string]*pb.LocalModelSourceFile{}
	spent := map[string]bool{}
	bodyGrants := map[string]int{}
	prepares := 0
	restoredBeforePrepare := false
	allNeededGranted := make(chan struct{}, 1)
	pod.sourceRuntime = declaredSourceRuntime{RuntimePreparationClient: runtime, before: func() {
		mu.Lock()
		defer mu.Unlock()
		if len(declared) != len(original.Files) || (!restoredBeforePrepare && len(bodyGrants) != 0) {
			t.Error("checkpoint transfer crossed before metadata or after premature source-body grant")
		}
	}}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		switch row := frame.Msg.(type) {
		case *pb.RecordOwnerFrame_ModelSourceFileRequest:
			request := row.ModelSourceFileRequest
			if request.OperationId != requestID {
				return true, fmt.Errorf("changed source operation")
			}
			mu.Lock()
			if request.Url != "" {
				if spent[request.Member] {
					t.Errorf("covered source carrier received a body grant: %s", request.Member)
				}
				bodyGrants[request.Member]++
				if len(bodyGrants) == 2 {
					select {
					case allNeededGranted <- struct{}{}:
					default:
					}
				}
			}
			header := filepath.Join(root, fmt.Sprintf("loop-header-%d", len(declared)))
			if old := declared[request.Member]; old != nil {
				header = old.HeaderPath
			}
			if err := os.WriteFile(header, request.Header, 0o600); err != nil {
				mu.Unlock()
				return true, err
			}
			declared[request.Member] = &pb.LocalModelSourceFile{Member: request.Member, ObjectId: request.ObjectId, Length: request.Length,
				HeaderPath: header, Path: filepath.Join(root, "loop-absent", request.Member)}
			state := pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED
			if spent[request.Member] {
				state = pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_CONVERTED
			}
			mu.Unlock()
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourceFileStatus{ModelSourceFileStatus: &pb.ModelSourceFileStatus{
				RecordOwnerEpoch: request.RecordOwnerEpoch, ControlStreamEpoch: request.ControlStreamEpoch, WorkerBootId: request.WorkerBootId,
				OperationId: requestID, SourceSelectionDigest: request.SourceSelectionDigest, Member: request.Member, ObjectId: request.ObjectId,
				Length: request.Length, CapabilityRevision: request.CapabilityRevision, State: state}}})
		case *pb.RecordOwnerFrame_ModelSourcePrepareRequest:
			request := row.ModelSourcePrepareRequest
			mu.Lock()
			if len(declared) != len(original.Files) || (prepares == 0 && len(bodyGrants) != 0) {
				t.Error("source probe did not precede body grants after all declarations")
			}
			native := &pb.PrepareModelSourceRequest{OperationId: request.OperationId, SourceSelectionDigest: request.SourceSelectionDigest, Profiles: request.Profiles, Checkpoints: request.Checkpoints}
			for _, file := range original.Files {
				native.Files = append(native.Files, proto.Clone(declared[file.Member]).(*pb.LocalModelSourceFile))
			}
			mu.Unlock()
			result, err := runtime.PrepareModelSource(ctx, native)
			if err != nil {
				return true, err
			}
			if result.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE || len(result.Checkpoints) != 1 || !proto.Equal(result.Checkpoints[0], acknowledged) {
				return true, fmt.Errorf("real replacement preparation changed acknowledged progress: %s", result.SafeCode)
			}
			mu.Lock()
			prepares++
			restoredBeforePrepare = true
			for _, member := range result.SpentMembers {
				spent[member] = true
			}
			mu.Unlock()
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourcePrepared{ModelSourcePrepared: &pb.ModelSourcePrepared{
				RecordOwnerEpoch: request.RecordOwnerEpoch, ControlStreamEpoch: request.ControlStreamEpoch, WorkerBootId: request.WorkerBootId,
				OperationId: requestID, SourceSelectionDigest: request.SourceSelectionDigest, Outcome: result.Outcome, Checkpoints: result.Checkpoints}}})
		}
		return false, nil
	}
	connected, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "source-replacement-loop-"+requestID, rentalWiring(connected, private), func(options *orchestrator.Options) {
		options.Store = store
		options.Cfg.HubURL = *publicationHub
		actual := cli.NewModelTransferOwner(options.Cfg, store, options.Log, auth)
		options.ModelTransfers = sourceFixtureAccess{ModelTransferOwner: actual, files: original.Files}
	})
	defer o.close()
	_, _, problem := o.c.Reconcile()
	fatal(t, problem)
	select {
	case <-allNeededGranted:
	case <-ctx.Done():
		t.Fatalf("source replacement loop did not reach uncovered members: %v", o.c.Events())
	}
	mu.Lock()
	if !restoredBeforePrepare || prepares < 1 {
		t.Error("no real restore probe")
	}
	for _, file := range original.Files {
		if file.Verified && bodyGrants[file.Member] != 0 {
			t.Errorf("restored carrier was downloaded again: %s", file.Member)
		}
		if !file.Verified && bodyGrants[file.Member] != 1 {
			t.Errorf("uncovered carrier did not receive one grant: %s", file.Member)
		}
	}
	mu.Unlock()
	pod.mu.Lock()
	offers := len(pod.offers)
	pod.mu.Unlock()
	if offers != 0 {
		t.Fatal("partial native preparation dispatched a producer")
	}
	fatal(t, o.c.CancelQueued(requestID, "complete partial source recovery proof"))
	t.Logf("production Creator loop: new claimed boot, same request and acknowledged head, all metadata before restore probe, zero body grants for 4 covered members; only 2 unconverted carriers granted; no producer offer")
}

type declaredSourceRuntime struct {
	pb.RuntimePreparationClient
	before func()
}

func (r declaredSourceRuntime) CheckpointPage(ctx context.Context, request *pb.CheckpointPageRequest, options ...grpc.CallOption) (*pb.CheckpointPageResult, error) {
	r.before()
	return r.RuntimePreparationClient.CheckpointPage(ctx, request, options...)
}
func (r declaredSourceRuntime) CheckpointTransfer(ctx context.Context, request *pb.CheckpointTransferRequest, options ...grpc.CallOption) (*pb.CheckpointTransferStatus, error) {
	r.before()
	return r.RuntimePreparationClient.CheckpointTransfer(ctx, request, options...)
}
