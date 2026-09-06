package producttest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var sourceCustodyBridge = flag.String("source-custody-host-bridge", "", "actual Go Host fixture bridge for the custody-only operator proof")

var sourceCheckpointConcurrent = flag.Bool("source-checkpoint-concurrency", false, "exercise four actual native PUT/GET bodies through a controlled R2 proxy")

var sourceCheckpointHelper = flag.String("source-checkpoint-helper", "testdata/public_source.py", "native partial-H3 source fixture and actual RuntimePreparation gRPC service")

// The source owner is the production CLI implementation, including link traversal,
// Hub grants, verification and durable acknowledgment. Its callbacks cross actual
// public Runtime gRPC handlers into TensorFS and R2; no callback invents a byte verdict.
func TestSourceCheckpointThroughPublicRuntimeAndHub(t *testing.T) {
	if *sourceCheckpointHelper == "" || *publicationHub == "" || *publicationHome == "" ||
		*publicationPython == "" || *publicationModel == "" {
		t.Skip("requires source-checkpoint-helper and publication hub/home/python/existing-model flags")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	root := t.TempDir()
	authCfg := config.Config{HubURL: *publicationHub, Home: *publicationHome}
	auth := accountauth.New(authCfg)
	client := hub.New(authCfg, "cozy-source-proof").WithTokenSource(auth)
	account, problem := client.CurrentAccount(ctx)
	fatal(t, problem)
	fixture, problem := hub.ParseRef(*publicationModel)
	fatal(t, problem)
	if fixture.Org != account.Name {
		t.Fatal("source checkpoint fixture must belong to the selected account")
	}
	_, problem = client.ModelCard(ctx, fixture)
	fatal(t, problem)

	command := exec.CommandContext(ctx, *publicationPython, *sourceCheckpointHelper, filepath.Join(root, "native"))
	if *sourceCheckpointConcurrent {
		command.Args = append(command.Args, "--transfer-proof")
	}
	command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE=1"}
	input, err := command.StdinPipe()
	must(t, err)
	output, err := command.StdoutPipe()
	must(t, err)
	command.Stderr = io.Discard // dependency errors must not print capabilities
	must(t, command.Start())
	defer func() { _ = input.Close(); cancel(); _ = command.Wait() }()
	decoder := json.NewDecoder(io.LimitReader(output, 2*pb.MaxInlineControlBytes))
	var initial struct {
		Request  string
		Software struct{ Runtime, Tensorfs, Commit string }
	}
	must(t, decoder.Decode(&initial))
	if initial.Software.Runtime == "" || initial.Software.Tensorfs == "" || len(initial.Software.Commit) != 40 {
		t.Fatal("source proof did not report published Runtime/TensorFS identity")
	}
	raw, err := base64.StdEncoding.DecodeString(initial.Request)
	must(t, err)
	preparedRequest := new(pb.PrepareModelSourceRequest)
	must(t, proto.Unmarshal(raw, preparedRequest))
	selection, err := canonical.Spell(preparedRequest.SourceSelectionDigest)
	must(t, err)
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: fixture.String(),
		Source: "hf://cozy-proof/tiny-h3@" + strings.Repeat("0", 40), SourceSelection: selection,
		SourceProfiles: map[string]string{}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}
	for _, profile := range preparedRequest.Profiles {
		intent.SourceProfiles[profile.Slot] = profile.Profile
	}
	for _, file := range preparedRequest.Files {
		header, err := os.ReadFile(file.HeaderPath)
		must(t, err)
		intent.SourceFiles = append(intent.SourceFiles, records.ModelTransferSourceFile{
			Member: file.Member, SHA256: strings.TrimPrefix(file.ObjectId, "sha256:"),
			Length: int64(file.Length), Header: header})
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	requestID := records.NewID("job")
	_, fresh, problem := store.Submit(records.Request{ID: requestID, IdemKey: requestID,
		BodyDigest: "sha256:" + strings.Repeat("c", 64), Package: "cozy/h3-package", Release: "1.0.7", PlanID: "sha256:" + strings.Repeat("35", 32),
		Worker: func() string {
			if *sourceCustodyBridge != "" {
				return "pr-11111111111111111111"
			}
			return podRental
		}(), Rental: true, RentalRequired: true,
		Entrypoint: "four-lane", State: "queued", Kind: "job", Payload: []byte("{}"),
		Outputs: "[]", WeightsOutputs: "[]", ModelTransfer: intent})
	fatal(t, problem)
	if !fresh {
		t.Fatal("source proof request was not newly recorded")
	}
	fatal(t, store.BeginModelTransferMaterialization(requestID))
	boot := "source-proof-boot"
	if *sourceCustodyBridge != "" {
		boot = "boot-1"
	}
	activeBoot := boot
	for _, file := range preparedRequest.Files {
		fatal(t, store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
			RequestID: requestID, Member: file.Member, ObjectID: file.ObjectId, WorkerBootID: boot,
			Length: int64(file.Length), CapabilityRevision: 1, State: "accepted"}))
	}
	_, err = fmt.Fprintf(input, "{\"operation_id\":%q}\n", requestID)
	must(t, err)
	var started struct {
		Address, Prepared  string
		RestoreAddress     string `json:"restore_address"`
		LoopRestoreAddress string `json:"loop_restore_address"`
	}
	must(t, decoder.Decode(&started))
	hostname, _, err := net.SplitHostPort(started.Address)
	must(t, err)
	if hostname != "127.0.0.1" {
		t.Fatal("native proof did not bind its private loopback service")
	}
	raw, err = base64.StdEncoding.DecodeString(started.Prepared)
	must(t, err)
	prepared := new(pb.PrepareModelSourceResult)
	must(t, proto.Unmarshal(raw, prepared))
	if prepared.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE || len(prepared.Checkpoints) != 1 {
		t.Fatal("native preparation did not return one actual partial checkpoint")
	}
	checkpoint := prepared.Checkpoints[0]
	head, err := canonical.Spell(checkpoint.Head.Digest)
	must(t, err)
	plan, err := canonical.Spell(checkpoint.PlanDigest)
	must(t, err)
	observed := records.ModelCheckpoint{Slot: checkpoint.Slot, HeadID: head,
		HeadLength: int64(checkpoint.Head.Length), PlanDigest: plan, Index: int64(checkpoint.Index), Bytes: int64(checkpoint.Bytes)}
	fatal(t, store.ObserveModelSourceCheckpoints(requestID, selection, boot, []records.ModelCheckpoint{observed}))
	if *sourceCustodyBridge != "" {
		preparedRequest.OperationId = requestID
		proveOperatorSourceCustody(t, ctx, root, store, auth, preparedRequest, started.Address, observed)
		return
	}
	connection, err := grpc.NewClient(started.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	must(t, err)
	defer connection.Close()
	runtime := pb.NewRuntimePreparationClient(connection)
	var pages, transfers atomic.Int64
	var gate *checkpointHTTPGate
	canceled, cancelTransfer := context.WithCancel(ctx)
	defer cancelTransfer()
	if *sourceCheckpointConcurrent {
		gate = newCheckpointHTTPGate(t, func() {
			progress, problem := store.ModelSourceProgress(requestID)
			if problem != nil || len(progress) != 1 || progress[0].Acknowledged != nil {
				t.Error("source custody was acknowledged before native HTTP bodies completed")
			}
			cancelTransfer()
		})
	}
	host := orchestrator.CheckpointHost{BootID: boot,
		Page: func(ctx context.Context, request *pb.CheckpointPageRequest) (*pb.CheckpointPageResult, *exit.Error) {
			request.RecordOwnerEpoch, request.WorkerBootId = 1, activeBoot
			answer, err := runtime.CheckpointPage(ctx, request)
			if err != nil {
				return nil, exit.Unavailablef("native checkpoint page RPC failed")
			}
			if answer.SafeCode != "" {
				return nil, exit.Named(exit.Failed, answer.SafeCode, "native checkpoint page refused")
			}
			pages.Add(1)
			return answer, nil
		},
		Transfer: func(ctx context.Context, request *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error) {
			request.RecordOwnerEpoch, request.WorkerBootId = 1, activeBoot
			if gate != nil {
				if grant := request.GetUploadGrant(); grant != nil {
					grant.Url = gate.route(http.MethodPut, grant.ObjectId, grant.Url)
				} else {
					id, _ := canonical.Spell(request.Object.Ref.Digest)
					request.Decision = &pb.CheckpointTransferRequest_DownloadUrl{DownloadUrl: gate.route(http.MethodGet, id, request.GetDownloadUrl())}
				}
			}
			answer, err := runtime.CheckpointTransfer(ctx, request)
			if err != nil {
				return nil, exit.Unavailablef("native checkpoint transfer RPC failed")
			}
			if answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED &&
				answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT && answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD {
				return nil, exit.Named(exit.Unavailable, answer.SafeCode, "native checkpoint upload refused (%s)", answer.SafeCode)
			}
			if answer.TransferId != request.TransferId || !bytes.Equal(answer.Head.Digest, request.Head.Digest) {
				return nil, exit.New(exit.Conflict, "native checkpoint transfer identity changed")
			}
			transfers.Add(1)
			return answer, nil
		}}
	owner := cli.NewModelTransferOwner(config.Config{HubURL: *publicationHub, Home: root}, store, io.Discard, auth)
	defer func() {
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if problem := owner.ReleaseCheckpoints(cleanup, requestID); problem != nil {
			t.Errorf("exact source publication cleanup failed: %s", problem.ErrName())
		}
	}()
	if gate != nil {
		ended := owner.SyncCheckpoints(canceled, requestID, host)
		if ended == nil || ended.Code != exit.Canceled {
			t.Fatalf("native transfer cancellation lost its type: %v", ended)
		}
		progress, problem := store.ModelSourceProgress(requestID)
		fatal(t, problem)
		if len(progress) != 1 || progress[0].Acknowledged != nil {
			t.Fatal("canceled Link advanced custody")
		}
	}
	if gate != nil {
		denied := owner.SyncCheckpoints(ctx, requestID, host)
		if denied == nil || denied.Code != exit.Unavailable {
			t.Fatalf("native denied PUT lost its typed failure: %v", denied)
		}
		failed, problem := store.ModelSourceProgress(requestID)
		fatal(t, problem)
		if len(failed) != 1 || failed[0].Acknowledged != nil {
			t.Fatal("failed Link advanced its custody acknowledgment")
		}
		held, problem := store.CheckpointPublications(requestID)
		fatal(t, problem)
		if len(held) == 0 {
			t.Fatal("failed Link dropped its recovery holds")
		}
		gate.mu.Lock()
		gate.refuse = false
		gate.mu.Unlock()
	}
	fatal(t, owner.SyncCheckpoints(ctx, requestID, host))
	progress, problem := store.ModelSourceProgress(requestID)
	fatal(t, problem)
	if pages.Load() == 0 || transfers.Load() == 0 || len(progress) != 1 || progress[0].Acknowledged == nil ||
		*progress[0].Acknowledged != observed {
		t.Fatal("native source upload did not reach exact durable owner acknowledgment")
	}
	if gate != nil {
		gate.mu.Lock()
		peak, moved := gate.peak, gate.moved
		gate.mu.Unlock()
		if peak != 4 || moved == 0 {
			t.Fatalf("native PUT concurrency/bytes: peak=%d moved=%d", peak, moved)
		}
		t.Logf("actual native PUT overlap: peak=%d, uploaded body bytes=%d", peak, moved)
	}

	// A replacement has no source bodies or CAS. Metadata is declared before restoring
	// the owner's acknowledged head; the same request/selection/profile remains authority.
	restored, err := grpc.NewClient(started.RestoreAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	must(t, err)
	defer restored.Close()
	runtime = pb.NewRuntimePreparationClient(restored)
	activeBoot = "source-proof-replacement-boot"
	host.BootID = activeBoot
	replacement := proto.Clone(preparedRequest).(*pb.PrepareModelSourceRequest)
	replacement.OperationId = requestID
	replacement.Checkpoints = []*pb.ModelSourceCheckpoint{checkpoint}
	for i, file := range replacement.Files {
		header, err := os.ReadFile(file.HeaderPath)
		must(t, err)
		file.Path = filepath.Join(root, "replacement-absent", fmt.Sprintf("body-%d", i))
		file.HeaderPath = filepath.Join(root, fmt.Sprintf("replacement-header-%d", i))
		must(t, os.WriteFile(file.HeaderPath, header, 0o600))
		file.Verified = false
		fatal(t, store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
			RequestID: requestID, Member: file.Member, ObjectID: file.ObjectId,
			WorkerBootID: activeBoot, Length: int64(file.Length), CapabilityRevision: 2, State: "accepted"}))
	}
	fatal(t, owner.RestoreSourceCheckpoints(ctx, requestID, host))
	recovered, err := runtime.PrepareModelSource(ctx, replacement)
	must(t, err)
	if recovered.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE ||
		len(recovered.Checkpoints) != 1 || !proto.Equal(recovered.Checkpoints[0], checkpoint) {
		t.Fatalf("replacement did not resume exact native progress without conversion: %v", recovered)
	}
	spent := map[string]bool{}
	for _, member := range recovered.SpentMembers {
		spent[member] = true
	}
	for _, file := range preparedRequest.Files {
		if file.Verified && !spent[file.Member] {
			t.Fatalf("replacement did not retire covered carrier %s", file.Member)
		}
	}
	for _, file := range replacement.Files {
		if _, err := os.Stat(file.Path); !os.IsNotExist(err) {
			t.Fatal("replacement unexpectedly gained a source body")
		}
	}
	if gate != nil {
		gate.mu.Lock()
		getPeak := gate.getPeak
		gate.mu.Unlock()
		if getPeak != 4 {
			t.Fatalf("native GET concurrency: peak=%d", getPeak)
		}
	}
	t.Logf("new boot resumed same operation %s: %d checkpointed bytes, %d spent members, zero source bodies present; partial profile remains incomplete", requestID, recovered.Checkpoints[0].Bytes, len(spent))

	held, problem := store.CheckpointPublications(requestID)
	fatal(t, problem)
	if len(held) == 0 {
		t.Fatal("source acknowledgment has no retained Hub publication")
	}
	t.Logf("public Runtime %s/TensorFS %s source checkpoint: %d native pages, %d successful native transfers, %d verified Hub holds, exact head %s acknowledged",
		initial.Software.Runtime, initial.Software.Tensorfs, pages.Load(), transfers.Load(), len(held), head)
	proveSourceReplacementLoop(t, ctx, root, store, auth, requestID, preparedRequest, checkpoint, started.LoopRestoreAddress)
}
